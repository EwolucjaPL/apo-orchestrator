import os
import re
import json
import time
import asyncio
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Request, Header
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from openai import AsyncOpenAI  # OpenRouter (Async)

# --------------------------------------------------------------------------------------
# KONFIGURACJA / ENV
# --------------------------------------------------------------------------------------
load_dotenv()

APP_TITLE = "APO Gateway"
APP_DESC = "Gateway + mini-RAG dla prawa oświatowego (APO)"
app = FastAPI(title=APP_TITLE, description=APP_DESC)

# Klucze / modele
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Brak OPENROUTER_API_KEY w środowisku")

LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-4o")
LLM_PLANNER_MODEL = os.getenv("LLM_PLANNER_MODEL", "mistralai/mistral-7b-instruct:free")

# Ścieżki – domyślnie na Persistent Disk
KNOWLEDGE_INDEX_PATH = os.getenv("KNOWLEDGE_INDEX_PATH", "/var/apo/data/index.json")
BULLETIN_PATH = os.getenv("BULLETIN_PATH", "/var/apo/data/bulletin.json")
DENYLIST_PATH = os.getenv("APO_KB_DENYLIST_PATH", "/var/apo/data/denylist.json")

# Mirror repo (kopiowane przez render.yaml na /var/apo/repo)
KB_DIR = os.getenv("KB_DIR", "/var/apo/repo/kb")
INSTRUCTIONS_DIR = os.getenv("INSTRUCTIONS_DIR", "/var/apo/repo/instructions")

# Inne ENV
LEGAL_STATUS_DEFAULT_DATE = os.getenv("LEGAL_STATUS_DEFAULT_DATE", "1 września 2025 r.")
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://0.0.0.0:8000")
ADMIN_KEY = os.getenv("APO_ADMIN_KEY")
ALLOW_UPLOADS = os.getenv("ALLOW_UPLOADS", "false").lower() == "true"

# Commit mirrora (zapisuje go startCommand w render.yaml, jeśli RENDER_GIT_COMMIT jest dostępny)
SYNCED_COMMIT_PATH = "/var/apo/.synced_commit"

# --------------------------------------------------------------------------------------
# KLIENT LLM (OpenRouter)
# --------------------------------------------------------------------------------------
_client: Optional[AsyncOpenAI] = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    return _client


async def llm_call(prompt: str, model: str = LLM_DEFAULT_MODEL, timeout: float = 45.0) -> str:
    async def _inner() -> str:
        client = get_client()
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    try:
        return await asyncio.wait_for(_inner(), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout podczas wywołania modelu AI")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Błąd wywołania modelu AI: {e}")

# --------------------------------------------------------------------------------------
# PROMPTY
# --------------------------------------------------------------------------------------
PROMPT_KATEGORYZACJA = (
    "Twoje jedyne zadanie: oceń, czy poniższe pytanie dotyczy wyłącznie polskiego prawa oświatowego. "
    "Domena zawiera m.in.: Kartę Nauczyciela, prawo oświatowe, nadzór pedagogiczny, uczniów, dyrektorów. "
    "Poza domeną: podatki, budowlanka, prawo autorskie, zamówienia publiczne, itp. "
    "Odpowiedz tylko 'TAK' lub 'NIE'.\n\nPytanie: \"{query}\""
)

PROMPT_ANALIZA_ZAPYTANIA = (
    "Rozbij zapytanie użytkownika na prosty plan działań w JSON. "
    "Dopuszczalne zadania: 'analiza_prawna', 'weryfikacja_cytatu', 'biuletyn_informacyjny'. "
    "Zwróć wyłącznie poprawny JSON z kluczem 'zadania' (lista stringów). Bez komentarzy.\n\n"
    "Zapytanie: \"{query}\""
)

# 9-sekcyjny szkielet odpowiedzi — edytor scali komponenty do czytelnego Markdown
PROMPT_SYNTEZA_ODPOWIEDZI = (
    "Jesteś redaktorem w kancelarii prawa oświatowego. ZŁÓŻ spójny, bardzo czytelny Markdown "
    "z wyraźnymi nagłówkami i **dużymi odstępami między sekcjami**. Użyj poniższego schematu "
    "i uzupełnij danymi z komponentów. Język: polski.\n\n"
    "=== KOMPONENTY ===\n"
    "[Analiza prawna]\n{analiza_prawna}\n\n"
    "[Weryfikacja cytatu]\n{wynik_weryfikacji}\n\n"
    "[Biuletyn (ostatnie zmiany)]\n{biuletyn_informacyjny}\n\n"
    "=== WYMAGANIA FORMATU ===\n"
    "1) **Weryfikacja pytania** — krótka parafraza intencji.\n"
    "2) **✅ Komunikat weryfikacji** — jednozdaniowe potwierdzenie zakresu.\n"
    "3) **Podstawa prawna ⚖️** — punktowo, akty/artykuły; dodaj: 'Stan prawny: "
    + LEGAL_STATUS_DEFAULT_DATE
    + "'.\n"
    "4) **Interpretacja prawna 💡** — 2–3 akapity, prosto.\n"
    "5) **Procedura krok po kroku 📝** — lista numerowana, max 7 kroków.\n"
    "6) **Odpowiedź wprost 🎯** — jednozdaniowe rozstrzygnięcie.\n"
    "7) **Proaktywna sugestia 💡** — 2–3 praktyczne wskazówki.\n"
    "8) **Disclaimer prawny ⚖️** — formuła ogólna, z datą stanu prawnego.\n"
    "9) **Źródła** — wypunktuj; jeśli brak konkretnych, wskaż 'akty prawne wskazane wyżej'.\n"
)

# --------------------------------------------------------------------------------------
# MODELE DANYCH
# --------------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)


class PlanZadania(BaseModel):
    zadania: List[str]


class SynthesisRequest(BaseModel):
    analiza_prawna: Optional[str] = None
    wynik_weryfikacji: Optional[str] = None
    biuletyn_informacyjny: Optional[str] = None


class SearchHit(BaseModel):
    id: str
    title: str
    book: Optional[str] = None
    chapter: Optional[str] = None
    score: float
    snippet: str


# --------------------------------------------------------------------------------------
# DANE: INDEKS, DENYLISTA, BIULETYN
# --------------------------------------------------------------------------------------
IndexEntry = Dict[str, Any]
_INDEX_METADATA: Dict[str, Any] = {}
_ENTRIES: List[IndexEntry] = []
_DENY_PATTERNS: List[str] = []


def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def _tokenize(t: str) -> List[str]:
    t = _normalize_text(t)
    t = re.sub(r"[^\wąćęłńóśźż\s]", " ", t)
    return [tok for tok in t.split() if len(tok) > 2]


def _load_index(path: str) -> None:
    global _INDEX_METADATA, _ENTRIES
    if not os.path.exists(path):
        raise RuntimeError(f"Nie znaleziono indeksu wiedzy pod ścieżką: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _INDEX_METADATA = data.get("metadata", {})
    _ENTRIES = data.get("entries", [])
    for e in _ENTRIES:
        e["_title_norm"] = _normalize_text(e.get("title", ""))
        e["_summary_norm"] = _normalize_text(e.get("summary", ""))
        e["_tokens"] = _tokenize(e.get("title", "") + " " + e.get("summary", ""))


def _load_denylist(path: str) -> None:
    """Wczytuje wzorce tytułów/źródeł, których NIE wolno ujawniać."""
    global _DENY_PATTERNS
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _DENY_PATTERNS = list(data.get("patterns", []))
        else:
            _DENY_PATTERNS = []
    except Exception:
        _DENY_PATTERNS = []


def _deny_match(title: str) -> bool:
    """True jeśli tytuł trafia w denylistę (prosty contains – w razie potrzeby rozbuduj do fnmatch)."""
    t = title.lower()
    for pat in _DENY_PATTERNS:
        p = pat.lower().strip()
        if not p:
            continue
        if p == "*":
            return True
        if p in t:
            return True
    return False


def _score_entry(query_tokens: List[str], entry: IndexEntry) -> float:
    if not query_tokens:
        return 0.0
    cnt = Counter(entry.get("_tokens", []))
    score = sum(cnt[tok] for tok in query_tokens)
    title = entry.get("_title_norm", "")
    q_join = " ".join(query_tokens)
    if q_join and q_join in title:
        score += 2.0
    return float(score)


def search_entries(query: str, k: int = 5) -> List[SearchHit]:
    q_tokens = _tokenize(query)
    scored: List[Tuple[float, IndexEntry]] = []
    for e in _ENTRIES:
        s = _score_entry(q_tokens, e)
        if s > 0:
            scored.append((s, e))
    scored.sort(key=lambda x: x[0], reverse=True)

    hits: List[SearchHit] = []
    for s, e in scored[: k * 2]:  # bierz trochę więcej i przefiltruj denylistę
        title = e.get("title", "")
        if _deny_match(title):
            continue
        snippet = e.get("summary", "")
        snippet = snippet[:700] + ("…" if len(snippet) > 700 else "")
        hits.append(
            SearchHit(
                id=e.get("id", ""),
                title=title,
                book=e.get("book"),
                chapter=e.get("chapter"),
                score=float(s),
                snippet=snippet,
            )
        )
        if len(hits) >= k:
            break
    return hits


def _safe_read_json(path: str, fallback: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return fallback


def _ensure_dirs_for(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_synced_commit() -> Optional[str]:
    try:
        if os.path.exists(SYNCED_COMMIT_PATH):
            return Path(SYNCED_COMMIT_PATH).read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


# Inicjalizacja przy starcie
_load_index(KNOWLEDGE_INDEX_PATH)
_load_denylist(DENYLIST_PATH)

# Log diagnostyczny ścieżek
print(f"[APO] KB_DIR={KB_DIR}")
print(f"[APO] INSTRUCTIONS_DIR={INSTRUCTIONS_DIR}")
print(f"[APO] INDEX_PATH={KNOWLEDGE_INDEX_PATH}")
print(f"[APO] BULLETIN_PATH={BULLETIN_PATH}")
print(f"[APO] DENYLIST_PATH={DENYLIST_PATH}")
print(f"[APO] SYNCED_COMMIT={_read_synced_commit() or '-'}")

# --------------------------------------------------------------------------------------
# PUBLICZNE ŹRÓDŁA: odświeżanie biuletynu (ISAP/MEN itp.)
# --------------------------------------------------------------------------------------
try:
    from public_sources import fetch_public_updates_last_year, refresh_public_sources
except Exception:
    # Minimalne stuby, gdy modułu brak
    async def fetch_public_updates_last_year() -> List[Dict[str, Any]]:
        return []

    async def refresh_public_sources(bulletin_path: str) -> Dict[str, Any]:
        _ensure_dirs_for(bulletin_path)
        now = datetime.utcnow().isoformat() + "Z"
        data = {"items": [], "updated_at": now}
        with open(bulletin_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"ok": True, "items": 0, "updated_at": now}

# --------------------------------------------------------------------------------------
# SANITY / FILTRY
# --------------------------------------------------------------------------------------
def sanitize_component(text: Optional[str]) -> str:
    if not text:
        return ""
    # usuń bloki codefence, ale nie połykaj treści między nimi
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # usuń nagłówki markdown, by uniknąć duplikacji H1/H2
    text = re.sub(r"(^|\n)\s*#{1,6}\s+.*", " ", text)
    # usuń role-tags
    text = text.replace("<|system|>", "").replace("<|assistant|>", "").replace("<|user|>", "")
    return re.sub(r"\s{2,}", " ", text).strip()

# --------------------------------------------------------------------------------------
# ENDPOINTY OPS
# --------------------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": APP_TITLE,
        "message": "APO Gateway aktywny",
        "docs": "/docs",
    }


@app.get("/live")
def live() -> Dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> Dict[str, Any]:
    # sprawdź możliwość stworzenia katalogu dla BULLETIN_PATH
    try:
        _ensure_dirs_for(BULLETIN_PATH)
        disk_ok = True
    except Exception:
        disk_ok = False
    return {
        "ready": bool(_ENTRIES) and bool(OPENROUTER_API_KEY) and disk_ok,
        "kb_loaded": bool(_ENTRIES),
        "has_api_key": bool(OPENROUTER_API_KEY),
        "disk_ok": disk_ok,
        "commit": _read_synced_commit(),
    }


@app.get("/health")
def health_check() -> Dict[str, Any]:
    bulletin_json = _safe_read_json(BULLETIN_PATH, {"items": [], "updated_at": None})
    return {
        "status": "ok",
        "entries": len(_ENTRIES),
        "kb_version": _INDEX_METADATA.get("version"),
        "model_default": LLM_DEFAULT_MODEL,
        "model_planner": LLM_PLANNER_MODEL,
        "has_api_key": bool(OPENROUTER_API_KEY),
        "bulletin_exists": os.path.exists(BULLETIN_PATH),
        "bulletin_items": len(bulletin_json.get("items", [])),
        "bulletin_updated_at": bulletin_json.get("updated_at"),
        "denylist_exists": os.path.exists(DENYLIST_PATH),
        "paths": {
            "index": KNOWLEDGE_INDEX_PATH,
            "bulletin": BULLETIN_PATH,
            "denylist": DENYLIST_PATH,
            "kb_dir": KB_DIR,
            "instructions_dir": INSTRUCTIONS_DIR,
        },
    }

# --------------------------------------------------------------------------------------
# GŁÓWNE AKCJE: analyze-query, knowledge/search, gate-and-format-response
# --------------------------------------------------------------------------------------
@app.post("/analyze-query")
async def analyze_query(request: QueryRequest) -> Dict[str, Any]:
    # 0) Kategoryzacja (TAK/NIE)
    k_prompt = PROMPT_KATEGORYZACJA.format(query=request.query)
    k_raw = (await llm_call(k_prompt, model=LLM_PLANNER_MODEL)).strip().upper()
    k_value = "TAK" if k_raw == "TAK" else ("NIE" if k_raw == "NIE" else "TAK")
    if k_value == "NIE":
        return {"zadania": ["ODRZUCONE_SPOZA_DOMENY"]}

    # 1) Plan JSON
    p_prompt = PROMPT_ANALIZA_ZAPYTANIA.format(query=request.query)
    p_raw = await llm_call(p_prompt, model=LLM_PLANNER_MODEL)
    m = re.search(r"\{[\s\S]*\}", p_raw)
    plan_json = m.group(0) if m else '{"zadania":["analiza_prawna"]}'
    try:
        plan = PlanZadania.model_validate_json(plan_json)
    except ValidationError:
        plan = PlanZadania(zadania=["analiza_prawna"])

    return plan.model_dump()


@app.get("/knowledge/search", response_model=List[SearchHit])
async def knowledge_search(
    q: str = Query(..., min_length=2),
    k: int = Query(5, ge=1, le=10),
):
    """Zwrot dopasowań z KB — bez ujawniania pozycji z denylisty."""
    return search_entries(q, k=k)


@app.post("/gate-and-format-response")
async def gate_and_format_response(payload: SynthesisRequest, req: Request):
    """
    Zwraca końcową odpowiedź APO w Markdown.
    Jeśli nagłówek Accept zawiera 'application/json', pakujemy string do JSON.
    """
    # Sentinel „poza domeną”
    if payload.analiza_prawna == "ODRZUCONE_SPOZA_DOMENY":
        content = (
            "### Komunikat APO\n\n"
            "Twoje pytanie wykracza poza zakres polskiego prawa oświatowego. "
            "Nie mogę przygotować odpowiedzi merytorycznej. "
            "Jeżeli chcesz, przekształcę pytanie na kontekst prawa oświatowego."
        )
        accept = (req.headers.get("accept") or "").lower()
        if "application/json" in accept:
            return JSONResponse(content=content)
        return PlainTextResponse(content=content, media_type="text/markdown; charset=utf-8")

    # Dodatkowy bezpiecznik: jeśli to metapytanie o „bazę wiedzy”, nie ujawniamy list tytułów.
    q_lower = (payload.analiza_prawna or "").lower()
    if any(p in q_lower for p in ["baza wiedzy", "jakie masz źródła", "podaj źródła", "podaj swoją bazę"]):
        neutral = (
            "### Informacja o zakresie wiedzy\n\n"
            "Korzystam z wewnętrznej bazy materiałów i aktów prawnych dotyczących polskiego prawa oświatowego. "
            "Nie udostępniam listy tytułów ani szczegółowych spisów treści. Mogę za to wskazać ramy prawne, "
            "podstawowe akty i ogólny zakres tematyczny, a w odpowiedziach przywołuję odpowiednie podstawy prawne."
        )
        accept = (req.headers.get("accept") or "").lower()
        if "application/json" in accept:
            return JSONResponse(content=neutral)
        return PlainTextResponse(content=neutral, media_type="text/markdown; charset=utf-8")

    analiza = sanitize_component(payload.analiza_prawna)
    wery = sanitize_component(payload.wynik_weryfikacji)
    biul = sanitize_component(payload.biuletyn_informacyjny)

    markdown = await llm_call(
        PROMPT_SYNTEZA_ODPOWIEDZI.format(
            analiza_prawna=analiza or "(brak danych z KB – zaznacz ostrożność)",
            wynik_weryfikacji=wery or "(brak danych)",
            biuletyn_informacyjny=biul or "(brak danych)",
        ),
        model=LLM_DEFAULT_MODEL,
    )

    # Negocjacja content-type
    accept = (req.headers.get("accept") or "").lower()
    if "application/json" in accept:
        return JSONResponse(content=markdown)  # JSON string
    return PlainTextResponse(content=markdown, media_type="text/markdown; charset=utf-8")

# --------------------------------------------------------------------------------------
# ADMIN
# --------------------------------------------------------------------------------------
@app.post("/admin/refresh-public-sources")
async def admin_refresh_public_sources(request: Request):
    if not ADMIN_KEY or request.headers.get("X-APO-Key") != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        _ensure_dirs_for(BULLETIN_PATH)
        res = await refresh_public_sources(BULLETIN_PATH)  # z public_sources.py
        return res
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/admin/reload-index")
async def admin_reload_index():
    try:
        _load_index(KNOWLEDGE_INDEX_PATH)
        return {"ok": True, "entries": len(_ENTRIES)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/admin/upload-index")
async def admin_upload_index(file: UploadFile = File(...)):
    if not ALLOW_UPLOADS:
        raise HTTPException(status_code=403, detail="Uploads disabled")
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
        _ensure_dirs_for(KNOWLEDGE_INDEX_PATH)
        with open(KNOWLEDGE_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        _load_index(KNOWLEDGE_INDEX_PATH)
        return {"ok": True, "entries": len(_ENTRIES)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/admin/reload-denylist")
async def admin_reload_denylist(x_apo_key: str | None = Header(None)):
    if not ADMIN_KEY or (x_apo_key or "") != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        _load_denylist(DENYLIST_PATH)
        return {"ok": True, "patterns": len(_DENY_PATTERNS)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/admin/test-disk")
def admin_test_disk(x_apo_key: str | None = Header(None)):
    if not ADMIN_KEY or (x_apo_key or "") != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        target_dir = Path(BULLETIN_PATH).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        tf = target_dir / f"__apo_write_test_{int(time.time())}.txt"
        tf.write_text("ok", encoding="utf-8")
        head = sorted(os.listdir(target_dir))[:20]
        return {"ok": True, "path": str(tf), "ls_head": head}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# --------------------------------------------------------------------------------------
# DEV ENTRYPOINT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Łagodne auto-init plików na starcie (przy lokalnym dev)
    try:
        for p, fallback in [
            (BULLETIN_PATH, {"items": [], "updated_at": None}),
            (DENYLIST_PATH, {"patterns": []}),
            (KNOWLEDGE_INDEX_PATH, {"metadata": {}, "entries": []}),
        ]:
            _ensure_dirs_for(p)
            if not os.path.exists(p):
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(fallback, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))