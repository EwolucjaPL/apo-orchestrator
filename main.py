import os
import re
import json
import time
import asyncio
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Request, Header, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, Response
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

# Mirror repo
KB_DIR = os.getenv("KB_DIR", "/var/apo/repo/kb")
INSTRUCTIONS_DIR = os.getenv("INSTRUCTIONS_DIR", "/var/apo/repo/instructions")

# Inne ENV
LEGAL_STATUS_DEFAULT_DATE = os.getenv("LEGAL_STATUS_DEFAULT_DATE", "1 września 2025 r.")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))
ADMIN_KEY = os.getenv("APO_ADMIN_KEY")
ALLOW_UPLOADS = os.getenv("ALLOW_UPLOADS", "false").lower() == "true"

SYNCED_COMMIT_PATH = "/var/apo/.synced_commit"

# --------------------------------------------------------------------------------------
# KLIENT LLM (OpenRouter) z retry
# --------------------------------------------------------------------------------------
_client: Optional[AsyncOpenAI] = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    return _client


async def llm_call(prompt: str, model: str = LLM_DEFAULT_MODEL, timeout: float = 45.0) -> str:
    async def _once() -> str:
        client = get_client()
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    last_err = None
    for attempt in range(3):  # retry z backoffem
        try:
            return await asyncio.wait_for(_once(), timeout=timeout)
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.5 * (2 ** attempt))
    raise HTTPException(status_code=502, detail=f"Błąd wywołania modelu AI: {last_err}")

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

PROMPT_SYNTEZA_ODPOWIEDZI = (
    "Jesteś redaktorem w kancelarii prawa oświatowego. ZŁÓŻ spójny, czytelny Markdown "
    "z dużymi odstępami między sekcjami. Schemat:\n\n"
    "1) **Weryfikacja pytania**\n"
    "2) **✅ Komunikat weryfikacji**\n"
    "3) **Podstawa prawna ⚖️** (dodaj: Stan prawny: " + LEGAL_STATUS_DEFAULT_DATE + ")\n"
    "4) **Interpretacja prawna 💡**\n"
    "5) **Procedura krok po kroku 📝**\n"
    "6) **Odpowiedź wprost 🎯**\n"
    "7) **Proaktywna sugestia 💡**\n"
    "8) **Disclaimer prawny ⚖️**\n"
    "9) **Źródła**\n\n"
    "=== KOMPONENTY ===\n"
    "[Analiza prawna]\n{analiza_prawna}\n\n"
    "[Weryfikacja cytatu]\n{wynik_weryfikacji}\n\n"
    "[Biuletyn]\n{biuletyn_informacyjny}\n"
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
# INDEKS / DENYLISTA / BIULETYN
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
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _INDEX_METADATA = data.get("metadata", {})
    _ENTRIES = data.get("entries", [])
    for e in _ENTRIES:
        e["_title_norm"] = _normalize_text(e.get("title", ""))
        e["_summary_norm"] = _normalize_text(e.get("summary", ""))
        e["_tokens"] = _tokenize(e.get("title", "") + " " + e.get("summary", ""))


def _load_denylist(path: str) -> None:
    global _DENY_PATTERNS
    if not os.path.exists(path):
        _DENY_PATTERNS = []
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _DENY_PATTERNS = list(data.get("patterns", []))


def _deny_match(title: str) -> bool:
    t = title.lower()
    for pat in _DENY_PATTERNS:
        if pat.lower().strip() in t:
            return True
    return False


def _score_entry(query_tokens: List[str], entry: IndexEntry) -> float:
    if not query_tokens:
        return 0.0
    cnt = Counter(entry.get("_tokens", []))
    score = sum(cnt[tok] for tok in query_tokens)
    if " ".join(query_tokens) in entry.get("_title_norm", ""):
        score += 2.0
    return float(score)


def search_entries(query: str, k: int = 5) -> List[SearchHit]:
    q_tokens = _tokenize(query)
    scored = [(s, e) for e in _ENTRIES if (s := _score_entry(q_tokens, e)) > 0]
    scored.sort(key=lambda x: x[0], reverse=True)

    hits: List[SearchHit] = []
    for s, e in scored:
        if _deny_match(e.get("title", "")):
            continue
        snippet = e.get("summary", "")[:700]
        if len(e.get("summary", "")) > 700:
            snippet += "…"
        hits.append(SearchHit(
            id=e.get("id", ""), title=e.get("title", ""), book=e.get("book"),
            chapter=e.get("chapter"), score=float(s), snippet=snippet
        ))
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
        return Path(SYNCED_COMMIT_PATH).read_text(encoding="utf-8").strip()
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# SANITIZER
# --------------------------------------------------------------------------------------
def sanitize_component(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"(^|\n)\s*#{1,6}\s+.*", " ", text)
    return text.replace("<|system|>", "").replace("<|assistant|>", "").replace("<|user|>", "").strip()

# --------------------------------------------------------------------------------------
# ENDPOINTY OPS
# --------------------------------------------------------------------------------------
def _method_injector(request: Request):
    root.request_method = request.method
    return None


@app.api_route("/", methods=["GET", "HEAD", "OPTIONS"])
def root(_=Depends(_method_injector)) -> Response:
    payload = {
        "status": "ok",
        "service": APP_TITLE,
        "message": "APO Gateway aktywny",
        "docs": "/docs",
    }
    if root.request_method == "HEAD":
        return Response(status_code=200)
    if root.request_method == "OPTIONS":
        return Response(status_code=204)
    return JSONResponse(payload, status_code=200)


@app.get("/live")
def live() -> Dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> Dict[str, Any]:
    disk_ok = os.access(Path(BULLETIN_PATH).parent, os.W_OK)
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
    disk_path = str(Path(BULLETIN_PATH).parent)
    return {
        "status": "ok",
        "entries": len(_ENTRIES),
        "kb_version": _INDEX_METADATA.get("version"),
        "model_default": LLM_DEFAULT_MODEL,
        "model_planner": LLM_PLANNER_MODEL,
        "has_api_key": bool(OPENROUTER_API_KEY),
        "bulletin": {
            "path": BULLETIN_PATH,
            "exists": os.path.exists(BULLETIN_PATH),
            "items": len(bulletin_json.get("items", [])),
            "updated_at": bulletin_json.get("updated_at"),
        },
        "denylist_exists": os.path.exists(DENYLIST_PATH),
        "disk": {"mount": disk_path, "writable": os.access(disk_path, os.W_OK)},
    }

# --------------------------------------------------------------------------------------
# GŁÓWNE AKCJE
# --------------------------------------------------------------------------------------
@app.post("/analyze-query")
async def analyze_query(request: QueryRequest) -> Dict[str, Any]:
    k_prompt = PROMPT_KATEGORYZACJA.format(query=request.query)
    k_raw = (await llm_call(k_prompt, model=LLM_PLANNER_MODEL)).strip().upper()
    if k_raw != "TAK":
        return {"zadania": ["ODRZUCONE_SPOZA_DOMENY"]}

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
async def knowledge_search(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=10)):
    return search_entries(q, k=k)


@app.post("/gate-and-format-response")
async def gate_and_format_response(payload: SynthesisRequest, req: Request):
    if payload.analiza_prawna == "ODRZUCONE_SPOZA_DOMENY":
        msg = (
            "### Komunikat APO\n\n"
            "Twoje pytanie wykracza poza zakres polskiego prawa oświatowego."
        )
        return PlainTextResponse(content=msg, media_type="text/markdown")

    analiza = sanitize_component(payload.analiza_prawna)
    wery = sanitize_component(payload.wynik_weryfikacji)
    biul = sanitize_component(payload.biuletyn_informacyjny)

    markdown = await llm_call(
        PROMPT_SYNTEZA_ODPOWIEDZI.format(
            analiza_prawna=analiza or "(brak danych)",
            wynik_weryfikacji=wery or "(brak danych)",
            biuletyn_informacyjny=biul or "(brak danych)",
        ),
        model=LLM_DEFAULT_MODEL,
    )
    accept = (req.headers.get("accept") or "").lower()
    if "application/json" in accept:
        return JSONResponse(content=markdown)
    return PlainTextResponse(content=markdown, media_type="text/markdown")

# --------------------------------------------------------------------------------------
# ADMIN
# --------------------------------------------------------------------------------------
@app.post("/admin/reload-index")
async def admin_reload_index():
    try:
        _load_index(KNOWLEDGE_INDEX_PATH)
        return {"ok": True, "entries": len(_ENTRIES)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/admin/reload-denylist")
async def admin_reload_denylist(x_apo_key: str | None = Header(None)):
    if not ADMIN_KEY or (x_apo_key or "") != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    _load_denylist(DENYLIST_PATH)
    return {"ok": True, "patterns": len(_DENY_PATTERNS)}

@app.post("/admin/test-disk")
def admin_test_disk(x_apo_key: str | None = Header(None)):
    if not ADMIN_KEY or (x_apo_key or "") != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    tf = Path(BULLETIN_PATH).parent / f"__apo_write_test_{int(time.time())}.txt"
    tf.write_text("ok", encoding="utf-8")
    return {"ok": True, "path": str(tf)}

# --------------------------------------------------------------------------------------
# DEV ENTRYPOINT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    for p, fallback in [
        (BULLETIN_PATH, {"items": [], "updated_at": None}),
        (DENYLIST_PATH, {"patterns": []}),
        (KNOWLEDGE_INDEX_PATH, {"metadata": {}, "entries": []}),
    ]:
        _ensure_dirs_for(p)
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(fallback, f, ensure_ascii=False, indent=2)

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))