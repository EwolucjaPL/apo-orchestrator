import os
import re
import json
import time
import asyncio
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query, Header, UploadFile, File, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

try:
    import orjson
    _json_loads = orjson.loads
    def _json_dumps(obj: Any) -> str:
        return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode("utf-8")
    ORJSON_OK = True
except Exception:
    _json_loads = json.loads
    def _json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    ORJSON_OK = False

# Optional OpenRouter client
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

# Opcjonalny moduł do pobierania źródeł publicznych (ISAP/MEN itp.)
try:
    import public_sources  # lokalny plik public_sources.py (opcjonalny)
except Exception:
    public_sources = None

load_dotenv()

# ------------------------------------------------------------------------------
# KONFIGURACJA ŚCIEŻEK I MODELI
# ------------------------------------------------------------------------------
APP_NAME = "APO Gateway"
APP_DESC = "Gateway + mini-RAG dla prawa oświatowego"

# Mirror repo (tylko do odczytu)
KB_DIR = os.getenv("KB_DIR", "/var/apo/repo/kb")
INSTRUCTIONS_DIR = os.getenv("INSTRUCTIONS_DIR", "/var/apo/repo/instructions")

# Dane dynamiczne (do zapisu/odczytu)
KNOWLEDGE_INDEX_PATH = os.getenv("KNOWLEDGE_INDEX_PATH", "/var/apo/data/index.json")
BULLETIN_PATH = os.getenv("BULLETIN_PATH", "/var/apo/data/bulletin.json")
DENYLIST_PATH = os.getenv("APO_KB_DENYLIST_PATH", "/var/apo/data/denylist.json")

# Inne ENV
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-4o")
LLM_PLANNER_MODEL = os.getenv("LLM_PLANNER_MODEL", "mistralai/mistral-7b-instruct:free")
LEGAL_STATUS_DEFAULT_DATE = os.getenv("LEGAL_STATUS_DEFAULT_DATE", "1 września 2025 r.")
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")

ADMIN_KEY = os.getenv("APO_ADMIN_KEY", "")

MAX_RETURN_SNIPPETS = 5
APP_START_TS = time.time()

# ------------------------------------------------------------------------------
# APLIKACJA
# ------------------------------------------------------------------------------
app = FastAPI(title=APP_NAME, description=APP_DESC)

# ------------------------------------------------------------------------------
# MODELE DANYCH
# ------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)

class PlanZadania(BaseModel):
    zadania: List[str] = Field(..., min_length=1)

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

# ------------------------------------------------------------------------------
# STORAGE / ŁADOWANIE INDEKSU I DENYLISTY
# ------------------------------------------------------------------------------
IndexEntry = Dict[str, Any]
_ENTRIES: List[IndexEntry] = []
_INDEX_METADATA: Dict[str, Any] = {}
_DENY_PATTERNS: List[str] = []

def _ensure_file(path: str, default_obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(_json_dumps(default_obj))

def _read_json(path: str, default_obj: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _json_loads(f.read())
    except Exception:
        _ensure_file(path, default_obj)
        return default_obj

def _write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_json_dumps(obj))
    os.replace(tmp, path)

def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()

def _tokenize(t: str) -> List[str]:
    t = _normalize_text(t)
    t = re.sub(r"[^\wąćęłńóśźż\s]", " ", t)
    return [tok for tok in t.split() if len(tok) > 2]

def _load_index() -> None:
    global _ENTRIES, _INDEX_METADATA
    data = _read_json(KNOWLEDGE_INDEX_PATH, {"metadata": {}, "entries": []})
    _INDEX_METADATA = data.get("metadata", {})
    _ENTRIES = data.get("entries", []) or []
    for e in _ENTRIES:
        e["_title_norm"] = _normalize_text(e.get("title", ""))
        base = (e.get("title", "") + " " + e.get("summary", "")) or ""
        e["_tokens"] = _tokenize(base)

def _load_denylist() -> None:
    global _DENY_PATTERNS
    data = _read_json(DENYLIST_PATH, {"patterns": []})
    _DENY_PATTERNS = data.get("patterns", []) or []

def _is_denied(title: str) -> bool:
    if not title:
        return False
    for pat in _DENY_PATTERNS:
        try:
            if re.search(pat, title, flags=re.I):
                return True
        except re.error:
            # zły regex – ignorujemy
            continue
    return False

# Ładowanie na starcie
_load_index()
_load_denylist()
_ = _read_json(BULLETIN_PATH, {"items": [], "updated_at": None})  # ensure exists

# ------------------------------------------------------------------------------
# PROSTE WYSZUKIWANIE (BM25-lite)
# ------------------------------------------------------------------------------
def _score_entry(query_tokens: List[str], entry: IndexEntry) -> float:
    if not query_tokens:
        return 0.0
    tokens = entry.get("_tokens", [])
    # prosty tf
    s = sum(1 for t in tokens if t in query_tokens)
    # bonus, gdy fraza w tytule
    if " ".join(query_tokens) in entry.get("_title_norm", ""):
        s += 2.0
    return float(s)

def search_entries(query: str, k: int = MAX_RETURN_SNIPPETS) -> List[SearchHit]:
    q_tokens = _tokenize(query)
    scored: List[Tuple[float, IndexEntry]] = []
    for e in _ENTRIES:
        sc = _score_entry(q_tokens, e)
        if sc > 0:
            scored.append((sc, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits: List[SearchHit] = []
    for sc, e in scored[:k]:
        title = e.get("title", "")
        display_title = "[Zastrzeżone źródło]" if _is_denied(title) else title
        snippet = (e.get("summary", "") or "")
        if len(snippet) > 800:
            snippet = snippet[:800] + "…"
        hits.append(
            SearchHit(
                id=e.get("id", ""),
                title=display_title,
                book=e.get("book"),
                chapter=e.get("chapter"),
                score=float(sc),
                snippet=snippet,
            )
        )
    return hits

# ------------------------------------------------------------------------------
# LLM KLIENT
# ------------------------------------------------------------------------------
_client: Optional[AsyncOpenAI] = None

def _get_client() -> Optional[AsyncOpenAI]:
    global _client
    if not OPENROUTER_API_KEY or AsyncOpenAI is None:
        return None
    if _client is None:
        _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    return _client

async def llm_call(prompt: str, model: str, timeout: float = 40.0) -> str:
    client = _get_client()
    if client is None:
        # środowisko bez klucza – zwracamy kontrolowany fallback
        return "(Uwaga: środowisko bez klucza API – odpowiedź syntetyczna na podstawie lokalnych reguł.)"
    async def _inner():
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content
    try:
        return await asyncio.wait_for(_inner(), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout podczas wywołania modelu")
    except Exception as e:
        # kontrolowany błąd bez „connector error”
        raise HTTPException(status_code=502, detail=f"Błąd wywołania modelu: {e}")

# ------------------------------------------------------------------------------
# PROMPTY I FORMATOWANIE 9-SEKCYJNE
# ------------------------------------------------------------------------------
PROMPT_KATEGORYZACJA = (
    "Twoje zadanie: odpowiedz wyłącznie 'TAK' lub 'NIE' — czy poniższe pytanie dotyczy wyłącznie polskiego prawa oświatowego "
    "(Karta Nauczyciela, Prawo oświatowe, nadzór pedagogiczny, kompetencje dyrektora/rady, prawa ucznia)? "
    "Inne dziedziny (podatki, budowlane, ZUS, VAT) = 'NIE'.\n\nPytanie: {q}"
)

PROMPT_ANALIZA = (
    "Zdekomponuj pytanie na prosty plan JSON pod kluczem 'zadania'. Dozwolone: 'analiza_prawna', 'weryfikacja_cytatu', 'biuletyn_informacyjny'. "
    "Zwróć wyłącznie poprawny JSON bez wyjaśnień.\n\nPytanie: {q}"
)

def format_response_9_sections(
    parafraza: str,
    potwierdzenie: str,
    podstawa: List[str],
    interpretacja: str,
    procedura: List[str],
    odpowiedz_wprost: str,
    sugestie: List[str],
    disclaimer: str,
    oferta: str
) -> str:
    # Większe odstępy: pusty wiersz nad i pod sekcjami, wyraźne nagłówki
    lines = []
    lines.append("## Weryfikacja pytania\n")
    lines.append(parafraza.strip() + "\n\n")
    lines.append("## Komunikat weryfikacji\n")
    lines.append(f"✅ {potwierdzenie.strip()}\n\n")
    lines.append("## Podstawa prawna ⚖️\n")
    if podstawa:
        for p in podstawa:
            lines.append(f"- {p}")
        lines.append(f"\nStan prawny: {LEGAL_STATUS_DEFAULT_DATE}\n\n")
    else:
        lines.append(f"- (brak wskazanych podstaw)\n\nStan prawny: {LEGAL_STATUS_DEFAULT_DATE}\n\n")
    lines.append("## Interpretacja prawna 💡\n")
    lines.append(interpretacja.strip() + "\n\n")
    lines.append("## Procedura krok po kroku 📝\n")
    if procedura:
        for i, step in enumerate(procedura, 1):
            lines.append(f"{i}. {step}")
        lines.append("\n")
    else:
        lines.append("(brak)\n\n")
    lines.append("## Odpowiedź wprost 🎯\n")
    lines.append(odpowiedz_wprost.strip() + "\n\n")
    lines.append("## Proaktywna sugestia 💡\n")
    if sugestie:
        for s in sugestie:
            lines.append(f"- {s}")
        lines.append("\n")
    else:
        lines.append("(brak)\n\n")
    lines.append("## Disclaimer prawny ⚖️\n")
    lines.append(disclaimer.strip() + "\n\n")
    lines.append("## Dodatkowa oferta wsparcia 🤝\n")
    lines.append(oferta.strip() + "\n")
    return "\n".join(lines)

PROMPT_SYNTEZA = (
    "Złóż końcową odpowiedź dla dyrektora/nauczyciela w ustrukturyzowanym formacie (9 sekcji: "
    "Weryfikacja pytania; Komunikat weryfikacji; Podstawa prawna; Interpretacja; Procedura; Odpowiedź wprost; "
    "Proaktywna sugestia; Disclaimer; Dodatkowa oferta). Użyj prostego, klarownego języka po polsku. "
    "Jeśli któryś komponent pusty, oprzyj się na pozostałych i ogólnych zasadach (oznacz ostrożność). "
    "Nie ujawniaj tytułów zastrzeżonych źródeł. "
    "Komponenty do wykorzystania poniżej.\n\n"
    "[Analiza prawna]\n{A}\n\n[Weryfikacja cytatu]\n{W}\n\n[Biuletyn]\n{B}\n"
)

def sanitize_component(text: Optional[str]) -> str:
    if not text:
        return ""
    # usuń code fences, role tagi, nagłówki systemowe
    t = re.sub(r"```.+?```", " ", text, flags=re.S)
    t = t.replace("<|system|>", "").replace("<|assistant|>", "").replace("<|user|>", "")
    t = re.sub(r"(^|\n)\s*#{1,6}.*", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()

# ------------------------------------------------------------------------------
# RATE LIMIT (prost y, per-proces)
# ------------------------------------------------------------------------------
_last_minute_calls = []
def _rate_limit():
    if not RATE_LIMIT_ENABLED:
        return
    now = time.time()
    # czyść okno 60s
    while _last_minute_calls and now - _last_minute_calls[0] > 60:
        _last_minute_calls.pop(0)
    if len(_last_minute_calls) >= RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="Zbyt wiele zapytań, spróbuj za chwilę.")
    _last_minute_calls.append(now)

# ------------------------------------------------------------------------------
# GUARDS
# ------------------------------------------------------------------------------
def require_admin(x_apo_key: Optional[str] = Header(None)) -> None:
    if not ADMIN_KEY or x_apo_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Brak uprawnień.")

# ------------------------------------------------------------------------------
# ENDPOINTY OPS
# ------------------------------------------------------------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return f"{APP_NAME} is up. See /docs"

@app.get("/live")
def live():
    return {"status": "alive", "uptime_s": int(time.time() - APP_START_TS)}

@app.get("/ready")
def ready():
    disk_ok = os.path.isdir("/var/apo")
    return {
        "ready": bool(_ENTRIES is not None),
        "kb_loaded": bool(len(_ENTRIES) >= 0),
        "has_api_key": bool(OPENROUTER_API_KEY),
        "disk_ok": disk_ok,
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "entries": len(_ENTRIES),
        "kb_version": _INDEX_METADATA.get("version"),
        "model_default": LLM_DEFAULT_MODEL,
        "model_planner": LLM_PLANNER_MODEL,
        "has_api_key": bool(OPENROUTER_API_KEY),
        "bm25": True,
        "rate_limit_rpm": RATE_LIMIT_RPM,
        "orjson": ORJSON_OK,
    }

# ------------------------------------------------------------------------------
# ANALYZE / SEARCH / SYNTH
# ------------------------------------------------------------------------------
@app.post("/analyze-query")
async def analyze_query(req: QueryRequest) -> Dict[str, Any]:
    _rate_limit()
    # Kategoryzacja
    k_prompt = PROMPT_KATEGORYZACJA.format(q=req.query)
    k_raw = (await llm_call(k_prompt, model=LLM_PLANNER_MODEL)).strip().upper()
    domain = "TAK" if k_raw == "TAK" else ("NIE" if k_raw == "NIE" else "TAK")
    if domain == "NIE":
        return {"zadania": ["ODRZUCONE_SPOZA_DOMENY"]}

    # Plan
    p_prompt = PROMPT_ANALIZA.format(q=req.query)
    p_raw = await llm_call(p_prompt, model=LLM_PLANNER_MODEL)
    m = re.search(r"\{[\s\S]*\}", p_raw)
    plan_json = m.group(0) if m else '{"zadania":["analiza_prawna"]}'
    try:
        plan = PlanZadania.model_validate_json(plan_json)
    except ValidationError:
        plan = PlanZadania(zadania=["analiza_prawna"])
    return plan.model_dump()

@app.get("/knowledge/search", response_model=List[SearchHit])
async def knowledge_search(q: str = Query(..., min_length=2), k: int = Query(MAX_RETURN_SNIPPETS, ge=1, le=10)):
    _rate_limit()
    # Nie ujawniamy surowych tytułów objętych denylistą
    return search_entries(q, k=k)

@app.post("/gate-and-format-response", response_class=PlainTextResponse)
async def gate_and_format_response(req: SynthesisRequest) -> str:
    _rate_limit()

    if req.analiza_prawna == "ODRZUCONE_SPOZA_DOMENY":
        # jasny komunikat out-of-domain (bez „connector error”)
        return format_response_9_sections(
            parafraza="Pytanie nie dotyczy polskiego prawa oświatowego.",
            potwierdzenie="Pytanie jest poza domeną asystenta prawa oświatowego.",
            podstawa=[],
            interpretacja="Zakres asystenta obejmuje wyłącznie Karta Nauczyciela, Prawo oświatowe oraz akty i procedury bezpośrednio z nimi związane.",
            procedura=[],
            odpowiedz_wprost="Nie mogę odpowiedzieć na to pytanie w tym trybie.",
            sugestie=["Doprecyzuj pytanie w obszarze prawa oświatowego."],
            disclaimer=f"Odpowiedź ogólna. Stan prawny: {LEGAL_STATUS_DEFAULT_DATE}.",
            oferta="Mogę pomóc sformułować pytanie w zakresie prawa oświatowego."
        )

    A = sanitize_component(req.analiza_prawna)
    W = sanitize_component(req.wynik_weryfikacji)
    B = sanitize_component(req.biuletyn_informacyjny)

    # Szablon LLM: prosimy o złożenie w 9 sekcjach
    prompt = PROMPT_SYNTEZA.format(A=A or "(brak danych z KB – oprzyj się na ogólnych zasadach i ostrożności)", W=W or "(brak danych)", B=B or "(brak danych)")
    try:
        llm_out = await llm_call(prompt, model=LLM_DEFAULT_MODEL)
        # Zakładamy, że model zwraca już 9-sekcyjny markdown. Jeśli nie – fallback.
        if not llm_out or "## Weryfikacja pytania" not in llm_out:
            raise ValueError("Brak oczekiwanego formatu, użyję fallbacku.")
        return llm_out
    except Exception:
        # Fallback – bez LLM sklejka minimalna
        return format_response_9_sections(
            parafraza="Parafraza pytania: (brak – tryb awaryjny).",
            potwierdzenie="Odpowiedź dotyczy zagadnienia prawa oświatowego.",
            podstawa=["(brak – tryb awaryjny)"],
            interpretacja=A or "(brak – tryb awaryjny)",
            procedura=[],
            odpowiedz_wprost="(brak – tryb awaryjny)",
            sugestie=[],
            disclaimer=f"Odpowiedź ogólna. Stan prawny: {LEGAL_STATUS_DEFAULT_DATE}.",
            oferta="Czy chcesz, abym przygotował wzór pisma lub listę kroków dla dyrektora?"
        )

# ------------------------------------------------------------------------------
# ADMIN
# ------------------------------------------------------------------------------
@app.post("/admin/reload-index")
def admin_reload_index(x_apo_key: Optional[str] = Header(None)):
    require_admin(x_apo_key)
    _load_index()
    return {"ok": True, "entries": len(_ENTRIES)}

@app.post("/admin/refresh-public-sources")
async def admin_refresh_public_sources(x_apo_key: Optional[str] = Header(None)):
    require_admin(x_apo_key)
    if public_sources is None:
        return {"ok": False, "message": "Brak modułu public_sources.py – pomijam."}
    try:
        updated = await public_sources.refresh_all(BULLETIN_PATH)
        return {"ok": True, "updated": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"refresh error: {e}")

# (opcjonalnie) upload indeksu – wyłączone w domyślnej konfiguracji bezpieczeństwa
ALLOW_UPLOADS = os.getenv("ALLOW_UPLOADS", "false").lower() == "true"

@app.post("/admin/upload-index")
async def admin_upload_index(
    x_apo_key: Optional[str] = Header(None),
    file: UploadFile = File(...)
):
    require_admin(x_apo_key)
    if not ALLOW_UPLOADS:
        raise HTTPException(status_code=403, detail="Uploads disabled")
    content = await file.read()
    try:
        data = _json_loads(content)
        # krótka walidacja
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("invalid index payload")
        _write_json(KNOWLEDGE_INDEX_PATH, data)
        _load_index()
        return {"ok": True, "entries": len(_ENTRIES)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid index: {e}")

# ------------------------------------------------------------------------------
# DEV ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)