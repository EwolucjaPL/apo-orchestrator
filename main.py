import os
import json
import re
import asyncio
import uuid
from time import monotonic
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query, Response, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
# --- ProxyHeadersMiddleware z bezpiecznym fallbackiem ---
try:
    from starlette.middleware.proxy_headers import ProxyHeadersMiddleware
    _PROXY_HEADERS_AVAILABLE = True
except Exception:
    ProxyHeadersMiddleware = None  # type: ignore
    _PROXY_HEADERS_AVAILABLE = False

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- ORJSON (szybszy serializer) ---
try:
    from fastapi.responses import ORJSONResponse
    _ORJSON = True
except Exception:
    ORJSONResponse = JSONResponse  # fallback bez orjson
    _ORJSON = False

# --------------------------------------------------------------------------------------
# KONFIGURACJA
# --------------------------------------------------------------------------------------
load_dotenv()

APP_TITLE = "APO Gateway"
APP_DESC = "Gateway + mini-RAG dla prawa oświatowego"

# FastAPI z szybkim serializerem (jeśli dostępny)
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESC,
    default_response_class=ORJSONResponse if _ORJSON else JSONResponse,
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-4o")
LLM_PLANNER_MODEL = os.getenv("LLM_PLANNER_MODEL", "mistralai/mistral-7b-instruct:free")
KNOWLEDGE_INDEX_PATH = os.getenv("KNOWLEDGE_INDEX_PATH", "index.json")
MAX_RETURN_SNIPPETS = int(os.getenv("MAX_RETURN_SNIPPETS", "5"))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "2000"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "2097152"))  # 2 MB
ALLOW_UPLOADS = os.getenv("ALLOW_UPLOADS", "false").lower() == "true"

# CORS i kompresja
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
ALLOW_ORIGINS = CORS_ORIGINS if CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type", "authorization", "x-request-id"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)

# Proxy headers (prawdziwe IP z X-Forwarded-For), tylko gdy dostępne
if _PROXY_HEADERS_AVAILABLE:
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Prosty rate-limit per IP (in-memory; dla 1 procesu)
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))
_RATE_LOG: Dict[str, List[float]] = {}

_client: Optional[AsyncOpenAI] = None

def get_client() -> AsyncOpenAI:
    """Lenient: brak klucza = 503 dopiero przy użyciu LLM (nie przy starcie)."""
    global _client
    if _client is None:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise HTTPException(status_code=503, detail="Brak OPENROUTER_API_KEY w środowisku")
        _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    return _client

def _make_request_id() -> str:
    return str(uuid.uuid4())

# Globalny middleware: X-Request-Id + rate-limit
@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or _make_request_id()

    # proste ograniczenie: N żądań / 60s / IP
    if RATE_LIMIT_ENABLED:
        ip = request.client.host if request.client else "unknown"
        now = monotonic()
        bucket = _RATE_LOG.setdefault(ip, [])
        cutoff = now - 60
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) >= RATE_LIMIT_RPM and request.url.path not in ("/health", "/live", "/ready", "/"):
            return ORJSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
                headers={"Retry-After": "30", "X-Request-Id": rid},
            )
        bucket.append(now)

    resp = await call_next(request)
    resp.headers["X-Request-Id"] = rid
    return resp

# Dodatkowe nagłówki bezpieczeństwa
@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp

# Handlery wyjątków z X-Request-Id
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    rid = request.headers.get("X-Request-Id") or _make_request_id()
    return ORJSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers={"X-Request-Id": rid})

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    rid = request.headers.get("X-Request-Id") or _make_request_id()
    return ORJSONResponse({"detail": "Internal server error"}, status_code=500, headers={"X-Request-Id": rid})

# --------------------------------------------------------------------------------------
# PROMPTY
# --------------------------------------------------------------------------------------
PROMPT_KATEGORYZACJA = (
    "Your only task is to assess if the following query is exclusively about Polish educational law. "
    "Your domain includes: Teacher's Charter, school management, student rights, pedagogical supervision. "
    "Topics like general civil law, copyright law, "
    "construction law, or public procurement law are OUTSIDE YOUR DOMAIN. "
    "Answer only 'TAK' or 'NIE'.\nQuery: \"{query}\""
)

PROMPT_ANALIZA_ZAPYTANIA = (
    "Your task is to decompose the user's query into a simple action plan in JSON format. "
    "Analyze the query and return a JSON with a list of tasks under the key 'zadania'. "
    "Allowed tasks are: 'analiza_prawna', 'weryfikacja_cytatu', 'biuletyn_informacyjny'. "
    "Respond ONLY with valid JSON, no explanations.\n\nQuery: \"{query}\""
)

# --- NOWY SCHEMAT ODPOWIEDZI (9 SEKCJI) ---
PROMPT_SYNTEZA_ODPOWIEDZI = (
    "You are Asystent Prawa Oświatowego. Assemble the verified components into a single, "
    "coherent, professional, and clear response in Markdown in Polish. "
    "FOLLOW EXACTLY THE SCHEMA BELOW (all 9 sections, in order). "
    "Do not skip any section. If a component is missing, write (brak danych). "
    "Do not use code blocks or backticks. Use the exact emoji and section names. "
    "Keep the language clear, practical, and concise.\n\n"
    "📑 Schemat odpowiedzi APO – nowa wersja\n"
    "1. **Weryfikacja pytania**\n"
    "- Krótkie parafrazowanie pytania użytkownika.\n\n"
    "2. **Komunikat weryfikacji**\n"
    "- ✅ Jednozdaniowe potwierdzenie, czego dotyczy odpowiedź.\n\n"
    "3. **Podstawa prawna ⚖️**\n"
    "- Lista punktowana: artykuły, ustawy, akty wykonawcze.\n"
    "- Dodaj „Stan prawny: [data]”.\n\n"
    "4. **Interpretacja prawna 💡**\n"
    "- 2–3 akapity wyjaśnienia prostym językiem (warunki, wyjątki, kto decyduje).\n\n"
    "5. **Procedura krok po kroku 📝**\n"
    "- Lista numerowana (1., 2., 3.), maksymalnie 5–7 kroków.\n\n"
    "6. **Odpowiedź wprost 🎯**\n"
    "- Jedno zdanie: Tak/Nie + kluczowy warunek.\n\n"
    "7. **Proaktywna sugestia 💡**\n"
    "- 2–3 praktyczne wskazówki (opcjonalnie wzmianka o wzorach dokumentów).\n\n"
    "8. **Disclaimer prawny ⚖️**\n"
    "- Standardowa formuła: odpowiedź ogólna, nie stanowi porady prawnej; podaj stan prawny.\n\n"
    "9. **Dodatkowa oferta wsparcia 🤝**\n"
    "- Pytanie otwierające, zachęcające do dalszej interakcji.\n\n"
    "== KOMPONENTY DO UŻYCIA ==\n"
    "[Analiza prawna]\n{analiza_prawna}\n\n"
    "[Wynik weryfikacji cytatu]\n{wynik_weryfikacji}\n\n"
    "[Biuletyn informacji – najnowsze zmiany]\n{biuletyn_informacyjny}\n\n"
    "Wymagania formatowania:\n"
    "- Nie używaj nagłówków (#, ##, ###); stosuj pogrubienia i listy.\n"
    "- Zawsze dodaj sekcję **Źródła:** na końcu odpowiedzi (ID z bazy/akt Dz.U.).\n"
    "- Gdy zadanie obejmuje weryfikację cytatu, wstaw 1–2 zdania LITERALNEGO cytatu przepisu z oznaczeniem (ustawa, art., ust., Dz.U.).\n"
    "- Nie powołuj się na źródła inne niż dostarczone w komponentach; nie używaj własnej pamięci modelu."
)

# --------------------------------------------------------------------------------------
# MODELE DANYCH
# --------------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=MAX_QUERY_CHARS)

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
# MINI-RAG (BM25 jeśli dostępny; w innym razie prosty TF)
# --------------------------------------------------------------------------------------
IndexEntry = Dict[str, Any]
_INDEX_METADATA: Dict[str, Any] = {}
_ENTRIES: List[IndexEntry] = []

_BM25 = None
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _BM25_AVAILABLE = True
except Exception:
    _BM25_AVAILABLE = False

def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()

def _tokenize(t: str) -> List[str]:
    t = _normalize_text(t)
    t = re.sub(r"[^\wąćęłńóśźż\s]", " ", t)
    return [tok for tok in t.split() if len(tok) > 2]

def _validate_index_payload(data: Dict[str, Any]) -> None:
    if "entries" not in data or not isinstance(data["entries"], list):
        raise ValueError("Brak pola 'entries' (lista).")
    for i, e in enumerate(data["entries"]):
        if not isinstance(e, dict):
            raise ValueError(f"entries[{i}] nie jest obiektem")
        for key in ("id", "title", "summary"):
            if key not in e or not isinstance(e[key], str) or not e[key].strip():
                raise ValueError(f"entries[{i}].{key} musi być niepustym stringiem")

def _build_bm25():
    global _BM25
    if _BM25_AVAILABLE and _ENTRIES:
        corpus = [e.get("_tokens", []) for e in _ENTRIES]
        _BM25 = BM25Okapi(corpus)

def _load_index(path: str) -> None:
    global _INDEX_METADATA, _ENTRIES
    if not os.path.exists(path):
        raise RuntimeError(f"Nie znaleziono indeksu wiedzy pod ścieżką: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        _validate_index_payload(data)
    _INDEX_METADATA = data.get("metadata", {})
    _ENTRIES = data.get("entries", [])
    for e in _ENTRIES:
        e["_title_norm"] = _normalize_text(e.get("title", ""))
        e["_summary_norm"] = _normalize_text(e.get("summary", ""))
        e["_tokens"] = _tokenize(e.get("title", "") + " " + e.get("summary", ""))
    _build_bm25()

_load_index(KNOWLEDGE_INDEX_PATH)

def _score_entry_tf(query_tokens: List[str], entry: IndexEntry) -> float:
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
    hits: List[SearchHit] = []

    if _BM25_AVAILABLE and _BM25 is not None:
        scores = _BM25.get_scores(q_tokens)
        ranked = sorted(zip(scores, _ENTRIES), key=lambda x: x[0], reverse=True)[:k]
    else:
        scored: List[Tuple[float, IndexEntry]] = []
        for e in _ENTRIES:
            s = _score_entry_tf(q_tokens, e)
            if s > 0:
                scored.append((s, e))
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)[:k]

    for s, e in ranked:
        snippet = e.get("summary", "")
        snippet = snippet[:600] + ("…" if len(snippet) > 600 else "")
        hits.append(
            SearchHit(
                id=e.get("id", ""),
                title=e.get("title", ""),
                book=e.get("book"),
                chapter=e.get("chapter"),
                score=float(s),
                snippet=snippet,
            )
        )
    return hits

# --------------------------------------------------------------------------------------
# WYWOŁANIA LLM (retry 1x)
# --------------------------------------------------------------------------------------
async def llm_call(prompt: str, model: str = LLM_DEFAULT_MODEL, timeout: float = 30.0) -> str:
    async def _once() -> str:
        client = get_client()
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    for i in range(2):  # 1 próba + 1 retry
        try:
            return await asyncio.wait_for(_once(), timeout=timeout)
        except asyncio.TimeoutError:
            if i == 1:
                raise HTTPException(status_code=504, detail="Timeout podczas wywołania modelu AI")
        except Exception as e:
            if i == 1:
                msg = str(e)
                if "429" in msg:
                    raise HTTPException(status_code=429, detail="OpenAI rate limit")
                if "503" in msg or "overloaded" in msg.lower():
                    raise HTTPException(status_code=503, detail="OpenAI service unavailable")
                raise HTTPException(status_code=502, detail=f"Błąd wywołania modelu AI: {e}")
            await asyncio.sleep(0.5)

def sanitize_component(text: Optional[str]) -> str:
    if not text:
        return ""
    # usuń potencjalne instrukcje sterujące i nagłówki
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"(^|\n)\s*#{1,6}.*", " ", text)
    text = text.replace("<|system|>", "").replace("<|assistant|>", "").replace("<|user|>", "")
    text = text.replace("\u2028", " ").replace("\u2029", " ")
    return text.strip()

def _all_components_empty(req: "SynthesisRequest") -> bool:
    def _empty(x: Optional[str]) -> bool:
        return (x is None) or (isinstance(x, str) and len(x.strip()) == 0)
    return _empty(req.analiza_prawna) and _empty(req.wynik_weryfikacji) and _empty(req.biuletyn_informacyjny)

# --------------------------------------------------------------------------------------
# ENDPOINTY API
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "APO Gateway",
        "status": "ok",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "ready": "/ready",
        "live": "/live"
    }

@app.get("/live")
def liveness() -> Dict[str, Any]:
    return {"status": "alive"}

@app.get("/ready")
def readiness() -> Dict[str, Any]:
    kb_ok = len(_ENTRIES) > 0
    has_key = bool(os.getenv("OPENROUTER_API_KEY"))
    return {"ready": kb_ok, "kb_loaded": kb_ok, "has_api_key": has_key}

@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "entries": len(_ENTRIES),
        "kb_version": _INDEX_METADATA.get("version"),
        "model_default": LLM_DEFAULT_MODEL,
        "model_planner": LLM_PLANNER_MODEL,
        "has_api_key": bool(os.getenv("OPENROUTER_API_KEY")),
        "bm25": bool(_BM25_AVAILABLE and _BM25 is not None),
        "rate_limit_rpm": RATE_LIMIT_RPM if RATE_LIMIT_ENABLED else 0,
        "orjson": _ORJSON,
    }

@app.post("/analyze-query")
async def analyze_query(request: QueryRequest) -> Dict[str, Any]:
    # sanityzacja zapytania
    q = (request.query or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="Puste zapytanie.")
    if len(q) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=413, detail=f"Zapytanie zbyt długie (>{MAX_QUERY_CHARS} znaków).")

    # 1) Kategoryzacja domeny
    k_prompt = PROMPT_KATEGORYZACJA.format(query=q)
    k_raw = (await llm_call(k_prompt, model=LLM_PLANNER_MODEL)).strip().upper()
    k_value = "TAK" if k_raw == "TAK" else ("NIE" if k_raw == "NIE" else "TAK")
    if k_value == "NIE":
        return {"zadania": ["ODRZUCONE_SPOZA_DOMENY"]}

    # 2) Plan zadań (JSON + walidacja)
    p_prompt = PROMPT_ANALIZA_ZAPYTANIA.format(query=q)
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
    return search_entries(q, k=k)

@app.post("/gate-and-format-response")
async def gate_and_format_response(request: SynthesisRequest):
    # sentinel: poza domeną
    if request.analiza_prawna == "ODRZUCONE_SPOZA_DOMENY":
        final_md = (
            "Dziękuję za Twoje pytanie. Nazywam się Asystent Prawa Oświatowego, a moja wiedza "
            "jest ograniczona wyłącznie do zagadnień polskiego prawa oświatowego. Twoje pytanie "
            "wykracza poza ten zakres – nie mogę udzielić informacji na ten temat."
        )
        return Response(content=final_md, media_type="text/markdown; charset=utf-8")

    # 400 jeśli wszystkie komponenty są puste
    if _all_components_empty(request):
        raise HTTPException(status_code=400, detail="Brak treści do zsyntezowania (wszystkie komponenty puste).")

    analiza = sanitize_component(request.analiza_prawna)
    wery = sanitize_component(request.wynik_weryfikacji)
    biul = sanitize_component(request.biuletyn_informacyjny)

    prompt = PROMPT_SYNTEZA_ODPOWIEDZI.format(
        analiza_prawna=analiza or "(brak danych)",
        wynik_weryfikacji=wery or "(brak danych)",
        biuletyn_informacyjny=biul or "(brak danych)",
    )
    final_md = await llm_call(prompt, model=LLM_DEFAULT_MODEL)
    return Response(content=final_md, media_type="text/markdown; charset=utf-8")

# --- Admin: wymuszenie przeładowania indeksu (bez uploadu) ---
@app.post("/admin/reload-index")
async def admin_reload_index():
    _load_index(KNOWLEDGE_INDEX_PATH)
    return {"ok": True, "entries": len(_ENTRIES)}

# --- Admin: upload nowego index.json (multipart) – domyślnie OFF ---
@app.post("/admin/upload-index")
async def admin_upload_index(file: UploadFile = File(...), request: Request = None):
    if not ALLOW_UPLOADS:
        raise HTTPException(status_code=403, detail="Upload wyłączony (ALLOW_UPLOADS=false).")

    # Content-Length guard
    if request:
        cl = request.headers.get("content-length")
        if cl and int(cl) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Plik za duży (limit {MAX_UPLOAD_BYTES} B)")

    if file.content_type not in ("application/json", "text/json", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Dozwolone tylko JSON (application/json)")

    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=415, detail="Dozwolone są tylko pliki .json")

    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
        _validate_index_payload(data)  # twardsza walidacja
        with open(KNOWLEDGE_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        _load_index(KNOWLEDGE_INDEX_PATH)
        return {"ok": True, "entries": len(_ENTRIES)}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Nieprawidłowy JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload nieudany: {e}")

# --------------------------------------------------------------------------------------
# DEV ENTRYPOINT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)