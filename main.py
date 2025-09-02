import os
import json
import re
import asyncio
import uuid
from time import monotonic
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, Response, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
try:
    from starlette.middleware.proxy_headers import ProxyHeadersMiddleware
    _PROXY_HEADERS_AVAILABLE = True
except Exception:
    ProxyHeadersMiddleware = None  # type: ignore
    _PROXY_HEADERS_AVAILABLE = False

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ==============================================================================
# KONFIGURACJA
# ==============================================================================
load_dotenv()

APP_TITLE = "APO Gateway"
APP_DESC = "Gateway + mini-RAG dla prawa oświatowego"

try:
    from fastapi.responses import ORJSONResponse
    _ORJSON = True
except Exception:
    ORJSONResponse = JSONResponse  # type: ignore
    _ORJSON = False

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESC,
    default_response_class=ORJSONResponse if _ORJSON else JSONResponse,
)

# ENV
LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-4o")
LLM_PLANNER_MODEL = os.getenv("LLM_PLANNER_MODEL", "mistralai/mistral-7b-instruct:free")
KNOWLEDGE_INDEX_PATH = os.getenv("KNOWLEDGE_INDEX_PATH", "index.json")
MAX_RETURN_SNIPPETS = int(os.getenv("MAX_RETURN_SNIPPETS", "5"))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "2000"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "2097152"))  # 2 MB
ALLOW_UPLOADS = os.getenv("ALLOW_UPLOADS", "false").lower() == "true"
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))
LEGAL_STATUS_DEFAULT_DATE = os.getenv("LEGAL_STATUS_DEFAULT_DATE", "1 września 2025 r.")
BULLETIN_PATH = os.getenv("BULLETIN_PATH", "/var/apo/bulletin.json")
ADMIN_KEY = os.getenv("APO_ADMIN_KEY")

# Denylista
DENYLIST_PATH = os.getenv("APO_KB_DENYLIST_PATH", "/var/apo/denylist.json")
ENV_DENYLIST = os.getenv("APO_KB_DENYLIST", "")

# CORS + kompresja
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
ALLOW_ORIGINS = CORS_ORIGINS if CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type", "authorization", "x-request-id", "x-apo-key"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)
if _PROXY_HEADERS_AVAILABLE:
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Rate limit
_RATE_LOG: Dict[str, List[float]] = {}

_client: Optional[AsyncOpenAI] = None
def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise HTTPException(status_code=503, detail="Brak OPENROUTER_API_KEY w środowisku")
        _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    return _client

def _make_request_id() -> str:
    return str(uuid.uuid4())

# ==============================================================================
# MIDDLEWARE: request-id, rate-limit, security headers
# ==============================================================================
@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or _make_request_id()
    if RATE_LIMIT_ENABLED:
        ip = request.client.host if request.client else "unknown"
        now = monotonic()
        bucket = _RATE_LOG.setdefault(ip, [])
        cutoff = now - 60
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) >= RATE_LIMIT_RPM and request.url.path not in ("/health", "/ready", "/live", "/"):
            return ORJSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
                headers={"Retry-After": "30", "X-Request-Id": rid},
            )
        bucket.append(now)
    resp = await call_next(request)
    resp.headers["X-Request-Id"] = rid
    return resp

@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp

# ==============================================================================
# PROMPTY
# ==============================================================================
PROMPT_KATEGORYZACJA = (
    "Your only task is to assess if the following query is exclusively about Polish educational law. "
    "Your domain includes: Teacher's Charter, school management, student rights, pedagogical supervision. "
    "Topics like general civil law, copyright law, construction law, or public procurement law are OUTSIDE YOUR DOMAIN. "
    "Answer only 'TAK' or 'NIE'.\nQuery: \"{query}\""
)

PROMPT_ANALIZA_ZAPYTANIA = (
    "Your task is to decompose the user's query into a simple action plan in JSON format. "
    "Analyze the query and return a JSON with a list of tasks under the key 'zadania'. "
    "Allowed tasks are: 'analiza_prawna', 'weryfikacja_cytatu', 'biuletyn_informacyjny'. "
    "Respond ONLY with valid JSON, no explanations.\n\nQuery: \"{query}\""
)

PROMPT_SYNTEZA_ODPOWIEDZI = (
    "You are Asystent Prawa Oświatowego. Assemble the verified components into a single, "
    "coherent, professional, and crystal-clear answer in Polish for school leaders. "
    "Use exactly nine sections shown below, in order. Keep it concise and practical. "
    "Never include meta-notes or code blocks; do not reveal internal IDs.\n\n"
    "SEKCJE (W TEJ KOLEJNOŚCI):\n"
    "1) **Weryfikacja pytania** – jedno zdanie parafrazy pytania użytkownika.\n"
    "2) **Komunikat weryfikacji** – linia zaczynająca się od ✅ i krótkie potwierdzenie ZAKRESU pytania (neutralne; NIE podawaj jeszcze rozstrzygnięcia!).\n"
    "3) **Podstawa prawna ⚖️** – lista punktowana artykułów/aktów; zakończ wierszem:\n"
    "   Stan prawny: [data] (jeśli brak danych w komponentach, wpisz: {stan_prawny_domyslny} (domyślny)).\n"
    "4) **Interpretacja prawna 💡** – 2–3 krótkie akapity wyjaśniające sens i wyjątki.\n"
    "5) **Procedura krok po kroku 📝** – lista numerowana 1–5 (max 7), praktyczne kroki.\n"
    "6) **Odpowiedź wprost 🎯** – JEDNO zdanie wytłuszczone: Tak/Nie + warunek.\n"
    "7) **Proaktywna sugestia 💡** – 2–3 krótkie wskazówki.\n"
    "8) **Disclaimer prawny ⚖️** – standard: odpowiedź ogólna; podaj stan prawny.\n"
    "9) **Dodatkowa oferta wsparcia 🤝** – pytanie otwierające.\n\n"
    "FORMATOWANIE (OBOWIĄZKOWE):\n"
    "- Przed KAŻDĄ sekcją wstaw poziomą linię: ---\n"
    "- Po treści każdej sekcji zostaw DWA puste wiersze (dla czytelności).\n"
    "- Nie używaj nagłówków #/##/###; sekcje pisz jako wytłuszczone tytuły (**) + tekst pod spodem.\n"
    "- W **Podstawa prawna ⚖️** użyj punktorów (–) z pełnymi nazwami aktów i artykułów.\n"
    "- W **Odpowiedź wprost 🎯**: całe zdanie wytłuszczone i w osobnym akapicie.\n"
    "- Na końcu dodaj blok **Źródła** oddzielony poziomą linią i wypisz wyłącznie publiczne akty/dokumenty "
    "(ISAP, Dz.U., MEN, komunikaty urzędowe). Nigdy nie ujawniaj tytułów ani kompozycji wewnętrznej bazy wiedzy.\n\n"
    "== KOMPONENTY ==\n"
    "[Analiza prawna]\n{analiza_prawna}\n\n"
    "[Wynik weryfikacji cytatu]\n{wynik_weryfikacji}\n\n"
    "[Biuletyn informacji – najnowsze zmiany]\n{biuletyn_informacyjny}\n\n"
    "ZASADY:\n"
    "- Jeśli brak któregoś komponentu, wpisz (brak danych) – nie wymyślaj treści.\n"
    "- Pisz krótko, jasno, zorientowanie na dyrektorów szkół.\n"
)

# ==============================================================================
# MODELE
# ==============================================================================
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
    score: float
    snippet: str
    book: Optional[str] = None
    chapter: Optional[str] = None

# ==============================================================================
# KB: ŁADOWANIE I WYSZUKIWANIE
# ==============================================================================
IndexEntry = Dict[str, Any]
_INDEX_METADATA: Dict[str, Any] = {}
_ENTRIES: List[IndexEntry] = []

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception:
    _BM25_AVAILABLE = False

_BM25 = None

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
    else:
        _BM25 = None

def _load_index(path: str) -> None:
    global _INDEX_METADATA, _ENTRIES
    if not os.path.exists(path):
        _INDEX_METADATA = {"version": "empty"}
        _ENTRIES = []
        _build_bm25()
        return
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

def _score_entry_tf(qt: List[str], entry: IndexEntry) -> float:
    if not qt:
        return 0.0
    toks = entry.get("_tokens", [])
    score = sum(1 for t in toks if t in qt)
    if " ".join(qt) in entry.get("_title_norm", ""):
        score += 2.0
    return float(score)

def search_entries(query: str, k: int = 5) -> List[SearchHit]:
    qt = _tokenize(query)
    ranked: List[Tuple[float, IndexEntry]] = []
    if _BM25 is not None:
        scores = _BM25.get_scores(qt)
        ranked = sorted(list(zip(map(float, scores), _ENTRIES)), key=lambda x: x[0], reverse=True)[:k]
    else:
        tmp: List[Tuple[float, IndexEntry]] = []
        for e in _ENTRIES:
            s = _score_entry_tf(qt, e)
            if s > 0:
                tmp.append((s, e))
        ranked = sorted(tmp, key=lambda x: x[0], reverse=True)[:k]

    hits: List[SearchHit] = []
    for s, e in ranked:
        snip = e.get("summary", "")
        snip = snip[:600] + ("…" if len(snip) > 600 else "")
        hits.append(
            SearchHit(
                id=e.get("id", ""),
                title=e.get("title", ""),
                book=e.get("book"),
                chapter=e.get("chapter"),
                score=float(s),
                snippet=snip,
            )
        )
    return hits

# ==============================================================================
# LLM + SANITY
# ==============================================================================
async def llm_call(prompt: str, model: str = LLM_DEFAULT_MODEL, timeout: float = 30.0) -> str:
    async def _once() -> str:
        client = get_client()
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    for i in range(2):
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
    txt = text
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)
    txt = re.sub(r"(^|\n)\s*#{1,6}.*", " ", txt)
    txt = txt.replace("<|system|>", "").replace("<|assistant|>", "").replace("<|user|>", "")
    txt = txt.replace("\u2028", " ").replace("\u2029", " ")
    return txt.strip()

# ==============================================================================
# META-PYTNIA O BAZĘ (NIE UJAWNIAMY SKŁADU)
# ==============================================================================
_KB_META_PATTERNS = [
    r"\bbaza wiedzy\b", r"\bco masz w bazie\b", r"\bpoka(ż|z) baz(ę|e)\b",
    r"\bspis (treści|tresci)\b", r"\blista (źródeł|zrodel)\b", r"\bwykaz (źródeł|zrodel)\b",
    r"\bjakich (źródeł|zrodel) używasz\b", r"\bsk(ą|a)d bierzesz dane\b",
    r"\bjakie dokumenty\b", r"\bjakie (źródła|zrodla)\b", r"\brepozytorium\b", r"\bzaplecze\b",
    r"\bbibliografia\b",
    r"\bknowledge base\b", r"\bbibliography\b", r"\byour sources\b", r"\bwhat sources\b", r"\blist your sources\b"
]
def _is_kb_meta_query(text: str) -> bool:
    t = (text or "").lower().strip()
    for pat in _KB_META_PATTERNS:
        if re.search(pat, t):
            return True
    return False

# ==============================================================================
# BIULETYN (oficjalne + nieoficjalne) + BACKFILL
# ==============================================================================
def _load_bulletin_text() -> str:
    p = Path(BULLETIN_PATH)
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data.get("items", [])
        if not items:
            return ""
        off = [i for i in items if i.get("source_type") == "official"]
        unoff = [i for i in items if i.get("source_type") == "unofficial"]

        lines: List[str] = []
        if off:
            lines.append("**Biuletyn (źródła oficjalne)**")
            for it in off[:5]:
                lines.append(f"- {it.get('title')} — {it.get('source')} ({it.get('date')})")
            lines.append("")
        if unoff:
            lines.append("**Dodatkowe komentarze (źródła nieoficjalne)**")
            for it in unoff[:3]:
                lines.append(f"- {it.get('title')} — {it.get('source')} ({it.get('date')})")
            lines.append("_Uwaga: pozycje z nieoficjalnych źródeł mają charakter informacyjny._")

        return "\n".join(lines).strip()
    except Exception:
        return ""

try:
    from public_sources import refresh_all as refresh_all_sources  # opcjonalny moduł
except Exception:
    refresh_all_sources = None

def _bulletin_is_empty() -> bool:
    p = Path(BULLETIN_PATH)
    if not p.exists():
        return True
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return len(data.get("items", [])) == 0
    except Exception:
        return True

def _backfill_bulletin_last_year() -> None:
    """Uzupełnia biuletyn pozycjami z ostatnich 12 miesięcy, jeśli jest pusty."""
    if refresh_all_sources is None:
        return
    try:
        since = (datetime.utcnow() - timedelta(days=365)).date().isoformat()
        payload = refresh_all_sources(since_date=since)  # jeśli nie obsługuje, niżej fallback
    except TypeError:
        raw = refresh_all_sources()
        items = raw.get("items", [])
        cutoff = datetime.utcnow() - timedelta(days=365)
        kept = []
        for it in items:
            try:
                d = datetime.fromisoformat(it.get("date", ""))
            except Exception:
                d = cutoff
            if d >= cutoff:
                kept.append(it)
        payload = {"items": kept, "updated_at": datetime.utcnow().isoformat()}

    Path(BULLETIN_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(BULLETIN_PATH).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# ==============================================================================
# FORMATOWANIE 9-SEKCYJNE + MASKOWANIE + DENYLISTA
# ==============================================================================
_SECTION_ORDER = [
    "Weryfikacja pytania",
    "Komunikat weryfikacji",
    "Podstawa prawna ⚖️",
    "Interpretacja prawna 💡",
    "Procedura krok po kroku 📝",
    "Odpowiedź wprost 🎯",
    "Proaktywna sugestia 💡",
    "Disclaimer prawny ⚖️",
    "Dodatkowa oferta wsparcia 🤝",
]
_SOURCES_HEADER = "Źródła"

def _md_sep() -> str:
    return "---\n"

def _dbl_nl() -> str:
    return "\n\n"

def enforce_nine_section_format(md: str) -> str:
    if not md:
        md = ""
    text = md.replace("\r\n", "\n").strip()

    pattern = r"\*\*\s*(.+?)\s*\*\*"
    parts = re.split(pattern, text)
    parsed: Dict[str, str] = {}
    if parts:
        it = iter(parts[1:])
        for head, content in zip(it, it):
            parsed[head.strip()] = content.strip()

    out_lines: List[str] = []
    for sec in _SECTION_ORDER:
        out_lines.append(_md_sep() + f"**{sec}**")
        content = parsed.get(sec, "").strip() or "(brak danych)"

        if sec == "Komunikat weryfikacji":
            if re.search(r"(?i)\b(tak|nie)\b.*\b(ma|może|nie może|przysługuje|nie przysługuje|obowiązuje)\b", content):
                wery = parsed.get("Weryfikacja pytania", "").strip()
                first_sent = wery.split(".")[0].strip() if wery else "prawa oświatowego"
                content = f"✅ Pytanie dotyczy: {first_sent if first_sent else 'prawa oświatowego'}."
            elif not content.startswith("✅"):
                content = "✅ Potwierdzam zakres pytania (prawo oświatowe)."

        if sec == "Podstawa prawna ⚖️" and "Stan prawny:" not in content:
            if not content.endswith("\n"):
                content += "\n"
            content += f"Stan prawny: {LEGAL_STATUS_DEFAULT_DATE} (domyślny)."

        if sec == "Disclaimer prawny ⚖️" and "Stan prawny:" not in content:
            if not content.endswith("\n"):
                content += "\n"
            content += f"Stan prawny: {LEGAL_STATUS_DEFAULT_DATE}."

        out_lines.append(content)
        out_lines.append(_dbl_nl())

    sources_content = parsed.get(_SOURCES_HEADER, "").strip()
    if not sources_content:
        sources_content = "– ISAP (akty prawne)\n– Dziennik Ustaw / RCL\n– MEN – komunikaty i rozporządzenia"

    out_lines.append(_md_sep() + f"**{_SOURCES_HEADER}**")
    out_lines.append(sources_content)
    return "\n".join(out_lines).strip()

_PRIVATE_PATTERNS = [
    r"\bautor_APO\b",
    r"\b[A-Za-z0-9/_-]+\.(md|json|pdf)\b",
    r"\bnowoczesne[-\s]zarzadzanie[-\s]szkola\b",
    r"\bprawa[-\s]i[-\s]obowiazki[-\s]dyrektora\b",
    r"\bkomentarz( |\b).*karta( |\b) nauczyciela\b",
]
def mask_private_kb_references(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    out = []
    for ln in lines:
        if any(re.search(pat, ln, flags=re.IGNORECASE) for pat in _PRIVATE_PATTERNS):
            out.append("")  # wytnij linię
        else:
            out.append(ln)
    return "\n".join(out)

_DENY_PATTERNS_RAW: List[str] = []
_DENY_REGEXES: List[re.Pattern] = []

def _load_denylist():
    global _DENY_PATTERNS_RAW, _DENY_REGEXES
    patterns: List[str] = []

    if ENV_DENYLIST.strip():
        parts = [p.strip() for p in ENV_DENYLIST.split(";") if p.strip()]
        patterns.extend(parts)

    try:
        p = Path(DENYLIST_PATH)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("patterns"), list):
                patterns.extend([str(x) for x in data["patterns"] if str(x).strip()])
    except Exception:
        pass

    seen = set()
    clean: List[str] = []
    for ptn in patterns:
        if ptn not in seen:
            seen.add(ptn)
            clean.append(ptn)

    _DENY_PATTERNS_RAW = clean
    _DENY_REGEXES = []
    for pat in clean:
        try:
            rx = re.compile(pat, flags=re.IGNORECASE | re.UNICODE)
        except Exception:
            rx = re.compile(re.escape(pat), flags=re.IGNORECASE | re.UNICODE)
        _DENY_REGEXES.append(rx)

_load_denylist()

def redact_denied_titles(text: str) -> str:
    if not text or not _DENY_REGEXES:
        return text
    lines = text.splitlines()
    redacted = []
    for ln in lines:
        ls = ln.strip()
        if not ls:
            redacted.append(ln)
            continue
        if any(rx.search(ls) for rx in _DENY_REGEXES):
            continue
        redacted.append(ln)
    return "\n".join(redacted)

def _redact_search_hits(hits: List[SearchHit]) -> List[SearchHit]:
    if not _DENY_REGEXES:
        return hits
    out: List[SearchHit] = []
    for h in hits:
        t = h.title or ""
        s = h.snippet or ""
        if any(rx.search(t) or rx.search(s) for rx in _DENY_REGEXES):
            continue
        out.append(h)
    return out

# ==============================================================================
# MARKDOWNY SCENARIUSZY KRAWĘDZIOWYCH
# ==============================================================================
def _refusal_markdown() -> str:
    parts: List[str] = []
    def add(sec_title: str, body: str):
        parts.append("---\n" + f"**{sec_title}**")
        parts.append(body)
        parts.append("\n\n")
    add("Weryfikacja pytania", "To pytanie nie dotyczy polskiego prawa oświatowego.")
    add("Komunikat weryfikacji", "✅ Pytanie wykracza poza zakres Asystenta Prawa Oświatowego (prawo oświatowe).")
    add("Podstawa prawna ⚖️", f"– (brak danych)\nStan prawny: {LEGAL_STATUS_DEFAULT_DATE} (domyślny).")
    add("Interpretacja prawna 💡", "Asystent Prawa Oświatowego udziela informacji wyłącznie w obszarze prawa oświatowego. "
                                   "Pytania z zakresu podatków, ubezpieczeń, prawa cywilnego lub gospodarczego nie są obsługiwane.")
    add("Procedura krok po kroku 📝", "1. Sformułuj pytanie dotyczące prawa oświatowego.\n"
                                      "2. W pytaniach spoza domeny skonsultuj specjalistę w danej dziedzinie.\n"
                                      "3. Podaj kontekst (typ szkoły, etap), aby doprecyzować odpowiedź.")
    add("Odpowiedź wprost 🎯", "**Nie – nie odpowiadam na pytania spoza domeny prawa oświatowego.**")
    add("Proaktywna sugestia 💡", "Rozważ zadanie pytania o Kartę Nauczyciela, Prawo oświatowe, statut szkoły, rady pedagogiczne.")
    add("Disclaimer prawny ⚖️", f"Odpowiedź ma charakter ogólny i dotyczy wyłącznie prawa oświatowego.\nStan prawny: {LEGAL_STATUS_DEFAULT_DATE}.")
    add("Dodatkowa oferta wsparcia 🤝", "Czy chcesz, abym pomógł przeformułować pytanie w zakresie prawa oświatowego?")
    parts.append("---\n**Źródła**\n– (brak – pytanie spoza domeny)")
    return "\n".join(parts).strip()

def _kb_scope_markdown() -> str:
    parts: List[str] = []
    def add(sec_title: str, body: str):
        parts.append("---\n" + f"**{sec_title}**")
        parts.append(body)
        parts.append("\n\n")
    add("Weryfikacja pytania", "Prośba o przedstawienie bazy wiedzy i źródeł.")
    add("Komunikat weryfikacji", "✅ Pytanie dotyczy zakresu tematycznego i rodzaju publicznych źródeł wykorzystywanych przez APO.")
    add("Podstawa prawna ⚖️", "– Karta Nauczyciela (ustawa z 26 stycznia 1982 r.)\n"
                               "– Prawo oświatowe (ustawa z 14 grudnia 2016 r.)\n"
                               "– Rozporządzenia MEN\n"
                               "– Orzecznictwo i komunikaty organów administracji\n"
                               f"Stan prawny: {LEGAL_STATUS_DEFAULT_DATE} (domyślny).")
    add("Interpretacja prawna 💡", "APO udziela odpowiedzi wyłącznie w obszarze polskiego prawa oświatowego. "
                                   "W odpowiedziach cytowane są wyłącznie publicznie dostępne akty i dokumenty. "
                                   "Skład wewnętrznych materiałów pomocniczych nie jest ujawniany.")
    add("Procedura krok po kroku 📝", "1. Podaj konkretne zagadnienie (np. artykuł KN, statut szkoły, arkusz).\n"
                                      "2. Otrzymasz analizę z podstawą prawną i krótką procedurą.\n"
                                      "3. W razie potrzeby doprecyzuj kontekst (typ szkoły, etap, rola pytającego).")
    add("Odpowiedź wprost 🎯", "**APO przedstawia jedynie ogólny zakres publicznych źródeł i nie ujawnia składu wewnętrznej bazy wiedzy.**")
    add("Proaktywna sugestia 💡", "Podaj konkretne pytanie z obszaru prawa oświatowego; przygotuję zwięzłą analizę z podstawą prawną.")
    add("Disclaimer prawny ⚖️", f"Odpowiedź ma charakter ogólny i dotyczy zakresu tematycznego.\nStan prawny: {LEGAL_STATUS_DEFAULT_DATE}.")
    add("Dodatkowa oferta wsparcia 🤝", "Czy chcesz, abym zaproponował katalog przykładowych tematów (dyrektor, rada pedagogiczna, KN, statut)?")
    parts.append("---\n**Źródła**\n– ISAP (akty prawne)\n– Dziennik Ustaw / RCL\n– MEN – komunikaty i rozporządzenia")
    return "\n".join(parts).strip()

# ==============================================================================
# ENDPOINTY STATUSOWE
# ==============================================================================
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
    disk_ok = True
    try:
        Path(BULLETIN_PATH).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        disk_ok = False
    return {"ready": kb_ok and has_key and disk_ok, "kb_loaded": kb_ok, "has_api_key": has_key, "disk_ok": disk_ok}

@app.get("/health")
def health_check() -> Dict[str, Any]:
    # spróbuj odczytać meta biuletynu
    bulletin_exists = Path(BULLETIN_PATH).exists()
    bulletin_items = 0
    bulletin_updated = None
    try:
        if bulletin_exists:
            data = json.loads(Path(BULLETIN_PATH).read_text(encoding="utf-8"))
            bulletin_items = len(data.get("items", []))
            bulletin_updated = data.get("updated_at")
    except Exception:
        pass

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
        "bulletin_exists": bulletin_exists,
        "bulletin_items": bulletin_items,
        "bulletin_updated_at": bulletin_updated,
        "denylist_loaded": len(_DENY_REGEXES) > 0
    }

# ==============================================================================
# PLANOWANIE
# ==============================================================================
@app.post("/analyze-query")
async def analyze_query(request: QueryRequest) -> Dict[str, Any]:
    q = (request.query or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="Puste zapytanie.")
    if len(q) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=413, detail=f"Zapytanie zbyt długie (>{MAX_QUERY_CHARS} znaków).")

    if _is_kb_meta_query(q):
        return {"zadania": ["META_KB_SCOPE_ONLY"]}

    k_prompt = PROMPT_KATEGORYZACJA.format(query=q)
    k_raw = (await llm_call(k_prompt, model=LLM_PLANNER_MODEL)).strip().upper()
    k_value = "TAK" if k_raw == "TAK" else ("NIE" if k_raw == "NIE" else "TAK")
    if k_value == "NIE":
        return {"zadania": ["ODRZUCONE_SPOZA_DOMENY"]}

    p_prompt = PROMPT_ANALIZA_ZAPYTANIA.format(query=q)
    p_raw = await llm_call(p_prompt, model=LLM_PLANNER_MODEL)
    m = re.search(r"\{[\s\S]*\}", p_raw)
    plan_json = m.group(0) if m else '{"zadania":["analiza_prawna"]}'
    try:
        plan = PlanZadania.model_validate_json(plan_json)
    except ValidationError:
        plan = PlanZadania(zadania=["analiza_prawna"])
    return plan.model_dump()

# ==============================================================================
# KB SEARCH
# ==============================================================================
@app.get("/knowledge/search", response_model=List[SearchHit])
async def knowledge_search(q: str = Query(..., min_length=2), k: int = Query(MAX_RETURN_SNIPPETS, ge=1, le=10)):
    res = search_entries(q, k=k)
    res = _redact_search_hits(res)
    return res

# ==============================================================================
# SYNTEZA
# ==============================================================================
def _all_components_empty(req: "SynthesisRequest") -> bool:
    return not ( (req.analiza_prawna and req.analiza_prawna.strip())
                 or (req.wynik_weryfikacji and req.wynik_weryfikacji.strip())
                 or (req.biuletyn_informacyjny and req.biuletyn_informacyjny.strip()) )

@app.post("/gate-and-format-response")
async def gate_and_format_response(request: SynthesisRequest):
    if request.analiza_prawna == "ODRZUCONE_SPOZA_DOMENY":
        md = _refusal_markdown()
        md = enforce_nine_section_format(md)
        md = mask_private_kb_references(md)
        md = redact_denied_titles(md)
        return Response(content=md, media_type="text/markdown; charset=utf-8")

    if request.analiza_prawna == "META_KB_SCOPE_ONLY":
        md = _kb_scope_markdown()
        md = enforce_nine_section_format(md)
        md = mask_private_kb_references(md)
        md = redact_denied_titles(md)
        return Response(content=md, media_type="text/markdown; charset=utf-8")

    if _all_components_empty(request):
        raise HTTPException(status_code=400, detail="Brak treści do zsyntezowania (wszystkie komponenty puste).")

    analiza = sanitize_component(request.analiza_prawna)
    wery = sanitize_component(request.wynik_weryfikacji)
    biul = sanitize_component(request.biuletyn_informacyjny) or _load_bulletin_text()

    prompt = PROMPT_SYNTEZA_ODPOWIEDZI.format(
        analiza_prawna=analiza or "(brak danych)",
        wynik_weryfikacji=wery or "(brak danych)",
        biuletyn_informacyjny=biul or "(brak danych)",
        stan_prawny_domyslny=LEGAL_STATUS_DEFAULT_DATE,
    )
    final_md = await llm_call(prompt, model=LLM_DEFAULT_MODEL)

    final_md = enforce_nine_section_format(final_md)
    final_md = mask_private_kb_references(final_md)
    final_md = redact_denied_titles(final_md)

    return Response(content=final_md, media_type="text/markdown; charset=utf-8")

# ==============================================================================
# ADMIN: KB / BIULETYN / DENYLISTA
# ==============================================================================
@app.post("/admin/reload-index")
async def admin_reload_index():
    _load_index(KNOWLEDGE_INDEX_PATH)
    return {"ok": True, "entries": len(_ENTRIES)}

@app.post("/admin/upload-index")
async def admin_upload_index(file: UploadFile = File(...), request: Request = None):
    if not ALLOW_UPLOADS:
        raise HTTPException(status_code=403, detail="Upload wyłączony (ALLOW_UPLOADS=false).")
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
        _validate_index_payload(data)
        with open(KNOWLEDGE_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        _load_index(KNOWLEDGE_INDEX_PATH)
        return {"ok": True, "entries": len(_ENTRIES)}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Nieprawidłowy JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload nieudany: {e}")

@app.post("/admin/refresh-public-sources")
async def refresh_public_sources_ep(request: Request):
    if not ADMIN_KEY or request.headers.get("X-APO-Key") != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if refresh_all_sources is None:
        raise HTTPException(status_code=501, detail="Brak modułu public_sources.refresh_all")
    payload = refresh_all_sources()
    Path(BULLETIN_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(BULLETIN_PATH).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "items": len(payload.get("items", [])), "updated": payload.get("updated_at")}

@app.post("/admin/reload-denylist")
async def admin_reload_denylist(request: Request):
    if not ADMIN_KEY or request.headers.get("X-APO-Key") != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    _load_denylist()
    return {"ok": True, "patterns": len(_DENY_REGEXES)}

# ==============================================================================
# STARTUP: BACKFILL BIULETYNU, JEŚLI PUSTY
# ==============================================================================
@app.on_event("startup")
async def _startup_backfill():
    try:
        if _bulletin_is_empty():
            _backfill_bulletin_last_year()
    except Exception:
        # best-effort: nie blokuj startu
        pass

# ==============================================================================
# DEV ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)