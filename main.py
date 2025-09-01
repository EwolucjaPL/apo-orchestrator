import os
import json
import re
import asyncio
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --------------------------------------------------------------------------------------
# KONFIGURACJA
# --------------------------------------------------------------------------------------
load_dotenv()
app = FastAPI(title="APO Gateway", description="Gateway + mini‑RAG dla prawa oświatowego")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Brak OPENROUTER_API_KEY w środowisku")

LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-4o")
LLM_PLANNER_MODEL = os.getenv("LLM_PLANNER_MODEL", "mistralai/mistral-7b-instruct:free")
KNOWLEDGE_INDEX_PATH = os.getenv("KNOWLEDGE_INDEX_PATH", "index.json")
MAX_RETURN_SNIPPETS = int(os.getenv("MAX_RETURN_SNIPPETS", "5"))

_client: Optional[AsyncOpenAI] = None

def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    return _client

# --------------------------------------------------------------------------------------
# PROMPTY
# --------------------------------------------------------------------------------------
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
    "You are an editor in a law firm specializing in educational law. Assemble the verified components into a single, "
    "coherent, professional, and clear response in Markdown in Polish. If some component is missing, proceed without it.\n\n"
    "== KOMPONENTY ==\n"
    "[Analiza prawna]\n{analiza_prawna}\n\n"
    "[Wynik weryfikacji cytatu]\n{wynik_weryfikacji}\n\n"
    "[Biuletyn informacji – najnowsze zmiany]\n{biuletyn_informacyjny}\n\n"
    "Wymagania: bądź zwięzły, rzeczowy, wskaż podstawy prawne, wypunktuj zalecenia dyrektora/nauczyciela."
)

# --------------------------------------------------------------------------------------
# MODELE DANYCH
# --------------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)

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
# MINI‑RAG: Ładowanie indeksu i proste wyszukiwanie słów kluczowych
# --------------------------------------------------------------------------------------
IndexEntry = Dict[str, Any]
_INDEX_METADATA: Dict[str, Any] = {}
_ENTRIES: List[IndexEntry] = []


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
    # wstępna normalizacja dla szybszego wyszukiwania
    for e in _ENTRIES:
        e["_title_norm"] = _normalize_text(e.get("title", ""))
        e["_summary_norm"] = _normalize_text(e.get("summary", ""))
        e["_tokens"] = _tokenize(e.get("title", "") + " " + e.get("summary", ""))


_load_index(KNOWLEDGE_INDEX_PATH)


# Bardzo prosty ranking: TF (częstość tokenów zapytania) + bonus za dopasowanie frazy w tytule

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
    for s, e in scored[:k]:
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
# NARZĘDZIA: wywołania LLM z twardymi zasadami
# --------------------------------------------------------------------------------------
async def llm_call(prompt: str, model: str = LLM_DEFAULT_MODEL, timeout: float = 30.0) -> str:
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


def sanitize_component(text: Optional[str]) -> str:
    if not text:
        return ""
    # usuwamy potencjalne próby sterowania modelem
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"(^|\n)\s*#{1,6}.*", " ", text)  # nagłówki
    text = text.replace("<|system|>", "").replace("<|assistant|>", "").replace("<|user|>", "")
    return text.strip()


# --------------------------------------------------------------------------------------
# ENDPOINTY API
# --------------------------------------------------------------------------------------
@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "entries": len(_ENTRIES),
        "kb_version": _INDEX_METADATA.get("version"),
    }


@app.post("/analyze-query")
async def analyze_query(request: QueryRequest) -> Dict[str, Any]:
    # 1) Kategoryzacja (TAK/NIE – twarde porównanie)
    k_prompt = PROMPT_KATEGORYZACJA.format(query=request.query)
    k_raw = (await llm_call(k_prompt, model=LLM_PLANNER_MODEL)).strip().upper()
    k_value = "TAK" if k_raw == "TAK" else ("NIE" if k_raw == "NIE" else "TAK")
    if k_value == "NIE":
        return {"zadania": ["ODRZUCONE_SPOZA_DOMENY"]}

    # 2) Plan zadań (wymuszony JSON + walidacja)
    p_prompt = PROMPT_ANALIZA_ZAPYTANIA.format(query=request.query)
    p_raw = await llm_call(p_prompt, model=LLM_PLANNER_MODEL)

    # wyłuskanie pierwszego poprawnego bloku JSON
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
async def gate_and_format_response(request: SynthesisRequest) -> str:
    if request.analiza_prawna == "ODRZUCONE_SPOZA_DOMENY":
        return (
            "Dziękuję za Twoje pytanie. Nazywam się Asystent Prawa Oświatowego, a moja wiedza "
            "jest ograniczona wyłącznie do zagadnień polskiego prawa oświatowego. Twoje pytanie "
            "wykracza poza ten zakres – nie mogę udzielić informacji na ten temat."
        )

    analiza = sanitize_component(request.analiza_prawna)
    wery = sanitize_component(request.wynik_weryfikacji)
    biul = sanitize_component(request.biuletyn_informacyjny)

    prompt = PROMPT_SYNTEZA_ODPOWIEDZI.format(
        analiza_prawna=analiza or "(brak danych z bazy – użyj ogólnych zasad i zaznacz niepewność)",
        wynik_weryfikacji=wery or "(brak danych)",
        biuletyn_informacyjny=biul or "(brak danych)",
    )
    return await llm_call(prompt, model=LLM_DEFAULT_MODEL)


# --------------------------------------------------------------------------------------
# DEV ENTRYPOINT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
