# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import openai

# --- Tworzenie Aplikacji ---
app = FastAPI(
    title="APO-Orchestrator API",
    description="API zarządzające logiką i bezpieczeństwem agenta APO.",
    version="1.0.0",
)

# --- Middleware CORS (Kluczowe dla bezpieczeństwa) ---
# W docelowym rozwiązaniu należy usunąć tę sekcję, jeśli nie jest potrzebna
# lub skonfigurować ją zgodnie z wymaganiami bezpieczeństwa.
# W tej wersji zostawiamy ją, aby upewnić się, że nie blokuje komunikacji.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.openai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Konfiguracja Klienta pod OpenRouter ---
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# --- Definicje Modeli Danych (Pydantic) ---

class PlanZadania(BaseModel):
    zadania: List[Literal["analiza_prawna", "biuletyn_men", "biuletyn_cke", "weryfikacja_cytatu"]]
    sygnatura: Optional[str] = None

class AnalizaRequest(BaseModel):
    query: str

class KomponentyDoSyntezy(BaseModel):
    analiza_prawna: Optional[str] = Field(default=None)
    wynik_weryfikacji: Optional[str] = Field(default=None)
    biuletyn_informacyjny: Optional[str] = Field(default=None)

# --- Logika Pomocnicza (Symulator LLM) ---

def wywolaj_llm_z_promptem(prompt: str, model_do_zadania: str) -> str:
    print(f"--- WYWOŁANIE MODELU '{model_do_zadania}' PRZEZ OPENROUTER ---\n{prompt}\n---------------------------")
    # W docelowym rozwiązaniu tutaj powinno znaleźć się rzeczywiste wywołanie API
    # response = client.chat.completions.create(...)
    # return response.choices[0].message.content
    if "PROMPT-01" in prompt:
        return '{"zadania": ["analiza_prawna", "weryfikacja_cytatu"], "sygnatura": "III CZP 99/25"}'
    if "PROMPT-04" in prompt:
        return """### Analiza Prawna\nZgodnie z Prawem oświatowym, organ prowadzący szkołę ponosi odpowiedzialność za jej działalność.\n\n### Weryfikacja Orzeczenia\nWeryfikacja orzeczenia o sygnaturze III CZP 99/25 zakończyła się niepowodzeniem. Taki wyrok nie został odnaleziony."""
    return "Symulowana odpowiedź LLM."

# --- Główne Endpointy API (Actions dla GPT-s) ---

@app.post("/analyze-query", response_model=PlanZadania)
async def analyze_query(request: AnalizaRequest):
    prompt_analityk = f"PROMPT-01-ANALIZA_ZAPYTANIA\nPytanie użytkownika:\n'''\n{request.query}\n'''"
    wynik_json_str = wywolaj_llm_z_promptem(prompt_analityk, "openai/gpt-4o")
    return PlanZadania.parse_raw(wynik_json_str)

@app.post("/gate-and-format-response", response_model=str)
async def gate_and_format_response(komponenty: KomponentyDoSyntezy):
    prompt_redaktor = f"""PROMPT-04-SYNTEZA_ODPOWIEDZI\nKomponent "Analiza Prawna":\n'''\n{komponenty.analiza_prawna or "Brak danych."}\n'''\nKomponent "Wynik Weryfikacji Cytatu":\n'''\n{komponenty.wynik_weryfikacji or "Brak danych."}\n'''\nKomponent "Biuletyn Informacyjny":\n'''\n{komponenty.biuletyn_informacyjny or "Brak danych."}\n'''"""
    return wywolaj_llm_z_promptem(prompt_redaktor, "mistralai/mistral-7b-instruct")

# --- NOWY, KLUCZOWY ELEMENT: ENDPOINT HEALTH CHECK DLA RENDER ---
@app.get("/health", summary="Sprawdza stan aplikacji")
def health_check():
    """Ten endpoint jest używany przez Render do weryfikacji, czy aplikacja działa poprawnie."""
    return {"status": "ok"}