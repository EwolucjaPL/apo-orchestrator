# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import openai

# --- Tworzenie Aplikacji ---
app = FastAPI(
    title="APO-Orchestrator API",
    description="API zarządzające logiką i bezpieczeństwem agenta APO.",
    version="1.0.0",
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
    analiza_prawna: Optional[str] = Field(default=None, description="Tekst analizy prawnej z wewnętrznej bazy wiedzy.")
    wynik_weryfikacji: Optional[str] = Field(default=None, description="Wynik weryfikacji cytatu, np. 'NIE ZNALEZIONO'.")
    biuletyn_informacyjny: Optional[str] = Field(default=None, description="Tekst biuletynu informacyjnego ze strony MEN lub CKE.")


# --- Logika Pomocnicza (Stubs - do rozbudowy) ---

def wywolaj_llm_z_promptem(prompt: str, model_do_zadania: str) -> str:
    """
    Funkcja-symulator wywołania LLM. W docelowym rozwiązaniu
    używałaby klienta OpenAI do uzyskania odpowiedzi.
    """
    print(f"--- WYWOŁANIE MODELU '{model_do_zadania}' PRZEZ OPENROUTER ---\n{prompt}\n---------------------------")
    
    if "PROMPT-01" in prompt:
        return '{"zadania": ["analiza_prawna", "weryfikacja_cytatu"], "sygnatura": "III CZP 99/25"}'
    if "PROMPT-04" in prompt:
        return """
### Analiza Prawna
Zgodnie z Prawem oświatowym, organ prowadzący szkołę ponosi odpowiedzialność za jej działalność.

### Weryfikacja Orzeczenia
Weryfikacja orzeczenia o sygnaturze III CZP 99/25 zakończyła się niepowodzeniem. Taki wyrok nie został odnaleziony.
"""
    return "Symulowana odpowiedź LLM."

# --- Definicje Końcówek API (Actions dla GPT-s) ---

@app.post("/analyze-query", response_model=PlanZadania, summary="Analizuje zapytanie użytkownika i tworzy plan działania")
async def analyze_query(request: AnalizaRequest):
    prompt_analityk = f"""
    PROMPT-01-ANALIZA_ZAPYTANIA
    Pytanie użytkownika:
    '''
    {request.query}
    '''
    """
    wynik_json_str = wywolaj_llm_z_promptem(prompt_analityk, "openai/gpt-4o")
    plan = PlanZadania.parse_raw(wynik_json_str)
    return plan

@app.post("/gate-and-format-response", response_model=str, summary="Składa komponenty w finalną, bezpieczną odpowiedź")
async def gate_and_format_response(komponenty: KomponentyDoSyntezy):
    prompt_redaktor = f"""
    PROMPT-04-SYNTEZA_ODPOWIEDZI
    Komponent "Analiza Prawna":
    '''
    {komponenty.analiza_prawna or "Brak danych."}
    '''
    Komponent "Wynik Weryfikacji Cytatu":
    '''
    {komponenty.wynik_weryfikacji or "Brak danych."}
    '''
    Komponent "Biuletyn Informacyjny":
    '''
    {komponenty.biuletyn_informacyjny or "Brak danych."}
    '''
    """
    finalna_odpowiedz = wywolaj_llm_z_promptem(prompt_redaktor, "mistralai/mistral-7b-instruct")
    return finalna_odpowiedz

# --- NOWY ELEMENT: ENDPOINT HEALTH CHECK DLA RENDER ---
@app.get("/health", summary="Sprawdza stan aplikacji")
def health_check():
    """
    Ten endpoint jest używany przez Render do weryfikacji,
    czy aplikacja działa poprawnie.
    """
    return {"status": "ok"}