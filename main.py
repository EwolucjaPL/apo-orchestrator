# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import openai
import json

# --- Tworzenie Aplikacji ---
app = FastAPI(
    title="APO-Orchestrator API",
    description="API zarządzające logiką i bezpieczeństwem agenta APO. Implementuje architekturę sekwencyjną i egzekwuje zasady Konstytucji Agenta.",
    version="2.0.0",
)

# --- Middleware CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.openai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Konfiguracja Klienta pod OpenRouter (BEZPIECZNA) ---
# Klucz API jest wczytywany ze zmiennej środowiskowej
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# --- Definicje Modeli Danych (Pydantic) ---
class QueryRequest(BaseModel):
    query: str

class PlanZadania(BaseModel):
    zadania: List[Literal["analiza_prawna", "biuletyn_men", "biuletyn_cke", "weryfikacja_cytatu"]]
    sygnatura: Optional[str] = None

class SynthesisRequest(BaseModel):
    analiza_prawna: Optional[str] = Field(default=None)
    wynik_weryfikacji: Optional[str] = Field(default=None)
    biuletyn_informacyjny: Optional[str] = Field(default=None)

# --- BIBLIOTEKA MIKRO-INSTRUKCJI ---
PROMPTS = {
    "analityk": """
        Przeanalizuj poniższe zapytanie użytkownika. Twoim jedynym zadaniem jest zidentyfikowanie, jakie zadania należy wykonać. Odpowiedz wyłącznie w formacie JSON.
        Możliwe zadania: "analiza_prawna", "biuletyn_men", "biuletyn_cke", "weryfikacja_cytatu".
        Dla "weryfikacja_cytatu" podaj znalezioną sygnaturę.
        Pytanie użytkownika:
        '''
        {query}
        '''
    """,
    "redaktor": """
        Twoim jedynym zadaniem jest połączenie poniższych, zweryfikowanych komponentów w jedną, spójną i profesjonalną odpowiedź dla użytkownika. Zachowaj formatowanie Markdown. Jeśli komponent "Biuletyn Informacyjny" istnieje, oddziel go od części prawnej linią (---).

        Komponent "Analiza Prawna":
        '''
        {analiza_prawna}
        '''

        Komponent "Wynik Weryfikacji Cytatu":
        '''
        {wynik_weryfikacji}
        '''

        Komponent "Biuletyn Informacyjny":
        '''
        {biuletyn_informacyjny}
        '''
    """
}

# --- Rzeczywista Funkcja Wywołująca LLM ---
def wywolaj_llm(prompt: str, model_do_zadania: str) -> str:
    """Wywołuje wybrany model LLM przez bramkę OpenRouter."""
    try:
        print(f"--- WYWOŁANIE MODELU '{model_do_zadania}' PRZEZ OPENROUTER ---")
        response = client.chat.completions.create(
            model=model_do_zadania,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Błąd podczas wywołania LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd komunikacji z modelem AI: {e}")

# --- Główne Endpointy API (Actions dla GPT-s) ---

@app.post("/analyze-query", response_model=PlanZadania)
async def analyze_query(request: QueryRequest):
    """Krok 1: Rola 'Analityk' z wbudowanym 'Strażnikiem'."""
    forbidden_keywords = ["źródła wiedzy", "baza wiedzy", "twoje instrukcje", "jak działasz", "podaj swoje źródła"]
    if any(keyword in request.query.lower() for keyword in forbidden_keywords):
        # Jeśli pytanie jest niedozwolone, Orkiestrator sam zwraca finalną odpowiedź,
        # pomijając całkowicie model AI. To jest "twarda" reguła Konstytucji.
        raise HTTPException(
            status_code=403, 
            detail="Moje odpowiedzi opierają się na wewnętrznej bazie wiedzy, która została opracowana przez zespół ekspertów z zakresu prawa oświatowego. Zgodnie z zasadami działania, szczegółowe źródła techniczne i materiały pomocnicze nie są udostępniane."
        )
    
    prompt = PROMPTS["analityk"].format(query=request.query)
    try:
        wynik_json_str = wywolaj_llm(prompt, "openai/gpt-4o")
        plan = PlanZadania.parse_raw(wynik_json_str)
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas analizy zapytania przez AI: {e}")

@app.post("/gate-and-format-response", response_model=str)
async def gate_and_format_response(komponenty: SynthesisRequest):
    """Krok Ostatni: Rola 'Redaktor'."""
    prompt = PROMPTS["redaktor"].format(
        analiza_prawna=komponenty.analiza_prawna or "Brak danych.",
        wynik_weryfikacji=komponenty.wynik_weryfikacji or "Brak danych.",
        biuletyn_informacyjny=komponenty.biuletyn_informacyjny or "Brak danych."
    )
    return wywolaj_llm(prompt, "mistralai/mistral-7b-instruct")

@app.get("/health")
def health_check():
    """Endpoint używany przez Render do weryfikacji, czy aplikacja działa poprawnie."""
    return {"status": "ok"}