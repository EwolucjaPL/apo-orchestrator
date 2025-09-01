# --- KROK 1: START PLIKU ---
print("--- [DIAGNOSTYKA] Krok 1: Uruchamianie pliku main.py ---")

import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import uvicorn

# --- KROK 2: PO IMPORTACH ---
print("--- [DIAGNOSTYKA] Krok 2: Importy zakończone pomyślnie. ---")

# --- KONFIGURACJA ---
load_dotenv()
app = FastAPI()

print("--- [DIAGNOSTYKA] Krok 3: Aplikacja FastAPI zainicjalizowana. ---")

# --- ZMIANA ARCHITEKTONICZNA: Leniwa Inicjalizacja Klienta AI ---
_client = None

def get_client():
    global _client
    if _client is None:
        print("--- [DIAGNOSTYKA] Próba inicjalizacji klienta AI (pierwsze wywołanie)... ---")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("--- [BŁĄD KRYTYCZNY] Nie znaleziono zmiennej środowiskowej OPENROUTER_API_KEY! ---")
            raise ValueError("API key for OpenRouter is not configured.")
        
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        print("--- [DIAGNOSTYKA] Klient AI zainicjalizowany pomyślnie. ---")
    return _client

# --- MIKRO-INSTRUKCJE (PROMPTY) ---
PROMPT_KATEGORYZACJA = "Your only task is to assess if the following query is exclusively about Polish educational law..."
PROMPT_ANALIZA_ZAPYTANIA = "Your task is to decompose the user's query into a simple action plan in JSON format..."
PROMPT_SYNTEZA_ODPOWIEDZI = "You are an editor in a law firm specializing in educational law..."

# --- MODELE DANYCH (PYDANTIC) ---
class QueryRequest(BaseModel):
    query: str
class SynthesisRequest(BaseModel):
    analiza_prawna: str | None = None
    wynik_weryfikacji: str | None = None
    biuletyn_informacyjny: str | None = None

# --- FUNKCJE POMOCNICZE ---
async def llm_call(prompt: str, model: str = "openai/gpt-4o"):
    try:
        client = get_client()
        response = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"--- [BŁĄD KRYTYCZNY] Błąd podczas wywołania LLM: {e} ---")
        raise HTTPException(status_code=500, detail=f"AI model call error: {e}")

# --- ENDPOINTY API ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/test-endpoint")
def test_endpoint():
    return {"message": "Test endpoint is working!"}

@app.post("/analyze-query")
async def analyze_query(request: QueryRequest):
    # ... (logika funkcji bez zmian) ...
    prompt_kategoryzacji = PROMPT_KATEGORYZACJA.format(query=request.query)
    kategoria = await llm_call(prompt_kategoryzacji, model="mistralai/mistral-7b-instruct:free")
    if "NIE" in kategoria.upper():
        return {"zadania": ["ODRZUCONE_SPOZA_DOMENY"], "sygnatura": ""}
    prompt_analizy = PROMPT_ANALIZA_ZAPYTANIA.format(query=request.query)
    plan_json_str = await llm_call(prompt_analizy)
    try:
        return json.loads(plan_json_str)
    except json.JSONDecodeError:
        return {"zadania": ["analiza_prawna"], "sygnatura": ""}

@app.post("/gate-and-format-response")
async def gate_and_format_response(request: SynthesisRequest):
    # ... (logika funkcji bez zmian) ...
    if request.analiza_prawna == "ODRZUCONE_SPOZA_DOMENY":
        return "Dziękuję za Twoje pytanie..."
    prompt_syntezy = PROMPT_SYNTEZA_ODPOWIEDZI.format(
        analiza_prawna=request.analiza_prawna or "Brak.",
        wynik_weryfikacji=request.wynik_weryfikacji or "Nie dotyczy.",
        biuletyn_informacyjny=request.biuletyn_informacyjny or "Brak."
    )
    return await llm_call(prompt_syntezy)

print("--- [DIAGNOSTYKA] Krok 4: Definicje endpointów zakończone. Aplikacja gotowa do uruchomienia przez Uvicorn. ---")