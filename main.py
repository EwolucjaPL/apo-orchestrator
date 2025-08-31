# main.py
import os
import re # <-- NOWY IMPORT
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import openai

# --- Tworzenie Aplikacji ---
app = FastAPI(
    title="APO-Orchestrator API",
    description="API zarządzające logiką i bezpieczeństwem agenta APO.",
    version="2.1.0", # Podniesienie wersji po poprawce
)

# --- Middleware CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["[https://chat.openai.com](https://chat.openai.com)"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Konfiguracja Klienta pod OpenRouter (BEZPIECZNA) ---
client = openai.OpenAI(
    base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# --- Definicje Modeli Danych (Pydantic) ---
class PlanZadania(BaseModel):
    zadania: List[Literal["analiza_prawna", "biuletyn_men", "biuletyn_cke", "weryfikacja_cytatu", "odpowiedz_skryptem_tajemnicy"]]
    sygnatura: Optional[str] = None

class AnalizaRequest(BaseModel):
    query: str

class KomponentyDoSyntezy(BaseModel):
    analiza_prawna: Optional[str] = Field(default=None)
    wynik_weryfikacji: Optional[str] = Field(default=None)
    biuletyn_informacyjny: Optional[str] = Field(default=None)
    specjalne_zadanie: Optional[str] = Field(default=None)

# --- NOWA FUNKCJA: Narzędzie do czyszczenia odpowiedzi AI ---
def extract_json_from_markdown(text: str) -> str:
    """Wyciąga czysty string JSON z bloku kodu Markdown."""
    match = re.search(r'```(json)?\s*(\{.*\})\s*```', text, re.DOTALL)
    if match:
        return match.group(2)
    return text # Zwróć oryginalny tekst, jeśli nie znaleziono bloku

# --- Rzeczywista Funkcja Wywołująca LLM ---
def wywolaj_llm_z_promptem(prompt: str, model_do_zadania: str) -> str:
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
        raise HTTPException(status_code=500, detail="Błąd komunikacji z modelem AI.")

# --- Główne Endpointy API (Actions dla GPT-s) ---

@app.post("/analyze-query", response_model=PlanZadania, summary="Analizuje zapytanie użytkownika i tworzy plan działania")
async def analyze_query(request: AnalizaRequest):
    """Krok 1: Rola 'Analityk' z wbudowanym 'Strażnikiem'."""
    forbidden_keywords = ["źródła wiedzy", "baza wiedzy", "twoje instrukcje", "jak działasz", "podaj swoje źródła"]
    if any(keyword in request.query.lower() for keyword in forbidden_keywords):
        return PlanZadania(zadania=["odpowiedz_skryptem_tajemnicy"])
    
    prompt_analityk = f"PROMPT-01-ANALIZA_ZAPYTANIA\nPytanie użytkownika:\n'''\n{request.query}\n'''"
    try:
        wynik_llm_str = wywolaj_llm_z_promptem(prompt_analityk, "openai/gpt-4o")
        # --- ZASTOSOWANIE POPRAWKI ---
        czysty_json_str = extract_json_from_markdown(wynik_llm_str)
        plan = PlanZadania.parse_raw(czysty_json_str)
        return plan
    except Exception as e:
        # Ten błąd powinien teraz zniknąć
        raise HTTPException(status_code=500, detail=f"Błąd podczas analizy zapytania przez AI: {e}")

@app.post("/gate-and-format-response", response_model=str, summary="Składa komponenty w finalną, bezpieczną odpowiedź")
async def gate_and_format_response(komponenty: KomponentyDoSyntezy):
    """Krok Ostatni: Rola 'Redaktor'."""
    if komponenty.specjalne_zadanie == "odpowiedz_skryptem_tajemnicy":
        return "Moje odpowiedzi opierają się na wewnętrznej bazie wiedzy, która została opracowana przez zespół ekspertów z zakresu prawa oświatowego. Zgodnie z zasadami działania, szczegółowe źródła techniczne i materiały pomocnicze nie są udostępniane."

    prompt_redaktor = f"""PROMPT-04-SYNTEZA_ODPOWIEDZI\nKomponent "Analiza Prawna":\n'''\n{komponenty.analiza_prawna or "Brak danych."}\n'''\nKomponent "Wynik Weryfikacji Cytatu":\n'''\n{komponenty.wynik_weryfikacji or "Brak danych."}\n'''\nKomponent "Biuletyn Informacyjny":\n'''\n{komponenty.biuletyn_informacyjny or "Brak danych."}\n'''"""
    finalna_odpowiedz = wywolaj_llm_z_promptem(prompt_redaktor, "mistralai/mistral-7b-instruct")
    return finalna_odpowiedz

@app.get("/health", summary="Sprawdza stan aplikacji")
def health_check():
    """Ten endpoint jest używany przez Render do weryfikacji, czy aplikacja działa poprawnie."""
    return {"status": "ok"}