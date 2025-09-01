import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import uvicorn

# --- KONFIGURACJA ---
load_dotenv()
app = FastAPI()

# --- ZMIANA ARCHITEKTONICZNA: Leniwa Inicjalizacja Klienta AI ---
# Klient nie jest już tworzony globalnie przy starcie aplikacji.
# Zostanie utworzony przy pierwszym wywołaniu funkcji, która go potrzebuje.
_client = None

def get_client():
    """
    Funkcja tworzy klienta AI przy pierwszym użyciu (leniwa inicjalizacja).
    Gwarantuje to, że zmienne środowiskowe są już załadowane.
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            # Ta wiadomość pojawi się w logach Render, jeśli klucz nie zostanie znaleziony
            print("KRYTYCZNY BŁĄD: Zmienna środowiskowa OPENROUTER_API_KEY nie jest ustawiona lub jest pusta!")
            raise ValueError("API key for OpenRouter is not configured.")
        
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client

# --- MIKRO-INSTRUKCJE (PROMPTY) ---
# ... (ta sekcja pozostaje bez zmian) ...
PROMPT_KATEGORYZACJA = """
Your only task is to assess if the following query is exclusively about Polish educational law. Your domain includes: Teacher's Charter, school management, student rights, pedagogical supervision. Topics like general civil law, copyright law, construction law, or public procurement law are OUTSIDE YOUR DOMAIN. Answer only 'TAK' or 'NIE'.
Query: "{query}"
"""
PROMPT_ANALIZA_ZAPYTANIA = """
Your task is to decompose the user's query into a simple action plan in JSON format. Analyze the query and return a JSON with a list of tasks and a case signature if applicable. Allowed tasks are: 'analiza_prawna', 'weryfikacja_cytatu', 'biuletyn_informacyjny'. For simple queries, return only 'analiza_prawna'.
Query: "{query}"
"""
PROMPT_SYNTEZA_ODPOWIEDZI = """
You are an editor in a law firm specializing in educational law. Your task is to assemble the following verified components into a single, coherent, professional, and clear response in Markdown format for a client. The response MUST be in Polish.
Here are the components:
- Legal analysis from the knowledge base: {analiza_prawna}
- Citation verification result: {wynik_weryfikacji}
- Information from the news bulletin (latest changes): {biuletyn_informacyjny}
Create a response that is readable and helpful for a school principal or teacher.
"""

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
        # Używamy funkcji get_client() zamiast globalnego obiektu client
        client = get_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        # Ten błąd będzie teraz widoczny w logach, jeśli wystąpi
        print(f"Błąd podczas wywołania LLM: {e}")
        raise HTTPException(status_code=500, detail=f"AI model call error: {e}")

# --- ENDPOINTY API ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze-query")
async def analyze_query(request: QueryRequest):
    prompt_kategoryzacji = PROMPT_KATEGORYZACJA.format(query=request.query)
    kategoria = await llm_call(prompt_kategoryzacji, model="mistralai/mistral-7b-instruct:free")
    if "NIE" in kategoria.upper():
        return {"zadania": ["ODRZUCONE_SPOZA_DOMENY"], "sygnatura": ""}
    
    prompt_analizy = PROMPT_ANALIZA_ZAPYTANIA.format(query=request.query)
    plan_json_str = await llm_call(prompt_analizy)
    try:
        plan_zadania = json.loads(plan_json_str)
        return plan_zadania
    except json.JSONDecodeError:
        return {"zadania": ["analiza_prawna"], "sygnatura": ""}

@app.post("/gate-and-format-response")
async def gate_and_format_response(request: SynthesisRequest):
    if request.analiza_prawna == "ODRZUCONE_SPOZA_DOMENY":
        return "Dziękuję za Twoje pytanie. Nazywam się Asystent Prawa Oświatowego, a moja wiedza jest specjalistycznie ograniczona wyłącznie do zagadnienie polskiego prawa oświatowego. Twoje pytanie dotyczy innej dziedziny prawa i wykracza poza ten zakres. Nie mogę udzielić informacji na ten temat."

    prompt_syntezy = PROMPT_SYNTEZA_ODPOWIEDZI.format(
        analiza_prawna=request.analiza_prawna or "Brak danych.",
        wynik_weryfikacji=request.wynik_weryfikacji or "Nie dotyczy.",
        biuletyn_informacyjny=request.biuletyn_informacyjny or "Brak danych."
    )
    final_response = await llm_call(prompt_syntezy)
    return final_response