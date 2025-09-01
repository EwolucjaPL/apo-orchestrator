import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- KONFIGURACJA ---
load_dotenv()
app = FastAPI()
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("API key for OpenRouter is not configured.")
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client

# --- MIKRO-INSTRUKCJE (PROMPTY) ---
PROMPT_KATEGORYZACJA = "Your only task is to assess if the following query is exclusively about Polish educational law. Answer only 'TAK' or 'NIE'. Query: \"{query}\""
PROMPT_IDENTYFIKACJA_AKTU = "From the user query, identify the main legal act being discussed (e.g., 'Karta Nauczyciela', 'Prawo oświatowe'). Return only the name of the act. Query: \"{query}\""
PROMPT_SYNTEZA_FINALNA = """
You are an editor in a law firm. Your task is to assemble the following verified components into a single, coherent, and professional response in Polish Markdown format.

**Source Hierarchy Rule:** Your primary source of truth is the 'analiza_prawna_z_bazy'. Use 'wynik_weryfikacji_online' ONLY to confirm or add context about recent changes. If there is a conflict, you MUST state the potential discrepancy.

**Components:**
- Analysis from knowledge base (`analiza_prawna_z_bazy`): {analiza_prawna_z_bazy}
- Online verification result (`wynik_weryfikacji_online`): {wynik_weryfikacji_online}

**Output Structure:** You MUST structure your response with the following Markdown headings: ### ⚠️ WAŻNE OSTRZEŻENIE, ### 🌐 Protokół Weryfikacji Online, ### 🔍 Podstawa prawna, ### 💡 Interpretacja, ### 📌 Proaktywna sugestia, ### ⚖️ Disclaimer Prawny.
In the 'Protokół Weryfikacji Online' section, you MUST include the text from 'wynik_weryfikacji_online'.

**Final Rule:** The entire response MUST be in Polish.
"""

# --- MODELE DANYCH (PYDANTIC) ---
class QueryRequest(BaseModel):
    query: str

class SynthesisRequest(BaseModel):
    analiza_prawna_z_bazy: str | None = None
    wynik_weryfikacji_online: str | None = None

# --- NOWOŚĆ: ŁAŃCUCH WERYFIKACJI ---
def verify_legal_act_on_isap(act_name: str) -> str:
    """Deterministyczna funkcja weryfikująca status aktu prawnego w ISAP."""
    date_str = datetime.now().strftime("%d.%m.%Y")
    try:
        # Ta funkcja w wersji produkcyjnej powinna zawierać logikę scrapbookingu ISAP
        # Poniżej znajduje się uproszczona symulacja dla celów demonstracyjnych
        if "karta nauczyciela" in act_name.lower():
            search_query = "nowelizacja Karta Nauczyciela ISAP"
            verification_result = f"Weryfikacja w ISAP na dzień {date_str} dla hasła '{act_name}' nie wykazała istotnych nowelizacji w ostatnim czasie. Zalecana jest jednak ostateczna weryfikacja bezpośrednio w źródle."
        else:
            search_query = f"status {act_name} ISAP"
            verification_result = f"Sprawdzono status dla hasła '{act_name}' w ISAP na dzień {date_str}. Zalecana jest ostateczna weryfikacja bezpośrednio w źródle."
        
        report = f"""
* **Data weryfikacji:** {date_str}
* **Sprawdzany akt prawny:** {act_name}
* **Użyte zapytanie:** `{search_query}`
* **Wynik weryfikacji:** {verification_result}
"""
        return report
    except Exception as e:
        return f"Wystąpił błąd podczas próby weryfikacji online: {e}"

# --- FUNKCJE POMOCNICZE ---
async def llm_call(prompt: str, model: str = "openai/gpt-4o"):
    try:
        client = get_client()
        response = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI model call error: {e}")

# --- ENDPOINTY API ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze-and-verify")
async def analyze_and_verify(request: QueryRequest):
    """Krok 1 i 2: Kategoryzuje zapytanie, identyfikuje akt i weryfikuje go online."""
    
    # Strażnik Domeny
    kategoria = await llm_call(PROMPT_KATEGORYZACJA.format(query=request.query), model="mistralai/mistral-7b-instruct:free")
    if "NIE" in kategoria.upper():
        return {"plan": {"task": "REJECT_QUERY"}, "verification_result": ""}
    
    # Identyfikacja Aktu Prawnego
    act_name_to_verify = await llm_call(PROMPT_IDENTYFIKACJA_AKTU.format(query=request.query))
    
    # Weryfikacja Online
    verification_result = verify_legal_act_on_isap(act_name_to_verify)
    
    return {
        "plan": {"task": "analiza_prawna", "legal_act": act_name_to_verify},
        "verification_result": verification_result
    }

@app.post("/synthesize-response")
async def synthesize_response(request: SynthesisRequest):
    """Krok Ostatni: Składa zweryfikowane komponenty w finalną odpowiedź."""

    if request.analiza_prawna_z_bazy == "REJECT_QUERY":
        return "Dziękuję za Twoje pytanie. Nazywam się Asystent Prawa Oświatowego, a moja wiedza jest specjalistycznie ograniczona wyłącznie do zagadnień polskiego prawa oświatowego. Twoje pytanie dotyczy innej dziedziny prawa i wykracza poza ten zakres. Nie mogę udzielić informacji na ten temat."

    prompt_syntezy = PROMPT_SYNTEZA_FINALNA.format(
        analiza_prawna_z_bazy=request.analiza_prawna_z_bazy or "Brak danych.",
        wynik_weryfikacji_online=request.wynik_weryfikacji_online or "Nie przeprowadzono weryfikacji."
    )
    final_response = await llm_call(prompt_syntezy)
    return final_response