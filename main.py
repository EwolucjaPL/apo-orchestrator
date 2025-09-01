import os
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
text = re.sub(r"(^|\n)\s*#{1,6}.*", " ", text) # nagłówki
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