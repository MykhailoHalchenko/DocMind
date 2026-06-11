import json
from openai import AsyncOpenAI
from config import settings

_fast_client = AsyncOpenAI(
    api_key=settings.fast_llm_api_key,
    base_url=settings.fast_llm_base_url,
)

CHUNK_CATEGORIZATION_PROMPT = """You are a scientific document classifier.
Given a text chunk, return ONLY a JSON object (no markdown) with:
- "category": one of ["Introduction", "Methodology", "Results", "Discussion", "Conclusion", "Literature Review", "Limitations", "Other"]
- "keywords": list of 3-5 key terms
- "sentiment": one of ["neutral", "positive", "negative"]

Example output:
{"category": "Methodology", "keywords": ["sample size", "randomized", "control group"], "sentiment": "neutral"}"""

USER_INTENT_PROMPT = """You are an intent classifier for a scientific RAG system.
Given a user query, return ONLY a JSON object (no markdown) with:
- "intent": one of ["SUMMARIZE", "FIND_FACT", "COMPARE", "METHODOLOGY", "GENERAL", "GREETING"]
- "filters": object with optional keys "category" (chunk category to filter by) or null
- "complexity": one of ["simple", "complex"]

Example output:
{"intent": "FIND_FACT", "filters": {"category": "Results"}, "complexity": "simple"}"""


async def classify_chunk(text: str) -> dict:
    response = await _fast_client.chat.completions.create(
        model=settings.fast_llm_model,
        messages=[
            {"role": "system", "content": CHUNK_CATEGORIZATION_PROMPT},
            {"role": "user", "content": text[:1500]},
        ],
        temperature=0,
        max_tokens=150,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"category": "Other", "keywords": [], "sentiment": "neutral"}


async def classify_user_intent(question: str) -> dict:
    response = await _fast_client.chat.completions.create(
        model=settings.fast_llm_model,
        messages=[
            {"role": "system", "content": USER_INTENT_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "GENERAL", "filters": None, "complexity": "simple"}


async def classify_chunks_batch(texts: list[str]) -> list[dict]:
    import asyncio
    tasks = [classify_chunk(t) for t in texts]
    return await asyncio.gather(*tasks)