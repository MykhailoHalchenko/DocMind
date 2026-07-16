import asyncio
import logging

from openai import AsyncOpenAI
from config import settings

try:
    from google import genai as google_genai
except Exception:  # pragma: no cover - optional dependency
    google_genai = None

_logger = logging.getLogger(__name__)

_strong_client = AsyncOpenAI(
    api_key=settings.strong_llm_api_key,
    base_url=settings.strong_llm_base_url,
)

_gemini_client = None
if google_genai is not None and settings.gemini_api_key:
    try:
        _gemini_client = google_genai.Client(api_key=settings.gemini_api_key)
    except Exception as e:  # pragma: no cover - optional dependency
        _logger.warning(f"Gemini client initialization failed: {e}")

RAG_SYSTEM_PROMPT = """You are a precise scientific assistant.
Rules:
1. Use ONLY the provided context to answer.
2. After each fact, cite the source like [chunk_id].
3. If the answer is not in the context, respond: "Insufficient data in the provided sources."
4. Never hallucinate or infer beyond the context."""

MAP_PROMPT = """Summarize the following text chunk in 2-3 sentences, preserving key facts, numbers, and findings:

{chunk}

Summary:"""

REDUCE_PROMPT = """You are given partial summaries of a scientific document. 
Combine them into one coherent final summary (5-7 sentences). 
Preserve all key findings, numbers, and citations.

Summaries:
{summaries}

Final summary:"""


def _should_use_gemini(provider: str | None) -> bool:
    return provider == "gemini" and _gemini_client is not None


def _extract_gemini_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                return part_text.strip()

    return str(response).strip()


async def _gemini_generate_text(prompt: str) -> str:
    if _gemini_client is None:
        raise RuntimeError("Gemini client is not configured")

    def _call() -> str:
        response = _gemini_client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        return _extract_gemini_text(response)

    return await asyncio.to_thread(_call)


async def rag_answer(question: str, context_chunks: list[dict], provider: str | None = None) -> str:
    context_parts = []
    for c in context_chunks:
        chunk_id = c.get("id", "?")
        text = c.get("text", "")
        context_parts.append(f"[{chunk_id}] {text}")

    context = "\n\n".join(context_parts)

    if _should_use_gemini(provider):
        gemini_prompt = f"""{RAG_SYSTEM_PROMPT}

Context:
{context}

Question: {question}"""
        return await _gemini_generate_text(gemini_prompt)

    response = await _strong_client.chat.completions.create(
        model=settings.strong_llm_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


async def _summarize_chunk(client: AsyncOpenAI, model: str, chunk_text: str) -> str:
    prompt = MAP_PROMPT.format(chunk=chunk_text[:3000])
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


async def _summarize_chunk_gemini(chunk_text: str) -> str:
    prompt = MAP_PROMPT.format(chunk=chunk_text[:3000])
    return await _gemini_generate_text(prompt)


async def map_reduce_summarize(chunks: list[str], provider: str | None = None) -> str:
    if _should_use_gemini(provider):
        map_tasks = [_summarize_chunk_gemini(c) for c in chunks]
        partial_summaries = await asyncio.gather(*map_tasks)

        combined = "\n\n".join(f"- {s}" for s in partial_summaries)
        prompt = REDUCE_PROMPT.format(summaries=combined)
        return await _gemini_generate_text(prompt)

    map_tasks = [_summarize_chunk(_strong_client, settings.strong_llm_model, c) for c in chunks]
    partial_summaries = await asyncio.gather(*map_tasks)

    combined = "\n\n".join(f"- {s}" for s in partial_summaries)
    prompt = REDUCE_PROMPT.format(summaries=combined)

    response = await _strong_client.chat.completions.create(
        model=settings.strong_llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()