from openai import AsyncOpenAI
from config import settings

_strong_client = AsyncOpenAI(
    api_key=settings.strong_llm_api_key,
    base_url=settings.strong_llm_base_url,
)

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


async def rag_answer(question: str, context_chunks: list[dict]) -> str:
    context_parts = []
    for c in context_chunks:
        chunk_id = c.get("id", "?")
        text = c.get("text", "")
        context_parts.append(f"[{chunk_id}] {text}")

    context = "\n\n".join(context_parts)

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


async def map_reduce_summarize(chunks: list[str]) -> str:
    import asyncio

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