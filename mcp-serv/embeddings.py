from openai import AsyncOpenAI
from config import settings

_client = AsyncOpenAI(api_key=settings.openai_api_key)


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = await _client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        dimensions=settings.embedding_dimensions,
    )
    return [item.embedding for item in response.data]


async def get_single_embedding(text: str) -> list[float]:
    results = await get_embeddings([text])
    return results[0]