import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from config import settings


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
) -> dict:
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.embedding_model,
        "input": texts,
        "dimensions": settings.embedding_dimensions,
    }

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        data = await _post_with_retry(
            client,
            "https://api.openai.com/v1/embeddings",
            payload,
        )

    return [item["embedding"] for item in data["data"]]


async def get_single_embedding(text: str) -> list[float]:
    results = await get_embeddings([text])
    return results[0]