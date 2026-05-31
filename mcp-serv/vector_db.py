import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)

from config import settings


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: dict


class VectorDatabase:
    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None

    async def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        return self._client

    async def ensure_collection_exists(self) -> None:
        client = await self._get_client()
        exists = await client.collection_exists(settings.qdrant_collection)
        if not exists:
            await client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )

    async def batch_insert(
        self,
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        client = await self._get_client()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
            for vector, payload in zip(vectors, payloads)
        ]
        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )

    async def semantic_search(
        self,
        query_vector: list[float],
        filters: dict | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        client = await self._get_client()
        limit = top_k or settings.top_k_results

        qdrant_filter: Filter | None = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        hits: list[ScoredPoint] = await client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            SearchResult(
                id=str(hit.id),
                score=hit.score,
                text=hit.payload.get("text", ""),
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
            )
            for hit in hits
        ]

    async def get_by_id(self, point_id: str) -> dict | None:
        client = await self._get_client()
        results = await client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[point_id],
            with_payload=True,
        )
        if not results:
            return None
        return results[0].payload

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


db = VectorDatabase()