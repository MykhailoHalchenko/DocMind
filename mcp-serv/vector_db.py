import asyncio
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
        self._lock = asyncio.Lock()

    async def _get_client(self) -> AsyncQdrantClient:
        if self._client is not None:
            return self._client
        async with self._lock:
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
        if len(vectors) != len(payloads):
            raise ValueError(
                f"vectors and payloads must have the same length (got {len(vectors)} and {len(payloads)})"
            )
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
                text=(payload := (hit.payload or {})).get("text", ""),
                metadata={k: v for k, v in payload.items() if k != "text"},
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

    async def delete_by_id(self, point_id: str) -> bool:
        """Delete a document by ID. Returns True if deleted."""
        client = await self._get_client()
        try:
            await client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=point_id,
            )
            return True
        except Exception:
            return False

    async def delete_by_filter(self, filters: dict) -> int:
        """Delete documents matching filter. Returns approximate count."""
        client = await self._get_client()
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)
        try:
            await client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=qdrant_filter,
            )
            return -1  # Qdrant doesn't return count
        except Exception:
            return 0

    async def get_collection_info(self) -> dict:
        """Get information about the collection."""
        client = await self._get_client()
        try:
            info = await client.get_collection(settings.qdrant_collection)
            return {
                "name": settings.qdrant_collection,
                "points_count": info.points_count or 0,
                "vectors_count": info.vectors_count or 0,
                "status": info.status.value if info.status else "unknown",
            }
        except Exception:
            return {"name": settings.qdrant_collection, "points_count": 0, "error": "Collection not found"}

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


db = VectorDatabase()