import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from config import settings


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: dict


class VectorDatabase:
    """Persistent FAISS index with a JSON sidecar for document metadata."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._records: list[dict] = []
        self._lock = asyncio.Lock()
        self._loaded = False

    @property
    def _index_path(self) -> Path:
        return Path(settings.faiss_index_path)

    @property
    def _metadata_path(self) -> Path:
        return Path(settings.faiss_metadata_path)

    def _load(self) -> None:
        if self._loaded:
            return
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index_path.exists() and self._metadata_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                records = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            except (OSError, RuntimeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Could not load the persisted FAISS database") from exc
            if not isinstance(records, list) or any(
                not isinstance(record, dict)
                or not isinstance(record.get("id"), str)
                or not isinstance(record.get("payload"), dict)
                for record in records
            ):
                raise RuntimeError("FAISS metadata sidecar has an invalid format")
            self._records = records
        else:
            self._index = faiss.IndexFlatIP(settings.embedding_dimensions)
            self._records = []
        if self._index.ntotal != len(self._records):
            raise RuntimeError("FAISS index and metadata sidecar are out of sync")
        self._loaded = True

    def _persist(self) -> None:
        assert self._index is not None
        index_temp_path = self._index_path.with_suffix(".faiss.tmp")
        metadata_temp_path = self._metadata_path.with_suffix(".json.tmp")
        try:
            faiss.write_index(self._index, str(index_temp_path))
            metadata_temp_path.write_text(
                json.dumps(self._records, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(index_temp_path, self._index_path)
            os.replace(metadata_temp_path, self._metadata_path)
        finally:
            index_temp_path.unlink(missing_ok=True)
            metadata_temp_path.unlink(missing_ok=True)

    async def ensure_collection_exists(self) -> None:
        async with self._lock:
            self._load()

    async def batch_insert(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        if len(vectors) != len(payloads):
            raise ValueError(
                f"vectors and payloads must have the same length (got {len(vectors)} and {len(payloads)})"
            )
        if not vectors:
            return
        async with self._lock:
            self._load()
            assert self._index is not None
            matrix = np.asarray(vectors, dtype="float32")
            if matrix.ndim != 2 or matrix.shape[1] != self._index.d:
                raise ValueError(f"Expected vectors with {self._index.d} dimensions")
            if not np.isfinite(matrix).all():
                raise ValueError("Vectors must contain only finite numbers")
            previous_index = faiss.clone_index(self._index)
            previous_records = self._records.copy()
            faiss.normalize_L2(matrix)
            self._index.add(matrix)
            self._records.extend(
                {"id": str(uuid.uuid4()), "payload": payload}
                for payload in payloads
            )
            try:
                self._persist()
            except Exception:
                self._index = previous_index
                self._records = previous_records
                raise

    async def semantic_search(
        self,
        query_vector: list[float],
        filters: dict | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        async with self._lock:
            self._load()
            assert self._index is not None
            if not self._records:
                return []
            query = np.asarray([query_vector], dtype="float32")
            if query.shape[1] != self._index.d:
                raise ValueError(f"Expected vectors with {self._index.d} dimensions")
            if not np.isfinite(query).all():
                raise ValueError("Query vector must contain only finite numbers")
            faiss.normalize_L2(query)
            limit = len(self._records) if filters else min(
                top_k or settings.top_k_results, len(self._records)
            )
            scores, indexes = self._index.search(query, limit)
            results = []
            for score, index in zip(scores[0], indexes[0]):
                if index < 0:
                    continue
                record = self._records[index]
                payload = record["payload"]
                if filters and any(payload.get(key) != value for key, value in filters.items()):
                    continue
                results.append(
                    SearchResult(
                        id=record["id"],
                        score=float(score),
                        text=payload.get("text", ""),
                        metadata={k: v for k, v in payload.items() if k != "text"},
                    )
                )
            return results

    async def get_by_id(self, point_id: str) -> dict | None:
        async with self._lock:
            self._load()
            return next(
                (record["payload"] for record in self._records if record["id"] == point_id),
                None,
            )

    async def _remove(self, predicate: Callable[[dict], bool]) -> int:
        async with self._lock:
            self._load()
            original_records = self._records
            kept_indexes = [
                index for index, record in enumerate(original_records)
                if not predicate(record)
            ]
            kept = [original_records[index] for index in kept_indexes]
            removed = len(original_records) - len(kept)
            if removed:
                assert self._index is not None
                vectors = np.zeros((len(kept), self._index.d), dtype="float32")
                for new_index, old_index in enumerate(kept_indexes):
                    self._index.reconstruct(old_index, vectors[new_index])
                previous_index = self._index
                previous_records = self._records
                self._index = faiss.IndexFlatIP(previous_index.d)
                if len(vectors):
                    self._index.add(vectors)
                self._records = kept
                try:
                    self._persist()
                except Exception:
                    self._index = previous_index
                    self._records = previous_records
                    raise
            return removed

    async def delete_by_id(self, point_id: str) -> bool:
        return await self._remove(lambda record: record["id"] == point_id) > 0

    async def delete_by_filter(self, filters: dict) -> int:
        return await self._remove(
            lambda record: all(
                record["payload"].get(key) == value for key, value in filters.items()
            )
        )

    async def get_collection_info(self) -> dict:
        async with self._lock:
            self._load()
            assert self._index is not None
            return {
                "name": self._index_path.stem,
                "points_count": len(self._records),
                "vectors_count": self._index.ntotal,
                "status": "ready",
                "backend": "faiss",
            }

    async def close(self) -> None:
        return None


db = VectorDatabase()
