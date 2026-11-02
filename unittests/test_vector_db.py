import sys
import tempfile
import unittest
from pathlib import Path

from unittests._support import configure_test_environment

configure_test_environment()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp-serv"))

import vector_db


class VectorDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_path = Path(self.temp_dir.name) / "index.faiss"
        self.metadata_path = Path(self.temp_dir.name) / "metadata.json"
        self.old_values = (
            vector_db.settings.faiss_index_path,
            vector_db.settings.faiss_metadata_path,
            vector_db.settings.embedding_dimensions,
        )
        vector_db.settings.faiss_index_path = str(self.index_path)
        vector_db.settings.faiss_metadata_path = str(self.metadata_path)
        vector_db.settings.embedding_dimensions = 3
        self.database = vector_db.VectorDatabase()

    async def asyncTearDown(self):
        (
            vector_db.settings.faiss_index_path,
            vector_db.settings.faiss_metadata_path,
            vector_db.settings.embedding_dimensions,
        ) = self.old_values
        self.temp_dir.cleanup()

    async def test_persists_searches_filters_and_deletes(self):
        await self.database.batch_insert(
            [[1, 0, 0], [0, 1, 0]],
            [{"text": "alpha", "kind": "a"}, {"text": "beta", "kind": "b"}],
        )
        results = await self.database.semantic_search([1, 0, 0], top_k=1)
        self.assertEqual(results[0].text, "alpha")

        filtered = await self.database.semantic_search(
            [1, 0, 0], filters={"kind": "b"}, top_k=1
        )
        self.assertEqual(filtered[0].text, "beta")
        self.assertEqual(await self.database.delete_by_filter({"kind": "a"}), 1)

        restored = vector_db.VectorDatabase()
        self.assertEqual((await restored.get_collection_info())["points_count"], 1)

    async def test_rejects_invalid_vectors_and_top_k(self):
        with self.assertRaises(ValueError):
            await self.database.batch_insert([[1, 0]], [{"text": "bad"}])
        with self.assertRaises(ValueError):
            await self.database.batch_insert([[float("nan"), 0, 0]], [{"text": "bad"}])
        with self.assertRaises(ValueError):
            await self.database.semantic_search([1, 0, 0], top_k=0)
