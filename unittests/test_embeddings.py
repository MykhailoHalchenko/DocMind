import unittest
from unittest.mock import AsyncMock, patch

from unittests._support import configure_test_environment

configure_test_environment()

import embeddings


class EmbeddingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_embeddings_returns_vectors_from_api_response(self):
        fake_response = type(
            "Response",
            (),
            {
                "data": [
                    type("Item", (), {"embedding": [0.1, 0.2, 0.3]})(),
                    type("Item", (), {"embedding": [0.4, 0.5, 0.6]})(),
                ]
            },
        )

        with patch.object(embeddings._client.embeddings, "create", new=AsyncMock(return_value=fake_response)):
            vectors = await embeddings.get_embeddings(["a", "b"])

        self.assertEqual(vectors, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    async def test_get_single_embedding_returns_first_vector(self):
        with patch.object(embeddings, "get_embeddings", new=AsyncMock(return_value=[[9.0, 8.0, 7.0]])):
            vector = await embeddings.get_single_embedding("query")

        self.assertEqual(vector, [9.0, 8.0, 7.0])


if __name__ == "__main__":
    unittest.main()