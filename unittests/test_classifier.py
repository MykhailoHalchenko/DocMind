import unittest
from unittest.mock import AsyncMock, patch

from unittests._support import configure_test_environment

configure_test_environment()

import classifier


class ClassifierTests(unittest.IsolatedAsyncioTestCase):
    def test_heuristic_detects_common_intents(self):
        self.assertEqual(classifier._classify_intent_heuristic("Please summarize this paper")["intent"], "SUMMARIZE")
        self.assertEqual(classifier._classify_intent_heuristic("How did they do the method?")["intent"], "METHODOLOGY")
        self.assertEqual(classifier._classify_intent_heuristic("Compare the results to the baseline")["intent"], "COMPARE")
        self.assertEqual(classifier._classify_intent_heuristic("Hello there")["intent"], "GREETING")

    async def test_classify_chunks_batch_preserves_order(self):
        responses = [{"category": "Results"}, {"category": "Methodology"}, {"category": "Other"}]

        async def fake_classify(text: str) -> dict:
            return responses.pop(0)

        with patch.object(classifier, "classify_chunk", side_effect=fake_classify):
            results = await classifier.classify_chunks_batch(["one", "two", "three"])

        self.assertEqual([result["category"] for result in results], ["Results", "Methodology", "Other"])


if __name__ == "__main__":
    unittest.main()