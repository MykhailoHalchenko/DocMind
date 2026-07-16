import unittest

from unittests._support import configure_test_environment

configure_test_environment()

from chunking import chunk_documents, clean_pdf_text, split_into_chunks


class ChunkingTests(unittest.TestCase):
    def test_clean_pdf_text_normalizes_headers_and_spacing(self):
        raw_text = "Figure 1 something\n\n1\nThis   is\tfine\fmore text"
        cleaned = clean_pdf_text(raw_text)

        self.assertNotIn("Figure 1", cleaned)
        self.assertNotIn("\f", cleaned)
        self.assertIn("This is fine", cleaned)

    def test_split_into_chunks_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_into_chunks("one two three", chunk_size=2, chunk_overlap=2)

    def test_chunk_documents_preserves_metadata_and_skips_empty(self):
        documents = [
            {"text": "alpha beta gamma delta", "source": "paper.pdf", "category": "Results"},
            {"text": "   ", "source": "ignored.pdf"},
        ]

        chunks = chunk_documents(documents, clean_text=False)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source"], "paper.pdf")
        self.assertEqual(chunks[0]["category"], "Results")
        self.assertTrue(all(chunk["text"].strip() for chunk in chunks))


if __name__ == "__main__":
    unittest.main()