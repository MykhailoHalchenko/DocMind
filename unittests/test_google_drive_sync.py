import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage import google_drive_sync as gds


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        if size is None or size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class GoogleDriveSyncTests(unittest.TestCase):
    def test_extract_drive_file_id(self):
        self.assertEqual(gds.extract_drive_file_id("https://docs.google.com/document/d/abc123DEF456ghi789JKL/edit"), "abc123DEF456ghi789JKL")
        self.assertEqual(gds.extract_drive_file_id("https://drive.google.com/open?id=abc123DEF456ghi789JKL"), "abc123DEF456ghi789JKL")
        self.assertEqual(gds.extract_drive_file_id("abc123DEF456ghi789JKL"), "abc123DEF456ghi789JKL")
        self.assertEqual(gds.extract_drive_file_id("abc123DEF4"), "abc123DEF4")

    def test_sync_google_drive_source_writes_storage_and_manifest(self):
        payload = b"first line\nsecond line\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(gds.urllib.request, "urlopen", return_value=FakeResponse(payload)):
                result = gds.sync_google_drive_source(
                    "https://docs.google.com/document/d/abc123DEF456ghi789JKL/edit",
                    storage_root=temp_dir,
                    file_name="paper.pdf",
                )

            self.assertEqual(result["status"], "synced")
            content_path = Path(result["file_path"])
            manifest_path = Path(result["manifest_path"])

            self.assertTrue(content_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(content_path.read_bytes(), payload)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["file_id"], "abc123DEF456ghi789JKL")
            self.assertEqual(manifest["sha256"], result["sha256"])
            self.assertEqual(manifest["dataset_type"], "txt")

    def test_sync_google_drive_source_reports_unchanged_when_hash_matches(self):
        payload = b"same content\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(gds.urllib.request, "urlopen", return_value=FakeResponse(payload)):
                first = gds.sync_google_drive_source(
                    "https://docs.google.com/document/d/abc123DEF456ghi789JKL/edit",
                    storage_root=temp_dir,
                    file_name="paper.pdf",
                )

            with patch.object(gds.urllib.request, "urlopen", return_value=FakeResponse(payload)):
                second = gds.sync_google_drive_source(
                    "https://docs.google.com/document/d/abc123DEF456ghi789JKL/edit",
                    storage_root=temp_dir,
                    file_name="paper.pdf",
                )

            self.assertEqual(first["status"], "synced")
            self.assertEqual(second["status"], "unchanged")
            self.assertEqual(Path(first["file_path"]).read_bytes(), payload)
            self.assertEqual(Path(second["file_path"]).read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
