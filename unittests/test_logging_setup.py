import logging
import unittest
from pathlib import Path

from unittests._support import configure_test_environment

root = configure_test_environment()

from logging_setup import configure_logging


class LoggingSetupTests(unittest.TestCase):
    def test_configure_logging_creates_project_log_file(self):
        log_file = configure_logging()
        self.assertEqual(log_file, root / "logging" / "docmind.log")

        logger = logging.getLogger("docmind.test")
        message = "logging smoke test"
        logger.info(message)

        for handler in logging.getLogger().handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        self.assertTrue(Path(log_file).exists())
        self.assertIn(message, Path(log_file).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()