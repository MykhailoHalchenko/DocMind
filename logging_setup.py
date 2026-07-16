import logging
from pathlib import Path


def configure_logging(level: int = logging.INFO) -> Path:
    """Configure project-wide logging to console and logging/docmind.log."""
    root_logger = logging.getLogger()
    log_dir = Path(__file__).resolve().parent / "logging"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "docmind.log"

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(getattr(handler, "baseFilename", "")) == log_file:
            root_logger.setLevel(level)
            return log_file

    root_logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_file