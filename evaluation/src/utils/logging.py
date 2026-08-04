from __future__ import annotations

import logging
from pathlib import Path


_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def setup_run_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
