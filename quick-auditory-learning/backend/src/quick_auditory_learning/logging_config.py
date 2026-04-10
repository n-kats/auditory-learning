from __future__ import annotations

import logging
from pathlib import Path

_CONFIGURED = False


def configure_logging(log_dir: Path) -> Path:
    global _CONFIGURED
    if _CONFIGURED:
        return log_dir / "backend.log"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "backend.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        child = logging.getLogger(name)
        child.handlers.clear()
        child.setLevel(logging.INFO)
        child.propagate = True

    _CONFIGURED = True
    return log_path
