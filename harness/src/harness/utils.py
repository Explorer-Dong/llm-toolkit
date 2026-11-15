"""Text preparation and atomic trajectory storage."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from harness.agent import Trajectory

__all__ = ["clean_text", "encode_id", "load_trajectory", "save_trajectory", "write_json"]


def clean_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + (" ..." if len(text) > limit else "")


def encode_id(task_id: str) -> str:
    if not task_id or task_id in {".", ".."}:
        raise ValueError("Sample id must be nonempty and different from '.' and '..'")
    return quote(task_id, safe="")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_trajectory(trajectory: Trajectory, path: Path) -> None:
    write_json(path, asdict(trajectory))


def load_trajectory(path: Path) -> Trajectory:
    from harness.agent import Trajectory

    return Trajectory(**json.loads(path.read_text(encoding="utf-8")))
