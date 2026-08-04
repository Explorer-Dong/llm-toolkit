from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.schema import to_jsonable


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
        file.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: expected object")
            records.append(loaded)
    return records


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(record), file, ensure_ascii=False, indent=2)
        file.write("\n")
