from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class Task:
    id: str
    input: dict[str, Any]
    answer: Any | None
    meta: dict[str, Any]


@dataclass
class Prediction:
    id: str
    benchmark: str
    model: str
    raw_output: str
    parsed_answer: Any | None
    gold_answer: Any | None
    score: float | None
    latency_sec: float | None
    error: str | None
    meta: dict[str, Any]


@dataclass
class RunConfig:
    benchmark: str
    model: str
    base_url: str
    api_key: str
    request_timeout: float
    split: str
    limit: int | None
    output_dir: str
    temperature: float
    top_p: float
    top_k: int | None
    max_tokens: int
    num_workers: int
    seed: int | None
    resume: bool
    dry_run: bool


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value
