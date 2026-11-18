"""Harness CLI options, text preparation, and atomic trajectory storage."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from ..agent import Trajectory

__all__ = [
    "add_harness_args",
    "clean_text",
    "encode_id",
    "write_json",
    "save_trajectory",
    "load_trajectory",
]


def add_harness_args(
    parser: argparse.ArgumentParser,
    *,
    max_turns: int = 1,
    task_timeout: float = 3600.0,
    sample_concurrency: int | None = None,
    tools: bool = False,
) -> None:
    """Register agent turn, sample concurrency, and optional tool options."""
    parser.add_argument(
        "--max-turns",
        type=int,
        default=max_turns,
        help="regular agent turns; one extra final-only request if exhausted",
    )
    parser.add_argument(
        "--task-timeout", type=float, default=task_timeout, help="episode budget in seconds; 0 disables it"
    )
    parser.add_argument(
        "--sample-concurrency", type=int, default=sample_concurrency, help="maximum active runtime instances"
    )
    if not tools:
        return
    from .tool import ToolConfig

    defaults = ToolConfig()
    parser.add_argument("--search-provider", choices=("deepseek", "serper"), default=defaults.search_provider)
    parser.add_argument("--fetch-provider", choices=("deepseek", "crawl4ai"), default=defaults.fetch_provider)
    parser.add_argument(
        "--search-results", type=int, choices=range(1, 51), metavar="N", default=defaults.search_results
    )
    parser.add_argument("--observation-chars", type=int, default=defaults.observation_chars)
    parser.add_argument("--ds-tool-url", default=defaults.ds_tool_url, help="DeepSeek tools base URL")
    parser.add_argument("--ds-tool-key", default=defaults.ds_tool_key, help="DeepSeek tools API key")
    parser.add_argument("--serper-api-key", default=defaults.serper_api_key)
    parser.add_argument("--crawl4ai-url", default=defaults.crawl4ai_url)
    parser.add_argument("--crawl4ai-token", default=defaults.crawl4ai_token)


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
    from ..agent import Trajectory

    record = json.loads(path.read_text(encoding="utf-8"))
    if "stop_reason" in record:
        record.setdefault("episode_status", record.pop("stop_reason"))
    record.setdefault(
        "finish_reason",
        next(
            (
                message.get("finish_reason")
                for message in reversed(record.get("messages", []))
                if message["role"] == "assistant"
            ),
            None,
        ),
    )
    return Trajectory(**record)
