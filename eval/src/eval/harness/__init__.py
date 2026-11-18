"""Benchmark-independent agent execution and runtime primitives."""

from __future__ import annotations

import argparse

__all__ = ["add_harness_args"]


def add_harness_args(
    parser: argparse.ArgumentParser,
    *,
    max_turns: int = 1,
    task_timeout: float = 3600.0,
    concurrency: int = 4,
    tools: bool = False,
) -> None:
    """Register agent turn, task concurrency, and optional tool options."""
    parser.add_argument(
        "--max-turns",
        type=int,
        default=max_turns,
        help="regular agent turns; one extra final-only request if exhausted",
    )
    parser.add_argument("--task-timeout", type=float, default=task_timeout, help="in seconds; 0 disables it")
    parser.add_argument("--concurrency", type=int, default=concurrency, help="maximum concurrent tasks")

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
