"""Shared evaluation scheduling and storage utilities."""

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from harness.utils import encode_id
from tqdm import tqdm

__all__ = ["SECRET_KEYS", "run_tasks", "trajectory_path", "create_directory"]

SECRET_KEYS = {"api_key", "serper_api_key", "crawl4ai_token", "ds_tool_key", "judge_api_key"}


async def run_tasks[T, R](
    items: list[T],
    solve: Callable[[T], Coroutine[Any, Any, R]],
    *,
    concurrency: int,
    desc: str,
    on_result: Callable[[R], None] | None = None,
) -> list[R]:
    """Bound active samples; cancel and drain admitted tasks on failure."""
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    task_iter = iter(enumerate(items))
    pending: dict[asyncio.Task[R], int] = {}
    results: dict[int, R] = {}
    try:
        with tqdm(total=len(items), desc=desc, unit="task") as progress:
            while pending or len(results) < len(items):
                while len(pending) < concurrency:
                    try:
                        index, item = next(task_iter)
                    except StopIteration:
                        break
                    pending[asyncio.create_task(solve(item))] = index
                completed, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for future in completed:
                    result = await future
                    results[pending.pop(future)] = result
                    if on_result is not None:
                        on_result(result)
                    progress.update(1)
    finally:
        for future in pending:
            future.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    return [results[index] for index in range(len(items))]


def trajectory_path(run_dir: Path, task_id: str) -> Path:
    return run_dir / "trajectories" / f"{encode_id(task_id)}.json"


def create_directory(directory: Path) -> Path:
    candidate = directory
    index = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            candidate = directory.with_name(f"{directory.name}-{index}")
            index += 1
        else:
            return candidate
