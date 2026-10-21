"""Run lifecycle: run directory, progress loop, and run artifacts."""

import argparse
import asyncio
import json
import os
import re
from collections.abc import Callable, Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from eval import OUTPUT_DIR
from eval.core.model import Context, build_llm_client, build_model_params

__all__ = ["prepare_run_dir", "run", "make_record", "build_summary"]


def prepare_run_dir(
    benchmark: str,
    args: argparse.Namespace,
    *,
    secrets: set[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Create outputs/<benchmark>/<timestamp>_<model>/ and write config.json with secrets filtered."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / benchmark / f"{timestamp}_{re.sub(r'[^A-Za-z0-9_.-]+', '-', args.model)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) for key, value in vars(args).items() if key not in secrets}
    if extra:
        config.update(extra)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


async def run(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    solve: Callable[[dict[str, Any], Context], Coroutine[Any, Any, dict[str, Any]]],
    output_path: Path,
    desc: str,
    *,
    max_active_tasks: int | None = None,
) -> list[dict[str, Any]]:
    "Run tasks with bounded admission and persist predictions incrementally."

    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if max_active_tasks is not None and max_active_tasks < 1:
        raise ValueError("max_active_tasks must be positive")
    client_llm = build_llm_client(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    ctx = Context(client_llm=client_llm, params=build_model_params(args), sem=asyncio.Semaphore(args.concurrency))
    records: list[dict[str, Any]] = []
    task_iter = iter(tasks)
    active_limit = min(len(tasks), max_active_tasks if max_active_tasks is not None else args.concurrency)
    pending: set[asyncio.Task[dict[str, Any]]] = set()
    for _ in range(active_limit):
        pending.add(asyncio.create_task(solve(next(task_iter), ctx)))
    try:
        with tqdm(total=len(tasks), desc=desc, unit="task") as progress:
            while pending:
                completed, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for future in completed:
                    pending.remove(future)
                    record = await future
                    records.append(record)
                    _write_predictions(output_path, records)
                    score = f"{sum(r['score'] for r in records)}/{len(records)}"
                    progress.set_postfix(score=score)
                    progress.update(1)
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        continue
                    pending.add(asyncio.create_task(solve(task, ctx)))
    finally:
        for future in pending:
            if not future.done():
                future.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await client_llm.close()
    return records


def make_record(
    task: dict[str, Any],
    *,
    extracted: Any,
    score: bool,
    usage: dict[str, int],
    finish_reasons: Any,
    latency_sec: float,
    sys_error: str | None,
    messages: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    "One predictions.json row; benchmark-specific fields ride along in **extra."

    record = {
        "id": task["id"],
        "answer": task["gold"],
        "extracted_prediction": extracted,
        "score": int(score),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "finish_reasons": finish_reasons,
        "latency_sec": round(latency_sec, 3),
        "sys_error": sys_error,
        "messages": messages,
    }
    record.update(extra)
    return record


def build_summary(benchmark: str, records: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "model": model_name,
        "num_tasks": len(records),
        "correct": sum(record["score"] for record in records),
        "score": round(sum(record["score"] for record in records) / len(records), 2) if records else 0.0,
        "sys_errors": sum(record["sys_error"] is not None for record in records),
    }


def _write_predictions(output_path: Path, records: list[dict[str, Any]]) -> None:
    "Rewrite the whole JSON array atomically."

    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, output_path)
