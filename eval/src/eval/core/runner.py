"""Run lifecycle: run directory, progress loop, and run artifacts."""

import argparse
import asyncio
import json
import os
import re
from collections.abc import Callable, Coroutine
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm

from eval import OUTPUT_DIR
from eval.core import harness
from eval.core.model import Context, ModelParams, build_llm_client, build_model_params
from eval.core.scoring import ScoreResult

__all__ = ["SystemRecord", "prepare_run_dir", "run", "make_record", "build_summary"]


@dataclass
class SystemRecord:
    """One task's agent result, score, and benchmark-specific metadata."""

    id: str
    agent_result: harness.AgentResult
    score_result: ScoreResult
    metadata: dict[str, Any]


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
    solve_task: Callable[..., Coroutine[Any, Any, SystemRecord]],
    output_path: Path,
    desc: str,
    *,
    use_tools: bool = False,
    use_judge: bool = False,
    max_active_tasks: int | None = None,
) -> list[SystemRecord]:
    """Manage shared clients, bind optional tools/judge, and run tasks with bounded admission."""

    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")

    if max_active_tasks is not None and max_active_tasks < 1:
        raise ValueError("max_active_tasks must be positive")

    async with AsyncExitStack() as resources:
        client_llm = build_llm_client(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
        resources.push_async_callback(client_llm.close)
        ctx = Context(client_llm=client_llm, params=build_model_params(args), sem=asyncio.Semaphore(args.concurrency))
        solver_kwargs: dict[str, Any] = {}
        if use_judge:
            judge = build_llm_client(api_key=args.judge_api_key, base_url=args.judge_base_url, timeout=args.timeout)
            resources.push_async_callback(judge.close)
            solver_kwargs["judge_ctx"] = Context(
                client_llm=judge,
                params=ModelParams(
                    model=args.judge_model,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=4096,
                    seed=args.seed,
                    enable_thinking=False,
                ),
                sem=ctx.sem,
            )
        if use_tools:
            client_search, client_fetch = await resources.enter_async_context(harness.tool_clients())
            solver_kwargs["tools"] = harness.make_tools(client_search, client_fetch, args)
        if solver_kwargs:
            solve_task = partial(solve_task, args=args, **solver_kwargs)
        records: list[SystemRecord] = []
        task_iter = iter(tasks)
        active_limit = min(len(tasks), max_active_tasks if max_active_tasks is not None else args.concurrency)
        pending: set[asyncio.Task[SystemRecord]] = set()
        for _ in range(active_limit):
            pending.add(asyncio.create_task(solve_task(next(task_iter), ctx)))

        try:
            with tqdm(total=len(tasks), desc=desc, unit="task") as progress:
                while pending:
                    completed, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for future in completed:
                        pending.remove(future)
                        record = await future
                        records.append(record)
                        _write_predictions(output_path, records)
                        score = f"{sum(r.score_result.score for r in records)}/{len(records)}"
                        progress.set_postfix(score=score)
                        progress.update(1)
                        try:
                            task = next(task_iter)
                        except StopIteration:
                            continue
                        pending.add(asyncio.create_task(solve_task(task, ctx)))
        finally:
            for future in pending:
                if not future.done():
                    future.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        return records


def make_record(
    task: dict[str, Any],
    result: harness.AgentResult,
    score_result: ScoreResult,
    **extra: Any,
) -> SystemRecord:
    """Combine the agent trace, score, and benchmark-specific metadata into one record."""

    sys_error = result.sys_error or score_result.sys_error
    if sys_error is not None:
        score_result = replace(score_result, score=0.0)
    return SystemRecord(
        id=str(task["id"]),
        agent_result=result,
        score_result=score_result,
        metadata=extra,
    )


def build_summary(benchmark: str, records: list[SystemRecord], model_name: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "model": model_name,
        "num_tasks": len(records),
        "correct": sum(record.score_result.score for record in records),
        "score": round(sum(record.score_result.score for record in records) / len(records), 2) if records else 0.0,
        "sys_errors": sum(
            record.agent_result.sys_error is not None or record.score_result.sys_error is not None for record in records
        ),
    }


def _write_predictions(output_path: Path, records: list[SystemRecord]) -> None:
    "Rewrite the whole JSON array atomically."

    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp_path, output_path)
