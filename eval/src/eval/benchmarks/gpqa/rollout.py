"""gpqa agent rollout with local dataset selection."""

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from ... import ROOT
from ...agent import Task, Trajectory, react
from ...harness import add_harness_args
from ...harness.tool import ToolBox
from ...harness.utils import save_trajectory
from ...model import add_model_args, build_model_params, make_client
from .._common.resume import finalize_rollout_dir, init_rollout_dir, parse_rollout_args
from .._common.storage import cleanup_cancelled_task, trajectory_path
from .._common.tasks import run_tasks

__all__ = ["parse_args", "load_tasks", "rollout"]


DATA = Path(__file__).parent / "data"
DATA_FILE = "gpqa.json"


CHOICE_LABELS = ("A", "B", "C", "D")

SYSTEM_PROMPT = """You are solving a multiple-choice problem.

Rules:
1. Solve step by step, DO NOT answer directly with your memory.
2. Do not use external tools.
3. At the very end, output exactly one JSON object.
4. Do not include any text after the JSON.

Final output format:
{"answer": "A"}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="gpqa")
    add_model_args(parser, temperature=1.0, top_k=20, max_tokens=81920, timeout=1800.0)
    add_harness_args(parser)
    parser.set_defaults(benchmark="gpqa", dataset_file=str(DATA / DATA_FILE))
    return parse_rollout_args(parser, argv)


def load_tasks(args: argparse.Namespace) -> list[Task]:
    path = Path(args.dataset_file)
    limit = args.limit
    max_turns = getattr(args, "max_turns", 1)
    timeout = getattr(args, "task_timeout", 3600.0)
    rows = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[Task] = []
    for row in rows:
        choices = "\n".join(f"{label}. {row['choices'][label]}" for label in CHOICE_LABELS)
        tasks.append(
            Task(
                id=str(row["id"]),
                system=SYSTEM_PROMPT,
                prompt=f"{row['question']}\n\n{choices}",
                max_turns=max_turns,
                timeout=timeout,
                gold=str(row["answer"]),
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    if not tasks:
        raise ValueError("No GPQA tasks matched the requested filters")
    return tasks


async def rollout(args: argparse.Namespace, tasks: list[Task]) -> list[Trajectory]:
    started = time.perf_counter()
    run_dir, config, saved_config, completed, pending = init_rollout_dir(args, tasks)
    new_traces: list[Trajectory] = []
    try:
        if pending:
            model_params = build_model_params(args)
            tools = ToolBox()
            async with make_client(args.base_url, args.api_key, timeout=args.timeout) as client:

                async def run_task(task: Task) -> Trajectory:
                    task_started = time.perf_counter()
                    try:
                        trace = await react(task, client, tools, model_params)
                    except asyncio.CancelledError:
                        cleanup_cancelled_task(run_dir, task.id)
                        raise
                    trace.runtime_latency_sec = round(time.perf_counter() - task_started, 6)
                    save_trajectory(trace, trajectory_path(run_dir, task.id))
                    return trace

                new_traces = await run_tasks(pending, run_task, concurrency=args.concurrency, desc=args.benchmark)
    finally:
        by_id = dict(completed)
        by_id.update({trace.task_id: trace for trace in new_traces})
        traces = [by_id[task.id] for task in tasks] if len(by_id) == len(tasks) else list(by_id.values())
        finalize_rollout_dir(
            run_dir,
            config,
            saved_config,
            started=started,
            resumed_count=len(completed),
            benchmark=args.benchmark,
            traces=traces,
        )
    return traces


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(rollout(args, load_tasks(args)))
