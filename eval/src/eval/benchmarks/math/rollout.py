"""math agent rollout with local dataset selection."""

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ... import ROOT
from ...agent import Task, Trajectory, run_episode
from ...harness import add_harness_args
from ...harness.tool import ToolBox
from ...harness.utils import save_trajectory
from ...model import add_model_args, build_model_params, make_context
from .._common.resume import finalize_rollout_dir, init_rollout_dir, parse_rollout_args
from .._common.storage import trajectory_path
from .._common.tasks import run_tasks

__all__ = ["parse_args", "load_tasks", "build_task", "run_rollout", "main"]


DATA = Path(__file__).parent / "data"
DATA_FILE = "math.json"


SUBJECTS = (
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
)

SYSTEM_PROMPT = r"""You are solving a competition mathematics problem.

Solve the problem carefully and show your reasoning. End with the final answer in a single \boxed{...} expression. Do not put any text after the boxed answer.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="math")
    add_model_args(parser, temperature=1.0, top_k=20, max_tokens=81920, timeout=1800.0)
    add_harness_args(parser)
    parser.set_defaults(benchmark="math", dataset_file=str(DATA / DATA_FILE))
    parser.add_argument("--subject", choices=SUBJECTS, help="only run problems from this subject")
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4, 5), help="only run problems at this level")
    return parse_rollout_args(parser, argv)


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.dataset_file)
    limit = args.limit
    subject = args.subject
    level = args.level
    tasks: list[dict[str, Any]] = []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"MATH-500 dataset must be a JSON array: {path}")
    for row in rows:
        if subject is not None and row.get("subject") != subject:
            continue
        row_level = int(row["level"]) if row.get("level") is not None else None
        if level is not None and row_level != level:
            continue
        if limit is not None and len(tasks) >= limit:
            break
        tasks.append(
            {
                "id": str(row["unique_id"]),
                "problem": str(row["problem"]),
                "gold": str(row["answer"]),
                "subject": str(row["subject"]),
                "level": row_level,
            }
        )
    if not tasks:
        raise ValueError("No MATH tasks matched the requested filters")
    for task in tasks:
        task["meta"] = {"subject": task["subject"], "level": task["level"]}
    return tasks


def build_task(sample: dict[str, Any], args: argparse.Namespace) -> Task:
    return Task(
        id=str(sample["id"]),
        system=SYSTEM_PROMPT,
        prompt=sample["problem"],
        max_turns=args.max_turns,
        timeout=args.task_timeout,
        meta=dict(sample["meta"]),
    )


async def run_rollout(args: argparse.Namespace, samples: list[dict[str, Any]]) -> list[Trajectory]:
    started = time.perf_counter()
    tasks = [build_task(sample, args) for sample in samples]
    run_dir, config, saved_config, completed, pending_pairs = init_rollout_dir(args, samples, tasks)
    new_traces: list[Trajectory] = []
    ctx = None
    try:
        if pending_pairs:
            ctx = make_context(
                args.base_url,
                args.api_key,
                build_model_params(args),
                args.concurrency,
                timeout=args.timeout,
            )
            tools = ToolBox()

            async def run_sample(pair: tuple[dict[str, Any], Task]) -> Trajectory:
                sample_started = time.perf_counter()
                _, task = pair
                trajectory_file = trajectory_path(run_dir, task.id)
                trace = Trajectory(
                    task.id,
                    messages=[{"role": "system", "content": task.system}, {"role": "user", "content": task.prompt}],
                    meta=dict(task.meta),
                )
                try:
                    trace = await run_episode(task, ctx, tools, out=trajectory_file)
                except asyncio.CancelledError:
                    trace.episode_status, trace.error = "cancelled", "CancelledError"
                    trace.runtime_latency_sec = round(time.perf_counter() - sample_started, 6)
                    save_trajectory(trace, trajectory_file)
                    raise
                trace.runtime_latency_sec = round(time.perf_counter() - sample_started, 6)
                save_trajectory(trace, trajectory_file)
                return trace

            new_traces = await run_tasks(
                pending_pairs, run_sample, concurrency=args.sample_concurrency, desc=args.benchmark
            )
    finally:
        try:
            if ctx is not None:
                await ctx.client.close()
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples = load_tasks(args)
    traces = asyncio.run(run_rollout(args, samples))
    return int(any(trace.error is not None for trace in traces))


if __name__ == "__main__":
    raise SystemExit(main())
