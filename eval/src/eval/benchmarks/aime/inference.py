"""aime agent rollout with local dataset selection."""

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from harness import Task
from harness.model import add_model_args

from eval import ROOT
from eval import inference as eval_inference

__all__ = ["parse_args", "load_tasks", "build_task", "main"]


DATA = Path(__file__).parent / "data"

SYSTEM_PROMPT = """You are solving a math problem.

Rules:
1. Solve step by step, DO NOT answer directly with your memory.
2. The final answer must be an integer.
3. At the very end, output exactly one JSON object.
4. Do not include any text after the JSON.

Final output format:
{"answer": 123}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="aime")
    add_model_args(parser, temperature=1.0, top_k=20, max_tokens=81920, timeout=1800.0)
    parser.set_defaults(benchmark="aime")
    parser.add_argument("--year", type=int, default=2026, help="AIME year to evaluate (loads data/aime<year>.json)")
    args = eval_inference.parse_args(parser, argv)
    args.dataset_file = str(DATA / f"aime{args.year}.json")
    return args


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.dataset_file)
    limit = args.limit
    rows = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for row in rows:
        tasks.append(
            {
                "id": str(row["id"]),
                "problem": str(row["problem"]),
                "gold": int(row["answer"]),
            }
        )
        if limit is not None and len(tasks) >= limit:
            break
    if not tasks:
        raise ValueError("No AIME tasks matched the requested filters")
    for task in tasks:
        task["meta"] = {}
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples = load_tasks(args)
    traces = asyncio.run(eval_inference.run_inference(args, samples, [build_task(sample, args) for sample in samples]))
    return int(any(trace.error is not None for trace in traces))


if __name__ == "__main__":
    raise SystemExit(main())
