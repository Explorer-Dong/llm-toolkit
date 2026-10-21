"""GPQA (Diamond subset): single-turn multiple-choice benchmark."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval import DATA_DIR, ROOT
from eval.core import harness, model, runner, scoring
from eval.core.model import Context

DATA_FILE = "gpqa_subset_diamond198.json"
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


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="GPQA")
    model.add_model_args(parser, temperature=1.0, top_k=20, max_tokens=81920, timeout=1800.0)
    parser.add_argument("--limit", type=int, default=None, help="run the first N matching tasks")
    return parser.parse_args()


def load_tasks(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for row in rows:
        tasks.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "choices": {label: str(row["choices"][label]) for label in CHOICE_LABELS},
                "gold": str(row["answer"]),
            }
        )
        if limit is not None and len(tasks) >= limit:
            break
    if not tasks:
        raise ValueError("No GPQA tasks matched the requested filters")
    return tasks


async def solve_task(task: dict[str, Any], ctx: Context) -> runner.SystemRecord:
    choices = "\n".join(f"{label}. {task['choices'][label]}" for label in CHOICE_LABELS)
    result = await harness.qa(ctx, f"{task['question']}\n\n{choices}", system_prompt=SYSTEM_PROMPT)

    score_result = scoring.verify(result, task["gold"], method="choice")
    return runner.make_record(
        task,
        result,
        score_result,
    )


def main() -> int:
    # load data
    args = parse_args()
    tasks = load_tasks(DATA_DIR / "GPQA" / DATA_FILE, args.limit)

    # prepare running folder
    run_dir = runner.prepare_run_dir(
        "GPQA",
        args,
        secrets={"api_key"},
        extra={"dataset_file": DATA_FILE, "system_prompt": SYSTEM_PROMPT},
    )
    print(f"Running GPQA tasks with {args.model}; output: {run_dir}")

    # run evaluation
    records = asyncio.run(
        runner.run(
            args,
            tasks,
            solve_task,
            run_dir / "predictions.json",
            "GPQA",
        )
    )

    # log summary after evaluation
    summary = runner.build_summary("GPQA", records, args.model)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
