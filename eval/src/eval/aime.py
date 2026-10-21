"""AIME: single-turn math benchmark."""

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

SYSTEM_PROMPT = """You are solving a math problem.

Rules:
1. Solve step by step, DO NOT answer directly with your memory.
2. The final answer must be an integer.
3. At the very end, output exactly one JSON object.
4. Do not include any text after the JSON.

Final output format:
{"answer": 123}
"""


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="AIME")
    model.add_model_args(parser, temperature=1.0, top_k=20, max_tokens=81920, timeout=1800.0)
    parser.add_argument("--limit", type=int, default=None, help="run the first N matching tasks")
    parser.add_argument("--year", type=int, default=2026, help="AIME year to evaluate (loads data/aime<year>.json)")
    return parser.parse_args()


def load_tasks(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
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
    return tasks


async def solve_task(task: dict[str, Any], ctx: Context) -> runner.SystemRecord:
    result = await harness.qa(ctx, task["problem"], system_prompt=SYSTEM_PROMPT)

    score_result = scoring.verify(result, task["gold"], method="integer")
    return runner.make_record(
        task,
        result,
        score_result,
    )


def main() -> int:
    # load data
    args = parse_args()
    dataset_file = f"aime{args.year}.json"
    tasks = load_tasks(DATA_DIR / "AIME" / dataset_file, args.limit)

    # prepare running folder
    run_dir = runner.prepare_run_dir(
        "AIME",
        args,
        secrets={"api_key"},
        extra={"dataset_file": dataset_file, "system_prompt": SYSTEM_PROMPT},
    )
    print(f"Running AIME{args.year} tasks with {args.model}; output: {run_dir}")

    # run evaluation
    records = asyncio.run(
        runner.run(
            args,
            tasks,
            solve_task,
            run_dir / "predictions.json",
            f"AIME{args.year}",
        )
    )

    # log summary after evaluation
    summary = runner.build_summary(f"AIME{args.year}", records, args.model)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
