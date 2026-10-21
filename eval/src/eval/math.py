"""MATH-500: single-turn competition mathematics benchmark."""

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

DATA_FILE = "math_subset_500.json"
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


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="MATH-500")
    model.add_model_args(parser, temperature=1.0, top_k=20, max_tokens=81920, timeout=1800.0)
    parser.add_argument("--limit", type=int, default=None, help="run the first N matching tasks")
    parser.add_argument("--subject", choices=SUBJECTS, help="only run problems from this subject")
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4, 5), help="only run problems at this level")
    return parser.parse_args()


def load_tasks(
    path: Path,
    limit: int | None,
    subject: str | None,
    level: int | None,
) -> list[dict[str, Any]]:
    tasks = []
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
    return tasks


async def solve_task(task: dict[str, Any], ctx: Context) -> runner.SystemRecord:
    result = await harness.qa(ctx, task["problem"], system_prompt=SYSTEM_PROMPT)

    score_result = scoring.verify(result, task["gold"], method="math")
    return runner.make_record(
        task,
        result,
        score_result,
        subject=task["subject"],
        level=task["level"],
    )


def main() -> int:
    # load data
    args = parse_args()
    tasks = load_tasks(DATA_DIR / "MATH" / DATA_FILE, args.limit, args.subject, args.level)

    # prepare running folder
    run_dir = runner.prepare_run_dir(
        "MATH",
        args,
        secrets={"api_key"},
        extra={
            "dataset_file": DATA_FILE,
            "system_prompt": SYSTEM_PROMPT,
            "scoring_method": scoring.MATH_SCORING_METHOD,
        },
    )
    print(f"Running {len(tasks)} MATH-500 tasks with {args.model}; output: {run_dir}")

    # run evaluation
    records = asyncio.run(
        runner.run(
            args,
            tasks,
            solve_task,
            run_dir / "predictions.json",
            "MATH-500",
        )
    )

    # log summary after evaluation
    summary = runner.build_summary("MATH-500", records, args.model)
    for field, key in (("subject", "by_subject"), ("level", "by_level")):
        groups = sorted(
            {record.metadata[field] for record in records},
            key=lambda value: (value is None, str(value)),
        )
        summary[key] = {}
        for group in groups:
            subset = [record for record in records if record.metadata[field] == group]
            summary[key]["unknown" if group is None else str(group)] = {
                "num_tasks": len(subset),
                "correct": sum(record.score_result.score for record in subset),
                "score": sum(record.score_result.score for record in subset) / len(subset),
            }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
