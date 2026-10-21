"""GPQA (Diamond subset): single-turn multiple-choice benchmark."""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval import DATA_DIR, ROOT
from eval.core import harness, model, runner
from eval.core.model import Context
from eval.core.utils import last_json_object

DATA_FILE = "gpqa_subset_diamond198.json"
CHOICE_LABELS = ("A", "B", "C", "D")
CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
ANSWER_CHOICE_RE = re.compile(r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([ABCD])[\"']?", re.IGNORECASE)

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


def scorer(prediction: str | None, gold: str) -> bool:
    "Scorer for GPQA"

    if prediction is None:
        return False
    parsed = _parse_answer(prediction)
    return parsed is not None and parsed == gold


async def solve_task(task: dict[str, Any], ctx: Context) -> dict[str, Any]:
    choices = "\n".join(f"{label}. {task['choices'][label]}" for label in CHOICE_LABELS)
    started = time.perf_counter()
    result = await harness.qa(ctx, f"{task['question']}\n\n{choices}", system_prompt=SYSTEM_PROMPT)

    correct = scorer(result.answer, task["gold"]) if result.sys_error is None else False
    return runner.make_record(
        task,
        extracted_prediction=_parse_answer(result.answer or ""),
        score=float(correct),
        usage=result.usage,
        finish_reasons=result.finish_reasons,
        latency_sec=time.perf_counter() - started,
        sys_error=result.sys_error,
        messages=result.messages,
        steps=result.steps,
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


def _normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in CHOICE_LABELS:
        return text
    match = CHOICE_RE.search(text)
    return match.group(1).upper() if match else None


def _parse_answer(text: str) -> str | None:
    parsed = last_json_object(text)
    if parsed is not None and "answer" in parsed:
        choice = _normalize_choice(parsed.get("answer"))
        if choice is not None:
            return choice
    answer_matches = ANSWER_CHOICE_RE.findall(text)
    if answer_matches:
        return answer_matches[-1].upper()
    choice_matches = CHOICE_RE.findall(text)
    return choice_matches[-1].upper() if choice_matches else None


if __name__ == "__main__":
    sys.exit(main())
