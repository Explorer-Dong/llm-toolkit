"""AIME: single-turn math benchmark."""

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
from eval.core import model, runner
from eval.core.model import Context
from eval.core.utils import last_json_object

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


def load_data(path: Path, limit: int | None) -> list[dict[str, Any]]:
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


def scorer(prediction: str | None, gold: int) -> bool:
    "Scorer for AIME"

    if prediction is None:
        return False
    parsed = _parse_answer(prediction)
    return parsed is not None and parsed == gold


async def solve(task: dict[str, Any], ctx: Context) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["problem"]},
    ]
    started = time.perf_counter()
    sys_error: str | None = None
    output: str | None = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    finish_reason: str | None = None

    try:
        content, _, reasoning, call_usage, finish_reason = await model.call_model(
            ctx.sem, ctx.client_llm, ctx.params, messages
        )
        if call_usage:
            for key in usage:
                usage[key] += call_usage[key]
        output = content.strip()
        if output:
            message: dict[str, Any] = {"role": "assistant", "content": output}
            if reasoning:
                message["reasoning_content"] = reasoning
            if call_usage:
                message["usage"] = call_usage
            messages.append(message)
    except Exception as exc:
        sys_error = f"{type(exc).__name__}: {exc}"

    correct = scorer(output, task["gold"]) if not sys_error else False
    return runner.make_record(
        task,
        extracted=_parse_answer(output or ""),
        score=correct,
        usage=usage,
        finish_reasons=finish_reason,
        latency_sec=time.perf_counter() - started,
        sys_error=sys_error,
        messages=messages,
    )


def main() -> int:
    # load data
    args = parse_args()
    dataset_file = f"aime{args.year}.json"
    tasks = load_data(DATA_DIR / "AIME" / dataset_file, args.limit)

    # prepare running folder
    run_dir = runner.prepare_run_dir(
        "AIME",
        args,
        secrets={"api_key"},
        extra={"dataset_file": dataset_file, "system_prompt": SYSTEM_PROMPT},
    )
    print(f"Running AIME{args.year} tasks with {args.model}; output: {run_dir}")

    # run evaluation
    records = asyncio.run(runner.run(args, tasks, solve, run_dir / "predictions.json", f"AIME{args.year}"))

    # log summary after evaluation
    summary = runner.build_summary(f"AIME{args.year}", records, args.model)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_answer(text: str) -> int | None:
    parsed = last_json_object(text)
    if parsed is not None and "answer" in parsed:
        value = parsed.get("answer")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
            return int(value.strip())
        return None
    matches = re.findall(r"[-+]?\d+", text)
    return int(matches[-1]) if matches else None


if __name__ == "__main__":
    sys.exit(main())
