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
from eval.core import model, runner
from eval.core.model import Context
from eval.core.utils import last_json_object

DATA_SUBDIR = "GPQA"
DATA_FILE = "diamond198.json"
CHOICE_LABELS = ("A", "B", "C", "D")
CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
ANSWER_CHOICE_RE = re.compile(r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([ABCD])[\"']?", re.IGNORECASE)

SYSTEM_PROMPT = """You are solving a multiple-choice benchmark problem.

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
    parser = argparse.ArgumentParser(description="Single-turn harness for GPQA")
    runner.add_common_args(parser, temperature=1.0, max_tokens=81920, timeout=1800.0)
    parser.add_argument("--top-k", type=int, default=20, help="0 = do not send top_k")
    return parser.parse_args()


def load_data(path: Path, limit: int | None) -> list[dict[str, Any]]:
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


async def solve(task: dict[str, Any], ctx: Context) -> dict[str, Any]:
    choices = "\n".join(f"{label}. {task['choices'][label]}" for label in CHOICE_LABELS)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{task['question']}\n\n{choices}"},
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
    tasks = load_data(DATA_DIR / DATA_SUBDIR / DATA_FILE, args.limit)

    # prepare running folder
    run_dir = runner.prepare_run_dir("GPQA", args, secrets={"api_key"})
    print(f"Running GPQA tasks with {args.model}; output: {run_dir}")

    # run evaluation
    params = model.ModelParams(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        enable_thinking=args.enable_thinking,
        extra_body={"top_k": args.top_k} if args.top_k else None,
    )
    records = asyncio.run(runner.run(args, tasks, solve, run_dir / "predictions.json", "GPQA", params))

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
