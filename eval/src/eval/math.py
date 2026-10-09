"""MATH-500: single-turn competition mathematics benchmark."""

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from eval import DATA_DIR, ROOT
from eval.core import model, runner
from eval.core.model import Context


class DatasetConfig(TypedDict):
    name: str
    file: str
    source: str
    revision: str
    sha256: str
    num_rows: int


DATASET: DatasetConfig = {
    "name": "MATH-500",
    "file": "math_subset_500.json",
    "source": "HuggingFaceH4/MATH-500",
    "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "sha256": "c0cf9575e4988436c3973add2b131c6f8e8489174eac89d297f5dd41bd1eb45d",
    "num_rows": 500,
}
SUBJECTS = (
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
)
EXTRACTION_CONFIG = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())
SCORING_METHOD = (
    "Math-Verify 0.9.0 expression extraction and symbolic equivalence; normalized exact-text fallback when parsing fails"
)

SYSTEM_PROMPT = r"""You are solving a competition mathematics problem.

Solve the problem carefully and show your reasoning. End with the final answer in a single \boxed{...} expression. Do not put any text after the boxed answer.
"""


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Single-turn harness for MATH-500")
    runner.add_common_args(parser, temperature=1.0, max_tokens=81920, timeout=1800.0)
    parser.add_argument("--subject", choices=SUBJECTS, help="only run problems from this subject")
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4, 5), help="only run problems at this level")
    parser.add_argument("--top-k", type=int, default=20, help="0 = do not send top_k")
    return parser.parse_args()


def load_data(
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


def _normalize_math_text(text: str) -> str:
    """Normalize common TeX spacing and shorthand before parsing or exact-text comparison."""
    normalized = text.strip().replace(r",\!", "")
    normalized = re.sub(r",\s*$", "", normalized)
    normalized = re.sub(r"\\(sin|cos|tan)\s*\^\s*(\d+)\s+([A-Za-z])", r"\\\1^{\2}(\3)", normalized)
    return normalized


def parse_answer(text: str) -> list[Any]:
    "Extract a mathematical answer and parse it into Math-Verify's comparable representation."
    return parse(_normalize_math_text(text), extraction_config=EXTRACTION_CONFIG, fallback_mode="no_fallback")


def parse_gold_answer(text: str) -> list[Any]:
    parsed = parse_answer(text)
    if parsed:
        return parsed
    # MATH-500 stores some reference answers as bare LaTeX without math delimiters.
    return parse_answer(f"${_normalize_math_text(text)}$")


def extract_answer(text: str) -> str | None:
    parsed = parse_answer(text)
    return str(parsed[-1]) if parsed else None


def scorer(prediction: str | None, gold: str) -> bool:
    "Score math answers by symbolic/numeric equivalence, with normalized text fallback."
    if not prediction or not gold.strip():
        return False
    parsed_prediction = parse_answer(prediction)
    parsed_gold = parse_gold_answer(gold)
    if _score_parsed(parsed_gold, parsed_prediction):
        return True
    return _normalize_exact_answer(prediction) == _normalize_exact_answer(gold)


def _boxed_content(text: str) -> str | None:
    r"""Extract the last balanced \boxed{...} value for exact-text fallback scoring."""
    values = []
    search_from = 0
    while True:
        start = text.find(r"\boxed", search_from)
        if start < 0:
            break
        open_brace = start + len(r"\boxed")
        while open_brace < len(text) and text[open_brace].isspace():
            open_brace += 1
        if open_brace >= len(text) or text[open_brace] != "{":
            search_from = open_brace
            if search_from <= start:
                break
            continue
        depth = 1
        index = open_brace + 1
        while index < len(text) and depth:
            backslashes = 0
            before = index - 1
            while before >= 0 and text[before] == "\\":
                backslashes += 1
                before -= 1
            if backslashes % 2 == 0:
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
            index += 1
        if depth:
            break
        values.append(text[open_brace + 1 : index - 1])
        search_from = index
    return values[-1] if values else None


def _normalize_exact_answer(text: str) -> str:
    value = _boxed_content(text) or text
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\[,!;:]", "", value)
    value = re.sub(r"\(([a-zA-Z])\)", r"\1", value)
    return re.sub(r"\s+", "", value).strip("$.,;:").casefold()


def _score_parsed(parsed_gold: list[Any], parsed_prediction: list[Any]) -> bool:
    return bool(parsed_gold and parsed_prediction and verify(parsed_gold, parsed_prediction))


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

    parsed_prediction = parse_answer(output) if output and sys_error is None else []
    extracted = str(parsed_prediction[-1]) if parsed_prediction else _boxed_content(output) if output else None
    correct = scorer(output, task["gold"]) if output and sys_error is None else False
    return runner.make_record(
        task,
        extracted=extracted,
        score=correct,
        usage=usage,
        finish_reasons=finish_reason,
        latency_sec=time.perf_counter() - started,
        sys_error=sys_error,
        messages=messages,
        subject=task["subject"],
        level=task["level"],
    )


def main() -> int:
    args = parse_args()
    dataset_path = DATA_DIR / "MATH" / DATASET["file"]
    actual_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual_sha256 != DATASET["sha256"]:
        raise ValueError(f"MATH-500 dataset hash mismatch: expected {DATASET['sha256']}, got {actual_sha256}")
    tasks = load_data(dataset_path, args.limit, args.subject, args.level)

    run_dir = runner.prepare_run_dir(
        "MATH",
        args,
        secrets={"api_key"},
        extra={
            "dataset_name": DATASET["name"],
            "dataset_file": DATASET["file"],
            "dataset_source": DATASET["source"],
            "dataset_revision": DATASET["revision"],
            "dataset_sha256": actual_sha256,
            "dataset_source_num_rows": DATASET["num_rows"],
            "system_prompt": SYSTEM_PROMPT,
            "scoring_method": SCORING_METHOD,
        },
    )
    print(f"Running {len(tasks)} MATH-500 tasks with {args.model}; output: {run_dir}")

    params = model.ModelParams(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        enable_thinking=args.enable_thinking,
        extra_body={"top_k": args.top_k} if args.top_k else None,
    )
    records = asyncio.run(runner.run(args, tasks, solve, run_dir / "predictions.json", DATASET["name"], params))

    summary = runner.build_summary(DATASET["name"], records, args.model)
    for field, key in (("subject", "by_subject"), ("level", "by_level")):
        groups = sorted({record[field] for record in records}, key=lambda value: (value is None, str(value)))
        summary[key] = {}
        for group in groups:
            subset = [record for record in records if record[field] == group]
            summary[key]["unknown" if group is None else str(group)] = {
                "num_tasks": len(subset),
                "correct": sum(record["score"] for record in subset),
                "score": sum(record["score"] for record in subset) / len(subset),
            }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
