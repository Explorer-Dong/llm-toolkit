"""MATH-500: single-turn competition mathematics benchmark."""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from eval import DATA_DIR, ROOT
from eval.core import harness, model, runner
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
EXTRACTION_CONFIG = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())
SCORING_METHOD = "Math-Verify 0.9.0 expression extraction and symbolic equivalence; normalized exact-text fallback when parsing fails"

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


def scorer(prediction: str | None, gold: str) -> bool:
    """Score math answers by symbolic/numeric equivalence, with normalized text fallback."""
    if not prediction or not gold.strip():
        return False
    parsed_prediction = _parse_answer(prediction)
    parsed_gold = _parse_gold_answer(gold)
    if parsed_gold and parsed_prediction and verify(parsed_gold, parsed_prediction):
        return True
    return _normalize_exact_answer(prediction) == _normalize_exact_answer(gold)


async def solve_task(task: dict[str, Any], ctx: Context) -> dict[str, Any]:
    started = time.perf_counter()
    result = await harness.qa(ctx, task["problem"], system_prompt=SYSTEM_PROMPT)

    parsed_prediction = _parse_answer(result.answer) if result.answer and result.sys_error is None else []
    if parsed_prediction:
        extracted = str(parsed_prediction[-1])
    elif result.answer:
        extracted = _extract_boxed_answer(result.answer)
    else:
        extracted = None
    correct = scorer(result.answer, task["gold"]) if result.answer and result.sys_error is None else False
    return runner.make_record(
        task,
        extracted_prediction=extracted,
        score=float(correct),
        usage=result.usage,
        finish_reasons=result.finish_reasons,
        latency_sec=time.perf_counter() - started,
        sys_error=result.sys_error,
        messages=result.messages,
        steps=result.steps,
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
            "scoring_method": SCORING_METHOD,
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


def _normalize_math_text(text: str) -> str:
    """Normalize common TeX spacing and shorthand before parsing or exact-text comparison."""
    normalized = text.strip().replace(r",\!", "")
    normalized = re.sub(r",\s*$", "", normalized)
    normalized = re.sub(r"\\(sin|cos|tan)\s*\^\s*(\d+)\s+([A-Za-z])", r"\\\1^{\2}(\3)", normalized)
    return normalized


def _parse_answer(text: str) -> list[Any]:
    """Extract a mathematical answer into Math-Verify's comparable representation."""
    return parse(_normalize_math_text(text), extraction_config=EXTRACTION_CONFIG, fallback_mode="no_fallback")


def _parse_gold_answer(text: str) -> list[Any]:
    parsed = _parse_answer(text)
    if parsed:
        return parsed
    # MATH-500 stores some reference answers as bare LaTeX without math delimiters.
    return _parse_answer(f"${_normalize_math_text(text)}$")


def _extract_boxed_answer(text: str) -> str | None:
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
    value = _extract_boxed_answer(text) or text
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\[,!;:]", "", value)
    value = re.sub(r"\(([a-zA-Z])\)", r"\1", value)
    return re.sub(r"\s+", "", value).strip("$.,;:").casefold()


if __name__ == "__main__":
    sys.exit(main())
