"""Independent math answer scoring from saved rollouts."""

import argparse
import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse
from math_verify import verify as math_verify

from ...agent import Task, Trajectory
from ...harness.utils import write_json
from .._common.metrics import ScoreRecord, ScoreResult, make_score_record
from .._common.resume import finalize_score_dir, init_score_dir, parse_score_args
from .._common.tasks import run_tasks
from .rollout import load_tasks

__all__ = ["parse_args", "score", "run_score"]

METHOD = "math"
_MATH_EXTRACTION_CONFIG = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())
MATH_SCORING_METHOD = "Math-Verify 0.9.0 expression extraction and symbolic equivalence; normalized exact-text fallback when parsing fails"
_logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score math rollouts.")
    return parse_score_args(parser, argv)


async def score(task: Task, trace: Trajectory, environment_outputs: dict | None, config: dict) -> ScoreResult:
    prediction, value = _verify_math(trace.final_answer, str(task.gold))
    return ScoreResult(task.gold, prediction, value if trace.error is None else 0.0, METHOD)


async def run_score(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    (
        output,
        tasks,
        trajectories,
        environment_outputs,
        rollout_config,
        score_config,
        saved_score_config,
        kept,
        prev_wall,
    ) = init_score_dir(
        args,
        load_tasks,
        benchmark="math",
        method=METHOD,
        group_by=("subject", "level"),
        metadata={"scoring_description": MATH_SCORING_METHOD},
    )
    records = list(kept.values())
    pending = [task for task in tasks if task.id not in kept]

    async def score_task(task: Task) -> ScoreRecord:
        trace = trajectories[task.id]
        result = ScoreResult(task.gold, trace.final_answer, 0.0, METHOD)
        scoring_sec = 0.0
        if trace.error is None:
            scoring_started = time.perf_counter()
            try:
                result = await score(task, trace, environment_outputs[task.id], rollout_config)
            except Exception as exc:
                result.sys_error = f"{type(exc).__name__}: {exc}"
            finally:
                scoring_sec = round(time.perf_counter() - scoring_started, 6)
            if result.sys_error is not None:
                _logger.error("Scoring %s failed: %s", task.id, result.sys_error)
        return make_score_record(task, trace, result, scoring_sec)

    def persist(record: ScoreRecord) -> None:
        records.append(record)
        write_json(output / "trajectories.json", [asdict(r) for r in records])

    new_ordered: list[ScoreRecord] = []
    completed = False
    try:
        if pending:
            new_ordered = await run_tasks(
                pending, score_task, concurrency=args.concurrency, desc="score math", on_result=persist
            )
        completed = True
    finally:
        ordered = None
        if completed:
            by_id = dict(kept)
            by_id.update({r.task_id: r for r in new_ordered})
            ordered = [by_id[task.id] for task in tasks]
        summary = finalize_score_dir(
            output,
            score_config,
            saved_score_config,
            kept,
            prev_wall,
            started,
            ordered,
            benchmark="math",
            rollout_config=rollout_config,
            group_by=("subject", "level"),
        )
    assert summary is not None
    return output, summary


def _verify_math(prediction: str | None, gold: str) -> tuple[str | None, float]:
    parsed_prediction = _parse_math_answer(prediction) if prediction else []
    extracted = str(parsed_prediction[-1]) if parsed_prediction else _extract_boxed_answer(prediction or "")
    if not prediction or not gold.strip():
        return extracted, 0.0
    parsed_gold = _parse_math_gold(gold)
    if parsed_gold and parsed_prediction and math_verify(parsed_gold, parsed_prediction):
        return extracted, 1.0
    return extracted, float(_normalize_math_answer(prediction) == _normalize_math_answer(gold))


def _normalize_math_text(text: str) -> str:
    """Normalize common TeX spacing and shorthand before parsing or exact-text comparison."""
    normalized = text.strip().replace(r",\!", "")
    normalized = re.sub(r",\s*$", "", normalized)
    normalized = re.sub(r"\\(sin|cos|tan)\s*\^\s*(\d+)\s+([A-Za-z])", r"\\\1^{\2}(\3)", normalized)
    return normalized


def _parse_math_answer(text: str) -> list[Any]:
    """Extract a mathematical answer into Math-Verify's comparable representation."""
    return parse(_normalize_math_text(text), extraction_config=_MATH_EXTRACTION_CONFIG, fallback_mode="no_fallback")


def _parse_math_gold(text: str) -> list[Any]:
    parsed = _parse_math_answer(text)
    if parsed:
        return parsed
    # MATH-500 stores some reference answers as bare LaTeX without math delimiters.
    return _parse_math_answer(f"${_normalize_math_text(text)}$")


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


def _normalize_math_answer(text: str) -> str:
    value = _extract_boxed_answer(text) or text
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\[,!;:]", "", value)
    value = re.sub(r"\(([a-zA-Z])\)", r"\1", value)
    return re.sub(r"\s+", "", value).strip("$.,;:").casefold()


if __name__ == "__main__":
    output, summary = asyncio.run(run_score(parse_args()))
    print(f"Scores: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
