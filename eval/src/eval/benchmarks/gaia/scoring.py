"""Independent gaia answer scoring from saved rollouts."""

import argparse
import asyncio
import json
import logging
import re
import string
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...agent import Trajectory
from ...harness.utils import write_json
from .._common.metrics import ScoreRecord, ScoreResult, make_score_record
from .._common.resume import finalize_score_dir, init_score_dir, parse_score_args
from .._common.tasks import run_tasks
from .rollout import load_tasks

__all__ = ["parse_args", "score", "main"]

METHOD = "normalized"
_logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score gaia rollouts.")
    return parse_score_args(parser, argv)


async def score(
    sample: dict[str, Any], trace: Trajectory, environment_outputs: dict | None, config: dict
) -> ScoreResult:
    prediction = _parse_text_answer(trace.final_answer or "")
    value = float(_verify_text(prediction, str(sample["gold"])))
    return ScoreResult(sample["gold"], prediction, value if trace.error is None else 0.0, METHOD)


def main(argv: Sequence[str] | None = None) -> int:
    output, summary = asyncio.run(_run_score(parse_args(argv)))
    print(f"Scores: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary["sys_errors"] > 0)


async def _run_score(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    (
        output,
        samples,
        trajectories,
        environment_outputs,
        rollout_config,
        score_config,
        saved_score_config,
        kept,
        prev_wall,
    ) = init_score_dir(args, load_tasks, benchmark="gaia", method=METHOD, group_by=("level",))
    records = list(kept.values())
    pending = [sample for sample in samples if str(sample["id"]) not in kept]

    async def score_sample(sample: dict[str, Any]) -> ScoreRecord:
        task_id = str(sample["id"])
        trace = trajectories[task_id]
        result = ScoreResult(sample["gold"], trace.final_answer, 0.0, METHOD)
        scoring_sec = 0.0
        if trace.error is None:
            scoring_started = time.perf_counter()
            try:
                result = await score(sample, trace, environment_outputs[task_id], rollout_config)
            except Exception as exc:
                result.sys_error = f"{type(exc).__name__}: {exc}"
            finally:
                scoring_sec = round(time.perf_counter() - scoring_started, 6)
            if result.sys_error is not None:
                _logger.error("Scoring %s failed: %s", task_id, result.sys_error)
        return make_score_record(sample, trace, result, scoring_sec)

    def persist(record: ScoreRecord) -> None:
        records.append(record)
        write_json(output / "trajectories.json", [asdict(r) for r in records])

    new_ordered: list[ScoreRecord] = []
    completed = False
    try:
        if pending:
            new_ordered = await run_tasks(
                pending, score_sample, concurrency=args.sample_concurrency, desc="score gaia", on_result=persist
            )
        completed = True
    finally:
        ordered = None
        if completed:
            by_id = dict(kept)
            by_id.update({r.task_id: r for r in new_ordered})
            ordered = [by_id[str(s["id"])] for s in samples]
        summary = finalize_score_dir(
            output,
            score_config,
            saved_score_config,
            kept,
            prev_wall,
            started,
            ordered,
            benchmark="gaia",
            rollout_config=rollout_config,
            group_by=("level",),
        )
    assert summary is not None
    return output, summary


def _parse_text_answer(text: str) -> str | None:
    parsed = _last_json_object(text)
    if parsed is None:
        return None
    value = parsed.get("answer")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _verify_text(prediction: str | None, gold: str) -> bool:
    """Compare numeric, list, and text answers after normalization."""

    prediction = prediction or ""
    gold_number = _normalize_number(gold)
    if gold_number is not None:
        prediction_number = _normalize_number(prediction)
        return prediction_number is not None and prediction_number == gold_number
    if "," in gold or ";" in gold:
        gold_parts = re.split(r"[,;]", gold)
        prediction_parts = re.split(r"[,;]", prediction)
        if len(gold_parts) != len(prediction_parts):
            return False
        for gold_part, prediction_part in zip(gold_parts, prediction_parts, strict=True):
            gold_number = _normalize_number(gold_part)
            if gold_number is not None:
                prediction_number = _normalize_number(prediction_part)
                if prediction_number is None or prediction_number != gold_number:
                    return False
            elif _normalize_text(prediction_part, remove_punctuation=False) != _normalize_text(
                gold_part, remove_punctuation=False
            ):
                return False
        return True
    return _normalize_text(prediction) == _normalize_text(gold)


def _normalize_text(value: str, remove_punctuation: bool = True) -> str:
    value = re.sub(r"\s", "", value).lower()
    if remove_punctuation:
        value = value.translate(str.maketrans("", "", string.punctuation))
    return value


def _normalize_number(value: str) -> float | None:
    value = value.strip()
    for char in ("$", "%", ","):
        value = value.replace(char, "")
    try:
        return float(value)
    except ValueError:
        return None


def _last_json_object(text: str) -> dict[str, Any] | None:
    "Return the last parseable JSON object embedded in text, if any."

    end_positions = [i for i, char in enumerate(text) if char == "}"]
    for end in reversed(end_positions):
        start = text.rfind("{", 0, end + 1)
        while start != -1:
            candidate = text[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                start = text.rfind("{", 0, start)
                continue
            if isinstance(parsed, dict):
                return parsed
            break
    return None


if __name__ == "__main__":
    raise SystemExit(main())
