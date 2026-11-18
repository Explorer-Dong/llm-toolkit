"""Independent gpqa answer scoring from saved rollouts."""

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

from ...agent import Task, Trajectory
from ...harness.utils import write_json
from .._common.metrics import ScoreRecord, ScoreResult, make_score_record
from .._common.resume import finalize_score_dir, init_score_dir, parse_score_args
from .._common.tasks import run_tasks
from .rollout import load_tasks

__all__ = ["parse_args", "score", "run_score"]

METHOD = "choice"
_CHOICE_LABELS = ("A", "B", "C", "D")
_CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_ANSWER_CHOICE_RE = re.compile(r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([ABCD])[\"']?", re.IGNORECASE)
_logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score gpqa rollouts.")
    return parse_score_args(parser, argv)


async def score(task: Task, trace: Trajectory, environment_outputs: dict | None, config: dict) -> ScoreResult:
    prediction = _parse_choice_answer(trace.final_answer or "")
    value = float(prediction is not None and prediction == task.gold)
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
    ) = init_score_dir(args, load_tasks, benchmark="gpqa", method=METHOD)
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
                pending, score_task, concurrency=args.concurrency, desc="score gpqa", on_result=persist
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
            benchmark="gpqa",
            rollout_config=rollout_config,
        )
    assert summary is not None
    return output, summary


def _normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in _CHOICE_LABELS:
        return text
    match = _CHOICE_RE.search(text)
    return match.group(1).upper() if match else None


def _parse_choice_answer(text: str) -> str | None:
    parsed = _last_json_object(text)
    if parsed is not None and "answer" in parsed:
        choice = _normalize_choice(parsed.get("answer"))
        if choice is not None:
            return choice
    answer_matches = _ANSWER_CHOICE_RE.findall(text)
    if answer_matches:
        return answer_matches[-1].upper()
    choice_matches = _CHOICE_RE.findall(text)
    return choice_matches[-1].upper() if choice_matches else None


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
    output, summary = asyncio.run(run_score(parse_args()))
    print(f"Scores: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
