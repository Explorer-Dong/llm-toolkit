"""Independent gpqa answer scoring from saved rollouts."""

import argparse
import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any

from harness import Trajectory

from eval import scoring as eval_scoring
from eval.scoring import ScoreResult

from .inference import load_tasks

__all__ = ["parse_args", "score", "main"]

METHOD = "choice"
_CHOICE_LABELS = ("A", "B", "C", "D")
_CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_ANSWER_CHOICE_RE = re.compile(r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([ABCD])[\"']?", re.IGNORECASE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score gpqa rollouts.")
    return eval_scoring.parse_score_args(parser, argv)


async def score(sample: dict[str, Any], trace: Trajectory, artifacts: dict | None, config: dict) -> ScoreResult:
    prediction = _parse_choice_answer(trace.final_answer or "")
    value = float(prediction is not None and prediction == sample["gold"])
    return ScoreResult(sample["gold"], prediction, value if trace.error is None else 0.0, METHOD)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output, summary = asyncio.run(
        eval_scoring.run_score(
            args,
            load_tasks,
            score,
            benchmark="gpqa",
            method=METHOD,
            group_by=(),
        )
    )
    print(f"Scores: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary["sys_errors"] > 0)


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
    raise SystemExit(main())
