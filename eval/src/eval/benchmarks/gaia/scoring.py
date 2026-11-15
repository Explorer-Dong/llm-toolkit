"""Independent gaia answer scoring from saved rollouts."""

import argparse
import asyncio
import json
import re
import string
from collections.abc import Sequence
from typing import Any

from harness import Trajectory

from eval import scoring as eval_scoring
from eval.scoring import ScoreResult

from .inference import load_tasks

__all__ = ["parse_args", "score", "main"]

METHOD = "normalized"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score gaia rollouts.")
    return eval_scoring.parse_score_args(parser, argv)


async def score(sample: dict[str, Any], trace: Trajectory, artifacts: dict | None, config: dict) -> ScoreResult:
    prediction = _parse_text_answer(trace.final_answer or "")
    value = float(_verify_text(prediction, str(sample["gold"])))
    return ScoreResult(sample["gold"], prediction, value if trace.error is None else 0.0, METHOD)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output, summary = asyncio.run(
        eval_scoring.run_score(
            args,
            load_tasks,
            score,
            benchmark="gaia",
            method=METHOD,
            group_by=("level",),
        )
    )
    print(f"Scores: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary["sys_errors"] > 0)


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
