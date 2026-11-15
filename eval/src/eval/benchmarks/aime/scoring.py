"""Independent aime answer scoring from saved rollouts."""

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

METHOD = "integer"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score aime rollouts.")
    return eval_scoring.parse_score_args(parser, argv)


async def score(sample: dict[str, Any], trace: Trajectory, artifacts: dict | None, config: dict) -> ScoreResult:
    prediction = _parse_integer_answer(trace.final_answer or "")
    value = float(prediction is not None and prediction == sample["gold"])
    return ScoreResult(sample["gold"], prediction, value if trace.error is None else 0.0, METHOD)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output, summary = asyncio.run(
        eval_scoring.run_score(
            args,
            load_tasks,
            score,
            benchmark="aime",
            method=METHOD,
            group_by=(),
        )
    )
    print(f"Scores: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary["sys_errors"] > 0)


def _parse_integer_answer(text: str) -> int | None:
    parsed = _last_json_object(text)
    if parsed is not None and "answer" in parsed:
        value = parsed.get("answer")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
            return int(value.strip())
        return None
    matches = re.findall(r"[-+]?\d+", text)
    return int(matches[-1]) if matches else None


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
