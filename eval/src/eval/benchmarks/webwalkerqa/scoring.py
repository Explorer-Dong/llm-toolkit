"""Independent webwalkerqa grading with a reference-answer LLM judge."""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ... import ROOT
from ...agent import Trajectory
from ...harness.utils import clean_text, write_json
from ...model import Context, ModelParams, call_model, make_context
from .._common.metrics import ScoreRecord, ScoreResult, make_score_record
from .._common.resume import finalize_score_dir, init_score_dir, parse_score_args
from .._common.tasks import run_tasks
from .rollout import load_tasks

__all__ = ["parse_args", "score", "main"]

METHOD = "judge"
_logger = logging.getLogger(__name__)
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Score saved rollouts with an LLM judge.")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o"))
    parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL"))
    parser.add_argument("--judge-api-key", default=os.getenv("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--concurrency", type=int, default=4, help="maximum concurrent judge requests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable model thinking (default: enabled)",
    )
    parser.add_argument(
        "--reasoning-effort", default="medium", metavar="LEVEL", help="reasoning effort (default: medium)"
    )
    args = parse_score_args(parser, argv)
    if not args.judge_model or args.judge_max_tokens < 1 or args.timeout <= 0:
        parser.error("judge model, positive token budget and timeout are required")
    return args


async def score(
    sample: dict[str, Any], trace: Trajectory, environment_outputs: dict | None, config: dict, *, ctx: Context
) -> ScoreResult:
    score_result = ScoreResult(sample["gold"], trace.final_answer, 0.0, METHOD)
    if trace.error is not None or trace.final_answer is None:
        return score_result
    prompt = JUDGE_PROMPT.format(
        question=sample["question"], response=trace.final_answer, correct_answer=sample["gold"]
    )
    score_result.messages.append({"role": "user", "content": prompt})
    try:
        turn = await call_model(ctx, score_result.messages)
        score_result.messages.append(turn.message())
        if turn.error is not None:
            raise turn.error
        content = turn.content
        extracted = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\s*(?:reasoning|correct):|\Z)", content, re.IGNORECASE | re.DOTALL
        )
        if extracted is not None:
            value = extracted.group(1).strip()
            score_result.extracted_prediction = None if value.lower() == "none" else value
        match = re.search(r"correct: (yes|no)", content, re.IGNORECASE)
        if match is None:
            raise ValueError(f"judge returned no verdict: {clean_text(content, 200)}")
        score_result.score = float(match.group(1).lower() == "yes")
    except Exception as exc:
        score_result.sys_error = f"{type(exc).__name__}: {exc}"
    return score_result


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
    ) = init_score_dir(
        args,
        load_tasks,
        benchmark="webwalkerqa",
        method=METHOD,
        group_by=("kind",),
        metadata={"judge_prompt": JUDGE_PROMPT},
    )
    records = list(kept.values())
    pending = [sample for sample in samples if str(sample["id"]) not in kept]
    ctx = None

    async def score_sample(sample: dict[str, Any]) -> ScoreRecord:
        nonlocal ctx
        task_id = str(sample["id"])
        trace = trajectories[task_id]
        result = ScoreResult(sample["gold"], trace.final_answer, 0.0, METHOD)
        scoring_sec = 0.0
        if trace.error is None:
            if ctx is None:
                ctx = make_context(
                    args.judge_base_url,
                    args.judge_api_key,
                    ModelParams(
                        args.judge_model,
                        0,
                        1,
                        args.judge_max_tokens,
                        args.seed,
                        enable_thinking=args.enable_thinking,
                        reasoning_effort=args.reasoning_effort,
                    ),
                    args.concurrency,
                    timeout=args.timeout,
                )
            scoring_started = time.perf_counter()
            try:
                result = await score(sample, trace, environment_outputs[task_id], rollout_config, ctx=ctx)
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
                pending, score_sample, concurrency=args.sample_concurrency, desc="score webwalkerqa", on_result=persist
            )
        completed = True
    finally:
        try:
            if ctx is not None:
                await ctx.client.close()
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
                benchmark="webwalkerqa",
                rollout_config=rollout_config,
                group_by=("kind",),
            )
    assert summary is not None
    return output, summary


if __name__ == "__main__":
    raise SystemExit(main())
