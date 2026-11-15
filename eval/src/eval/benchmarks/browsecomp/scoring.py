"""Independent browsecomp grading with a reference-answer LLM judge."""

import argparse
import asyncio
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from harness import Trajectory
from harness.model import Context, ModelParams, add_thinking_args, call_model, make_context
from harness.utils import clean_text

from eval import ROOT
from eval import scoring as eval_scoring
from eval.scoring import ScoreResult

from .inference import load_tasks

__all__ = ["parse_args", "score", "main"]

METHOD = "judge"
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
    add_thinking_args(parser)
    args = eval_scoring.parse_score_args(parser, argv)
    if not args.judge_model or args.judge_max_tokens < 1 or args.timeout <= 0:
        parser.error("judge model, positive token budget and timeout are required")
    return args


async def score(
    sample: dict[str, Any], trace: Trajectory, artifacts: dict | None, config: dict, *, ctx: Context
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
    ctx = None

    async def evaluate(sample, trace, artifacts, config):
        nonlocal ctx
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
        return await score(sample, trace, artifacts, config, ctx=ctx)

    try:
        return await eval_scoring.run_score(
            args,
            load_tasks,
            evaluate,
            benchmark="browsecomp",
            method=METHOD,
            group_by=("topic",),
            metadata={"judge_prompt": JUDGE_PROMPT},
        )
    finally:
        if ctx is not None:
            await ctx.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
