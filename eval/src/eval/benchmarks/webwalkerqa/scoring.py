"""Independent webwalkerqa grading with a reference-answer LLM judge."""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ... import ROOT
from ...agent import Task, Trajectory
from ...harness.utils import clean_text, write_json
from ...model import call_model, make_client
from .._common.metrics import ScoreRecord, ScoreResult, make_score_record
from .._common.resume import finalize_score_dir, init_score_dir, parse_score_args
from .._common.tasks import run_tasks
from .rollout import load_tasks

__all__ = ["parse_args", "score", "run_score"]

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
    args = parse_score_args(parser, argv, concurrency=4)
    if not args.judge_model or args.judge_max_tokens < 1 or args.timeout <= 0:
        parser.error("judge model, positive token budget and timeout are required")
    return args


async def score(
    task: Task,
    trace: Trajectory,
    environment_outputs: dict | None,
    config: dict,
    *,
    client: AsyncOpenAI,
    model_params: Mapping[str, Any],
) -> ScoreResult:
    score_result = ScoreResult(task.gold, trace.final_answer, 0.0, METHOD)
    if trace.error is not None or trace.final_answer is None:
        return score_result
    prompt = JUDGE_PROMPT.format(question=task.prompt, response=trace.final_answer, correct_answer=task.gold)
    score_result.messages.append({"role": "user", "content": prompt})
    try:
        turn = await call_model(client, score_result.messages, **model_params)
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
        benchmark="webwalkerqa",
        method=METHOD,
        group_by=("kind",),
        metadata={"judge_prompt": JUDGE_PROMPT},
    )
    records = list(kept.values())
    pending = [task for task in tasks if task.id not in kept]
    client = None
    model_params = {
        "model": args.judge_model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": args.judge_max_tokens,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
    }

    async def score_task(task: Task) -> ScoreRecord:
        nonlocal client
        trace = trajectories[task.id]
        result = ScoreResult(task.gold, trace.final_answer, 0.0, METHOD)
        scoring_sec = 0.0
        if trace.error is None:
            if client is None:
                client = make_client(args.judge_base_url, args.judge_api_key, timeout=args.timeout)
            scoring_started = time.perf_counter()
            try:
                result = await score(
                    task,
                    trace,
                    environment_outputs[task.id],
                    rollout_config,
                    client=client,
                    model_params=model_params,
                )
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
                pending, score_task, concurrency=args.concurrency, desc="score webwalkerqa", on_result=persist
            )
        completed = True
    finally:
        try:
            if client is not None:
                await client.close()
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
                benchmark="webwalkerqa",
                rollout_config=rollout_config,
                group_by=("kind",),
            )
    assert summary is not None
    return output, summary


if __name__ == "__main__":
    output, summary = asyncio.run(run_score(parse_args()))
    print(f"Scores: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
