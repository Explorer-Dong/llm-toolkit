"""Grade saved SWE-bench patches with official tests in fresh runtime instances."""

import argparse
import asyncio
import json
import logging
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...agent import Task, Trajectory
from ...harness.utils import encode_id, write_json
from .._common.metrics import ScoreRecord, ScoreResult, make_score_record
from .._common.resume import finalize_score_dir, init_score_dir, parse_score_args
from .._common.tasks import run_tasks
from .rollout import SWEEnv, load_tasks

__all__ = ["parse_args", "score", "run_score"]

METHOD = "swebench_verified"
_logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score SWE-bench saved patches with official tests.")
    parser.add_argument("--test-timeout", type=float, default=600)
    args = parse_score_args(parser, argv, concurrency=2)
    if args.test_timeout <= 0:
        parser.error("--test-timeout must be positive")
    return args


async def score(task: Task, traj: Trajectory, environment_outputs, config, *, args: argparse.Namespace) -> ScoreResult:
    from swebench.harness.grading import get_eval_report
    from swebench.types import TestSpec

    if not environment_outputs or environment_outputs["patch_file"] != "patch.diff":
        raise ValueError("Missing collected patch")
    legacy_layout = "collect_environment_outputs" not in config and "collect_artifacts" in config
    environment_root = Path(args.run_dir) / ("artifacts" if legacy_layout else "environment_outputs")
    patch = (environment_root / encode_id(task.id) / "patch.diff").read_text(encoding="utf-8")
    result = ScoreResult(1, None, 0.0, METHOD)
    raw = task.raw
    async with SWEEnv(environment_outputs["image"], raw["base_commit"]) as fresh:
        await fresh.write_file(".git/harness.patch", patch)
        apply_started = time.perf_counter()
        applied = await fresh.exec("git apply --verbose .git/harness.patch") if patch.strip() else None
        if applied is not None:
            result.messages.append(
                {
                    "role": "tool",
                    "tool_name": "apply_patch",
                    "content": applied.stdout,
                    "stderr": applied.stderr,
                    "exit_code": applied.exit_code,
                    "is_error": applied.exit_code != 0,
                    "latency_sec": round(time.perf_counter() - apply_started, 6),
                    "scoring_container": fresh.container,
                }
            )
        if applied is not None and applied.exit_code:
            result.extracted_prediction = "patch_failed"
            return result
        await fresh.write_file(".git/harness-eval.sh", raw["eval_script"])
        test_started = time.perf_counter()
        tested = await fresh.exec("/bin/bash .git/harness-eval.sh 2>&1", timeout=args.test_timeout)
        test_sec = round(time.perf_counter() - test_started, 6)
        container = fresh.container
    spec = TestSpec(
        instance_id=task.id,
        image=environment_outputs["image"],
        eval_script_list=[],
        repo=raw["repo"],
        version=raw["version"],
        FAIL_TO_PASS=raw["FAIL_TO_PASS"],
        PASS_TO_PASS=raw["PASS_TO_PASS"],
        log_parser=raw["log_parser"],
        eval_type=raw["eval_type"],
    )
    with tempfile.TemporaryDirectory(prefix="swebench-grade-") as temporary:
        log = Path(temporary) / "test_output.txt"
        log.write_text(tested.stdout + tested.stderr, encoding="utf-8")
        report = get_eval_report(
            spec,
            {"instance_id": task.id, "model_name_or_path": config["model"], "model_patch": patch},
            str(log),
            include_tests_status=True,
        )[task.id]
    result.extracted_prediction = "resolved" if report["resolved"] else "unresolved"
    result.score = float(report["resolved"])
    if report.get("infra_failure"):
        result.sys_error = report.get("infra_failure_reason", "SWE-bench environment failure")
    elif not report["patch_successfully_applied"]:
        result.sys_error = "SWE-bench evaluation produced no valid test results"
    result.messages.append(
        {
            "role": "tool",
            "tool_name": "tests",
            "content": tested.stdout,
            "stderr": tested.stderr,
            "exit_code": tested.exit_code,
            "report": report,
            "is_error": result.sys_error is not None,
            "latency_sec": test_sec,
            "scoring_container": container,
        }
    )
    return result


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
    ) = init_score_dir(args, load_tasks, benchmark="swe_bench", method=METHOD, group_by=("repo",))
    records = list(kept.values())
    pending = [task for task in tasks if task.id not in kept]

    async def score_task(task: Task) -> ScoreRecord:
        trace: Trajectory = trajectories[task.id]
        result = ScoreResult(task.gold, trace.final_answer, 0.0, METHOD)
        scoring_sec = 0.0
        if trace.error is None:
            scoring_started = time.perf_counter()
            try:
                result = await score(task, trace, environment_outputs[task.id], rollout_config, args=args)
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
                pending, score_task, concurrency=args.concurrency, desc="score swe_bench", on_result=persist
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
            benchmark="swe_bench",
            rollout_config=rollout_config,
            group_by=("repo",),
        )
    assert summary is not None
    return output, summary


if __name__ == "__main__":
    output, summary = asyncio.run(run_score(parse_args()))
    print(f"Scores: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
