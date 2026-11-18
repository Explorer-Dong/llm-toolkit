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

from ...agent import Trajectory
from ...harness.utils import encode_id, write_json
from .._common.metrics import ScoreRecord, ScoreResult, make_score_record
from .._common.resume import finalize_score_dir, init_score_dir, parse_score_args
from .._common.tasks import run_tasks
from .rollout import SWEEnv, load_tasks

__all__ = ["parse_args", "score", "main"]

METHOD = "swebench_verified"
_logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score SWE-bench saved patches with official tests.")
    parser.add_argument("--test-timeout", type=float, default=600)
    args = parse_score_args(parser, argv, sample_concurrency=2)
    if args.test_timeout <= 0:
        parser.error("--test-timeout must be positive")
    return args


async def score(sample, traj, environment_outputs, config, *, args: argparse.Namespace) -> ScoreResult:
    from swebench.harness.grading import get_eval_report
    from swebench.types import TestSpec

    if not environment_outputs or environment_outputs["patch_file"] != "patch.diff":
        raise ValueError("Missing collected patch")
    legacy_layout = "collect_environment_outputs" not in config and "collect_artifacts" in config
    environment_root = Path(args.run_dir) / ("artifacts" if legacy_layout else "environment_outputs")
    patch = (environment_root / encode_id(str(sample["id"])) / "patch.diff").read_text(encoding="utf-8")
    result = ScoreResult(1, None, 0.0, METHOD)
    async with SWEEnv(environment_outputs["image"], sample["base_commit"]) as fresh:
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
        await fresh.write_file(".git/harness-eval.sh", sample["eval_script"])
        test_started = time.perf_counter()
        tested = await fresh.exec("/bin/bash .git/harness-eval.sh 2>&1", timeout=args.test_timeout)
        test_sec = round(time.perf_counter() - test_started, 6)
        container = fresh.container
    spec = TestSpec(
        instance_id=sample["id"],
        image=environment_outputs["image"],
        eval_script_list=[],
        repo=sample["repo"],
        version=sample["version"],
        FAIL_TO_PASS=sample["FAIL_TO_PASS"],
        PASS_TO_PASS=sample["PASS_TO_PASS"],
        log_parser=sample["log_parser"],
        eval_type=sample["eval_type"],
    )
    with tempfile.TemporaryDirectory(prefix="swebench-grade-") as temporary:
        log = Path(temporary) / "test_output.txt"
        log.write_text(tested.stdout + tested.stderr, encoding="utf-8")
        report = get_eval_report(
            spec,
            {"instance_id": sample["id"], "model_name_or_path": config["model"], "model_patch": patch},
            str(log),
            include_tests_status=True,
        )[sample["id"]]
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
    ) = init_score_dir(args, load_tasks, benchmark="swe_bench", method=METHOD, group_by=("repo",))
    records = list(kept.values())
    pending = [sample for sample in samples if str(sample["id"]) not in kept]

    async def score_sample(sample: dict[str, Any]) -> ScoreRecord:
        task_id = str(sample["id"])
        trace: Trajectory = trajectories[task_id]
        result = ScoreResult(sample["gold"], trace.final_answer, 0.0, METHOD)
        scoring_sec = 0.0
        if trace.error is None:
            scoring_started = time.perf_counter()
            try:
                result = await score(sample, trace, environment_outputs[task_id], rollout_config, args=args)
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
                pending, score_sample, concurrency=args.sample_concurrency, desc="score swe_bench", on_result=persist
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
            benchmark="swe_bench",
            rollout_config=rollout_config,
            group_by=("repo",),
        )
    assert summary is not None
    return output, summary


if __name__ == "__main__":
    raise SystemExit(main())
