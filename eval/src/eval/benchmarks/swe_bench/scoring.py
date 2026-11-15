"""Grade saved SWE-bench patches with official tests in fresh runtime instances."""

import argparse
import asyncio
import json
import tempfile
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from harness.utils import encode_id

from eval import scoring as eval_scoring
from eval.scoring import ScoreResult

from .inference import SWEEnv, load_tasks

__all__ = ["parse_args", "score", "main"]

METHOD = "swebench_verified"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score SWE-bench saved patches with official tests.")
    parser.add_argument("--test-timeout", type=float, default=600)
    args = eval_scoring.parse_score_args(parser, argv, sample_concurrency=2)
    if args.test_timeout <= 0:
        parser.error("--test-timeout must be positive")
    return args


async def score(sample, traj, artifacts, config, *, args: argparse.Namespace) -> ScoreResult:
    from swebench.harness.grading import get_eval_report
    from swebench.types import TestSpec

    if not artifacts or artifacts["patch_file"] != "patch.diff":
        raise ValueError("Missing collected patch")
    patch = (Path(args.run_dir) / "artifacts" / encode_id(str(sample["id"])) / "patch.diff").read_text(encoding="utf-8")
    result = ScoreResult(1, None, 0.0, METHOD)
    async with SWEEnv(artifacts["image"], sample["base_commit"]) as fresh:
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
        image=artifacts["image"],
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
    args = parse_args(argv)
    output, summary = asyncio.run(
        eval_scoring.run_score(
            args,
            load_tasks,
            partial(score, args=args),
            benchmark="swe_bench",
            method=METHOD,
            group_by=("repo",),
        )
    )
    print(f"Scores: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary["sys_errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
