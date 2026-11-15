"""Real limit-N inference and independent scoring for all benchmark commands."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from harness import load_trajectory
from harness.model import add_thinking_args

from eval import OUTPUT_DIR, ROOT

__all__ = ["main"]

BENCHMARKS = ("aime", "gpqa", "math", "gaia", "browsecomp", "webwalkerqa", "swe_bench")
RETRIEVAL_BENCHMARKS = {"gaia", "browsecomp", "webwalkerqa"}
JUDGE_BENCHMARKS = {"browsecomp", "webwalkerqa"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real model/MCP smoke checks and independent score processes")
    parser.add_argument("--benchmark", choices=BENCHMARKS, action="append")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--task-timeout", type=float, default=300)
    parser.add_argument("--output-dir", type=Path)
    add_thinking_args(parser)
    args = parser.parse_args()
    if min(args.limit, args.parallel, args.max_tokens, args.max_turns, args.timeout, args.task_timeout) <= 0:
        parser.error("smoke limits and timeouts must be positive")
    output = args.output_dir or OUTPUT_DIR / f"smoke_two_stage_{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        pending = {
            executor.submit(_run_benchmark, benchmark, args, output): benchmark
            for benchmark in args.benchmark or BENCHMARKS
        }
        for future in as_completed(pending):
            benchmark = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"benchmark": benchmark, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps(sorted(results, key=lambda item: item["benchmark"]), ensure_ascii=False), flush=True)
    return int(any(not result["passed"] for result in results))


def _run_benchmark(benchmark: str, args: argparse.Namespace, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "eval.inference",
        benchmark,
        "--limit",
        str(2 if benchmark == "swe_bench" else args.limit),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        "0",
        "--top-k",
        "0",
        "--timeout",
        str(args.timeout),
        "--task-timeout",
        str(args.task_timeout),
        "--concurrency",
        "1",
        "--sample-concurrency",
        "1",
        "--output-dir",
        str(output / benchmark),
        "--reasoning-effort",
        args.reasoning_effort,
    ]
    if not args.enable_thinking:
        command.append("--disable-thinking")
    if benchmark in RETRIEVAL_BENCHMARKS:
        command.extend(["--max-turns", str(args.max_turns)])
    if benchmark == "swe_bench":
        command.extend(
            [
                "--subset",
                "easy",
                "--sample-concurrency",
                "2",
                "--concurrency",
                "2",
                "--max-turns",
                "40",
                "--task-timeout",
                "900",
            ]
        )
    started = time.perf_counter()
    inference = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800 if benchmark == "swe_bench" else args.task_timeout + 180,
        check=False,
    )
    inference_sec = round(time.perf_counter() - started, 3)
    match = re.search(r"output: (.+)", inference.stdout)
    if match is None:
        raise RuntimeError(f"inference failed: {inference.stderr[-2000:]}")
    run_dir = Path(match.group(1).strip())
    config = json.loads((run_dir / "config.json").read_text())
    if (run_dir / "samples.json").exists():
        raise RuntimeError("Inference saved a redundant dataset copy")
    if benchmark != "swe_bench" and (run_dir / "artifacts").exists():
        raise RuntimeError("Inference created an empty evidence directory")
    paths = sorted((run_dir / "trajectories").glob("*.json"))
    trajectories = [load_trajectory(path) for path in paths]
    if {trajectory.task_id for trajectory in trajectories} != set(config["sample_ids"]):
        raise RuntimeError("trajectory IDs differ from the saved sample IDs")
    for trajectory in trajectories:
        assistants = [message for message in trajectory.messages if message["role"] == "assistant"]
        if trajectory.turns != len(assistants):
            raise RuntimeError("trajectory turn count differs from the saved model messages")
        for key, count in trajectory.usage.items():
            if count != sum((message.get("usage") or {}).get(key, 0) for message in assistants):
                raise RuntimeError(f"trajectory {key} differs from per-turn usage")
    inference_files = [path for path in run_dir.rglob("*") if path.is_file()]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inference_files}
    started = time.perf_counter()
    score_command = [
        sys.executable,
        "-m",
        "eval.scoring",
        benchmark,
        str(run_dir),
        "--sample-concurrency",
        "2" if benchmark == "swe_bench" else "1",
    ]
    if benchmark in JUDGE_BENCHMARKS:
        score_command.extend(["--timeout", str(args.timeout), "--concurrency", "1"])
        score_command.extend(["--reasoning-effort", args.reasoning_effort])
        if not args.enable_thinking:
            score_command.append("--disable-thinking")
    scoring = subprocess.run(
        score_command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1500 if benchmark == "swe_bench" else args.timeout * (args.limit * 4 + 1),
        check=False,
    )
    scoring_sec = round(time.perf_counter() - started, 3)
    match = re.search(r"Scores: (.+)", scoring.stdout)
    if match is None:
        raise RuntimeError(f"scoring failed: {scoring.stderr[-2000:]}")
    score_dir = Path(match.group(1).strip())
    if not re.fullmatch(r"\d{8}-\d{6}(?:-\d+)?", score_dir.name):
        raise RuntimeError("Score timestamps must use second precision")
    if {path.name for path in score_dir.iterdir()} != {"config.json", "trajectories.json", "summary.json"}:
        raise RuntimeError("Score output files differ from the result contract")
    summary = json.loads((score_dir / "summary.json").read_text())
    records = json.loads((score_dir / "trajectories.json").read_text())
    if {record["task_id"] for record in records} != set(config["sample_ids"]):
        raise RuntimeError("Score records differ from the selected sample IDs")
    cost = summary["costs"]
    inferred_tokens = sum(t.usage["prompt_tokens"] + t.usage["completion_tokens"] for t in trajectories)
    scored_models = [m for r in records for m in r["score_result"]["messages"] if m["role"] == "assistant"]
    scored_tokens = sum(
        (m.get("usage") or {}).get(key, 0) for m in scored_models for key in ("prompt_tokens", "completion_tokens")
    )
    if (cost["inference"]["tokens"]["total_tokens"], cost["scoring"]["tokens"]["total_tokens"]) != (
        inferred_tokens,
        scored_tokens,
    ):
        raise RuntimeError("Summary tokens differ from the saved model responses")
    if cost["total"]["calls"]["model"] != sum(t.turns for t in trajectories) + len(scored_models):
        raise RuntimeError("Summary model calls differ from saved turns")
    if not math.isclose(
        cost["total"]["tokens"]["mean_per_task"]["total_tokens"], (inferred_tokens + scored_tokens) / len(records)
    ):
        raise RuntimeError("Token means must include all selected samples")
    if cost["inference"]["time_sec"]["wall"] != config["wall_time_sec"]:
        raise RuntimeError("Inference wall time differs from the run manifest")
    if any(t.runtime_latency_sec is None for t in trajectories):
        raise RuntimeError("A trajectory is missing runtime timing")
    if any(
        cost[phase]["time_sec"][key] is None
        for phase in ("inference", "scoring", "total")
        for key in ("sample", "model", "model_queue", "tool", "wall")
    ):
        raise RuntimeError("A summary is missing recorded timing")
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inference_files}
    unchanged = before == after
    result = {
        "benchmark": benchmark,
        "passed": inference.returncode == scoring.returncode == 0 and unchanged and summary["sys_errors"] == 0,
        "inference_exit_code": inference.returncode,
        "scoring_exit_code": scoring.returncode,
        "inference_sec": inference_sec,
        "scoring_sec": scoring_sec,
        "run_dir": str(run_dir),
        "score_dir": str(score_dir),
        "inference_artifacts_unchanged": unchanged,
        "num_without_final_answer": sum(trajectory.final_answer is None for trajectory in trajectories),
        "tokens": {phase: cost[phase]["tokens"]["total_tokens"] for phase in cost},
        "model_calls": {phase: cost[phase]["calls"]["model"] for phase in cost},
        "runtime_calls": {phase: cost[phase]["calls"]["tool"] for phase in cost},
        **{
            key: summary[key]
            for key in ("num_tasks", "correct", "score", "agent_errors", "scoring_errors", "sys_errors", "tool_errors")
        },
    }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
