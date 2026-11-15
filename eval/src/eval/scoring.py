"""Independent trajectory scoring, evidence checks, and cost summaries."""

import argparse
import hashlib
import importlib
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from harness import Trajectory, load_trajectory
from harness.utils import encode_id, write_json

from eval.utils import SECRET_KEYS, create_directory, run_tasks, trajectory_path

__all__ = ["ScoreResult", "ScoreRecord", "parse_score_args", "run_score", "build_summary", "main"]


@dataclass
class ScoreResult:
    reference_answer: str | int
    extracted_prediction: str | int | None
    score: float
    method: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    sys_error: str | None = None


@dataclass
class ScoreRecord:
    task_id: str
    score_result: ScoreResult
    agent_error: str | None = None
    tool_errors: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)


def parse_score_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    sample_concurrency: int = 1,
) -> argparse.Namespace:
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-concurrency", type=int, default=sample_concurrency)
    args = parser.parse_args(argv)
    if args.sample_concurrency < 1 or getattr(args, "concurrency", 1) < 1:
        parser.error("concurrency limits must be positive")
    return args


async def run_score(
    args: argparse.Namespace,
    load_tasks: Callable[[argparse.Namespace], list[dict[str, Any]]],
    evaluate: Callable[[dict[str, Any], Trajectory, dict | None, dict], Awaitable[ScoreResult]],
    *,
    benchmark: str,
    method: str,
    group_by: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["benchmark"] != benchmark:
        raise ValueError(f"Expected a {benchmark!r} run, got {config['benchmark']!r}")
    if args.sample_concurrency < 1:
        raise ValueError("Concurrency must be positive")
    data_file = Path(config["dataset_file"])
    if not data_file.is_absolute():
        data_file = Path(__file__).resolve().parent / data_file
    dataset_sha256 = hashlib.sha256(data_file.read_bytes()).hexdigest()
    if dataset_sha256 != config["dataset_sha256"]:
        raise ValueError("Dataset checksum differs from the inference dataset")
    samples = load_tasks(argparse.Namespace(**(config | {"dataset_file": str(data_file)})))
    ids = [str(sample["id"]) for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Sample ids must be unique")
    if ids != config["sample_ids"]:
        raise ValueError("Dataset selection does not match the inference sample ids")
    trajectories: dict[str, Trajectory] = {}
    artifacts: dict[str, dict | None] = {}
    hashes = {"config.json": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    for task_id in ids:
        path = trajectory_path(run_dir, task_id)
        if not path.is_file():
            raise ValueError(f"Missing trajectory for sample {task_id!r}")
        trace = load_trajectory(path)
        if trace.task_id != task_id:
            raise ValueError(f"Trajectory id does not match sample {task_id!r}")
        trajectories[task_id] = trace
        hashes[str(path.relative_to(run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[task_id] = None
        if not config["collect_artifacts"]:
            continue
        directory = run_dir / "artifacts" / encode_id(task_id)
        if directory.is_symlink():
            raise ValueError("Artifact directories must be regular directories")
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["task_id"] != task_id:
            raise ValueError(f"Artifact id does not match sample {task_id!r}")
        if manifest["error"] and trace.error is None:
            raise ValueError(f"Sample {task_id!r} has incomplete collection")
        actual_files = set()
        for file in directory.rglob("*"):
            if file.is_symlink():
                raise ValueError("Artifact files must stay inside their sample directory")
            if file.is_file() and file != manifest_path:
                actual_files.add(str(file.relative_to(directory)))
        if actual_files != set(manifest["files"]):
            raise ValueError(f"Artifact files do not match the manifest for {task_id!r}")
        for relative, expected_hash in manifest["files"].items():
            file = directory / relative
            actual_hash = hashlib.sha256(file.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Artifact checksum mismatch for {task_id!r}: {relative}")
            hashes[str(file.relative_to(run_dir))] = actual_hash
        hashes[str(manifest_path.relative_to(run_dir))] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        artifacts[task_id] = manifest["data"]
    if set((run_dir / "trajectories").glob("*.json")) != {trajectory_path(run_dir, i) for i in ids}:
        raise ValueError("Trajectory files do not match the saved sample ids")
    expected_artifacts = {run_dir / "artifacts" / encode_id(i) for i in ids} if config["collect_artifacts"] else set()
    actual_artifacts = set((run_dir / "artifacts").iterdir()) if (run_dir / "artifacts").exists() else set()
    if actual_artifacts != expected_artifacts:
        raise ValueError("Artifact directories do not match the saved sample ids")

    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "scores" / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    )
    if run_dir.is_relative_to(output) or (
        output.is_relative_to(run_dir) and not output.is_relative_to(run_dir / "scores")
    ):
        raise ValueError("Score output must be separate from inference artifacts")
    if data_file.is_relative_to(output):
        raise ValueError("Score output must be separate from the dataset")
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = create_directory(output)
    score_config = {key: value for key, value in vars(args).items() if key not in SECRET_KEYS}
    score_config.update(
        source_run_dir=str(run_dir),
        benchmark=benchmark,
        scoring_method=method,
        group_by=group_by,
        source_sha256=hashes,
        dataset_file=str(data_file),
        dataset_sha256=dataset_sha256,
    )
    score_config.update(metadata or {})
    write_json(output / "config.json", score_config)
    records: list[ScoreRecord] = []
    write_json(output / "trajectories.json", [])

    async def score_sample(sample: dict[str, Any]) -> ScoreRecord:
        task_id = str(sample["id"])
        trace = trajectories[task_id]
        result = ScoreResult(sample["gold"], trace.final_answer, 0.0, method)
        scoring_sec = 0.0
        if trace.error is None:
            scoring_started = time.perf_counter()
            try:
                result = await evaluate(sample, trace, artifacts[task_id], config)
            except Exception as exc:
                result.sys_error = f"{type(exc).__name__}: {exc}"
            finally:
                scoring_sec = round(time.perf_counter() - scoring_started, 6)
        return ScoreRecord(
            task_id,
            result,
            agent_error=trace.error,
            tool_errors=sum(bool(m.get("is_error")) for m in trace.messages if m["role"] == "tool"),
            metadata=sample.get("meta", {}),
            costs={
                "inference": _collect_costs(
                    trace.messages,
                    sample_sec=trace.runtime_latency_sec,
                    episode_sec=trace.latency_sec,
                    usage=trace.usage,
                    model_calls=trace.turns,
                ),
                "scoring": _collect_costs(result.messages, sample_sec=scoring_sec, episode_sec=scoring_sec),
            },
        )

    def persist(record: ScoreRecord) -> None:
        records.append(record)
        write_json(output / "trajectories.json", [asdict(r) for r in records])

    ordered = await run_tasks(
        samples, score_sample, concurrency=args.sample_concurrency, desc=f"score {benchmark}", on_result=persist
    )
    write_json(output / "trajectories.json", [asdict(r) for r in ordered])
    for relative, expected_hash in hashes.items():
        if hashlib.sha256((run_dir / relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Scoring changed inference evidence: {relative}")
    if hashlib.sha256(data_file.read_bytes()).hexdigest() != dataset_sha256:
        raise ValueError("Scoring changed the dataset")
    summary = build_summary(
        benchmark,
        ordered,
        config["model"],
        group_by=group_by,
        inference_wall_sec=config.get("wall_time_sec"),
        scoring_wall_sec=time.perf_counter() - started,
    )
    write_json(output / "summary.json", summary)
    return output, summary


def build_summary(
    benchmark: str,
    records: list[ScoreRecord],
    model_name: str,
    *,
    group_by: tuple[str, ...] = (),
    inference_wall_sec: float | None = None,
    scoring_wall_sec: float | None = None,
) -> dict[str, Any]:
    correct = sum(record.score_result.score for record in records)
    summary: dict[str, Any] = {
        "benchmark": benchmark,
        "model": model_name,
        "num_tasks": len(records),
        "correct": correct,
        "score": round(correct / len(records), 4) if records else 0.0,
        "agent_errors": sum(record.agent_error is not None for record in records),
        "scoring_errors": sum(record.score_result.sys_error is not None for record in records),
        "sys_errors": sum(
            record.agent_error is not None or record.score_result.sys_error is not None for record in records
        ),
        "tool_errors": sum(record.tool_errors for record in records),
    }
    phase_costs = {phase: [record.costs.get(phase, {}) for record in records] for phase in ("inference", "scoring")}
    wall_times = {
        "inference": inference_wall_sec,
        "scoring": scoring_wall_sec,
        "total": (
            inference_wall_sec + scoring_wall_sec
            if inference_wall_sec is not None and scoring_wall_sec is not None
            else None
        ),
    }
    phase_costs["total"] = phase_costs["inference"] + phase_costs["scoring"]
    summary["costs"] = {
        phase: _summarize_costs(costs, num_tasks=len(records), wall_sec=wall_times[phase])
        for phase, costs in phase_costs.items()
    }
    for field_name in group_by:
        grouped: dict[str, list[ScoreRecord]] = {}
        for record in records:
            value = record.metadata.get(field_name)
            grouped.setdefault("unknown" if value is None else str(value), []).append(record)
        summary[f"by_{field_name}"] = {
            key: {
                "num_tasks": len(subset),
                "correct": sum(r.score_result.score for r in subset),
                "score": sum(r.score_result.score for r in subset) / len(subset),
                "costs": build_summary(benchmark, subset, model_name)["costs"],
            }
            for key, subset in sorted(grouped.items())
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark scoring.")
    parser.add_argument("benchmark", help="directory name under eval/benchmarks, e.g. aime")
    args = parser.parse_args(sys.argv[1:2])
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.benchmark):
        parser.error("benchmark names must use snake_case")
    module = importlib.import_module(f"eval.benchmarks.{args.benchmark}.scoring")
    return module.main(sys.argv[2:])


def _collect_costs(
    messages: list[dict[str, Any]],
    *,
    sample_sec: float | None,
    episode_sec: float,
    usage: dict[str, int] | None = None,
    model_calls: int | None = None,
) -> dict[str, Any]:
    models = [message for message in messages if message["role"] == "assistant"]
    tools = [message for message in messages if message["role"] == "tool"]
    tokens = Counter(usage) if usage is not None else Counter()
    if usage is None:
        for message in models:
            tokens.update(message.get("usage") or {})
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens"):
        tokens.setdefault(key, 0)
    tokens["total_tokens"] = tokens["prompt_tokens"] + tokens["completion_tokens"]
    model_sec = (
        sum(message["latency_sec"] - message["queue_sec"] for message in models)
        if all(message.get("latency_sec") is not None and message.get("queue_sec") is not None for message in models)
        else None
    )
    queue_sec = (
        sum(message["queue_sec"] for message in models)
        if all(message.get("queue_sec") is not None for message in models)
        else None
    )
    tool_sec = (
        sum(message["latency_sec"] for message in tools)
        if all(message.get("latency_sec") is not None for message in tools)
        else None
    )
    names = {call["id"]: call["function"]["name"] for message in models for call in message.get("tool_calls", ())}
    by_tool: dict[str, Any] = {}
    for message in tools:
        name = message.get("tool_name") or names.get(message.get("tool_call_id"), "unknown")
        entry = by_tool.setdefault(name, {"calls": 0, "errors": 0, "time_sec": 0.0})
        entry["calls"] += 1
        entry["errors"] += bool(message.get("is_error"))
        duration = message.get("latency_sec")
        entry["time_sec"] = (
            entry["time_sec"] + duration if entry["time_sec"] is not None and duration is not None else None
        )
    return {
        "tokens": dict(tokens),
        "time_sec": {
            "sample": sample_sec,
            "episode": episode_sec,
            "model": model_sec,
            "model_queue": queue_sec,
            "tool": tool_sec,
        },
        "calls": {
            "model": model_calls if model_calls is not None else len(models),
            "model_with_usage": sum(message.get("usage") is not None for message in models),
            "tool": len(tools),
            "tool_errors": sum(bool(message.get("is_error")) for message in tools),
        },
        "by_tool": by_tool,
    }


def _summarize_costs(costs: list[dict[str, Any]], *, num_tasks: int, wall_sec: float | None) -> dict[str, Any]:
    tokens: Counter[str] = Counter(
        {
            key: 0
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "cached_tokens",
                "total_tokens",
            )
        }
    )
    calls: Counter[str] = Counter({key: 0 for key in ("model", "model_with_usage", "tool", "tool_errors")})
    durations: dict[str, float | None] = dict.fromkeys(("sample", "episode", "model", "model_queue", "tool"), 0.0)
    by_tool: dict[str, Any] = {}
    for cost in costs:
        tokens.update(cost.get("tokens", {}))
        calls.update(cost.get("calls", {}))
        for key, previous in durations.items():
            value = cost.get("time_sec", {}).get(key)
            durations[key] = previous + value if previous is not None and value is not None else None
        for name, tool in cost.get("by_tool", {}).items():
            entry = by_tool.setdefault(name, {"calls": 0, "errors": 0, "time_sec": 0.0})
            entry["calls"] += tool["calls"]
            entry["errors"] += tool["errors"]
            entry["time_sec"] = (
                entry["time_sec"] + tool["time_sec"]
                if entry["time_sec"] is not None and tool["time_sec"] is not None
                else None
            )
    for entry in by_tool.values():
        entry["mean_calls_per_task"] = entry["calls"] / num_tasks if num_tasks else 0.0
        entry["mean_time_sec_per_call"] = (
            entry["time_sec"] / entry["calls"] if entry["time_sec"] is not None and entry["calls"] else None
        )
    return {
        "tokens": {
            **tokens,
            "mean_per_task": {key: value / num_tasks if num_tasks else 0.0 for key, value in tokens.items()},
            "mean_per_model_call": {
                key: value / calls["model"] if calls["model"] else 0.0 for key, value in tokens.items()
            },
        },
        "time_sec": {
            **durations,
            "wall": wall_sec,
            "mean_per_task": {
                key: value / num_tasks if value is not None and num_tasks else None for key, value in durations.items()
            },
            "mean_per_model_call": durations["model"] / calls["model"]
            if durations["model"] is not None and calls["model"]
            else None,
            "mean_queue_per_model_call": durations["model_queue"] / calls["model"]
            if durations["model_queue"] is not None and calls["model"]
            else None,
            "mean_per_tool_call": durations["tool"] / calls["tool"]
            if durations["tool"] is not None and calls["tool"]
            else None,
        },
        "calls": {
            **calls,
            "mean_per_task": {key: value / num_tasks if num_tasks else 0.0 for key, value in calls.items()},
        },
        "by_tool": dict(sorted(by_tool.items())),
    }


if __name__ == "__main__":
    raise SystemExit(main())
