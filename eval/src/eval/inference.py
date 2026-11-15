"""Benchmark inference, runtime lifecycles, and trajectory storage."""

import argparse
import asyncio
import hashlib
import importlib
import re
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from harness import (
    Env,
    HttpSpec,
    NoEnv,
    StdioSpec,
    Task,
    Trajectory,
    env_server,
    load_trajectory,
    open_toolbox,
    run_episode,
    save_trajectory,
)
from harness.model import build_model_params, make_context
from harness.utils import encode_id, write_json
from mcp.server.fastmcp import FastMCP

from eval import OUTPUT_DIR
from eval.utils import SECRET_KEYS, create_directory, run_tasks, trajectory_path

__all__ = ["parse_args", "run_inference", "main"]


def parse_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    max_turns: int = 1,
    sample_concurrency: int | None = None,
) -> argparse.Namespace:
    parser.add_argument("--max-turns", "--max-steps", dest="max_turns", type=int, default=max_turns)
    parser.add_argument("--task-timeout", type=float, default=3600, help="episode budget in seconds; 0 disables it")
    parser.add_argument(
        "--sample-concurrency", type=int, default=sample_concurrency, help="maximum active runtime instances"
    )
    parser.add_argument("--limit", type=int, help="run the first N matching samples")
    parser.add_argument("--output-dir", help="new inference directory; defaults to outputs/<benchmark>/<run>")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model or MODEL_NAME is required")
    if args.concurrency < 1 or (args.sample_concurrency is not None and args.sample_concurrency < 1):
        parser.error("concurrency limits must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_turns < 1 or args.task_timeout < 0:
        parser.error("max_turns must be positive and task_timeout must be nonnegative")
    args.sample_concurrency = args.sample_concurrency or args.concurrency
    return args


async def run_inference(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    tasks: list[Task],
    *,
    shared_servers: Sequence[StdioSpec | HttpSpec | FastMCP] = (),
    make_env: Callable[[dict[str, Any]], Env] | None = None,
    collect: Callable[[dict[str, Any], Trajectory, Env, Path], Awaitable[dict | None]] | None = None,
) -> list[Trajectory]:
    started = time.perf_counter()
    if args.concurrency < 1 or args.sample_concurrency < 1:
        raise ValueError("Concurrency limits must be positive")
    pairs = list(zip(samples, tasks, strict=True))
    if any(task.id != str(sample["id"]) for sample, task in pairs):
        raise ValueError("Task id must match its sample id")
    ids = [str(sample["id"]) for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Sample ids must be unique")
    for task_id in ids:
        encode_id(task_id)
    data_file = Path(args.dataset_file).resolve()
    dataset_sha256 = hashlib.sha256(data_file.read_bytes()).hexdigest()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    model = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model).lower()
    run_dir = (
        Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR / args.benchmark / f"{model}_{timestamp}"
    )
    if args.output_dir:
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = create_directory(run_dir)
    config = {key: value for key, value in vars(args).items() if key not in SECRET_KEYS}
    package = Path(__file__).resolve().parent
    config.update(
        dataset_file=str(data_file.relative_to(package)) if data_file.is_relative_to(package) else str(data_file),
        dataset_sha256=dataset_sha256,
        sample_ids=ids,
        collect_artifacts=collect is not None,
    )
    write_json(run_dir / "config.json", config)
    ctx = make_context(args.base_url, args.api_key, build_model_params(args), args.concurrency, timeout=args.timeout)
    try:
        async with open_toolbox(*shared_servers) as shared_tools:

            async def run_sample(pair: tuple[dict[str, Any], Task]) -> Trajectory:
                sample_started = time.perf_counter()
                sample, task = pair
                trajectory_file = trajectory_path(run_dir, task.id)
                artifact_dir = run_dir / "artifacts" / encode_id(task.id)
                if collect is not None:
                    artifact_dir.mkdir(parents=True, exist_ok=False)
                trace = Trajectory(
                    task.id,
                    messages=[{"role": "system", "content": task.system}, {"role": "user", "content": task.prompt}],
                    meta=dict(task.meta),
                )
                phase = "setup"
                episode_failure = False
                collection_error = None
                artifacts = None
                try:
                    env = make_env(sample) if make_env is not None else NoEnv()
                    async with env:
                        try:
                            phase = "tools"
                            sample_servers = [env_server(env)] if env.caps else []
                            async with shared_tools.scoped(*sample_servers) as tools:
                                phase = "episode"
                                try:
                                    trace = await run_episode(task, ctx, tools, out=trajectory_file)
                                except Exception:
                                    episode_failure = True
                                    raise
                                phase = "tools_cleanup"
                        finally:
                            if trajectory_file.exists():
                                trace = load_trajectory(trajectory_file)
                            previous_phase = phase
                            phase = "collect"
                            if collect is not None:
                                try:
                                    artifacts = await collect(sample, trace, env, artifact_dir)
                                except BaseException as exc:
                                    collection_error = f"{type(exc).__name__}: {exc}"
                                    raise
                                finally:
                                    _save_artifacts(artifact_dir, task.id, artifacts, collection_error)
                            phase = previous_phase
                        phase = "teardown"
                except asyncio.CancelledError:
                    trace.stop_reason, trace.error = "cancelled", "CancelledError"
                    trace.runtime_latency_sec = round(time.perf_counter() - sample_started, 6)
                    save_trajectory(trace, trajectory_file)
                    if collect is not None and not (artifact_dir / "manifest.json").exists():
                        _save_artifacts(artifact_dir, task.id, None, "Cancelled before collection")
                    raise
                except Exception as exc:
                    if episode_failure:
                        raise
                    trace.stop_reason = "environment_error"
                    detail = f"{phase}: {type(exc).__name__}: {exc}"
                    trace.error = f"{trace.error}; {detail}" if trace.error else detail
                    if collect is not None and not (artifact_dir / "manifest.json").exists():
                        _save_artifacts(artifact_dir, task.id, None, detail)
                trace.runtime_latency_sec = round(time.perf_counter() - sample_started, 6)
                save_trajectory(trace, trajectory_file)
                return trace

            traces = await run_tasks(pairs, run_sample, concurrency=args.sample_concurrency, desc=args.benchmark)
    finally:
        try:
            await ctx.client.close()
        finally:
            config["wall_time_sec"] = round(time.perf_counter() - started, 6)
            write_json(run_dir / "config.json", config)
    errors = sum(trace.error is not None for trace in traces)
    print(f"Saved {len(traces)} trajectories; system errors: {errors}; output: {run_dir}")
    print(f"Score separately: eval-scoring {args.benchmark} {run_dir}")
    return traces


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark inference.")
    parser.add_argument("benchmark", help="directory name under eval/benchmarks, e.g. aime")
    args = parser.parse_args(sys.argv[1:2])
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.benchmark):
        parser.error("benchmark names must use snake_case")
    module = importlib.import_module(f"eval.benchmarks.{args.benchmark}.inference")
    return module.main(sys.argv[2:])


def _save_artifacts(directory: Path, task_id: str, data: dict | None, error: str | None) -> None:
    files = {}
    for file in sorted(directory.rglob("*")):
        if file.name == "manifest.json" and file.parent == directory:
            continue
        if file.is_symlink():
            raise ValueError("Collected evidence must contain regular files and directories")
        if file.is_file():
            files[str(file.relative_to(directory))] = hashlib.sha256(file.read_bytes()).hexdigest()
    write_json(directory / "manifest.json", {"task_id": task_id, "data": data, "files": files, "error": error})


if __name__ == "__main__":
    raise SystemExit(main())
