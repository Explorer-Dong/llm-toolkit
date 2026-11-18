"""swe_bench agent rollout with local dataset selection."""

import argparse
import asyncio
import json
import logging
import re
import shlex
import time
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from ... import ROOT
from ...agent import Task, Trajectory, react
from ...harness import add_harness_args
from ...harness.env import DockerEnv
from ...harness.tool import env_server, open_toolbox
from ...harness.utils import save_trajectory
from ...model import add_model_args, build_model_params, make_client
from .._common.resume import filter_pending_tasks, finalize_rollout_dir, init_rollout_dir, parse_rollout_args
from .._common.storage import (
    cleanup_cancelled_task,
    prepare_environment_dir,
    save_environment_result,
    trajectory_path,
)
from .._common.tasks import run_tasks

__all__ = ["SWEEnv", "parse_args", "load_tasks", "prepare", "collect", "rollout"]


DATA = Path(__file__).parent / "data"
DATA_FILE = "verified_easy.json"
_logger = logging.getLogger(__name__)

SYSTEM = """Fix the reported issue in the repository at /testbed using the provided tools.
Inspect the source, implement a focused fix, and run relevant tests. Preserve existing behavior.
Shell commands start in /testbed with its Python environment activated.
Finish with a brief summary after editing the working tree. Your code changes will be graded separately.
"""


class SWEEnv(DockerEnv):
    def __init__(self, image: str, base_commit: str) -> None:
        super().__init__(image, workdir="/testbed")
        self.base_commit = base_commit

    async def setup(self) -> None:
        await super().setup()
        result = await self.exec(
            "git reset --hard " + shlex.quote(self.base_commit) + " && git config core.fileMode false"
        )
        if result.exit_code:
            raise RuntimeError(f"SWE-bench environment initialization failed: {result.stderr}")

    async def exec(self, cmd: str, timeout: float = 60):
        return await super().exec(
            ". /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && " + cmd,
            timeout=timeout,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="swe_bench")
    add_model_args(parser, max_tokens=8192)
    add_harness_args(parser, max_turns=40, concurrency=2)
    parser.set_defaults(benchmark="swe_bench", dataset_file=str(DATA / DATA_FILE))
    parser.add_argument("--subset", choices=("easy",), default="easy")
    parser.add_argument("--instance-id", action="append", help="select specific instance IDs")
    return parse_rollout_args(parser, argv)


def load_tasks(args: argparse.Namespace) -> list[Task]:
    rows = json.loads(Path(args.dataset_file).read_text(encoding="utf-8"))
    if args.instance_id:
        rows = [row for row in rows if row["instance_id"] in args.instance_id]
        missing = set(args.instance_id) - {row["instance_id"] for row in rows}
        if missing:
            raise ValueError(f"Unknown Easy instance IDs: {sorted(missing)}")
    rows = rows[: args.limit]
    if not rows:
        raise ValueError("No SWE-bench Verified Easy instances selected")
    max_turns = getattr(args, "max_turns", 40)
    timeout = getattr(args, "task_timeout", 3600.0)
    tasks: list[Task] = []
    for row in rows:
        if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
            raise ValueError("Invalid base commit")
        tasks.append(
            Task(
                id=row["instance_id"],
                system=SYSTEM,
                prompt=row["problem_statement"],
                max_turns=max_turns,
                timeout=timeout,
                meta={"repo": row["repo"], "difficulty": row["difficulty"], "base_commit": row["base_commit"]},
                gold=1,
                raw=row,
            )
        )
    return tasks


async def prepare(tasks: list[Task], *, concurrency: int) -> dict[str, str]:
    images = list(dict.fromkeys(task.raw["image"] for task in tasks))
    prepared = await run_tasks(images, _prepare_image, concurrency=concurrency, desc="SWE images")
    return dict(zip(images, prepared, strict=True))


async def collect(task: Task, trace: Trajectory, env: SWEEnv, directory: Path) -> dict:
    result = await env.exec(
        "git add -N . && git -c core.fileMode=false diff --binary " + shlex.quote(task.raw["base_commit"])
    )
    if result.exit_code:
        raise RuntimeError(f"Patch collection failed: {result.stderr}")
    (directory / "patch.diff").write_text(result.stdout, encoding="utf-8")
    return {"patch_file": "patch.diff", "image": env.image, "rollout_container": env.container}


async def rollout(args: argparse.Namespace, tasks: list[Task]) -> list[Trajectory]:
    started = time.perf_counter()
    pending_images = filter_pending_tasks(args, tasks, collect_enabled=True)
    images = await prepare(pending_images, concurrency=args.concurrency) if pending_images else {}
    run_dir, config, saved_config, completed, pending = init_rollout_dir(args, tasks, collect_enabled=True)
    new_traces: list[Trajectory] = []
    try:
        if pending:
            model_params = build_model_params(args)
            async with make_client(args.base_url, args.api_key, timeout=args.timeout) as client:

                async def run_task(task: Task) -> Trajectory:
                    task_started = time.perf_counter()
                    trajectory_file = trajectory_path(run_dir, task.id)
                    environment_dir = prepare_environment_dir(run_dir, task.id)
                    trace = Trajectory(
                        task.id,
                        messages=[{"role": "system", "content": task.system}, {"role": "user", "content": task.prompt}],
                        meta=dict(task.meta),
                    )
                    phase = "setup"
                    cancelled = False
                    collection_error = None
                    environment_outputs = None
                    try:
                        env = SWEEnv(images[task.raw["image"]], task.raw["base_commit"])
                        async with env:
                            try:
                                task = await env.prepare_task(task)
                                phase = "tools"
                                async with open_toolbox(env_server(env)) as tools:
                                    phase = "episode"
                                    trace = await react(task, client, tools, model_params, env=env)
                                    phase = "tools_cleanup"
                            except asyncio.CancelledError:
                                cancelled = True
                                raise
                            finally:
                                if not cancelled:
                                    previous_phase = phase
                                    phase = "collect"
                                    try:
                                        environment_outputs = await collect(task, trace, env, environment_dir)
                                    except BaseException as exc:
                                        collection_error = f"{type(exc).__name__}: {exc}"
                                        raise
                                    finally:
                                        save_environment_result(
                                            environment_dir, task.id, environment_outputs, collection_error
                                        )
                                    phase = previous_phase
                            phase = "teardown"
                    except asyncio.CancelledError:
                        cleanup_cancelled_task(run_dir, task.id)
                        raise
                    except Exception as exc:
                        _logger.exception("Runtime %s failed during %s", task.id, phase)
                        trace.episode_status = "environment_error"
                        detail = f"{phase}: {type(exc).__name__}: {exc}"
                        trace.error = f"{trace.error}; {detail}" if trace.error else detail
                        if not (environment_dir / "environment_result.json").exists():
                            save_environment_result(environment_dir, task.id, None, detail)
                    trace.runtime_latency_sec = round(time.perf_counter() - task_started, 6)
                    save_trajectory(trace, trajectory_file)
                    return trace

                new_traces = await run_tasks(pending, run_task, concurrency=args.concurrency, desc=args.benchmark)
    finally:
        by_id = dict(completed)
        by_id.update({trace.task_id: trace for trace in new_traces})
        traces = [by_id[task.id] for task in tasks] if len(by_id) == len(tasks) else list(by_id.values())
        finalize_rollout_dir(
            run_dir,
            config,
            saved_config,
            started=started,
            resumed_count=len(completed),
            benchmark=args.benchmark,
            traces=traces,
        )
    return traces


async def _prepare_image(image: str) -> str:
    for command in (("image", "inspect", image), ("pull", image), ("image", "inspect", image)):
        process = await asyncio.create_subprocess_exec(
            "docker", *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await process.communicate()
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode == 0 and command[0] == "image":
            info = json.loads(stdout)[0]
            return (info.get("RepoDigests") or [info["Id"]])[0]
        if process.returncode and command[0] == "pull":
            raise RuntimeError(f"Failed to prepare {image}: {stderr.decode(errors='replace')}")
    raise RuntimeError(f"Image is unavailable after pulling: {image}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(rollout(args, load_tasks(args)))
