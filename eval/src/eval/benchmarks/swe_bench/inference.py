"""swe_bench agent rollout with local dataset selection."""

import argparse
import asyncio
import json
import re
import shlex
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from harness import DockerEnv, Task
from harness.model import add_model_args

from eval import ROOT
from eval import inference as eval_inference
from eval.utils import run_tasks

__all__ = ["SWEEnv", "parse_args", "load_tasks", "build_task", "prepare", "collect", "main"]


DATA = Path(__file__).parent / "data"
DATA_FILE = "verified_easy.json"

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
    add_model_args(parser, max_tokens=8192, concurrency=2)
    parser.set_defaults(benchmark="swe_bench", dataset_file=str(DATA / DATA_FILE))
    parser.add_argument("--subset", choices=("easy",), default="easy")
    parser.add_argument("--instance-id", action="append", help="select specific instance IDs")
    return eval_inference.parse_args(parser, argv, max_turns=40, sample_concurrency=2)


def load_tasks(args: argparse.Namespace) -> list[dict]:
    rows = json.loads(Path(args.dataset_file).read_text(encoding="utf-8"))
    if args.instance_id:
        rows = [row for row in rows if row["instance_id"] in args.instance_id]
        missing = set(args.instance_id) - {row["instance_id"] for row in rows}
        if missing:
            raise ValueError(f"Unknown Easy instance IDs: {sorted(missing)}")
    rows = rows[: args.limit]
    if not rows:
        raise ValueError("No SWE-bench Verified Easy instances selected")
    for row in rows:
        row["id"] = row["instance_id"]
        row["gold"] = 1
        row["meta"] = {"repo": row["repo"], "difficulty": row["difficulty"], "base_commit": row["base_commit"]}
        if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
            raise ValueError("Invalid base commit")
    return rows


def build_task(sample: dict, args: argparse.Namespace) -> Task:
    return Task(
        sample["id"],
        SYSTEM,
        sample["problem_statement"],
        args.max_turns,
        args.task_timeout,
        meta=dict(sample["meta"]),
    )


async def prepare(samples: list[dict], *, concurrency: int) -> dict[str, str]:
    images = list(dict.fromkeys(sample["image"] for sample in samples))
    prepared = await run_tasks(images, _prepare_image, concurrency=concurrency, desc="SWE images")
    return dict(zip(images, prepared, strict=True))


async def collect(sample, traj, env, directory: Path) -> dict:
    result = await env.exec(
        "git add -N . && git -c core.fileMode=false diff --binary " + shlex.quote(sample["base_commit"])
    )
    if result.exit_code:
        raise RuntimeError(f"Patch collection failed: {result.stderr}")
    (directory / "patch.diff").write_text(result.stdout, encoding="utf-8")
    return {"patch_file": "patch.diff", "image": env.image, "inference_container": env.container}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples = load_tasks(args)

    async def evaluate():
        images = await prepare(samples, concurrency=args.sample_concurrency)
        return await eval_inference.run_inference(
            args,
            samples,
            [build_task(sample, args) for sample in samples],
            make_env=lambda sample: SWEEnv(images[sample["image"]], sample["base_commit"]),
            collect=collect,
        )

    traces = asyncio.run(evaluate())
    return int(any(trace.error is not None for trace in traces))


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
    raise SystemExit(main())
