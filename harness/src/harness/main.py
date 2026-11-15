"""Episode exception handling, atomic persistence, and standalone debugging CLI."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from harness.agent import Task, Trajectory, react
from harness.model import Context, add_model_args, build_model_params, make_context
from harness.tool import ToolBox, ToolConfig, add_tool_args, common_server, open_toolbox
from harness.utils import save_trajectory

__all__ = ["cli", "run_episode"]


async def run_episode(task: Task, ctx: Context, tools: ToolBox, out: Path | None = None) -> Trajectory:
    trace = Trajectory(task.id)
    started = time.perf_counter()
    timeout = asyncio.timeout(task.timeout or None)
    try:
        async with timeout:
            await react(task, ctx, tools, trajectory=trace)
    except asyncio.CancelledError:
        trace.stop_reason, trace.error = "cancelled", "CancelledError"
        raise
    except TimeoutError as exc:
        trace.stop_reason = "timeout" if timeout.expired() else "error"
        trace.error = f"task_timeout ({task.timeout}s)" if timeout.expired() else f"TimeoutError: {exc}"
    except Exception as exc:
        trace.stop_reason, trace.error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        trace.latency_sec = round(time.perf_counter() - started, 3)
        if out is not None:
            save_trajectory(trace, out)
    return trace


def cli() -> int:
    parser = argparse.ArgumentParser(description="Run one ReAct episode or serve common tools over MCP.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve-common", help="serve common tools over stdio MCP")
    add_tool_args(serve)
    run = commands.add_parser("run")
    run.add_argument("task", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--common-tools", action="store_true", help="enable common search and fetch tools")
    add_model_args(run)
    add_tool_args(run)
    args = parser.parse_args()
    config = ToolConfig.from_args(args)
    if args.command == "serve-common":
        common_server(config).run(transport="stdio")
        return 0
    if not args.model:
        parser.error("--model or MODEL_NAME is required")

    async def run_task() -> Trajectory:
        task = Task(**json.loads(args.task.read_text(encoding="utf-8")))
        ctx = make_context(
            args.base_url, args.api_key, build_model_params(args), args.concurrency, timeout=args.timeout
        )
        try:
            servers = [common_server(config)] if args.common_tools else []
            async with open_toolbox(*servers) as tools:
                return await run_episode(task, ctx, tools, args.out)
        finally:
            await ctx.client.close()

    trace = asyncio.run(run_task())
    print(f"{trace.stop_reason}: {args.out}")
    return int(trace.error is not None)


if __name__ == "__main__":
    raise SystemExit(cli())
