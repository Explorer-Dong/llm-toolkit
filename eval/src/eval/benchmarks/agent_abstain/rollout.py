"""AgentAbstain tasks over isolated simulated environments."""

import argparse
import asyncio
import copy
import json
import logging
import sys
import time
from collections.abc import Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.types import CallToolResult, TextContent, Tool

from ... import ROOT
from ...agent import Task, Trajectory, run_episode
from ...harness import add_harness_args
from ...harness.env import Env
from ...harness.tool import env_server, open_toolbox
from ...harness.utils import load_trajectory, save_trajectory
from ...model import add_model_args, build_model_params, make_context
from .._common.resume import finalize_rollout_dir, init_rollout_dir, parse_rollout_args
from .._common.storage import prepare_environment_dir, save_environment_result, trajectory_path
from .._common.tasks import run_tasks

__all__ = ["AgentAbstainEnv", "parse_args", "load_tasks", "build_task", "collect", "run_rollout", "main"]
DATA = Path(__file__).parent / "data"
DATA_FILE = "agent_abstain.json"
_TOOL_CALL_ACTIVE = ContextVar("agent_abstain_tool_call_active", default=False)
_logger = logging.getLogger(__name__)

SCENARIOS = (
    ("S1", "missing_critical_parameter", "关键参数缺失"),
    ("S2", "ambiguous_action_specification", "指令歧义"),
    ("S3", "conflicting_constraints", "约束冲突"),
    ("S4", "high_stakes_action", "高风险操作"),
    ("S5", "insufficient_tool_capability", "工具能力不足"),
    ("S6", "critical_tool_failure", "关键工具失败"),
    ("S7", "conflicting_evidence", "证据冲突"),
    ("S8", "emergent_risk_discovery", "运行中风险发现"),
)


class AgentAbstainEnv(Env):
    caps = frozenset({"tools"})

    def __init__(self, sample: dict) -> None:
        self.sample = sample
        self.environment: Any = None

    async def setup(self) -> None:
        from .data.runtime.multi import build_multi_environment

        if self.environment is not None:
            raise RuntimeError("AgentAbstain environment is already open")
        logger = logging.getLogger("fastmcp.server.server")
        if _filter_tool_error_log not in logger.filters:
            logger.addFilter(_filter_tool_error_log)
        self.environment = build_multi_environment(
            self.sample["env_types"], copy.deepcopy(self.sample["initial_states"])
        )
        task = self.sample["task_yaml"]
        for broken in task.get("tool_overrides", {}).get("broken_tools", []):
            await self.environment.abreak_tool(broken["name"], broken.get("error", "Service unavailable"))
        if task.get("available_tools") is not None:
            allowed = {tool["name"] if isinstance(tool, dict) else tool for tool in task["available_tools"]}
            for schema in self.environment.get_tool_schemas():
                if schema["name"] not in allowed:
                    self.environment.hide_tool(schema["name"])

    async def teardown(self) -> None:
        self.environment = None

    async def list_tools(self) -> list[Tool]:
        tools = await self.environment.mcp.list_tools()
        return [tool.to_mcp_tool() for tool in tools]

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        from fastmcp.exceptions import ToolError as MCPToolError
        from fastmcp.exceptions import ValidationError as MCPValidationError

        from .data.runtime.base import ToolError

        token = _TOOL_CALL_ACTIVE.set(True)
        try:
            result = await self.environment.mcp.call_tool(name, arguments)
            return CallToolResult(content=result.content, structuredContent=result.structured_content)
        except (MCPToolError, MCPValidationError, ToolError) as exc:
            if not _is_agent_tool_error(exc):
                raise
            return CallToolResult(content=[TextContent(type="text", text=str(exc))], isError=True)
        finally:
            _TOOL_CALL_ACTIVE.reset(token)

    def export(self) -> dict:
        if self.environment is None:
            raise RuntimeError("AgentAbstain environment is not open")
        return _serialize({"state": self.environment.state, "execution_log": self.environment.get_execution_log()})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="AgentAbstain")
    add_model_args(parser, max_tokens=16384, timeout=600, concurrency=4)
    add_harness_args(parser, max_turns=30, task_timeout=900)
    parser.set_defaults(
        benchmark="agent_abstain",
        dataset_file=str(DATA / DATA_FILE),
        top_p=None,
        seed=None,
        max_retries=2,
        max_tokens_field="max_completion_tokens",
        extra_body={"store": False},
    )
    parser.add_argument("--dataset-file", default=str(DATA / DATA_FILE))
    parser.add_argument("--category", choices=[category for _, category, _ in SCENARIOS])
    parser.add_argument("--task-id", action="append", help="pair task ID, e.g. preview_001")
    parser.add_argument("--task-type", choices=("act", "abstain"))
    parser.add_argument("--action-type", choices=("informational", "operational"))
    args = parse_rollout_args(parser, argv)
    if args.model.rsplit("/", 1)[-1].lower() == "qwen3.8-27b" and args.reasoning_effort not in {
        "low",
        "medium",
        "xhigh",
    }:
        parser.error("Qwen3.8-27B reasoning effort must be low, medium or xhigh")
    return args


def load_tasks(args: argparse.Namespace) -> list[dict]:
    rows = json.loads(Path(args.dataset_file).read_text(encoding="utf-8"))
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate AgentAbstain sample IDs")
    categories = {category for _, category, _ in SCENARIOS}
    for row in rows:
        meta = row["meta"]
        if meta["category"] not in categories or meta["task_type"] not in {"act", "abstain"}:
            raise ValueError("Invalid AgentAbstain task identity")
        expected_id = f"{meta['category']}/{meta['task_id']}/{meta['task_type']}"
        if row["id"] != expected_id or row["gold"] != f"correct_{meta['task_type']}":
            raise ValueError("AgentAbstain labels differ from sample identity")
    for field in ("category", "task_type", "action_type"):
        value = getattr(args, field, None)
        if value is not None:
            rows = [row for row in rows if row["meta"][field] == value]
    if getattr(args, "task_id", None):
        missing = set(args.task_id) - {row["meta"]["task_id"] for row in rows}
        if missing:
            raise ValueError(f"Unknown AgentAbstain task IDs: {sorted(missing)}")
        rows = [row for row in rows if row["meta"]["task_id"] in args.task_id]
    rows = rows[: args.limit]
    if not rows:
        raise ValueError("No AgentAbstain tasks selected")
    return rows


def build_task(sample: dict, args: argparse.Namespace) -> Task:
    return Task(
        sample["id"],
        sample["system"],
        sample["prompt"],
        args.max_turns,
        args.task_timeout,
        meta=dict(sample["meta"]),
        strict_response=True,
    )


async def collect(sample, trace, env, directory: Path) -> dict:
    return env.export()


async def run_rollout(args: argparse.Namespace, samples: list[dict[str, Any]]) -> list[Trajectory]:
    started = time.perf_counter()
    tasks = [build_task(sample, args) for sample in samples]
    run_dir, config, saved_config, completed, pending_pairs = init_rollout_dir(
        args, samples, tasks, collect_enabled=True
    )
    new_traces: list[Trajectory] = []
    ctx = None
    try:
        if pending_pairs:
            ctx = make_context(
                args.base_url,
                args.api_key,
                build_model_params(args),
                args.concurrency,
                timeout=args.timeout,
                max_retries=getattr(args, "max_retries", 3),
            )

            async def run_sample(pair: tuple[dict[str, Any], Task]) -> Trajectory:
                sample_started = time.perf_counter()
                sample, task = pair
                trajectory_file = trajectory_path(run_dir, task.id)
                environment_dir = prepare_environment_dir(run_dir, task.id)
                trace = Trajectory(
                    task.id,
                    messages=[{"role": "system", "content": task.system}, {"role": "user", "content": task.prompt}],
                    meta=dict(task.meta),
                )
                phase = "setup"
                episode_failure = False
                collection_error = None
                environment_outputs = None
                try:
                    env = AgentAbstainEnv(sample)
                    async with env:
                        try:
                            task = await env.prepare_task(task)
                            phase = "tools"
                            async with open_toolbox(env_server(env)) as tools:
                                phase = "episode"
                                try:
                                    trace = await run_episode(task, ctx, tools, out=trajectory_file, env=env)
                                except Exception:
                                    episode_failure = True
                                    raise
                                phase = "tools_cleanup"
                        finally:
                            if trajectory_file.exists():
                                trace = load_trajectory(trajectory_file)
                            previous_phase = phase
                            phase = "collect"
                            try:
                                environment_outputs = await collect(sample, trace, env, environment_dir)
                            except BaseException as exc:
                                collection_error = f"{type(exc).__name__}: {exc}"
                                raise
                            finally:
                                save_environment_result(environment_dir, task.id, environment_outputs, collection_error)
                            phase = previous_phase
                        phase = "teardown"
                except asyncio.CancelledError:
                    trace.episode_status, trace.error = "cancelled", "CancelledError"
                    trace.runtime_latency_sec = round(time.perf_counter() - sample_started, 6)
                    save_trajectory(trace, trajectory_file)
                    if not (environment_dir / "environment_result.json").exists():
                        save_environment_result(environment_dir, task.id, None, "Cancelled before collection")
                    raise
                except Exception as exc:
                    if episode_failure:
                        raise
                    _logger.exception("Runtime %s failed during %s", task.id, phase)
                    trace.episode_status = "environment_error"
                    detail = f"{phase}: {type(exc).__name__}: {exc}"
                    trace.error = f"{trace.error}; {detail}" if trace.error else detail
                    if not (environment_dir / "environment_result.json").exists():
                        save_environment_result(environment_dir, task.id, None, detail)
                trace.runtime_latency_sec = round(time.perf_counter() - sample_started, 6)
                save_trajectory(trace, trajectory_file)
                return trace

            new_traces = await run_tasks(
                pending_pairs, run_sample, concurrency=args.sample_concurrency, desc=args.benchmark
            )
    finally:
        try:
            if ctx is not None:
                await ctx.client.close()
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples = load_tasks(args)
    traces = asyncio.run(run_rollout(args, samples))
    return int(any(trace.error is not None for trace in traces))


def _filter_tool_error_log(record: logging.LogRecord) -> bool:
    if not _TOOL_CALL_ACTIVE.get():
        return True
    return not _is_agent_tool_error(record.exc_info[1] if record.exc_info else sys.exception())


def _is_agent_tool_error(error: BaseException | None) -> bool:
    from fastmcp.exceptions import ToolError as MCPToolError
    from fastmcp.exceptions import ValidationError as MCPValidationError

    from .data.runtime.base import ToolError

    # Mounted servers log wrapped errors again without attaching exc_info.
    while isinstance(error, MCPToolError) and error.__cause__ is not None:
        error = error.__cause__
    return isinstance(error, (MCPToolError, MCPValidationError, ToolError))


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize(value.to_dict())
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
