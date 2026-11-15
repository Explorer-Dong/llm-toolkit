"""Task records and a bounded ReAct loop over an already-open ToolBox."""

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any

from harness.model import Context, call_model
from harness.tool import ToolBox, ToolResponse

__all__ = ["Task", "Trajectory", "react"]


@dataclass(frozen=True)
class Task:
    id: str
    system: str
    prompt: str
    max_turns: int = 1
    timeout: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Task id must be nonempty")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self.timeout is not None and self.timeout < 0:
            raise ValueError("Task timeout must be nonnegative")


@dataclass
class Trajectory:
    task_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    stop_reason: str = "max_turns"
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    )
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    finish_reasons: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    latency_sec: float = 0.0
    tools: list[dict[str, Any]] = field(default_factory=list)
    runtime_latency_sec: float | None = None


async def react(task: Task, ctx: Context, tools: ToolBox, *, trajectory: Trajectory | None = None) -> Trajectory:
    """Propagate runtime exceptions; run_episode owns normalization and persistence.

    The optional record lets the outer wrapper retain a partially completed loop.
    """
    trace = trajectory if trajectory is not None else Trajectory(task.id)
    trace.meta = copy.deepcopy(task.meta)
    trace.messages = [{"role": "system", "content": task.system}, {"role": "user", "content": task.prompt}]
    trace.tools = tools.schemas
    for _ in range(task.max_turns):
        trace.turns += 1
        turn = await call_model(ctx, trace.messages, trace.tools)
        trace.messages.append(turn.message())
        for key, value in (turn.usage or {}).items():
            trace.usage[key] = trace.usage.get(key, 0) + value
        reason = turn.finish_reason or "unknown"
        trace.finish_reasons[reason] = trace.finish_reasons.get(reason, 0) + 1
        if turn.error is not None:
            raise turn.error
        if turn.tool_calls:
            for call in turn.tool_calls:
                started = time.perf_counter()
                observation: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "tool_name": call["function"]["name"],
                    "content": None,
                }
                try:
                    response = await _call_tool(tools, call)
                    observation.update(content=response.content, is_error=response.is_error)
                    if response.raw is not None:
                        observation["tool_result"] = response.raw
                    if response.fatal is not None:
                        raise RuntimeError(response.fatal)
                except BaseException as exc:
                    observation.update(is_error=True, error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    observation["latency_sec"] = round(time.perf_counter() - started, 6)
                    trace.messages.append(observation)
            continue
        if turn.content.strip():
            trace.final_answer = turn.content.strip()
            trace.stop_reason = "final_answer"
            break
        if trace.turns < task.max_turns:
            trace.messages.append(
                {"role": "user", "content": "Your reply was empty. Use a tool or provide the final answer."}
            )
    return trace


async def _call_tool(tools: ToolBox, call: dict[str, Any]) -> ToolResponse:
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return ToolResponse("Tool arguments must be a valid JSON object. Correct them and retry.", is_error=True)
    if not isinstance(args, dict):
        return ToolResponse("Tool arguments must be a JSON object. Correct them and retry.", is_error=True)
    return await tools.call(call["function"]["name"], args)
