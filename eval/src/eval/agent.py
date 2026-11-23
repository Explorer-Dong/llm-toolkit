"""Task records and bounded ReAct episode execution."""

import asyncio
import copy
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from .harness.env import Env
from .harness.tool import ToolBox, ToolResponse
from .model import call_model

__all__ = ["TAIL_VERIFICATION", "Task", "Trajectory", "ModelResponseError", "react"]
_logger = logging.getLogger(__name__)
_CONTEXT_LENGTH_ERROR_CODES = frozenset({"context_length_exceeded"})
_CONTEXT_LENGTH_ERROR_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context length",
    "context window",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "range of input length",
)

TAIL_VERIFICATION = (
    "[Verification Check] Before your next action:\n"
    "1. Check requirements: if the request relies on unstated personal preferences, has unresolved ambiguity, "
    "or contains conflicting constraints that tools cannot resolve, stop calling tools and abstain now.\n"
    "2. Check evidence: if tool outputs reveal contradictions, critical failures, emergent risks, "
    "or only a partial match to the user's constraints, never guess defaults or settle for an approximate option—"
    "keep searching if an exact match may still exist, or abstain and report the exact gap.\n"
    "3. If all requirements are clear and verified, proceed with the task."
)


@dataclass(frozen=True)
class Task:
    id: str
    system: str
    prompt: str
    max_turns: int = 1
    timeout: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    gold: Any = field(default=None, repr=False, compare=False)
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    final_prompt: str | None = (
        "No steps remain. Do not call any more tools. Reply now with only the final answer "
        "in the format required by the system prompt."
    )
    strict_response: bool = False

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
    episode_status: str = "max_turns"
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
    finish_reason: str | None = None


class ModelResponseError(ValueError):
    """The model response violates the task's response contract."""


async def react(
    task: Task,
    client: AsyncOpenAI,
    tools: ToolBox,
    model_params: Mapping[str, Any],
    *,
    env: Env | None = None,
) -> Trajectory:
    """Run one bounded ReAct episode; return a completed trace or propagate cancellation."""
    trace = Trajectory(
        task.id,
        messages=[{"role": "system", "content": task.system}, {"role": "user", "content": task.prompt}],
        meta=copy.deepcopy(task.meta),
    )
    episode_started = time.perf_counter()
    timeout = asyncio.timeout(task.timeout or None)
    try:
        async with timeout:
            trace.tools = tools.schemas
            commit_intercepted = False
            for index in range(task.max_turns + 1):
                final_turn = index == task.max_turns or (env is not None and env.terminal)
                if final_turn and task.final_prompt is not None:
                    trace.messages.append({"role": "user", "content": task.final_prompt})
                trace.turns += 1
                tail_prompt = getattr(env, "tail_verification", None)
                request_messages = (
                    _with_tail_verification(trace.messages, tail_prompt)
                    if not final_turn and tail_prompt and not trace.messages[-1].get("intercepted")
                    else trace.messages
                )
                turn = await call_model(client, request_messages, None if final_turn else trace.tools, **model_params)
                trace.messages.append(turn.message())
                trace.finish_reason = turn.finish_reason
                for key, value in (turn.usage or {}).items():
                    trace.usage[key] = trace.usage.get(key, 0) + value
                if turn.finish_reason is not None:
                    trace.finish_reasons[turn.finish_reason] = trace.finish_reasons.get(turn.finish_reason, 0) + 1
                if turn.error is not None:
                    raise turn.error
                if task.strict_response:
                    if turn.finish_reason in {"length", "max_tokens", "error", "aborted"}:
                        response_error_type = (
                            RuntimeError if turn.finish_reason in {"error", "aborted"} else ModelResponseError
                        )
                        raise response_error_type(f"Incomplete model response: {turn.finish_reason}")
                    if final_turn and turn.tool_calls:
                        raise ModelResponseError(
                            "Model attempted tools after the episode ended; these calls were not executed"
                        )
                    if not turn.tool_calls and not turn.content.strip():
                        raise ModelResponseError("Empty model response; no final text or executable tool calls")
                if env is not None:
                    env.observe_turn(turn, index + 1)
                if turn.tool_calls and not final_turn:
                    can_intercept = not commit_intercepted and index + 1 < task.max_turns and env is not None
                    turn_intercepted = False
                    for call in turn.tool_calls:
                        tool_started = time.perf_counter()
                        observation: dict[str, Any] = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "tool_name": call["function"]["name"],
                            "content": None,
                        }
                        try:
                            parsed_args = _parse_tool_args(call["function"]["arguments"])
                            if can_intercept and (
                                turn_intercepted
                                or (
                                    parsed_args is not None
                                    and env.is_commit_call(call["function"]["name"], parsed_args)
                                )
                            ):
                                turn_intercepted = True
                                observation.update(
                                    content=_pre_commit_prompt(call["function"]["name"], parsed_args or {}),
                                    is_error=False,
                                    intercepted=True,
                                )
                            else:
                                response = await _call_tool(
                                    tools, env.prepare_tool_call(call) if env is not None else call
                                )
                                observation.update(content=response.content, is_error=response.is_error)
                                if response.raw is not None:
                                    observation["tool_result"] = response.raw
                                if response.fatal is not None:
                                    raise RuntimeError(response.fatal)
                        except BaseException as exc:
                            observation.update(is_error=True, error=f"{type(exc).__name__}: {exc}")
                            raise
                        finally:
                            observation["latency_sec"] = round(time.perf_counter() - tool_started, 6)
                            trace.messages.append(observation)
                    if turn_intercepted:
                        commit_intercepted = True
                    continue
                if turn.content.strip():
                    trace.final_answer = turn.content if task.strict_response else turn.content.strip()
                    trace.episode_status = "final_answer"
                    break
                if trace.turns < task.max_turns:
                    trace.messages.append(
                        {"role": "user", "content": "Your reply was empty. Use a tool or provide the final answer."}
                    )
    except ModelResponseError as exc:
        trace.episode_status, trace.error = "invalid_response", f"{type(exc).__name__}: {exc}"
    except TimeoutError as exc:
        trace.episode_status = "timeout" if timeout.expired() else "error"
        trace.error = f"task_timeout ({task.timeout}s)" if timeout.expired() else f"TimeoutError: {exc}"
        _logger.error("Episode %s failed: %s", task.id, trace.error)
    except Exception as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
        if trace.turns > 1 and _is_context_length_error(exc):
            trace.episode_status = "context_length_exceeded"
        else:
            trace.episode_status = "error"
            _logger.exception("Episode %s failed", task.id)
    finally:
        trace.latency_sec = round(time.perf_counter() - episode_started, 3)
    return trace


def _is_context_length_error(exc: BaseException) -> bool:
    if not isinstance(exc, APIStatusError) or exc.status_code not in {400, 413, 422}:
        return False
    if exc.code in _CONTEXT_LENGTH_ERROR_CODES:
        return True
    if isinstance(exc.body, Mapping):
        nested = exc.body.get("error")
        if isinstance(nested, Mapping) and nested.get("code") in _CONTEXT_LENGTH_ERROR_CODES:
            return True
    text = str(exc).lower()
    return any(marker in text for marker in _CONTEXT_LENGTH_ERROR_MARKERS)


async def _call_tool(tools: ToolBox, call: dict[str, Any]) -> ToolResponse:
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return ToolResponse("Tool arguments must be a valid JSON object. Correct them and retry.", is_error=True)
    if not isinstance(args, dict):
        return ToolResponse("Tool arguments must be a JSON object. Correct them and retry.", is_error=True)
    return await tools.call(call["function"]["name"], args)


def _with_tail_verification(messages: list[dict[str, Any]], tail_prompt: str) -> list[dict[str, Any]]:
    last = messages[-1]
    content = last.get("content") or ""
    return [*messages[:-1], {**last, "content": f"{content}\n\n{tail_prompt}" if content else tail_prompt}]


def _parse_tool_args(raw_arguments: str | None) -> dict[str, Any] | None:
    try:
        args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def _pre_commit_prompt(name: str, arguments: dict[str, Any]) -> str:
    return (
        f"[PRE-COMMIT CHECK] Execution of `{name}` with arguments "
        f"`{json.dumps(arguments, ensure_ascii=False)}` is paused for a one-time verification.\n"
        "Before confirming this consequential action, audit your decision against the user's request "
        "and all observations so far:\n"
        "1. No unverified assumptions: Did you guess any default for a missing parameter, "
        "subjective preference, or ambiguous instruction?\n"
        "2. Strict constraint & evidence match: Does the target satisfy 100% of the user's constraints "
        "without conflicts, discrepancies, or emergent risks in the tool outputs?\n"
        "If every requirement is strictly verified, repeat this tool call in your next turn to execute it. "
        "If any gap, conflict, or uncertainty exists, do not repeat this call—"
        "either continue read-only exploration or stop calling tools and abstain."
    )
