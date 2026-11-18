"""Score records, phase cost accounting, and benchmark summary aggregation."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ...agent import Task, Trajectory

__all__ = [
    "ScoreResult",
    "ScoreRecord",
    "collect_costs",
    "make_score_record",
    "load_score_record",
    "build_summary",
]


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


def collect_costs(
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


def make_score_record(task: Task, trace: Trajectory, result: ScoreResult, scoring_sec: float) -> ScoreRecord:
    return ScoreRecord(
        task.id,
        result,
        agent_error=trace.error,
        tool_errors=sum(bool(m.get("is_error")) for m in trace.messages if m["role"] == "tool"),
        metadata=dict(task.meta),
        costs={
            "rollout": collect_costs(
                trace.messages,
                sample_sec=trace.runtime_latency_sec,
                episode_sec=trace.latency_sec,
                usage=trace.usage,
                model_calls=trace.turns,
            ),
            "scoring": collect_costs(result.messages, sample_sec=scoring_sec, episode_sec=scoring_sec),
        },
    )


def load_score_record(raw: dict[str, Any]) -> ScoreRecord:
    result_raw = raw["score_result"]
    score_result = ScoreResult(
        reference_answer=result_raw["reference_answer"],
        extracted_prediction=result_raw.get("extracted_prediction"),
        score=float(result_raw["score"]),
        method=result_raw["method"],
        messages=list(result_raw.get("messages") or ()),
        sys_error=result_raw.get("sys_error"),
    )
    return ScoreRecord(
        task_id=str(raw["task_id"]),
        score_result=score_result,
        agent_error=raw.get("agent_error"),
        tool_errors=int(raw.get("tool_errors", 0)),
        metadata=dict(raw.get("metadata") or {}),
        costs=dict(raw.get("costs") or {}),
    )


def build_summary(
    benchmark: str,
    records: list[ScoreRecord],
    model_name: str,
    *,
    group_by: tuple[str, ...] = (),
    rollout_wall_sec: float | None = None,
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
    phase_costs = {phase: [record.costs.get(phase, {}) for record in records] for phase in ("rollout", "scoring")}
    wall_times = {
        "rollout": rollout_wall_sec,
        "scoring": scoring_wall_sec,
        "total": (
            rollout_wall_sec + scoring_wall_sec
            if rollout_wall_sec is not None and scoring_wall_sec is not None
            else None
        ),
    }
    phase_costs["total"] = phase_costs["rollout"] + phase_costs["scoring"]
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
