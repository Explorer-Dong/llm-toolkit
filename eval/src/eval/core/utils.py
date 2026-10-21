"""Neutral helpers shared by benchmarks."""

import json
import re
import string
from typing import Any

__all__ = [
    "append_assistant_message",
    "clean_text",
    "last_json_object",
    "normalize_number",
    "normalize_text",
    "record_model_call",
    "record_token_usage",
]


def append_assistant_message(
    messages: list[dict[str, Any]], content: str, reasoning: str, usage: dict[str, int] | None
) -> None:
    """Append a model response with optional log-only reasoning and usage fields."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if usage:
        message["usage"] = usage
    messages.append(message)


def record_token_usage(usage: dict[str, int], call_usage: dict[str, int] | None) -> None:
    """Add one model call's token counts to the run or task totals."""
    if call_usage:
        for key in usage:
            usage[key] += call_usage[key]


def record_model_call(
    usage: dict[str, int],
    finish_reasons: dict[str, int],
    call_usage: dict[str, int] | None,
    finish_reason: str | None,
) -> None:
    """Update aggregate usage and finish-reason counts for one model call."""
    record_token_usage(usage, call_usage)
    name = finish_reason or "unknown"
    finish_reasons[name] = finish_reasons.get(name, 0) + 1


def clean_text(text: str, limit: int) -> str:
    """Collapse whitespace and truncate text to a readable limit."""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + (" ..." if len(text) > limit else "")


def last_json_object(text: str) -> dict[str, Any] | None:
    "Return the last parseable JSON object embedded in text, if any."

    end_positions = [i for i, char in enumerate(text) if char == "}"]
    for end in reversed(end_positions):
        start = text.rfind("{", 0, end + 1)
        while start != -1:
            candidate = text[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                start = text.rfind("{", 0, start)
                continue
            if isinstance(parsed, dict):
                return parsed
            break
    return None


def normalize_text(value: str, remove_punctuation: bool = True) -> str:
    value = re.sub(r"\s", "", value).lower()
    if remove_punctuation:
        value = value.translate(str.maketrans("", "", string.punctuation))
    return value


def normalize_number(value: str) -> float | None:
    value = value.strip()
    for char in ("$", "%", ","):
        value = value.replace(char, "")
    try:
        return float(value)
    except ValueError:
        return None
