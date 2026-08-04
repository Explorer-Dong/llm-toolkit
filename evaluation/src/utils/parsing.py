from __future__ import annotations

import json
import re
from typing import Any


CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
ANSWER_CHOICE_RE = re.compile(
    r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([ABCD])[\"']?",
    re.IGNORECASE,
)
INTEGER_RE = re.compile(r"[-+]?\d+")


def extract_last_json_object(text: str) -> dict[str, Any] | None:
    end_positions = [index for index, char in enumerate(text) if char == "}"]
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


def normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = CHOICE_RE.search(text)
    if match:
        return match.group(1).upper()
    return None


def parse_choice_answer(text: str) -> tuple[str | None, str | None]:
    parsed_json = extract_last_json_object(text)
    if parsed_json is not None and "answer" in parsed_json:
        choice = normalize_choice(parsed_json.get("answer"))
        if choice is not None:
            return choice, None

    answer_matches = ANSWER_CHOICE_RE.findall(text)
    if answer_matches:
        return answer_matches[-1].upper(), None

    choice_matches = CHOICE_RE.findall(text)
    if choice_matches:
        return choice_matches[-1].upper(), None

    return None, "Could not parse A/B/C/D answer from model output"


def parse_integer_answer(text: str) -> tuple[int | None, str | None]:
    parsed_json = extract_last_json_object(text)
    if parsed_json is not None and "answer" in parsed_json:
        value = parsed_json.get("answer")
        if isinstance(value, int):
            return value, None
        if isinstance(value, str):
            stripped = value.strip()
            if re.fullmatch(r"[-+]?\d+", stripped):
                return int(stripped), None
        return None, f"JSON answer is not an integer: {value!r}"

    integer_matches = INTEGER_RE.findall(text)
    if integer_matches:
        return int(integer_matches[-1]), None

    return None, "Could not parse integer answer from model output"
