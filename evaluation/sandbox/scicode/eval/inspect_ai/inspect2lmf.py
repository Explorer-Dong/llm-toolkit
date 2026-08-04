"""Convert successful SciCode Inspect AI trajectories to LlamaFactory OpenAI data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

ATTACHMENT_PREFIX = "attachment://"
SKIPPED_STEP_INDICES = {
    ("13", 5),
    ("62", 0),
    ("76", 2),
}


class ConversionError(Exception):
    """Raised when an Inspect trajectory cannot be converted safely."""


@dataclass
class ConversionStats:
    samples: int = 0
    metadata_steps: int = 0
    executable_steps: int = 0
    fixed_skipped_steps: int = 0
    model_events: int = 0
    exported_records: int = 0
    reasoning_records: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)
    issue_counts: Counter[str] = field(default_factory=Counter)


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "-")


def background_directory(task_args: dict[str, Any]) -> str:
    return "with_background" if task_args.get("with_background", False) else "without_background"


def is_skipped_step(problem_id: str, index: int) -> bool:
    return (problem_id, index) in SKIPPED_STEP_INDICES


def resolve_attachment(value: str, attachments: dict[str, Any]) -> str:
    resolved = value
    seen_ids: set[str] = set()

    while resolved.startswith(ATTACHMENT_PREFIX):
        attachment_id = resolved.removeprefix(ATTACHMENT_PREFIX)
        if attachment_id in seen_ids:
            raise ConversionError(f"Recursive attachment reference: {attachment_id}")
        seen_ids.add(attachment_id)

        attachment = attachments.get(attachment_id)
        if not isinstance(attachment, str):
            raise ConversionError(f"Missing text attachment: {attachment_id}")
        resolved = attachment

    return resolved


def resolve_content(value: Any, attachments: dict[str, Any]) -> str:
    if isinstance(value, str):
        return resolve_attachment(value, attachments)

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(resolve_attachment(item, attachments))
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(resolve_attachment(item["text"], attachments))
            else:
                raise ConversionError("Unsupported message content block")
        return "".join(parts)

    raise ConversionError("Unsupported message content")


def model_events(sample: dict[str, Any]) -> list[dict[str, Any]]:
    events = sample.get("events")
    if not isinstance(events, list):
        raise ConversionError("Sample has no events list")
    return [event for event in events if event.get("event") == "model"]


def executable_steps(sample: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        raise ConversionError("Sample has no metadata")

    problem_id = str(metadata.get("problem_id", sample.get("id", "")))
    sub_steps = metadata.get("sub_steps")
    if not isinstance(sub_steps, list):
        raise ConversionError(f"Problem {problem_id} has no sub_steps")

    steps: list[dict[str, Any]] = []
    for index, sub_step in enumerate(sub_steps):
        if not isinstance(sub_step, dict):
            raise ConversionError(f"Problem {problem_id} has non-object sub_step at index {index}")
        if not is_skipped_step(problem_id, index):
            steps.append(cast(dict[str, Any], sub_step))
    return steps


def extract_prompt(sample: dict[str, Any], event: dict[str, Any]) -> str:
    events_data = sample.get("events_data")
    if not isinstance(events_data, dict):
        raise ConversionError("Sample has no events_data")

    messages = events_data.get("messages")
    if not isinstance(messages, list):
        raise ConversionError("Sample has no event messages")

    attachments = sample.get("attachments", {})
    if not isinstance(attachments, dict):
        raise ConversionError("Sample attachments are invalid")

    user_messages: list[str] = []
    refs = event.get("input_refs")
    if not isinstance(refs, list):
        raise ConversionError("Model event has no input_refs")

    for ref_group in refs:
        if not isinstance(ref_group, list) or len(ref_group) != 2 or not all(isinstance(index, int) for index in ref_group):
            raise ConversionError("Model event input_refs are invalid")
        start, end = ref_group
        if start < 0 or end < start or end > len(messages):
            raise ConversionError("Model event input reference is out of range")
        for index in range(start, end):
            message = messages[index]
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            user_messages.append(resolve_content(message.get("content"), attachments))

    if not user_messages:
        raise ConversionError("Model event has no referenced user prompt")
    return "\n\n".join(user_messages)


def extract_assistant(sample: dict[str, Any], event: dict[str, Any]) -> tuple[str, str | None]:
    attachments = sample.get("attachments", {})
    if not isinstance(attachments, dict):
        raise ConversionError("Sample attachments are invalid")

    output = event.get("output")
    if not isinstance(output, dict):
        raise ConversionError("Model event has no output")
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ConversionError("Model event has no output choice")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ConversionError("Model event output choice has no message")

    content_blocks = message.get("content")
    text_parts: list[str] = []
    reasoning_parts: list[str] = []

    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(resolve_attachment(block["text"], attachments))
            elif block.get("type") == "reasoning" and isinstance(block.get("reasoning"), str):
                reasoning_parts.append(resolve_attachment(block["reasoning"], attachments))

    if not text_parts:
        raw_message = event.get("call", {}).get("response", {}).get("choices", [{}])[0].get("message", {})
        if isinstance(raw_message, dict) and isinstance(raw_message.get("content"), str):
            text_parts.append(resolve_attachment(raw_message["content"], attachments))
        if isinstance(raw_message, dict) and isinstance(raw_message.get("reasoning_content"), str):
            reasoning_parts.append(resolve_attachment(raw_message["reasoning_content"], attachments))

    content = "".join(text_parts).strip()
    reasoning = "".join(reasoning_parts).strip() or None
    if not content:
        raise ConversionError("Model event has no visible assistant response")
    return content, reasoning


def read_step_status(
    tmp_root: Path,
    model_name: str,
    background_dir: str,
    step_id: str,
) -> str:
    status_path = tmp_root / model_name / "evaluation_logs" / background_dir / f"{step_id}.log"
    if not status_path.is_file():
        return "missing"

    status = status_path.read_text(encoding="utf-8").strip()
    return status if status in {"pass", "fail", "time out"} else "unexpected"


def build_record(
    problem_id: str,
    step_id: str,
    model: str,
    prompt: str,
    content: str,
    reasoning: str | None,
) -> dict[str, Any]:
    assistant_message: dict[str, str] = {
        "role": "assistant",
        "content": content,
    }
    if reasoning is not None:
        assistant_message["reasoning_content"] = reasoning

    return {
        # "id": f"scicode-{problem_id}-{step_id}",
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
            assistant_message,
        ],
        "source": {
            "benchmark": "scicode",
            "problem_id": problem_id,
            "step_id": step_id,
            "status": "pass",
            "model": model,
        },
    }


def conversion_issue(stats: ConversionStats, strict: bool, issue: str, message: str) -> None:
    stats.issue_counts[issue] += 1
    if strict:
        raise ConversionError(message)
    print(f"warning: {message}", file=sys.stderr)


def convert(
    inspect_log: dict[str, Any],
    tmp_root: Path,
    strict: bool,
) -> tuple[list[dict[str, Any]], ConversionStats, str, str]:
    eval_data = inspect_log.get("eval")
    if not isinstance(eval_data, dict):
        raise ConversionError("Inspect log has no eval object")

    model = eval_data.get("model")
    if not isinstance(model, str) or not model:
        raise ConversionError("Inspect log eval.model is missing")
    model_name = sanitize_model_name(model)

    task_args = eval_data.get("task_args")
    if not isinstance(task_args, dict):
        raise ConversionError("Inspect log eval.task_args is missing")
    background_dir = background_directory(task_args)

    samples = inspect_log.get("samples")
    if not isinstance(samples, list):
        raise ConversionError("Inspect log has no samples list")

    records: list[dict[str, Any]] = []
    stats = ConversionStats(samples=len(samples))

    for sample in samples:
        if not isinstance(sample, dict):
            conversion_issue(stats, strict, "invalid_sample", "Skipping non-object sample")
            continue

        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            conversion_issue(stats, strict, "missing_metadata", "Skipping sample without metadata")
            continue

        problem_id = str(metadata.get("problem_id", sample.get("id", "")))
        sub_steps = metadata.get("sub_steps")
        if not isinstance(sub_steps, list):
            conversion_issue(
                stats,
                strict,
                "missing_sub_steps",
                f"Skipping problem {problem_id} without sub_steps",
            )
            continue

        stats.metadata_steps += len(sub_steps)
        stats.fixed_skipped_steps += sum(is_skipped_step(problem_id, index) for index in range(len(sub_steps)))

        try:
            steps = executable_steps(sample)
            events = model_events(sample)
        except ConversionError as error:
            conversion_issue(stats, strict, "invalid_sample", f"Problem {problem_id}: {error}")
            continue

        stats.executable_steps += len(steps)
        stats.model_events += len(events)
        if len(steps) != len(events):
            conversion_issue(
                stats,
                strict,
                "event_step_count_mismatch",
                f"Problem {problem_id} has {len(events)} model events for {len(steps)} executable steps",
            )

        for event, step in zip(events, steps, strict=False):
            step_id = str(step.get("step_number", ""))
            if not step_id:
                conversion_issue(
                    stats,
                    strict,
                    "missing_step_number",
                    f"Problem {problem_id} has executable step without step_number",
                )
                continue

            status = read_step_status(tmp_root, model_name, background_dir, step_id)
            stats.status_counts[status] += 1
            if status != "pass":
                continue

            try:
                prompt = extract_prompt(sample, event)
                content, reasoning = extract_assistant(sample, event)
            except ConversionError as error:
                conversion_issue(
                    stats,
                    strict,
                    "invalid_model_event",
                    f"Problem {problem_id} step {step_id}: {error}",
                )
                continue

            records.append(build_record(problem_id, step_id, model, prompt, content, reasoning))
            stats.exported_records += 1
            if reasoning is not None:
                stats.reasoning_records += 1

    return records, stats, model_name, background_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert successful SciCode Inspect AI trajectories to LlamaFactory OpenAI data.")
    parser.add_argument("--input", type=Path, required=True, help="Inspect AI JSON log")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON dataset")
    parser.add_argument("--tmp-root", type=Path, default=Path("tmp"), help="Root containing model evaluation_logs")
    parser.add_argument("--strict", action="store_true", help="Fail on malformed trajectories, missing status files, or event/step mismatches")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        inspect_log = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(inspect_log, dict):
            raise ConversionError("Inspect log top level must be a JSON object")

        records, stats, model_name, background_dir = convert(
            inspect_log,
            args.tmp_root,
            args.strict,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    except (ConversionError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Exported {stats.exported_records} successful records from {stats.samples} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
