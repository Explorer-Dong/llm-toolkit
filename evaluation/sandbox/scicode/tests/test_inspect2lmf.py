import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "eval" / "inspect_ai" / "inspect2lmf.py"
SPEC = importlib.util.spec_from_file_location("inspect2lmf", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
inspect2lmf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inspect2lmf
SPEC.loader.exec_module(inspect2lmf)


def model_event() -> dict:
    return {
        "event": "model",
        "input_refs": [[0, 1]],
        "output": {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "reasoning", "reasoning": "attachment://reasoning"},
                            {"type": "text", "text": "attachment://response"},
                        ]
                    }
                }
            ]
        },
    }


def inspect_log(problem_id: str = "1", steps: list[str] | None = None) -> dict:
    steps = steps or ["1.1"]
    return {
        "eval": {
            "model": "sglang/test-model",
            "task_args": {"with_background": True},
        },
        "samples": [
            {
                "id": problem_id,
                "metadata": {
                    "problem_id": problem_id,
                    "sub_steps": [{"step_number": step} for step in steps],
                },
                "attachments": {
                    "prompt": "Solve the function.",
                    "response": "```python\nreturn 1\n```",
                    "reasoning": "The implementation is direct.",
                },
                "events_data": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "attachment://prompt",
                        }
                    ]
                },
                "events": [model_event()],
            }
        ],
    }


def write_status(tmp_root: Path, step_id: str, status: str) -> None:
    status_path = (
        tmp_root
        / "sglang-test-model"
        / "evaluation_logs"
        / "with_background"
        / f"{step_id}.log"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(status, encoding="utf-8")


def test_convert_resolves_attachments_and_exports_reasoning(tmp_path: Path) -> None:
    write_status(tmp_path, "1.1", "pass")

    records, stats, _, _ = inspect2lmf.convert(inspect_log(), tmp_path, strict=True)

    assert stats.exported_records == 1
    assert records == [
        {
            "id": "scicode-1-1.1",
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Solve the function."},
                {
                    "role": "assistant",
                    "content": "```python\nreturn 1\n```",
                    "reasoning_content": "The implementation is direct.",
                },
            ],
            "source": {
                "benchmark": "scicode",
                "problem_id": "1",
                "step_id": "1.1",
                "status": "pass",
                "model": "sglang/test-model",
            },
        }
    ]


@pytest.mark.parametrize("status", ["fail", "time out", "unexpected"])
def test_convert_exports_only_exact_pass(tmp_path: Path, status: str) -> None:
    write_status(tmp_path, "1.1", status)

    records, stats, _, _ = inspect2lmf.convert(inspect_log(), tmp_path, strict=True)

    assert records == []
    assert stats.status_counts[status] == 1


def test_convert_omits_empty_reasoning(tmp_path: Path) -> None:
    log = inspect_log()
    log["samples"][0]["events"][0]["output"]["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "attachment://response"}
    ]
    write_status(tmp_path, "1.1", "pass")

    records, _, _, _ = inspect2lmf.convert(log, tmp_path, strict=True)

    assert records[0]["messages"][2] == {
        "role": "assistant",
        "content": "```python\nreturn 1\n```",
    }


def test_extract_assistant_uses_raw_response_fallback() -> None:
    sample = {
        "attachments": {
            "response": "answer",
            "reasoning": "reasoning",
        }
    }
    event = {
        "output": {"choices": [{"message": {"content": []}}]},
        "call": {
            "response": {
                "choices": [
                    {
                        "message": {
                            "content": "attachment://response",
                            "reasoning_content": "attachment://reasoning",
                        }
                    }
                ]
            }
        },
    }

    assert inspect2lmf.extract_assistant(sample, event) == ("answer", "reasoning")


def test_skipped_steps_do_not_consume_model_event_positions(tmp_path: Path) -> None:
    steps = [f"13.{number}" for number in range(1, 8)]
    log = inspect_log(problem_id="13", steps=steps)
    sample = log["samples"][0]
    sample["events"] = [model_event() for _ in range(6)]
    sample["events_data"]["messages"] = [
        {"role": "user", "content": "attachment://prompt"} for _ in range(6)
    ]
    for step in ("13.1", "13.2", "13.3", "13.4", "13.5", "13.7"):
        write_status(tmp_path, step, "pass")

    records, stats, _, _ = inspect2lmf.convert(log, tmp_path, strict=True)

    assert [record["source"]["step_id"] for record in records] == [
        "13.1",
        "13.2",
        "13.3",
        "13.4",
        "13.5",
        "13.7",
    ]
    assert stats.fixed_skipped_steps == 1


def test_mismatch_is_skipped_or_raises_in_strict_mode(tmp_path: Path) -> None:
    log = inspect_log(steps=["1.1", "1.2"])
    write_status(tmp_path, "1.1", "pass")

    records, stats, _, _ = inspect2lmf.convert(log, tmp_path, strict=False)
    assert len(records) == 1
    assert stats.issue_counts["event_step_count_mismatch"] == 1

    with pytest.raises(inspect2lmf.ConversionError, match="model events"):
        inspect2lmf.convert(log, tmp_path, strict=True)


def test_main_writes_openai_data_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "trace.json"
    output_path = tmp_path / "data.json"
    report_path = tmp_path / "data_report.md"
    input_path.write_text(json.dumps(inspect_log()), encoding="utf-8")
    write_status(tmp_path / "tmp", "1.1", "pass")
    monkeypatch.setattr(
        "sys.argv",
        [
            "inspect2lmf.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--tmp-root",
            str(tmp_path / "tmp"),
            "--report",
            str(report_path),
        ],
    )

    assert inspect2lmf.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))[0]["messages"][2]["role"] == "assistant"
    assert "关键字段映射" in report_path.read_text(encoding="utf-8")
