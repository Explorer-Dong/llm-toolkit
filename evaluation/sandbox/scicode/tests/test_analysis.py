import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "eval" / "inspect_ai" / "analysis.py"
SPEC = importlib.util.spec_from_file_location("analysis", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def record(step_id: str, status: str = "pass", model: str = "sglang/test-model") -> dict:
    return {
        "source": {
            "step_id": step_id,
            "status": status,
            "model": model,
        }
    }


def configure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    logs_dir = tmp_path / "logs-lmf"
    monkeypatch.setattr(analysis, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(analysis, "REPORT_PATH", tmp_path / "tmp" / "run_report.md")
    return logs_dir


def write_log(logs_dir: Path, name: str, records: list[dict]) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / name
    log_path.write_text(json.dumps(records), encoding="utf-8")
    return log_path


def test_extract_results_uses_source_fields_and_statuses() -> None:
    model, results = analysis.extract_results(
        [record("11.1"), record("11.2", "fail"), record("11.3", "time out")]
    )

    assert model == "test-model"
    assert results == {"11.1": "✅", "11.2": "❌", "11.3": "time out"}


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([], "at least one record"),
        ([{}], "without source"),
        ([record("1.1"), record("1.1")], "duplicate subproblem ID"),
        ([record("1.1", "unknown")], "unexpected status"),
        ([record("1.1"), record("1.2", model="vllm/other")], "multiple models"),
    ],
)
def test_extract_results_rejects_invalid_records(records: list[dict], message: str) -> None:
    with pytest.raises(analysis.AnalysisError, match=message):
        analysis.extract_results(records)


def test_update_report_adds_complete_subproblem_set_for_one_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs_dir = configure_paths(monkeypatch, tmp_path)
    log_path = write_log(logs_dir, "test.json", [record("11.1"), record("11.2")])

    model, subproblem_count = analysis.update_report(log_path)

    assert (model, subproblem_count) == ("test-model", len(analysis.SUBPROBLEM_IDS))
    report = analysis.REPORT_PATH.read_text(encoding="utf-8")
    assert report.startswith("| Sub ID | test-model |\n| :---: | :---: |\n| 2.1 | ❌ |\n")
    assert "| 11.1 | ✅ |\n" in report
    assert "| 11.2 | ✅ |\n" in report


def test_update_report_appends_duplicate_model_with_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs_dir = configure_paths(monkeypatch, tmp_path)
    first_log = write_log(logs_dir, "first.json", [record("11.1")])
    second_log = write_log(logs_dir, "second.json", [record("11.2")])

    assert analysis.update_report(first_log)[0] == "test-model"
    assert analysis.update_report(second_log)[0] == "test-model (1)"

    report = analysis.REPORT_PATH.read_text(encoding="utf-8")
    assert report.startswith("| Sub ID | test-model | test-model (1) |\n")
    assert "| 11.1 | ✅ | ❌ |\n" in report
    assert "| 11.2 | ❌ | ✅ |\n" in report


def test_validate_log_path_rejects_paths_outside_logs_lmf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs_dir = configure_paths(monkeypatch, tmp_path)
    outside_path = tmp_path / "outside.json"
    outside_path.write_text("[]", encoding="utf-8")
    nested_path = logs_dir / "nested" / "log.json"
    nested_path.parent.mkdir(parents=True)
    nested_path.write_text("[]", encoding="utf-8")

    with pytest.raises(analysis.AnalysisError, match="directly under"):
        analysis.validate_log_path(outside_path)
    with pytest.raises(analysis.AnalysisError, match="directly under"):
        analysis.validate_log_path(nested_path)
