"""Write a SciCode Markdown report from converted LlamaFactory logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPORT_HEADER = "Sub ID"
SKIPPED_SUBPROBLEMS = {"13.6", "62.1", "76.3"}
SUBPROBLEM_IDS = (
    "2.1", "5.1", "8.1", "9.1", "11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8",
    "11.9", "11.10", "11.11", "11.12", "12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7", "12.8",
    "12.9", "12.10", "12.11", "12.12", "12.13", "12.14", "13.1", "13.2", "13.3", "13.4", "13.5", "13.7",
    "13.8", "13.9", "13.10", "13.11", "13.12", "13.13", "13.14", "13.15", "14.1", "14.2", "15.1", "15.2",
    "16.1", "16.2", "17.1", "17.2", "18.1", "18.2", "20.1", "20.2", "21.1", "21.2", "21.3", "22.1",
    "22.2", "22.3", "23.1", "23.2", "23.3", "24.1", "24.2", "24.3", "25.1", "25.2", "25.3", "26.1",
    "26.2", "26.3", "27.1", "27.2", "27.3", "28.1", "28.2", "28.3", "30.1", "30.2", "30.3", "31.1",
    "31.2", "31.3", "32.1", "32.2", "32.3", "33.1", "33.2", "33.3", "34.1", "34.2", "34.3", "35.1",
    "35.2", "35.3", "36.1", "36.2", "36.3", "37.1", "37.2", "37.3", "39.1", "39.2", "39.3", "40.1",
    "40.2", "40.3", "41.1", "41.2", "41.3", "42.1", "42.2", "42.3", "43.1", "43.2", "43.3", "45.1",
    "45.2", "45.3", "45.4", "46.1", "46.2", "46.3", "46.4", "48.1", "48.2", "48.3", "48.4", "50.1",
    "50.2", "50.3", "50.4", "52.1", "52.2", "52.3", "52.4", "53.1", "53.2", "53.3", "53.4", "54.1",
    "54.2", "54.3", "54.4", "55.1", "55.2", "55.3", "55.4", "56.1", "56.2", "56.3", "56.4", "57.1",
    "57.2", "57.3", "57.4", "57.5", "58.1", "58.2", "58.3", "58.4", "58.5", "59.1", "59.2", "59.3",
    "59.4", "59.5", "60.1", "60.2", "60.3", "60.4", "60.5", "61.1", "61.2", "61.3", "61.4", "61.5",
    "62.2", "62.3", "62.4", "62.5", "62.6", "63.1", "63.2", "63.3", "63.4", "63.5", "63.6", "64.1",
    "64.2", "64.3", "64.4", "64.5", "64.6", "65.1", "65.2", "65.3", "65.4", "65.5", "65.6", "66.1",
    "66.2", "66.3", "66.4", "66.5", "66.6", "67.1", "67.2", "67.3", "67.4", "67.5", "67.6", "68.1",
    "68.2", "68.3", "68.4", "68.5", "68.6", "68.7", "68.8", "69.1", "69.2", "69.3", "69.4", "69.5",
    "69.6", "69.7", "69.8", "71.1", "71.2", "71.3", "71.4", "71.5", "71.6", "71.7", "71.8", "71.9",
    "72.1", "72.2", "72.3", "72.4", "72.5", "72.6", "72.7", "72.8", "72.9", "73.1", "73.2", "73.3",
    "73.4", "73.5", "73.6", "73.7", "73.8", "73.9", "74.1", "75.1", "75.2", "75.3", "76.1", "76.2",
    "76.4", "77.1", "77.2", "77.3", "77.4", "77.5", "77.6", "77.7", "77.8", "77.9", "77.10", "77.11",
    "77.12", "79.1", "79.2", "79.3", "79.4", "80.1", "80.2", "80.3", "80.4", "80.5", "80.6", "80.7",
)
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs-lmf"
REPORT_PATH = SCRIPT_DIR / "tmp" / "run_report.md"
STATUS_SYMBOLS = {
    "pass": "✅",
    "fail": "❌",
    "time out": "time out",
}


class AnalysisError(Exception):
    """Raised when converted LlamaFactory logs cannot be converted to a report."""


def display_model_name(model: str) -> str:
    """Return the model name without the provider prefix."""
    return model.rsplit("/", maxsplit=1)[-1]


def unique_model_name(model: str, headers: list[str]) -> str:
    """Return a report column name that does not duplicate an existing model."""
    if model not in headers:
        return model

    index = 1
    while f"{model} ({index})" in headers:
        index += 1
    return f"{model} ({index})"


def natural_sort_key(value: str) -> tuple[int, int, int, str]:
    """Sort SciCode subproblem IDs such as ``11.2`` before ``11.10``."""
    match = re.fullmatch(r"(\d+)\.(\d+)", value)
    if match:
        return (0, int(match.group(1)), int(match.group(2)), "")
    return (1, 0, 0, value)


def extract_results(log: list[Any]) -> tuple[str, dict[str, str]]:
    """Extract per-subproblem statuses from one converted LlamaFactory log."""
    if not log:
        raise AnalysisError("Converted LlamaFactory log must contain at least one record")

    model: str | None = None
    results: dict[str, str] = {}
    seen_subproblem_ids: set[str] = set()
    for record in log:
        if not isinstance(record, dict):
            raise AnalysisError("Converted LlamaFactory log contains a non-object record")

        source = record.get("source")
        if not isinstance(source, dict):
            raise AnalysisError("Converted LlamaFactory log contains a record without source")

        record_model = source.get("model")
        if not isinstance(record_model, str) or not record_model:
            raise AnalysisError("Converted LlamaFactory log contains a record without source.model")
        if model is None:
            model = record_model
        elif record_model != model:
            raise AnalysisError("Converted LlamaFactory log contains records from multiple models")

        subproblem_id = source.get("step_id")
        if not isinstance(subproblem_id, str) or not subproblem_id:
            raise AnalysisError("Converted LlamaFactory log contains a record without source.step_id")
        if subproblem_id in seen_subproblem_ids:
            raise AnalysisError(f"Converted LlamaFactory log contains duplicate subproblem ID '{subproblem_id}'")
        seen_subproblem_ids.add(subproblem_id)

        status = source.get("status")
        if not isinstance(status, str) or status not in STATUS_SYMBOLS:
            raise AnalysisError(f"Subproblem {subproblem_id} has unexpected status '{status}'")

        if subproblem_id not in SKIPPED_SUBPROBLEMS:
            results[subproblem_id] = STATUS_SYMBOLS[status]

    assert model is not None
    return display_model_name(model), results


def read_log(log_path: Path) -> tuple[str, dict[str, str]]:
    """Read and validate one converted LlamaFactory JSON log."""
    try:
        log = json.loads(log_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AnalysisError(f"Cannot read log {log_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise AnalysisError(f"Log file is not valid JSON: {log_path}: {error}") from error

    if not isinstance(log, list):
        raise AnalysisError(f"Converted LlamaFactory log {log_path} top level must be a JSON array")
    return extract_results(log)


def render_report(headers: list[str], rows: dict[str, list[str]]) -> str:
    """Render a stable Markdown table containing all reported subproblems."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(":---:" for _ in headers) + " |",
    ]
    for sample_id in sorted(rows, key=natural_sort_key):
        lines.append("| " + " | ".join([sample_id, *rows[sample_id]]) + " |")
    return "\n".join(lines) + "\n"


def parse_table_row(line: str) -> list[str]:
    """Parse a simple pipe-delimited Markdown table row."""
    if not line.startswith("|") or not line.endswith("|"):
        raise AnalysisError(f"Invalid Markdown table row: {line}")
    return [cell.strip() for cell in line[1:-1].split("|")]


def read_markdown_table(report_path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Read the existing report table, or create its initial structure."""
    if not report_path.exists() or not report_path.read_text(encoding="utf-8").strip():
        return [REPORT_HEADER], {}

    lines = [line.strip() for line in report_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise AnalysisError(f"Report {report_path} must contain a Markdown table header and separator")

    headers = parse_table_row(lines[0])
    separator = parse_table_row(lines[1])
    if not headers or headers[0] != REPORT_HEADER:
        raise AnalysisError(f"Report {report_path} must start with a '{REPORT_HEADER}' column")
    if len(separator) != len(headers) or any(not re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        raise AnalysisError(f"Report {report_path} has an invalid Markdown table separator")

    rows: dict[str, list[str]] = {}
    for line in lines[2:]:
        cells = parse_table_row(line)
        if len(cells) != len(headers):
            raise AnalysisError(f"Report {report_path} has a row with {len(cells)} cells; expected {len(headers)}")
        if not cells[0]:
            raise AnalysisError(f"Report {report_path} has a row without a subproblem ID")
        if cells[0] in rows:
            raise AnalysisError(f"Report {report_path} has duplicate subproblem ID '{cells[0]}'")
        rows[cells[0]] = cells[1:]
    return headers, rows


def validate_log_path(log_path: Path) -> Path:
    """Return a converted JSON log path directly under ``logs-lmf``."""
    resolved_log_path = log_path.resolve()
    if resolved_log_path.parent != LOGS_DIR.resolve() or resolved_log_path.suffix != ".json":
        raise AnalysisError(f"Log must be a JSON file directly under {LOGS_DIR}")
    return resolved_log_path


def update_report(log_path: Path) -> tuple[str, int]:
    """Append one converted LlamaFactory log to the complete SciCode report."""
    model, results = read_log(validate_log_path(log_path))
    headers, rows = read_markdown_table(REPORT_PATH)
    column_name = unique_model_name(model, headers)
    headers.append(column_name)
    column_index = len(headers) - 2
    for row in rows.values():
        row.append("❌")

    expected_columns = len(headers) - 1
    for subproblem_id in SUBPROBLEM_IDS:
        row = rows.setdefault(subproblem_id, ["❌"] * expected_columns)
        if len(row) != expected_columns:
            raise AnalysisError(f"Report row {subproblem_id} does not match the existing header")
        row[column_index] = results.get(subproblem_id, "❌")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(headers, rows), encoding="utf-8")
    return column_name, len(SUBPROBLEM_IDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a SciCode report from one converted LlamaFactory log.")
    parser.add_argument("--log", type=Path, required=True, help="JSON log directly under logs-lmf/")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model, subproblem_count = update_report(args.log)
    except (AnalysisError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {subproblem_count} subproblems for {model} to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
