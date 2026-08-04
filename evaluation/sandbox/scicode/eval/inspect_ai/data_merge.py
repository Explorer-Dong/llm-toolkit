"""Merge SciCode evaluation logs, preferring Qwen's correct answers.

For each task that Qwen3.5-35B-A3B passed, discard records for the same task from
GLM-5.2-FP8 and Kimi-K3. Records are identified by their ``source.step_id``.
"""

import argparse
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs-lmf"
QWEN_LOG = LOGS_DIR / "2026-07-27T15-28-02-00-00_Qwen3.5-35B-A3B.json"
GLM_LOG = LOGS_DIR / "2026-07-27T15-30-23-00-00_GLM-5.2-FP8.json"
KIMI_LOG = LOGS_DIR / "2026-07-29T01-45-06-00-00_Kimi-K3.json"
DEFAULT_OUTPUT = LOGS_DIR / "merged_prefer_qwen.json"

Record = dict[str, Any]


def load_records(path: Path) -> list[Record]:
    """Load and validate one evaluation log."""
    with path.open(encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}[{index}] must be a JSON object")
        source = record.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("step_id"), str):
            raise ValueError(f"{path}[{index}] is missing source.step_id")

    return records


def passed_step_ids(records: list[Record]) -> set[str]:
    """Return task IDs that the Qwen log marks as passed."""
    return {
        record["source"]["step_id"]
        for record in records
        if record["source"].get("status") == "pass"
    }


def merge_records(
    qwen_records: list[Record], glm_records: list[Record], kimi_records: list[Record]
) -> tuple[list[Record], int, int]:
    """Merge logs while removing non-Qwen records Qwen already solved."""
    qwen_passed_ids = passed_step_ids(qwen_records)
    retained_glm = [
        record for record in glm_records if record["source"]["step_id"] not in qwen_passed_ids
    ]
    retained_kimi = [
        record for record in kimi_records if record["source"]["step_id"] not in qwen_passed_ids
    ]
    merged_records = [*qwen_records, *retained_glm, *retained_kimi]

    return merged_records, len(glm_records) - len(retained_glm), len(kimi_records) - len(retained_kimi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-log", type=Path, default=QWEN_LOG)
    parser.add_argument("--glm-log", type=Path, default=GLM_LOG)
    parser.add_argument("--kimi-log", type=Path, default=KIMI_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    qwen_records = load_records(args.qwen_log)
    glm_records = load_records(args.glm_log)
    kimi_records = load_records(args.kimi_log)
    merged_records, removed_glm, removed_kimi = merge_records(
        qwen_records, glm_records, kimi_records
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(merged_records, file, ensure_ascii=False, indent=2)
        file.write("\n")

    qwen_passed_ids = passed_step_ids(qwen_records)
    glm_passed_ids = passed_step_ids(glm_records)
    kimi_passed_ids = passed_step_ids(kimi_records)
    kimi_passed_qwen_missed_ids = kimi_passed_ids - qwen_passed_ids
    glm_passed_qwen_missed_ids = glm_passed_ids - qwen_passed_ids

    print(f"Qwen3.5-35B-A3B 做对的题数: {len(qwen_passed_ids)}")
    print(
        "Kimi-K3 做对而 Qwen3.5-35B-A3B 做错的题数: "
        f"{len(kimi_passed_qwen_missed_ids)}"
    )
    print(
        "GLM-5.2-FP8 做对而 Qwen3.5-35B-A3B 做错的题数: "
        f"{len(glm_passed_qwen_missed_ids)}"
    )
    print(f"Merged records: {len(merged_records)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
