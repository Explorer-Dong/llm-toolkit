"""GPQA-Diamond benchmark, self-contained (ported from the former shared
harness src/benchmarks/gpqa_diamond.py + src/main.py; behavior preserved).

Usage:
    uv run main.py --model Qwen/Qwen3.6-27B --base-url http://localhost:8000/v1 \
        --api-key EMPTY --limit 1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATASET_REPO = "fingertap/GPQA-Diamond"

PROMPT_TEMPLATE = """You are solving a multiple-choice benchmark problem.

Rules:
1. Think carefully.
2. Do not use external tools.
3. At the very end, output exactly one JSON object.
4. Do not include any text after the JSON.

Question:
{question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Final output format:
{{"answer": "A"}}
"""


# ---------------------------------------------------------------- dataset ---

def load_tasks(split: str, limit: int | None) -> list[dict]:
    try:
        dataset = _load_local_or_hf(split=split, local_files_only=limit is not None)
    except Exception as exc:
        if limit is None:
            raise
        print(f"WARNING: dataset unavailable, using fallback sample ({exc})")
        rows = [{"id": "fallback-gpqa-diamond-0",
                 "question": "Fallback smoke-test question: Which option is the letter A?",
                 "choice_a": "A", "choice_b": "B", "choice_c": "C", "choice_d": "D",
                 "answer": "A", "fallback_sample": True}]
        return [_row_to_task(row, 0) for row in rows]

    tasks = []
    for index, row in enumerate(dataset):
        tasks.append(_row_to_task(dict(row), index))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def _load_local_or_hf(split: str, local_files_only: bool) -> Any:
    local_dir = HERE / "data"
    if local_dir.exists() and any(p.name not in {".cache", ".git"} for p in local_dir.iterdir()):
        try:
            return load_dataset(str(local_dir), split=split)
        except Exception:
            return _load_from_files(local_dir)
    from datasets import DownloadConfig
    dc = DownloadConfig(local_files_only=local_files_only, max_retries=0)
    return load_dataset(DATASET_REPO, split=split, download_config=dc)


def _load_from_files(local_dir: Path) -> Any:
    parquets = sorted(str(p) for p in local_dir.rglob("*.parquet"))
    if parquets:
        return load_dataset("parquet", data_files={"train": parquets})["train"]
    jsons = sorted(str(p) for p in local_dir.rglob("*.jsonl")) + sorted(str(p) for p in local_dir.rglob("*.json"))
    if jsons:
        return load_dataset("json", data_files={"train": jsons})["train"]
    raise RuntimeError(f"no parquet/json files under {local_dir}")


def _row_to_task(row: dict[str, Any], index: int) -> dict:
    question = str(_get_field(row, ["question", "Question", "prompt", "Problem"]))
    task_id = str(row.get("id") or row.get("qid") or row.get("question_id") or index)

    choices = _get_direct_choices(row)
    if choices is not None:
        answer = _get_gold_label(row, choices)
    else:
        parsed = _parse_choices_from_question(question)
        if parsed is not None:
            question, choices = parsed
            answer = _get_gold_label(row, choices)
        else:
            choices, answer = _build_choices_from_correct_and_incorrect(row, index)

    return {"id": task_id, "question": question, "choices": choices,
            "gold_answer": answer, "fallback_sample": bool(row.get("fallback_sample", False))}


def _get_direct_choices(row: dict[str, Any]) -> dict[str, str] | None:
    choice_fields = {
        "A": ["choice_a", "Choice A", "A", "option_a", "Option A"],
        "B": ["choice_b", "Choice B", "B", "option_b", "Option B"],
        "C": ["choice_c", "Choice C", "C", "option_c", "Option C"],
        "D": ["choice_d", "Choice D", "D", "option_d", "Option D"],
    }
    choices: dict[str, str] = {}
    for label, candidates in choice_fields.items():
        value = _find_field(row, candidates)
        if value is None:
            choices = {}
            break
        choices[label] = str(value)
    if len(choices) == 4:
        return choices

    list_choices = row.get("choices") or row.get("options")
    if isinstance(list_choices, list) and len(list_choices) >= 4:
        return {label: str(list_choices[i]) for i, label in enumerate(["A", "B", "C", "D"])}
    return None


def _parse_choices_from_question(question: str) -> tuple[str, dict[str, str]] | None:
    pattern = re.compile(
        r"(?ims)^\s*A\.\s*(?P<A>.*?)\s*^\s*B\.\s*(?P<B>.*?)\s*^\s*C\.\s*(?P<C>.*?)\s*^\s*D\.\s*(?P<D>.*)\s*$"
    )
    match = pattern.search(question)
    if not match:
        return None
    stem = question[: match.start()].strip()
    choices = {label: match.group(label).strip() for label in ["A", "B", "C", "D"]}
    return stem, choices


def _build_choices_from_correct_and_incorrect(row: dict[str, Any], index: int) -> tuple[dict[str, str], str]:
    correct = _get_field(row, ["Correct Answer", "correct_answer", "answer", "gold_answer"])
    incorrects = [
        _get_field(row, ["Incorrect Answer 1", "incorrect_answer_1", "wrong_answer_1"]),
        _get_field(row, ["Incorrect Answer 2", "incorrect_answer_2", "wrong_answer_2"]),
        _get_field(row, ["Incorrect Answer 3", "incorrect_answer_3", "wrong_answer_3"]),
    ]
    options = [("correct", str(correct))] + [("incorrect", str(v)) for v in incorrects]
    random.Random(index).shuffle(options)

    labels = ["A", "B", "C", "D"]
    choices = {label: text for label, (_, text) in zip(labels, options, strict=True)}
    answer = next(label for label, (kind, _) in zip(labels, options, strict=True) if kind == "correct")
    return choices, answer


def _get_gold_label(row: dict[str, Any], choices: dict[str, str]) -> str:
    raw_gold = _find_field(row, ["answer", "correct_answer", "label", "gold", "target", "gold_answer", "Correct Answer"])
    label = normalize_choice(raw_gold)
    if label is not None:
        return label
    if raw_gold is not None:
        raw_text = str(raw_gold).strip()
        for label, choice in choices.items():
            if str(choice).strip() == raw_text:
                return label
    raise KeyError(
        "Could not determine GPQA gold answer. Expected answer label or answer text. "
        f"Available fields: {sorted(row.keys())}"
    )


def _get_field(row: dict[str, Any], candidates: list[str]) -> Any:
    value = _find_field(row, candidates)
    if value is None:
        raise KeyError(f"Could not find any of fields {candidates}. Available: {sorted(row.keys())}")
    return value


def _find_field(row: dict[str, Any], candidates: list[str]) -> Any | None:
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return row[candidate]
    return None


# ---------------------------------------------------------------- parsing ---

CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
ANSWER_CHOICE_RE = re.compile(r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([ABCD])[\"']?", re.IGNORECASE)


def extract_last_json_object(text: str) -> dict[str, Any] | None:
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


# ------------------------------------------------------------------- main ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPQA-Diamond benchmark runner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="first N questions (0 = all)")
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20, help="0 = do not send top_k")
    parser.add_argument("--max-tokens", type=int, default=81920)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace) -> Path:
    base = Path(args.output_dir)
    if args.resume and (base / "predictions.jsonl").exists():
        return base
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base / f"{timestamp}_gpqa_diamond_{_safe_name(args.model)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "-" for c in value)


def main() -> int:
    args = parse_args()
    tasks = load_tasks(split=args.split, limit=args.limit or None)

    run_dir = make_run_dir(args)
    (run_dir / "predictions.jsonl").touch(exist_ok=True)
    (run_dir / "errors.jsonl").touch(exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({k: v for k, v in vars(args).items()}, ensure_ascii=False, indent=2))

    if args.dry_run:
        print(json.dumps({"dry_run": True, "num_tasks": len(tasks), "first_task": tasks[0]},
                         ensure_ascii=False, indent=2))
        return 0

    if args.num_workers != 1:
        print(f"WARNING: num_workers={args.num_workers} requested; running sequentially.")

    done_ids = {json.loads(l)["id"] for l in (run_dir / "predictions.jsonl").open() if l.strip()}
    todo = [t for t in tasks if t["id"] not in done_ids]
    print(f"gpqa_diamond/{args.model}: {len(todo)} to run, {len(done_ids)} already done -> {run_dir}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.request_timeout)
    extra_body: dict[str, Any] = {}
    if args.top_k:
        extra_body["top_k"] = args.top_k
    if args.seed is not None:
        extra_body["seed"] = args.seed

    for task in tqdm(todo, desc="gpqa_diamond"):
        prompt = PROMPT_TEMPLATE.format(
            question=task["question"],
            choice_a=task["choices"]["A"], choice_b=task["choices"]["B"],
            choice_c=task["choices"]["C"], choice_d=task["choices"]["D"],
        )
        try:
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body=extra_body or None,
            )
            raw_output = response.choices[0].message.content or ""
            latency = round(time.perf_counter() - started, 3)
            parsed_answer, parse_error = parse_choice_answer(raw_output)
            gold_answer = normalize_choice(task["gold_answer"])
            gold_error = None if gold_answer is not None else f"Gold answer is not A/B/C/D: {task['gold_answer']!r}"
            error = parse_error or gold_error
            score = None
            if parsed_answer is not None and gold_answer is not None:
                score = 1.0 if parsed_answer == gold_answer else 0.0
        except Exception as exc:  # noqa: BLE001 - record, never abort
            raw_output, parsed_answer, gold_answer, score, latency = "", None, None, None, None
            error = f"{type(exc).__name__}: {exc}"

        record = {"id": task["id"], "benchmark": "gpqa_diamond", "model": args.model,
                  "raw_output": raw_output, "parsed_answer": parsed_answer,
                  "gold_answer": gold_answer, "score": score,
                  "latency_sec": latency, "error": error,
                  "fallback_sample": task["fallback_sample"]}
        with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if error is not None:
            with (run_dir / "errors.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    preds = [json.loads(l) for l in (run_dir / "predictions.jsonl").open() if l.strip()]
    num_total = len(preds)
    num_failed = sum(1 for p in preds if p.get("error") is not None)
    accuracy = sum(p.get("score") or 0.0 for p in preds) / num_total if num_total else 0.0
    summary = {"benchmark": "gpqa_diamond", "model": args.model, "num_total": num_total,
               "num_success": num_total - num_failed, "num_failed": num_failed,
               "accuracy": accuracy, "debug_only": bool(args.limit)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
