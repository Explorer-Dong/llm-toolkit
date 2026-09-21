"""AIME26 benchmark, self-contained (ported from the former shared harness
src/benchmarks/aime26.py + src/main.py; behavior preserved).

Usage:
    uv run main.py --model Qwen/Qwen3.6-27B --base-url http://localhost:8000/v1 \
        --api-key EMPTY --limit 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATASET_REPO = "MathArena/aime_2026"

PROMPT_TEMPLATE = """You are solving an AIME-style math problem.

Rules:
1. Solve step by step.
2. The final answer must be an integer.
3. At the very end, output exactly one JSON object.
4. Do not include any text after the JSON.

Problem:
{problem}

Final output format:
{{"answer": 123}}
"""


# ---------------------------------------------------------------- dataset ---

def load_tasks(split: str, limit: int | None) -> list[dict]:
    try:
        dataset = _load_local_or_hf(split=split, local_files_only=limit is not None)
    except Exception as exc:
        if limit is None:
            raise
        print(f"WARNING: dataset unavailable, using fallback sample ({exc})")
        rows = [{"id": "fallback-aime26-0",
                 "problem": "Fallback smoke-test problem: compute 100 + 23.",
                 "answer": 123, "fallback_sample": True}]
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
            return _load_from_files(local_dir, split)
    return load_dataset(DATASET_REPO, split=split, download_config=_dc(local_files_only))


def _load_from_files(local_dir: Path, split: str) -> Any:
    parquets = sorted(str(p) for p in local_dir.rglob("*.parquet"))
    if parquets:
        return load_dataset("parquet", data_files={"train": parquets})["train"]
    jsons = sorted(str(p) for p in local_dir.rglob("*.jsonl")) + sorted(str(p) for p in local_dir.rglob("*.json"))
    if jsons:
        return load_dataset("json", data_files={"train": jsons})["train"]
    raise RuntimeError(f"no parquet/json files under {local_dir}")


def _dc(local_files_only: bool) -> Any:
    from datasets import DownloadConfig
    return DownloadConfig(local_files_only=local_files_only, max_retries=0)


def _row_to_task(row: dict[str, Any], index: int) -> dict:
    problem = _get_field(row, ["problem", "question", "Problem", "Question"])
    answer = _get_field(row, ["answer", "Answer", "final_answer", "Final Answer"])
    return {
        "id": str(row.get("id") or row.get("problem_id") or index),
        "problem": str(problem),
        "gold_answer": answer,
        "fallback_sample": bool(row.get("fallback_sample", False)),
    }


def _get_field(row: dict[str, Any], candidates: list[str]) -> Any:
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return row[candidate]
    raise KeyError(f"Could not find any of fields {candidates}. Available: {sorted(row.keys())}")


# ---------------------------------------------------------------- parsing ---

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


def parse_integer_answer(text: str) -> tuple[int | None, str | None]:
    parsed_json = extract_last_json_object(text)
    if parsed_json is not None and "answer" in parsed_json:
        value = parsed_json.get("answer")
        if isinstance(value, int):
            return value, None
        if isinstance(value, str):
            import re
            if re.fullmatch(r"[-+]?\d+", value.strip()):
                return int(value.strip()), None
        return None, f"JSON answer is not an integer: {value!r}"
    import re
    matches = re.findall(r"[-+]?\d+", text)
    if matches:
        return int(matches[-1]), None
    return None, "Could not parse integer answer from model output"


def parse_gold_integer(value: Any) -> tuple[int | None, str | None]:
    if isinstance(value, int):
        return value, None
    text = str(value).strip()
    if text.isdigit() or (text.startswith(("+", "-")) and text[1:].isdigit()):
        return int(text), None
    parsed, error = parse_integer_answer(text)
    if parsed is not None:
        return parsed, None
    return None, f"Gold answer is not an integer: {value!r}; {error}"


# ------------------------------------------------------------------- main ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIME26 benchmark runner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="first N problems (0 = all)")
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
    run_dir = base / f"{timestamp}_aime26_{_safe_name(args.model)}"
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
    config = {k: v for k, v in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))

    if args.dry_run:
        print(json.dumps({"dry_run": True, "num_tasks": len(tasks), "first_task": tasks[0]},
                         ensure_ascii=False, indent=2))
        return 0

    if args.num_workers != 1:
        print(f"WARNING: num_workers={args.num_workers} requested; running sequentially.")

    done_ids = {json.loads(l)["id"] for l in (run_dir / "predictions.jsonl").open() if l.strip()}
    todo = [t for t in tasks if t["id"] not in done_ids]
    print(f"aime26/{args.model}: {len(todo)} to run, {len(done_ids)} already done -> {run_dir}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.request_timeout)
    extra_body: dict[str, Any] = {}
    if args.top_k:
        extra_body["top_k"] = args.top_k
    if args.seed is not None:
        extra_body["seed"] = args.seed

    for task in tqdm(todo, desc="aime26"):
        prompt = PROMPT_TEMPLATE.format(problem=task["problem"])
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
            parsed_answer, parse_error = parse_integer_answer(raw_output)
            gold_answer, gold_error = parse_gold_integer(task["gold_answer"])
            error = parse_error or gold_error
            score = None
            if parsed_answer is not None and gold_answer is not None:
                score = 1.0 if parsed_answer == gold_answer else 0.0
        except Exception as exc:  # noqa: BLE001 - record, never abort
            raw_output, parsed_answer, gold_answer, score, latency = "", None, None, None, None
            error = f"{type(exc).__name__}: {exc}"

        record = {"id": task["id"], "benchmark": "aime26", "model": args.model,
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
    summary = {"benchmark": "aime26", "model": args.model, "num_total": num_total,
               "num_success": num_total - num_failed, "num_failed": num_failed,
               "accuracy": accuracy, "debug_only": bool(args.limit)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
