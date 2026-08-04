from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.benchmarks import BENCHMARKS, get_benchmark
from src.utils.call_llm import LLMClient
from src.utils.jsonl import append_jsonl, read_jsonl, write_json
from src.utils.logging import get_logger, setup_run_logging
from src.utils.schema import Prediction, RunConfig, Task, to_jsonable

DEFAULT_SPLIT = "test"


def main() -> int:
    args = parse_args()
    resume_config_data = load_resume_config(args) if args.resume else {}
    run_config = build_run_config(args, resume_config_data)
    benchmark = get_benchmark(run_config.benchmark)

    run_dir = make_run_dir(run_config, resume=run_config.resume)
    setup_run_logging(run_dir)
    logger = get_logger(__name__)

    run_config.output_dir = str(run_dir)
    if run_config.num_workers != 1:
        logger.warning("num_workers=%s was requested; this prototype runs sequentially.", run_config.num_workers)

    write_json(run_dir / "config.json", build_config_record(run_config))
    (run_dir / "predictions.jsonl").touch(exist_ok=True)
    (run_dir / "errors.jsonl").touch(exist_ok=True)

    logger.info("Loading benchmark %s", run_config.benchmark)
    tasks = benchmark.load_tasks(split=run_config.split, limit=run_config.limit)
    if run_config.limit is not None:
        tasks = tasks[: run_config.limit]
    enrich_tasks(tasks, run_config, run_dir)

    if run_config.dry_run:
        summary = {
            "benchmark": run_config.benchmark,
            "model": run_config.model,
            "num_total": len(tasks),
            "num_success": 0,
            "num_failed": 0,
            "dry_run": True,
            "debug_only": run_config.limit is not None,
            "message": "Dry run loaded tasks and wrote config without calling the model.",
            "first_task": to_jsonable(tasks[0]) if tasks else None,
            "config": generation_config_summary(run_config),
        }
        write_json(run_dir / "summary.json", summary)
        logger.info("Dry run complete: %s", run_dir)
        return 0

    client = LLMClient(
        model=run_config.model,
        base_url=run_config.base_url,
        api_key=run_config.api_key,
        request_timeout=run_config.request_timeout,
    )
    completed_ids = load_completed_ids(run_dir) if run_config.resume else set()
    predictions: list[Prediction] = []

    for task in tqdm(tasks, desc=f"Running {benchmark.name}"):
        if task.id in completed_ids:
            logger.info("Skipping completed task %s due to --resume", task.id)
            continue

        try:
            prediction = benchmark.run_one(task=task, client=client, config=run_config)
        except Exception as exc:
            logger.exception("Task %s failed", task.id)
            prediction = error_prediction(task, run_config, exc)

        prediction.meta.setdefault("run_config", generation_config_summary(run_config))
        prediction.meta.setdefault("debug_only", run_config.limit is not None)
        predictions.append(prediction)
        append_jsonl(run_dir / "predictions.jsonl", to_jsonable(prediction))
        if prediction.error is not None:
            append_jsonl(run_dir / "errors.jsonl", to_jsonable(prediction))

    if run_config.resume:
        predictions = load_all_predictions(run_dir)

    summary = benchmark.score(predictions, run_dir)
    summary.setdefault("benchmark", run_config.benchmark)
    summary.setdefault("model", run_config.model)
    summary.setdefault("debug_only", run_config.limit is not None)
    summary.setdefault("config", generation_config_summary(run_config))
    write_json(run_dir / "summary.json", summary)
    logger.info("Run complete: %s", run_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal local LLM evaluation runner")
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=positive_float, default=None, help="Request timeout in seconds; must be greater than zero.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def load_resume_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir is None:
        raise ValueError("--resume requires --output-dir so config.json can be loaded from that run directory")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"--resume requires existing config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")

    run_config = loaded.get("run_config", {})
    if not isinstance(run_config, dict):
        raise ValueError(f"Config file run_config must be a JSON object: {config_path}")
    return run_config


def build_run_config(args: argparse.Namespace, resume_config_data: dict[str, Any]) -> RunConfig:
    return RunConfig(
        benchmark=args.benchmark,
        model=str(cli_or_resume(args.model, resume_config_data, "model", "Qwen/Qwen3.6-27B")),
        base_url=str(cli_or_resume(args.base_url, resume_config_data, "base_url", "http://localhost:8000/v1")),
        api_key=str(cli_or_resume(args.api_key, resume_config_data, "api_key", "EMPTY")),
        request_timeout=float(cli_or_resume(args.request_timeout, resume_config_data, "request_timeout", 1800.0)),
        split=str(cli_or_resume(args.split, resume_config_data, "split", DEFAULT_SPLIT)),
        limit=cli_or_resume(args.limit, resume_config_data, "limit", None),
        output_dir=str(cli_or_resume(args.output_dir, resume_config_data, "output_dir", "outputs")),
        temperature=float(cli_or_resume(args.temperature, resume_config_data, "temperature", 1.0)),
        top_p=float(cli_or_resume(args.top_p, resume_config_data, "top_p", 0.95)),
        top_k=cli_or_resume(args.top_k, resume_config_data, "top_k", 20),
        max_tokens=int(cli_or_resume(args.max_tokens, resume_config_data, "max_tokens", 81920)),
        num_workers=int(cli_or_resume(args.num_workers, resume_config_data, "num_workers", 1)),
        seed=cli_or_resume(args.seed, resume_config_data, "seed", None),
        resume=args.resume,
        dry_run=args.dry_run,
    )


def cli_or_resume(cli_value: Any, resume_config_data: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return resume_config_data.get(key, default)


def make_run_dir(config: RunConfig, resume: bool) -> Path:
    output_base = Path(config.output_dir)
    if not output_base.is_absolute():
        output_base = PROJECT_ROOT / output_base

    if resume and (output_base / "predictions.jsonl").exists():
        output_base.mkdir(parents=True, exist_ok=True)
        return output_base

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{config.benchmark}_{safe_name(config.model)}"
    run_dir = output_base / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = output_base / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)


def build_config_record(run_config: RunConfig) -> dict[str, Any]:
    return {
        "run_config": to_jsonable(run_config),
        "generation": generation_config_summary(run_config),
    }


def generation_config_summary(config: RunConfig) -> dict[str, Any]:
    return {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_tokens": config.max_tokens,
        "request_timeout": config.request_timeout,
    }


def enrich_tasks(tasks: list[Task], config: RunConfig, run_dir: Path) -> None:
    for task in tasks:
        task.meta = copy.deepcopy(task.meta)
        task.meta["run_config"] = generation_config_summary(config)
        task.meta["debug_only"] = config.limit is not None
        task.meta["run_output_dir"] = str(run_dir)


def load_completed_ids(run_dir: Path) -> set[str]:
    return {str(record.get("id")) for record in read_jsonl(run_dir / "predictions.jsonl") if record.get("id")}


def load_all_predictions(run_dir: Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    for record in read_jsonl(run_dir / "predictions.jsonl"):
        predictions.append(
            Prediction(
                id=str(record.get("id")),
                benchmark=str(record.get("benchmark")),
                model=str(record.get("model")),
                raw_output=str(record.get("raw_output", "")),
                parsed_answer=record.get("parsed_answer"),
                gold_answer=record.get("gold_answer"),
                score=record.get("score"),
                latency_sec=record.get("latency_sec"),
                error=record.get("error"),
                meta=dict(record.get("meta", {}) or {}),
            )
        )
    return predictions


def error_prediction(task: Task, config: RunConfig, exc: Exception) -> Prediction:
    return Prediction(
        id=task.id,
        benchmark=config.benchmark,
        model=config.model,
        raw_output="",
        parsed_answer=None,
        gold_answer=task.answer,
        score=None,
        latency_sec=None,
        error=f"{type(exc).__name__}: {exc}",
        meta=task.meta,
    )


if __name__ == "__main__":
    raise SystemExit(main())
