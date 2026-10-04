import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"

SYSTEM_PROMPT = """You are solving a math problem.

Rules:
1. Solve step by step.
2. The final answer must be an integer.
3. At the very end, output exactly one JSON object.
4. Do not include any text after the JSON.
5. Do not answer directly with your memory.

Final output format:
{"answer": 123}
"""


class ModelUtils:
    """Utils for model during interaction."""

    @staticmethod
    def last_json_object(text: str) -> dict[str, Any] | None:
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

    @staticmethod
    def parse_answer(text: str) -> int | None:
        parsed = ModelUtils.last_json_object(text)
        if parsed is not None and "answer" in parsed:
            value = parsed.get("answer")
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
                return int(value.strip())
            return None
        matches = re.findall(r"[-+]?\d+", text)
        return int(matches[-1]) if matches else None

    @staticmethod
    async def call_model(
        request_sem: asyncio.Semaphore, client: AsyncOpenAI, request: dict[str, Any]
    ) -> tuple[str, str | None]:
        async with request_sem:
            stream = await client.chat.completions.create(**request)
            parts: list[str] = []
            finish_reason: str | None = None
            try:
                async for chunk in stream:
                    for choice in chunk.choices or ():
                        content = getattr(choice.delta, "content", None)
                        if content:
                            parts.append(content)
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
            finally:
                await stream.close()
            return "".join(parts), finish_reason


class EvalUtil:
    @staticmethod
    def parse_args() -> argparse.Namespace:
        load_dotenv(HERE / ".env")
        parser = argparse.ArgumentParser(description="Single-turn harness for AIME")
        parser.add_argument("--year", type=int, default=2026, help="AIME year to evaluate (loads data/aime<year>.json)")
        parser.add_argument("--model", default=os.getenv("MODEL_NAME"), required=not os.getenv("MODEL_NAME"))
        parser.add_argument("--base-url", default=os.getenv("BASE_URL"), required=not os.getenv("BASE_URL"))
        parser.add_argument("--api-key", default=os.getenv("API_KEY", "EMPTY"), help="API key for the model endpoint")
        parser.add_argument("--limit", type=int, default=None, help="run the first N tasks")
        parser.add_argument("--concurrency", type=int, default=4, help="maximum concurrent model requests")
        parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature for model requests")
        parser.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling cutoff")
        parser.add_argument("--top-k", type=int, default=20, help="0 = do not send top_k")
        parser.add_argument("--max-tokens", type=int, default=81920, help="completion token limit per model request")
        parser.add_argument("--seed", type=int, default=42, help="sampling seed for reproducible runs")
        parser.add_argument("--timeout", type=float, default=1800.0, help="per model request timeout in seconds")
        return parser.parse_args()

    @staticmethod
    def load_tasks(path: Path, limit: int | None) -> list[dict[str, Any]]:
        rows = json.loads(path.read_text(encoding="utf-8"))
        tasks = []
        for row in rows:
            tasks.append(
                {
                    "id": str(row["id"]),
                    "problem": str(row["problem"]),
                    "gold": int(row["answer"]),
                }
            )
            if limit is not None and len(tasks) >= limit:
                break
        if not tasks:
            raise ValueError("No AIME tasks matched the requested filters")
        return tasks

    @staticmethod
    def write_predictions(output_path: Path, records: list[dict[str, Any]]) -> None:
        "Rewrite the whole JSON array atomically."

        tmp_path = output_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, output_path)

    @staticmethod
    async def run(args: argparse.Namespace, tasks: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
        client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout, max_retries=3)
        request_sem = asyncio.Semaphore(args.concurrency)
        records: list[dict[str, Any]] = []
        pending = [
            asyncio.create_task(EvalUtil._run_helper(task, client, args.model, request_sem, args)) for task in tasks
        ]
        try:
            with tqdm(total=len(pending), desc=f"AIME{args.year}", unit="task") as progress:
                for future in asyncio.as_completed(pending):
                    record = await future
                    records.append(record)
                    EvalUtil.write_predictions(output_path, records)
                    progress.set_postfix(score=f"{sum(r['score'] for r in records)}/{len(records)}")
                    progress.update(1)
        finally:
            for future in pending:
                if not future.done():
                    future.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await client.close()
        return records

    @staticmethod
    async def _run_helper(
        task: dict[str, Any],
        client: AsyncOpenAI,
        model: str,
        request_sem: asyncio.Semaphore,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["problem"]},
        ]
        sys_error: str | None = None
        output: str | None = None
        finish_reason: str | None = None
        started = time.perf_counter()

        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "stream": True,
        }
        if args.top_k:
            request["extra_body"] = {"top_k": args.top_k}
        if args.seed is not None:
            request["seed"] = args.seed

        try:
            output, finish_reason = await ModelUtils.call_model(request_sem, client, request)
            output = output.strip()
        except Exception as exc:
            sys_error = f"{type(exc).__name__}: {exc}"

        correct = EvalUtil._rubric(output, task["gold"]) if not sys_error else False
        return {
            "id": task["id"],
            "problem": task["problem"],
            "gold": task["gold"],
            "prediction": ModelUtils.parse_answer(output or ""),
            "output": output,
            "finish_reason": finish_reason,
            "correct": correct,
            "score": int(correct),
            "latency_sec": round(time.perf_counter() - started, 3),
            "sys_error": sys_error,
        }

    @staticmethod
    def _rubric(prediction: str | None, gold: int) -> bool:
        "Rubric for AIME"

        if prediction is None:
            return False
        parsed = ModelUtils.parse_answer(prediction)
        return parsed is not None and parsed == gold

    @staticmethod
    def build_summary(records: list[dict[str, Any]], model: str, output_path: Path) -> dict[str, Any]:
        return {
            "benchmark": "AIME",
            "model": model,
            "num_tasks": len(records),
            "correct": sum(record["score"] for record in records),
            "score": sum(record["score"] for record in records) / len(records) if records else 0.0,
            "sys_errors": sum(record["sys_error"] is not None for record in records),
            "output": str(output_path),
        }


def main() -> int:
    # load data
    args = EvalUtil.parse_args()
    data_path = DATA_DIR / f"aime{args.year}.json"
    tasks = EvalUtil.load_tasks(data_path, args.limit)

    # prepare running folder
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / f"{timestamp}_{re.sub(r'[^A-Za-z0-9_.-]+', '-', args.model)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # log config before evaluation
    config = {key: str(value) for key, value in vars(args).items() if key != "api_key"}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Running AIME{args.year} tasks with {args.model}; output: {run_dir}")

    # run evaluation
    output_path = run_dir / "predictions.json"
    records = asyncio.run(EvalUtil.run(args, tasks, output_path))

    # log summary after evaluation
    summary = EvalUtil.build_summary(records, args.model, output_path)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
