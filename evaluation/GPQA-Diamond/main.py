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
DATA_FILE = "diamond198.json"
OUTPUT_DIR = HERE / "outputs"
LLM_MAX_RETRIES = 3
CHOICE_LABELS = ("A", "B", "C", "D")

SYSTEM_PROMPT = """You are solving a multiple-choice benchmark problem.

Rules:
1. Think carefully.
2. Do not use external tools.
3. At the very end, output exactly one JSON object.
4. Do not include any text after the JSON.

Final output format:
{"answer": "A"}
"""


class ModelUtils:
    """Utils for model during interaction."""

    CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
    ANSWER_CHOICE_RE = re.compile(r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([ABCD])[\"']?", re.IGNORECASE)

    @staticmethod
    async def call_model(
        request_sem: asyncio.Semaphore,
        client: AsyncOpenAI,
        request: dict[str, Any],
    ) -> tuple[str, str, dict[str, int] | None, str | None]:
        "Stream the single chat completion; return (content, reasoning, usage, finish_reason)."

        async with request_sem:
            stream = await client.chat.completions.create(**request)
            parts: list[str] = []
            reasoning_parts: list[str] = []
            usage: dict[str, int] | None = None
            finish_reason: str | None = None
            try:
                async for chunk in stream:
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        details = getattr(chunk_usage, "completion_tokens_details", None)
                        usage = {
                            "prompt_tokens": chunk_usage.prompt_tokens or 0,
                            "completion_tokens": chunk_usage.completion_tokens or 0,
                            "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
                        }
                    for choice in chunk.choices or ():
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                        delta = choice.delta
                        content = getattr(delta, "content", None)
                        if content:
                            parts.append(content)
                        # some servers stream reasoning as `reasoning`, others as `reasoning_content`
                        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                        if reasoning:
                            reasoning_parts.append(reasoning)
            finally:
                await stream.close()
        return "".join(parts), "".join(reasoning_parts), usage, finish_reason

    @staticmethod
    def parse_answer(text: str) -> str | None:
        parsed = ModelUtils._last_json_object(text)
        if parsed is not None and "answer" in parsed:
            choice = ModelUtils._normalize_choice(parsed.get("answer"))
            if choice is not None:
                return choice
        answer_matches = ModelUtils.ANSWER_CHOICE_RE.findall(text)
        if answer_matches:
            return answer_matches[-1].upper()
        choice_matches = ModelUtils.CHOICE_RE.findall(text)
        return choice_matches[-1].upper() if choice_matches else None

    @staticmethod
    def _last_json_object(text: str) -> dict[str, Any] | None:
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
    def _normalize_choice(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().upper()
        if text in CHOICE_LABELS:
            return text
        match = ModelUtils.CHOICE_RE.search(text)
        return match.group(1).upper() if match else None


class EvalUtil:
    @staticmethod
    def parse_args() -> argparse.Namespace:
        load_dotenv(HERE / ".env")
        parser = argparse.ArgumentParser(description="Single-turn harness for GPQA-Diamond")
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
                    "question": str(row["question"]),
                    "choices": {label: str(row["choices"][label]) for label in CHOICE_LABELS},
                    "gold": str(row["answer"]),
                }
            )
            if limit is not None and len(tasks) >= limit:
                break
        if not tasks:
            raise ValueError("No GPQA-Diamond tasks matched the requested filters")
        return tasks

    @staticmethod
    async def run(args: argparse.Namespace, tasks: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
        client = AsyncOpenAI(
            api_key=args.api_key, base_url=args.base_url, timeout=args.timeout, max_retries=LLM_MAX_RETRIES
        )
        request_sem = asyncio.Semaphore(args.concurrency)
        records: list[dict[str, Any]] = []
        pending = [
            asyncio.create_task(EvalUtil._run_helper(task, client, args.model, request_sem, args)) for task in tasks
        ]
        try:
            with tqdm(total=len(pending), desc="GPQA Diamond", unit="task") as progress:
                for future in asyncio.as_completed(pending):
                    record = await future
                    records.append(record)
                    EvalUtil._write_predictions(output_path, records)
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
    def build_summary(records: list[dict[str, Any]], model: str, output_path: Path) -> dict[str, Any]:
        return {
            "benchmark": "GPQA-Diamond",
            "model": model,
            "num_tasks": len(records),
            "correct": sum(record["score"] for record in records),
            "score": sum(record["score"] for record in records) / len(records) if records else 0.0,
            "sys_errors": sum(record["sys_error"] is not None for record in records),
            "output": str(output_path),
        }

    @staticmethod
    def _write_predictions(output_path: Path, records: list[dict[str, Any]]) -> None:
        "Rewrite the whole JSON array atomically."

        tmp_path = output_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, output_path)

    @staticmethod
    async def _run_helper(
        task: dict[str, Any],
        client: AsyncOpenAI,
        model: str,
        request_sem: asyncio.Semaphore,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        choices = "\n".join(f"{label}. {task['choices'][label]}" for label in CHOICE_LABELS)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{task['question']}\n\n{choices}"},
        ]
        sys_error: str | None = None
        output: str | None = None
        reasoning = ""
        usage: dict[str, int] | None = None
        finish_reason: str | None = None
        started = time.perf_counter()

        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if args.top_k:
            request["extra_body"] = {"top_k": args.top_k}
        if args.seed is not None:
            request["seed"] = args.seed

        try:
            output, reasoning, usage, finish_reason = await ModelUtils.call_model(request_sem, client, request)
            output = output.strip()
            if output:
                message: dict[str, Any] = {"role": "assistant", "content": output}
                if reasoning:
                    message["reasoning_content"] = reasoning
                if usage:
                    message["usage"] = usage
                messages.append(message)
        except Exception as exc:
            sys_error = f"{type(exc).__name__}: {exc}"

        if usage is None:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        correct = EvalUtil._rubric(output, task["gold"]) if not sys_error else False
        return {
            "id": task["id"],
            "answer": task["gold"],
            "extracted_prediction": ModelUtils.parse_answer(output or ""),
            "score": int(correct),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
            "finish_reasons": finish_reason,
            "latency_sec": round(time.perf_counter() - started, 3),
            "sys_error": sys_error,
            "messages": messages,
        }

    @staticmethod
    def _rubric(prediction: str | None, gold: str) -> bool:
        "Rubric for GPQA-Diamond"

        if prediction is None:
            return False
        parsed = ModelUtils.parse_answer(prediction)
        return parsed is not None and parsed == gold


def main() -> int:
    # load data
    args = EvalUtil.parse_args()
    data_path = DATA_DIR / DATA_FILE
    tasks = EvalUtil.load_tasks(data_path, args.limit)

    # prepare running folder
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / f"{timestamp}_{re.sub(r'[^A-Za-z0-9_.-]+', '-', args.model)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # log config before evaluation
    config = {key: str(value) for key, value in vars(args).items() if key != "api_key"}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Running GPQA-Diamond tasks with {args.model}; output: {run_dir}")

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
