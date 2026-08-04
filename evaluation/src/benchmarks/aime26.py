from __future__ import annotations

from pathlib import Path
from typing import Any

from src.benchmarks.base import Benchmark
from src.utils.call_llm import LLMClient
from src.utils.local_data import load_local_or_hf_dataset
from src.utils.parsing import parse_integer_answer
from src.utils.schema import Prediction, RunConfig, Task


AIME_DATASET = "MathArena/aime_2026"


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


class AIME26Benchmark(Benchmark):
    name = "aime26"

    def load_tasks(self, split: str, limit: int | None) -> list[Task]:
        try:
            dataset = load_local_or_hf_dataset(
                repo_id=AIME_DATASET,
                local_name="aime26",
                split=split,
                local_files_only=limit is not None,
            )
        except Exception as exc:
            if limit is None:
                raise
            dataset = _fallback_rows(str(exc))

        tasks: list[Task] = []
        for index, row in enumerate(dataset):
            row_dict = dict(row)
            problem = _get_field(row_dict, ["problem", "question", "Problem", "Question"])
            answer = _get_field(row_dict, ["answer", "Answer", "final_answer", "Final Answer"])
            tasks.append(
                Task(
                    id=str(row_dict.get("id") or row_dict.get("problem_id") or index),
                    input={"problem": str(problem)},
                    answer=answer,
                    meta={
                        "dataset": AIME_DATASET,
                        "split": split,
                        "raw": row_dict,
                        "fallback_sample": bool(row_dict.get("fallback_sample", False)),
                    },
                )
            )
            if limit is not None and len(tasks) >= limit:
                break
        return tasks

    def run_one(
        self,
        task: Task,
        client: LLMClient,
        config: RunConfig,
    ) -> Prediction:
        prompt = PROMPT_TEMPLATE.format(problem=task.input["problem"])
        raw_output, latency_sec = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        parsed_answer, parse_error = parse_integer_answer(raw_output)
        gold_answer, gold_error = _parse_gold_integer(task.answer)

        error = parse_error or gold_error
        score = 0.0
        if parsed_answer is not None and gold_answer is not None:
            score = 1.0 if parsed_answer == gold_answer else 0.0

        return Prediction(
            id=task.id,
            benchmark=self.name,
            model=config.model,
            raw_output=raw_output,
            parsed_answer=parsed_answer,
            gold_answer=gold_answer,
            score=score,
            latency_sec=latency_sec,
            error=error,
            meta=task.meta,
        )

    def score(
        self,
        predictions: list[Prediction],
        output_dir: Path,
    ) -> dict[str, Any]:
        del output_dir
        num_total = len(predictions)
        num_failed = sum(1 for prediction in predictions if prediction.error is not None)
        accuracy = sum(prediction.score or 0.0 for prediction in predictions) / num_total if num_total else 0.0
        config = predictions[0].meta.get("run_config", {}) if predictions else {}
        return {
            "benchmark": self.name,
            "model": predictions[0].model if predictions else None,
            "num_total": num_total,
            "num_success": num_total - num_failed,
            "num_failed": num_failed,
            "accuracy": accuracy,
            "debug_only": any(prediction.meta.get("debug_only", False) for prediction in predictions),
            "config": config,
        }


def _get_field(row: dict[str, Any], candidates: list[str]) -> Any:
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return row[candidate]
    raise KeyError(
        f"Could not find any of fields {candidates}. Available fields: {sorted(row.keys())}"
    )


def _parse_gold_integer(value: Any) -> tuple[int | None, str | None]:
    if isinstance(value, int):
        return value, None
    text = str(value).strip()
    if text.isdigit() or (text.startswith(("+", "-")) and text[1:].isdigit()):
        return int(text), None
    parsed, error = parse_integer_answer(text)
    if parsed is not None:
        return parsed, None
    return None, f"Gold answer is not an integer: {value!r}; {error}"


def _fallback_rows(error: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "fallback-aime26-0",
            "problem": "Fallback smoke-test problem: compute 100 + 23.",
            "answer": 123,
            "fallback_sample": True,
            "dataset_error": error,
        }
    ]
