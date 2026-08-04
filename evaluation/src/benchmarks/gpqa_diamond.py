from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from src.benchmarks.base import Benchmark
from src.utils.call_llm import LLMClient
from src.utils.local_data import load_local_or_hf_dataset
from src.utils.parsing import normalize_choice, parse_choice_answer
from src.utils.schema import Prediction, RunConfig, Task


GPQA_DATASET = "fingertap/GPQA-Diamond"


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


class GPQADiamondBenchmark(Benchmark):
    name = "gpqa_diamond"

    def load_tasks(self, split: str, limit: int | None) -> list[Task]:
        try:
            dataset = load_local_or_hf_dataset(
                repo_id=GPQA_DATASET,
                local_name="gpqa_diamond",
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
            task = _row_to_task(row_dict, index=index, split=split)
            tasks.append(task)
            if limit is not None and len(tasks) >= limit:
                break
        return tasks

    def run_one(
        self,
        task: Task,
        client: LLMClient,
        config: RunConfig,
    ) -> Prediction:
        prompt = PROMPT_TEMPLATE.format(
            question=task.input["question"],
            choice_a=task.input["choices"]["A"],
            choice_b=task.input["choices"]["B"],
            choice_c=task.input["choices"]["C"],
            choice_d=task.input["choices"]["D"],
        )
        raw_output, latency_sec = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        parsed_answer, parse_error = parse_choice_answer(raw_output)
        gold_answer = normalize_choice(task.answer)
        gold_error = None if gold_answer is not None else f"Gold answer is not A/B/C/D: {task.answer!r}"

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


def _row_to_task(row: dict[str, Any], index: int, split: str) -> Task:
    question = str(_get_field(row, ["question", "Question", "prompt", "Problem"]))
    task_id = str(row.get("id") or row.get("qid") or row.get("question_id") or index)

    choices = _get_direct_choices(row)
    if choices is not None:
        answer = _get_gold_label(row, choices)
    else:
        parsed_from_question = _parse_choices_from_question(question)
        if parsed_from_question is not None:
            question, choices = parsed_from_question
            answer = _get_gold_label(row, choices)
        else:
            choices, answer = _build_choices_from_correct_and_incorrect(row, index)

    return Task(
        id=task_id,
        input={"question": question, "choices": choices},
        answer=answer,
        meta={
            "dataset": GPQA_DATASET,
            "split": split,
            "raw": row,
            "fallback_sample": bool(row.get("fallback_sample", False)),
        },
    )


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
        return {label: str(list_choices[index]) for index, label in enumerate(["A", "B", "C", "D"])}
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
    options = [("correct", str(correct))] + [("incorrect", str(value)) for value in incorrects]
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
        raise KeyError(
            f"Could not find any of fields {candidates}. Available fields: {sorted(row.keys())}"
        )
    return value


def _find_field(row: dict[str, Any], candidates: list[str]) -> Any | None:
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return row[candidate]
    return None


def _fallback_rows(error: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "fallback-gpqa-diamond-0",
            "question": "Fallback smoke-test question: Which option is the letter A?",
            "choice_a": "A",
            "choice_b": "B",
            "choice_c": "C",
            "choice_d": "D",
            "answer": "A",
            "fallback_sample": True,
            "dataset_error": error,
        }
    ]
