"""Answer verification and LLM judging shared by benchmarks."""

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse
from math_verify import verify as math_verify

from eval.core.harness import AgentResult
from eval.core.model import Context, call_model
from eval.core.utils import clean_text, last_json_object, normalize_number, normalize_text

__all__ = ["JUDGE_PROMPT", "MATH_SCORING_METHOD", "ScoreResult", "verify", "judge"]

_CHOICE_LABELS = ("A", "B", "C", "D")
_CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_ANSWER_CHOICE_RE = re.compile(r"[\"']?answer[\"']?\s*[:=]\s*[\"']?([ABCD])[\"']?", re.IGNORECASE)
_MATH_EXTRACTION_CONFIG = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())
MATH_SCORING_METHOD = "Math-Verify 0.9.0 expression extraction and symbolic equivalence; normalized exact-text fallback when parsing fails"

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
"""


@dataclass
class ScoreResult:
    """A reference, extracted prediction, score, and any judge response or error."""

    reference_answer: str | int
    extracted_prediction: str | int | None
    score: float
    method: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    sys_error: str | None = None


def verify(
    result: AgentResult,
    gold: str | int,
    *,
    method: Literal["integer", "choice", "normalized", "math"],
) -> ScoreResult:
    """Verify an agent's final content using the selected answer format."""
    final_content = result.final_content
    if method == "integer":
        prediction = _parse_integer_answer(final_content or "")
        score = float(prediction is not None and prediction == gold)
    elif method == "choice":
        prediction = _parse_choice_answer(final_content or "")
        score = float(prediction is not None and prediction == gold)
    elif method == "normalized":
        prediction = _parse_text_answer(final_content or "")
        score = float(_verify_text(prediction, str(gold)))
    elif method == "math":
        prediction, score = _verify_math(final_content, str(gold))
    else:
        raise ValueError(f"Unknown verification method: {method}")
    return ScoreResult(gold, prediction, score if result.sys_error is None else 0.0, method)


async def judge(ctx: Context, result: AgentResult, gold: str | int, *, question: str) -> ScoreResult:
    """Judge final content through the shared model call and retain its full trace."""
    score_result = ScoreResult(gold, result.final_content, 0.0, "judge")
    if result.sys_error is not None or result.final_content is None:
        return score_result
    prompt = JUDGE_PROMPT.format(question=question, response=result.final_content, correct_answer=gold)
    score_result.messages.append({"role": "user", "content": prompt})
    try:
        content, _, reasoning, usage, finish_reason = await call_model(
            ctx.sem, ctx.client_llm, ctx.params, score_result.messages
        )
        score_result.messages.append(
            {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning,
                "usage": usage,
                "finish_reason": finish_reason or "unknown",
            }
        )
        extracted = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\s*(?:reasoning|correct):|\Z)", content, re.IGNORECASE | re.DOTALL
        )
        if extracted is not None:
            value = extracted.group(1).strip()
            score_result.extracted_prediction = None if value.lower() == "none" else value
        match = re.search(r"correct: (yes|no)", content, re.IGNORECASE)
        if match is None:
            raise ValueError(f"judge returned no verdict: {clean_text(content, 200)}")
        score_result.score = float(match.group(1).lower() == "yes")
    except Exception as exc:
        score_result.sys_error = f"{type(exc).__name__}: {exc}"
    return score_result


def _parse_integer_answer(text: str) -> int | None:
    parsed = last_json_object(text)
    if parsed is not None and "answer" in parsed:
        value = parsed.get("answer")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
            return int(value.strip())
        return None
    matches = re.findall(r"[-+]?\d+", text)
    return int(matches[-1]) if matches else None


def _normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in _CHOICE_LABELS:
        return text
    match = _CHOICE_RE.search(text)
    return match.group(1).upper() if match else None


def _parse_choice_answer(text: str) -> str | None:
    parsed = last_json_object(text)
    if parsed is not None and "answer" in parsed:
        choice = _normalize_choice(parsed.get("answer"))
        if choice is not None:
            return choice
    answer_matches = _ANSWER_CHOICE_RE.findall(text)
    if answer_matches:
        return answer_matches[-1].upper()
    choice_matches = _CHOICE_RE.findall(text)
    return choice_matches[-1].upper() if choice_matches else None


def _parse_text_answer(text: str) -> str | None:
    parsed = last_json_object(text)
    if parsed is None:
        return None
    value = parsed.get("answer")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _verify_text(prediction: str | None, gold: str) -> bool:
    """Compare numeric, list, and text answers after normalization."""

    prediction = prediction or ""
    gold_number = normalize_number(gold)
    if gold_number is not None:
        prediction_number = normalize_number(prediction)
        return prediction_number is not None and prediction_number == gold_number
    if "," in gold or ";" in gold:
        gold_parts = re.split(r"[,;]", gold)
        prediction_parts = re.split(r"[,;]", prediction)
        if len(gold_parts) != len(prediction_parts):
            return False
        for gold_part, prediction_part in zip(gold_parts, prediction_parts, strict=True):
            gold_number = normalize_number(gold_part)
            if gold_number is not None:
                prediction_number = normalize_number(prediction_part)
                if prediction_number is None or prediction_number != gold_number:
                    return False
            elif normalize_text(prediction_part, remove_punctuation=False) != normalize_text(
                gold_part, remove_punctuation=False
            ):
                return False
        return True
    return normalize_text(prediction) == normalize_text(gold)


def _verify_math(prediction: str | None, gold: str) -> tuple[str | None, float]:
    parsed_prediction = _parse_math_answer(prediction) if prediction else []
    extracted = str(parsed_prediction[-1]) if parsed_prediction else _extract_boxed_answer(prediction or "")
    if not prediction or not gold.strip():
        return extracted, 0.0
    parsed_gold = _parse_math_gold(gold)
    if parsed_gold and parsed_prediction and math_verify(parsed_gold, parsed_prediction):
        return extracted, 1.0
    return extracted, float(_normalize_math_answer(prediction) == _normalize_math_answer(gold))


def _normalize_math_text(text: str) -> str:
    """Normalize common TeX spacing and shorthand before parsing or exact-text comparison."""
    normalized = text.strip().replace(r",\!", "")
    normalized = re.sub(r",\s*$", "", normalized)
    normalized = re.sub(r"\\(sin|cos|tan)\s*\^\s*(\d+)\s+([A-Za-z])", r"\\\1^{\2}(\3)", normalized)
    return normalized


def _parse_math_answer(text: str) -> list[Any]:
    """Extract a mathematical answer into Math-Verify's comparable representation."""
    return parse(_normalize_math_text(text), extraction_config=_MATH_EXTRACTION_CONFIG, fallback_mode="no_fallback")


def _parse_math_gold(text: str) -> list[Any]:
    parsed = _parse_math_answer(text)
    if parsed:
        return parsed
    # MATH-500 stores some reference answers as bare LaTeX without math delimiters.
    return _parse_math_answer(f"${_normalize_math_text(text)}$")


def _extract_boxed_answer(text: str) -> str | None:
    r"""Extract the last balanced \boxed{...} value for exact-text fallback scoring."""
    values = []
    search_from = 0
    while True:
        start = text.find(r"\boxed", search_from)
        if start < 0:
            break
        open_brace = start + len(r"\boxed")
        while open_brace < len(text) and text[open_brace].isspace():
            open_brace += 1
        if open_brace >= len(text) or text[open_brace] != "{":
            search_from = open_brace
            if search_from <= start:
                break
            continue
        depth = 1
        index = open_brace + 1
        while index < len(text) and depth:
            backslashes = 0
            before = index - 1
            while before >= 0 and text[before] == "\\":
                backslashes += 1
                before -= 1
            if backslashes % 2 == 0:
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
            index += 1
        if depth:
            break
        values.append(text[open_brace + 1 : index - 1])
        search_from = index
    return values[-1] if values else None


def _normalize_math_answer(text: str) -> str:
    value = _extract_boxed_answer(text) or text
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\[,!;:]", "", value)
    value = re.sub(r"\(([a-zA-Z])\)", r"\1", value)
    return re.sub(r"\s+", "", value).strip("$.,;:").casefold()
