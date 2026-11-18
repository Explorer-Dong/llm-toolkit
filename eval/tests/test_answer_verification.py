"""Contracts for deterministic answer extraction and verification."""

import unittest

from eval.agent import Trajectory
from eval.benchmarks._common.metrics import ScoreResult
from eval.benchmarks.aime import scoring as aime
from eval.benchmarks.gaia import scoring as gaia
from eval.benchmarks.gpqa import scoring as gpqa
from eval.benchmarks.math import scoring as math

__all__ = []


class IntegerVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_agent_keeps_prediction_but_receives_zero_score(self):
        result = _agent_result('{"answer": 123}', sys_error="request failed")
        self.assertEqual(await aime.score({"gold": 123}, result, None, {}), ScoreResult(123, 123, 0.0, "integer"))

    async def test_json_integer_and_signed_string_answers(self):
        self.assertEqual(
            await aime.score({"gold": 123}, _agent_result('{"answer": 123}'), None, {}),
            ScoreResult(123, 123, 1.0, "integer"),
        )
        self.assertEqual(
            await aime.score({"gold": 123}, _agent_result('{"answer": " +00123 "}'), None, {}),
            ScoreResult(123, 123, 1.0, "integer"),
        )

    async def test_plain_text_fallback_uses_last_integer(self):
        self.assertEqual(
            await aime.score({"gold": 34}, _agent_result("First 12, then the answer is 34"), None, {}),
            ScoreResult(34, 34, 1.0, "integer"),
        )

    async def test_wrong_answer_retains_the_extracted_prediction(self):
        result = await aime.score({"gold": 123}, _agent_result('{"answer": 122}'), None, {})
        self.assertEqual((result.extracted_prediction, result.score), (122, 0.0))
        self.assertIsInstance(result.score, float)

    async def test_invalid_json_answer_types_and_missing_content(self):
        for content in ('{"answer": true}', '{"answer": 1.5}', '{"answer": "unknown"}', None, ""):
            with self.subTest(content=content):
                self.assertEqual(
                    await aime.score({"gold": 1}, _agent_result(content), None, {}),
                    ScoreResult(1, None, 0.0, "integer"),
                )


class ChoiceVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_choice_is_normalized_to_uppercase(self):
        self.assertEqual(
            await gpqa.score({"gold": "C"}, _agent_result('{"answer": " c "}'), None, {}),
            ScoreResult("C", "C", 1.0, "choice"),
        )
        self.assertEqual(
            await gpqa.score({"gold": "A"}, _agent_result('{"answer": "Option A"}'), None, {}),
            ScoreResult("A", "A", 1.0, "choice"),
        )

    async def test_answer_label_takes_priority_over_other_choices(self):
        self.assertEqual(
            await gpqa.score({"gold": "B"}, _agent_result("A is wrong. answer: B. Consider C."), None, {}),
            ScoreResult("B", "B", 1.0, "choice"),
        )

    async def test_plain_text_fallback_uses_last_choice(self):
        self.assertEqual(
            await gpqa.score({"gold": "D"}, _agent_result("Consider A, then choose D"), None, {}),
            ScoreResult("D", "D", 1.0, "choice"),
        )

    async def test_wrong_and_missing_choices(self):
        self.assertEqual(
            await gpqa.score({"gold": "A"}, _agent_result('{"answer": "B"}'), None, {}),
            ScoreResult("A", "B", 0.0, "choice"),
        )
        self.assertEqual(
            await gpqa.score({"gold": "A"}, _agent_result("unknown"), None, {}),
            ScoreResult("A", None, 0.0, "choice"),
        )
        self.assertEqual(
            await gpqa.score({"gold": "A"}, _agent_result(None), None, {}),
            ScoreResult("A", None, 0.0, "choice"),
        )


class NormalizedVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_numeric_formatting_and_leading_zeros(self):
        self.assertEqual(
            await gaia.score({"gold": "321"}, _agent_result('{"answer": "00321"}'), None, {}),
            ScoreResult("321", "00321", 1.0, "normalized"),
        )
        self.assertEqual(
            await gaia.score({"gold": "1000"}, _agent_result('{"answer": "$1,000"}'), None, {}),
            ScoreResult("1000", "$1,000", 1.0, "normalized"),
        )

    async def test_text_ignores_case_whitespace_and_punctuation(self):
        self.assertEqual(
            await gaia.score({"gold": "new york"}, _agent_result('{"answer": " New York! "}'), None, {}),
            ScoreResult("new york", "New York!", 1.0, "normalized"),
        )

    async def test_lists_preserve_order_and_allow_comma_or_semicolon(self):
        self.assertEqual(
            await gaia.score({"gold": "Paris,42"}, _agent_result('{"answer": "Paris; 0042"}'), None, {}),
            ScoreResult("Paris,42", "Paris; 0042", 1.0, "normalized"),
        )
        self.assertEqual(
            (await gaia.score({"gold": "Paris,42"}, _agent_result('{"answer": "42; Paris"}'), None, {})).score, 0.0
        )
        self.assertEqual(
            (await gaia.score({"gold": "Paris,42"}, _agent_result('{"answer": "Paris"}'), None, {})).score, 0.0
        )

    async def test_list_items_keep_punctuation_significant(self):
        self.assertEqual(
            (await gaia.score({"gold": "St. Louis;42"}, _agent_result('{"answer": "St Louis;42"}'), None, {})).score,
            0.0,
        )

    async def test_plain_text_does_not_trigger_an_extraction_request(self):
        self.assertEqual(
            await gaia.score({"gold": "Paris"}, _agent_result("The answer is Paris"), None, {}),
            ScoreResult("Paris", None, 0.0, "normalized"),
        )


class MathVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_symbolic_equivalence_and_bare_latex_reference(self):
        self.assertEqual(
            await math.score({"gold": r"\frac{1}{2}"}, _agent_result(r"\boxed{\frac{2}{4}}"), None, {}),
            ScoreResult(r"\frac{1}{2}", "1/2", 1.0, "math"),
        )
        self.assertEqual(
            await math.score({"gold": "2"}, _agent_result(r"\boxed{\sqrt{4}}"), None, {}),
            ScoreResult("2", "sqrt(4)", 1.0, "math"),
        )

    async def test_multiple_boxed_answers_preserve_math_verify_extraction(self):
        self.assertEqual(
            await math.score({"gold": "2"}, _agent_result(r"First \boxed{1}, then \boxed{2}"), None, {}),
            ScoreResult("2", "{1, 2}", 1.0, "math"),
        )

    async def test_exact_text_fallback_keeps_non_symbolic_answers(self):
        self.assertEqual(
            (await math.score({"gold": "No solution"}, _agent_result(r"\boxed{\text{no solution}}"), None, {})).score,
            1.0,
        )

    async def test_wrong_and_missing_math_answers(self):
        self.assertEqual(
            await math.score({"gold": "2"}, _agent_result(r"\boxed{3}"), None, {}),
            ScoreResult("2", "3", 0.0, "math"),
        )
        self.assertEqual(
            await math.score({"gold": "2"}, _agent_result(None), None, {}),
            ScoreResult("2", None, 0.0, "math"),
        )
        self.assertEqual(
            await math.score({"gold": ""}, _agent_result(r"\boxed{2}"), None, {}),
            ScoreResult("", "2", 0.0, "math"),
        )


def _agent_result(final_content, *, sys_error=None):
    return Trajectory(task_id="test", final_answer=final_content, error=sys_error)


if __name__ == "__main__":
    unittest.main()
