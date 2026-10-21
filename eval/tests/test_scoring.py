"""Contracts for deterministic answer extraction and verification."""

import unittest

from eval.core import harness, scoring

__all__ = []


class IntegerVerificationTests(unittest.TestCase):
    def test_failed_agent_keeps_prediction_but_receives_zero_score(self):
        result = _agent_result('{"answer": 123}', sys_error="request failed")
        self.assertEqual(scoring.verify(result, 123, method="integer"), scoring.ScoreResult(123, 123, 0.0, "integer"))

    def test_json_integer_and_signed_string_answers(self):
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": 123}'), 123, method="integer"),
            scoring.ScoreResult(123, 123, 1.0, "integer"),
        )
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": " +00123 "}'), 123, method="integer"),
            scoring.ScoreResult(123, 123, 1.0, "integer"),
        )

    def test_plain_text_fallback_uses_last_integer(self):
        self.assertEqual(
            scoring.verify(_agent_result("First 12, then the answer is 34"), 34, method="integer"),
            scoring.ScoreResult(34, 34, 1.0, "integer"),
        )

    def test_wrong_answer_retains_the_extracted_prediction(self):
        result = scoring.verify(_agent_result('{"answer": 122}'), 123, method="integer")
        self.assertEqual((result.extracted_prediction, result.score), (122, 0.0))
        self.assertIsInstance(result.score, float)

    def test_invalid_json_answer_types_and_missing_content(self):
        for content in ('{"answer": true}', '{"answer": 1.5}', '{"answer": "unknown"}', None, ""):
            with self.subTest(content=content):
                self.assertEqual(
                    scoring.verify(_agent_result(content), 1, method="integer"),
                    scoring.ScoreResult(1, None, 0.0, "integer"),
                )


class ChoiceVerificationTests(unittest.TestCase):
    def test_json_choice_is_normalized_to_uppercase(self):
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": " c "}'), "C", method="choice"),
            scoring.ScoreResult("C", "C", 1.0, "choice"),
        )
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": "Option A"}'), "A", method="choice"),
            scoring.ScoreResult("A", "A", 1.0, "choice"),
        )

    def test_answer_label_takes_priority_over_other_choices(self):
        self.assertEqual(
            scoring.verify(_agent_result("A is wrong. answer: B. Consider C."), "B", method="choice"),
            scoring.ScoreResult("B", "B", 1.0, "choice"),
        )

    def test_plain_text_fallback_uses_last_choice(self):
        self.assertEqual(
            scoring.verify(_agent_result("Consider A, then choose D"), "D", method="choice"),
            scoring.ScoreResult("D", "D", 1.0, "choice"),
        )

    def test_wrong_and_missing_choices(self):
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": "B"}'), "A", method="choice"),
            scoring.ScoreResult("A", "B", 0.0, "choice"),
        )
        self.assertEqual(
            scoring.verify(_agent_result("unknown"), "A", method="choice"),
            scoring.ScoreResult("A", None, 0.0, "choice"),
        )
        self.assertEqual(
            scoring.verify(_agent_result(None), "A", method="choice"), scoring.ScoreResult("A", None, 0.0, "choice")
        )


class NormalizedVerificationTests(unittest.TestCase):
    def test_numeric_formatting_and_leading_zeros(self):
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": "00321"}'), "321", method="normalized"),
            scoring.ScoreResult("321", "00321", 1.0, "normalized"),
        )
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": "$1,000"}'), "1000", method="normalized"),
            scoring.ScoreResult("1000", "$1,000", 1.0, "normalized"),
        )

    def test_text_ignores_case_whitespace_and_punctuation(self):
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": " New York! "}'), "new york", method="normalized"),
            scoring.ScoreResult("new york", "New York!", 1.0, "normalized"),
        )

    def test_lists_preserve_order_and_allow_comma_or_semicolon(self):
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": "Paris; 0042"}'), "Paris,42", method="normalized"),
            scoring.ScoreResult("Paris,42", "Paris; 0042", 1.0, "normalized"),
        )
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": "42; Paris"}'), "Paris,42", method="normalized").score, 0.0
        )
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": "Paris"}'), "Paris,42", method="normalized").score, 0.0
        )

    def test_list_items_keep_punctuation_significant(self):
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": "St Louis;42"}'), "St. Louis;42", method="normalized").score, 0.0
        )

    def test_plain_text_does_not_trigger_an_extraction_request(self):
        self.assertEqual(
            scoring.verify(_agent_result("The answer is Paris"), "Paris", method="normalized"),
            scoring.ScoreResult("Paris", None, 0.0, "normalized"),
        )


class MathVerificationTests(unittest.TestCase):
    def test_symbolic_equivalence_and_bare_latex_reference(self):
        self.assertEqual(
            scoring.verify(_agent_result(r"\boxed{\frac{2}{4}}"), r"\frac{1}{2}", method="math"),
            scoring.ScoreResult(r"\frac{1}{2}", "1/2", 1.0, "math"),
        )
        self.assertEqual(
            scoring.verify(_agent_result(r"\boxed{\sqrt{4}}"), "2", method="math"),
            scoring.ScoreResult("2", "sqrt(4)", 1.0, "math"),
        )

    def test_multiple_boxed_answers_preserve_math_verify_extraction(self):
        self.assertEqual(
            scoring.verify(_agent_result(r"First \boxed{1}, then \boxed{2}"), "2", method="math"),
            scoring.ScoreResult("2", "{1, 2}", 1.0, "math"),
        )

    def test_exact_text_fallback_keeps_non_symbolic_answers(self):
        self.assertEqual(
            scoring.verify(_agent_result(r"\boxed{\text{no solution}}"), "No solution", method="math").score, 1.0
        )

    def test_wrong_and_missing_math_answers(self):
        self.assertEqual(
            scoring.verify(_agent_result(r"\boxed{3}"), "2", method="math"), scoring.ScoreResult("2", "3", 0.0, "math")
        )
        self.assertEqual(
            scoring.verify(_agent_result(None), "2", method="math"), scoring.ScoreResult("2", None, 0.0, "math")
        )
        self.assertEqual(
            scoring.verify(_agent_result(r"\boxed{2}"), "", method="math"), scoring.ScoreResult("", "2", 0.0, "math")
        )


def _agent_result(final_content, *, sys_error=None):
    return harness.AgentResult(
        [],
        final_content,
        dict.fromkeys(("prompt_tokens", "completion_tokens", "reasoning_tokens"), 0),
        {},
        1,
        sys_error,
    )


if __name__ == "__main__":
    unittest.main()
