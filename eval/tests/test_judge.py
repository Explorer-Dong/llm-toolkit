"""Offline LLM judge contracts; inference is already represented by a trajectory."""

import asyncio
import unittest
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from eval import model
from eval.agent import Task, Trajectory
from eval.benchmarks._common.metrics import ScoreResult
from eval.benchmarks.browsecomp import scoring as browsecomp
from eval.benchmarks.webwalkerqa import scoring as webwalkerqa

__all__ = []
PARAMS: dict[str, Any] = {"model": "test-model", "temperature": 0, "top_p": 0.95, "max_tokens": 8192, "seed": 42}
JUDGE_PARAMS: dict[str, Any] = {
    **PARAMS,
    "model": "test-judge",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 4096,
    "enable_thinking": False,
}
USAGE = {"prompt_tokens": 10, "completion_tokens": 4, "reasoning_tokens": 2}


class BrowseCompJudgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scorer = browsecomp

    def test_judge_cli_thinking_defaults_and_override(self):
        with patch.object(self.scorer, "load_dotenv"):
            defaults = self.scorer.parse_args(["saved-run"])
            disabled = self.scorer.parse_args(["saved-run", "--no-enable-thinking", "--reasoning-effort", "high"])
        self.assertTrue(defaults.enable_thinking)
        self.assertEqual(defaults.reasoning_effort, "medium")
        self.assertFalse(disabled.enable_thinking)
        self.assertEqual(disabled.reasoning_effort, "high")

    async def test_failed_stream_is_closed_and_error_is_recorded(self):
        client, stream = _make_client([], error=RuntimeError("judge stream failed"))
        result = await self.scorer.score(
            _task(),
            _agent_result("prediction"),
            None,
            {},
            client=client,
            model_params=JUDGE_PARAMS,
        )
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.sys_error, "RuntimeError: judge stream failed")
        self.assertEqual(len(result.messages), 2)
        stream.close.assert_awaited_once()

    async def test_uses_shared_model_call_and_keeps_full_judge_trace(self):
        content = "extracted_final_answer: Paris\nreasoning: " + "x" * 2400 + "\ncorrect: YES"
        client, stream = _make_client(
            [
                _chunk(content=content, reasoning_content="judge reasoning", finish_reason="stop"),
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=4,
                        completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
                    ),
                ),
            ]
        )
        with patch.object(self.scorer, "call_model", wraps=model.call_model) as call:
            result = await self.scorer.score(
                _task("Capital?", "Paris"),
                _agent_result("The capital is Paris."),
                None,
                {},
                client=client,
                model_params=JUDGE_PARAMS,
            )
        self.assertIsInstance(result, ScoreResult)
        self.assertEqual(
            (result.reference_answer, result.extracted_prediction, result.score, result.method),
            ("Paris", "Paris", 1.0, "judge"),
        )
        self.assertIsInstance(result.score, float)
        self.assertIsNone(result.sys_error)
        self.assertEqual(
            {key: value for key, value in result.messages[-1].items() if key not in {"latency_sec", "queue_sec"}},
            {
                "role": "assistant",
                "content": content,
                "reasoning_content": "judge reasoning",
                "usage": USAGE,
                "finish_reason": "stop",
            },
        )
        self.assertGreaterEqual(result.messages[-1]["latency_sec"], result.messages[-1]["queue_sec"])
        call.assert_awaited_once_with(client, result.messages, **JUDGE_PARAMS)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["messages"],
            [
                {
                    "role": "user",
                    "content": self.scorer.JUDGE_PROMPT.format(
                        question="Capital?", response="The capital is Paris.", correct_answer="Paris"
                    ),
                }
            ],
        )
        self.assertEqual((request["model"], request["temperature"], request["max_tokens"]), ("test-judge", 0, 4096))
        self.assertEqual(request["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(request["stream_options"], {"include_usage": True})
        stream.close.assert_awaited_once()

    async def test_no_verdict_keeps_the_response_and_records_error(self):
        client, stream = _make_client([_chunk(content="no verdict", finish_reason="length")])
        result = await self.scorer.score(
            _task(),
            _agent_result("prediction"),
            None,
            {},
            client=client,
            model_params=JUDGE_PARAMS,
        )
        self.assertEqual(result.score, 0.0)
        self.assertIn("ValueError: judge returned no verdict", result.sys_error or "")
        self.assertEqual(result.messages[-1]["content"], "no verdict")
        self.assertEqual(result.messages[-1]["finish_reason"], "length")
        stream.close.assert_awaited_once()

    async def test_no_answer_extracted_is_distinct_from_raw_prediction(self):
        client, _ = _make_client([_chunk(content="extracted_final_answer: None\nreasoning: No answer.\ncorrect: no")])
        result = await self.scorer.score(
            _task(),
            _agent_result("I do not know."),
            None,
            {},
            client=client,
            model_params=JUDGE_PARAMS,
        )
        self.assertEqual(result.score, 0.0)
        self.assertIsNone(result.extracted_prediction)
        self.assertIsNone(result.messages[-1]["finish_reason"])

    async def test_agent_failure_does_not_start_judging(self):
        client, _ = _make_client([])
        result = await self.scorer.score(
            _task(),
            _agent_result("partial", sys_error="task_timeout"),
            None,
            {},
            client=client,
            model_params=JUDGE_PARAMS,
        )
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.messages, [])
        client.chat.completions.create.assert_not_awaited()

    async def test_missing_final_content_does_not_start_judging(self):
        client, _ = _make_client([])
        result = await self.scorer.score(
            _task(),
            _agent_result(None),
            None,
            {},
            client=client,
            model_params=JUDGE_PARAMS,
        )
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.messages, [])
        client.chat.completions.create.assert_not_awaited()

    async def test_task_cancellation_propagates(self):
        with (
            patch.object(self.scorer, "call_model", new_callable=AsyncMock, side_effect=asyncio.CancelledError),
            self.assertRaises(asyncio.CancelledError),
        ):
            await self.scorer.score(
                _task(),
                _agent_result("prediction"),
                None,
                {},
                client=MagicMock(),
                model_params=JUDGE_PARAMS,
            )


class WebWalkerJudgeTests(BrowseCompJudgeTests):
    def setUp(self):
        self.scorer = webwalkerqa


def _task(question="question", gold="gold"):
    return Task("sample", "system", question, gold=gold)


def _agent_result(final_content, *, sys_error=None):
    return Trajectory(task_id="sample", final_answer=final_content, error=sys_error)


def _chunk(*, content=None, reasoning=None, reasoning_content=None, calls=(), finish_reason=None):
    delta = SimpleNamespace(content=content, reasoning=reasoning, reasoning_content=reasoning_content, tool_calls=calls)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


def _tool_call(index, name, arguments):
    return SimpleNamespace(index=index, id=None, function=SimpleNamespace(name=name, arguments=arguments))


def _make_client(chunks, *, error=None):
    stream = MagicMock()
    stream.__aiter__.side_effect = partial(_stream_chunks, chunks, error)
    stream.close = AsyncMock()
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=stream)
    return client, stream


async def _stream_chunks(chunks, error):
    for chunk in chunks:
        yield chunk
    if error is not None:
        raise error


if __name__ == "__main__":
    unittest.main()
