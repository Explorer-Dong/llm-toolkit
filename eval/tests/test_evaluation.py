"""Offline contracts for shared model options, response capture, and benchmark scoring."""

import argparse
import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from eval import aime, browsecomp, gaia, gpqa, math, webwalkerqa
from eval.core import harness, model, runner, scoring

__all__ = []

TEST_ENV = {
    "MODEL_NAME": "test-model",
    "BASE_URL": "https://model.example/v1",
    "API_KEY": "model-secret",
    "DS_TOOL_URL": "https://tools.example/v1/tools",
    "DS_TOOL_KEY": "tool-secret",
    "JUDGE_MODEL": "test-judge",
    "JUDGE_BASE_URL": "https://judge.example/v1",
    "JUDGE_API_KEY": "judge-secret",
}
PARAMS = model.ModelParams(model="test-model", temperature=0, top_p=0.95, max_tokens=8192, seed=42)
JUDGE_PARAMS = replace(PARAMS, model="test-judge", temperature=0.0, top_p=1.0, max_tokens=4096, enable_thinking=False)
USAGE = {"prompt_tokens": 10, "completion_tokens": 4, "reasoning_tokens": 2}


class ModelArgsTests(unittest.TestCase):
    def test_model_options_load_environment_and_optional_judge(self):
        parser = argparse.ArgumentParser()
        with patch.dict(os.environ, TEST_ENV, clear=True):
            model.add_model_args(parser, include_judge=True)
        args = parser.parse_args([])
        self.assertEqual(
            (args.model, args.base_url, args.api_key), ("test-model", TEST_ENV["BASE_URL"], "model-secret")
        )
        self.assertEqual(
            (args.judge_model, args.judge_base_url, args.judge_api_key),
            ("test-judge", TEST_ENV["JUDGE_BASE_URL"], "judge-secret"),
        )
        self.assertEqual(model.build_model_params(args), replace(PARAMS, enable_thinking=False))

    def test_model_options_accept_overrides_without_judge(self):
        parser = argparse.ArgumentParser()
        model.add_model_args(parser)
        args = parser.parse_args(
            ["--model", "other", "--top-k", "9", "--temperature", "0.7", "--enable-thinking", "--concurrency", "2"]
        )
        params = model.build_model_params(args)
        self.assertEqual((params.model, params.temperature, params.enable_thinking), ("other", 0.7, True))
        self.assertEqual(params.extra_body, {"top_k": 9})
        self.assertEqual(args.concurrency, 2)
        self.assertFalse(hasattr(args, "judge_model"))


class AgentArgsTests(unittest.TestCase):
    def test_agent_defaults_and_overrides(self):
        parser = argparse.ArgumentParser()
        harness.add_paradigm_args(parser)
        args = parser.parse_args([])
        self.assertEqual(
            (args.max_steps, args.search_results, args.observation_chars, args.task_timeout), (16, 8, 12000, 3600)
        )
        args = parser.parse_args(
            [
                "--max-steps",
                "3",
                "--search-results",
                "50",
                "--observation-chars",
                "100",
                "--task-timeout",
                "0",
            ]
        )
        self.assertEqual(
            (args.max_steps, args.search_results, args.observation_chars, args.task_timeout), (3, 50, 100, 0)
        )

    def test_search_result_limits_are_checked_by_argparse(self):
        parser = argparse.ArgumentParser()
        harness.add_paradigm_args(parser)
        for value in ("0", "51", "invalid"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    parser.parse_args(["--search-results", value])
                self.assertEqual(exc.exception.code, 2)

    def test_model_only_parser_omits_tool_and_agent_options(self):
        parser = argparse.ArgumentParser()
        model.add_model_args(parser)
        args = parser.parse_args([])
        self.assertFalse(hasattr(args, "max_steps"))
        self.assertFalse(hasattr(args, "search_provider"))


class BenchmarkCliTests(unittest.TestCase):
    def test_aime_model_options_reach_runner(self):
        self._assert_entrypoint(aime, top_k=20, max_tokens=81920, temperature=1.0)

    def test_gpqa_model_options_reach_runner(self):
        self._assert_entrypoint(gpqa, top_k=20, max_tokens=81920, temperature=1.0)

    def test_math_model_options_reach_runner(self):
        self._assert_entrypoint(math, top_k=20, max_tokens=81920, temperature=1.0)

    def test_gaia_model_options_reach_runner(self):
        self._assert_entrypoint(gaia, top_k=0, max_tokens=8192, temperature=0.0)

    def test_browsecomp_model_options_reach_runner(self):
        self._assert_entrypoint(browsecomp, top_k=0, max_tokens=32768, temperature=0.0)

    def test_webwalkerqa_model_options_reach_runner(self):
        self._assert_entrypoint(webwalkerqa, top_k=0, max_tokens=32768, temperature=0.0)

    def _assert_entrypoint(self, module, *, top_k, max_tokens, temperature):
        for flags, expected_top_k in (([], top_k), (["--top-k", "0"], 0), (["--top-k", "7"], 7)):
            with self.subTest(flags=flags):
                params = _run_main(module, flags)
                self.assertEqual(params.extra_body, {"top_k": expected_top_k} if expected_top_k else None)
                self.assertEqual((params.max_tokens, params.temperature), (max_tokens, temperature))
                self.assertFalse(params.enable_thinking)


class ModelCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_captures_content_reasoning_usage_and_fragmented_tools(self):
        chunks = [
            _chunk(content="answer", reasoning="first", calls=[_tool_call(1, "fetch", '{"url":')]),
            _chunk(reasoning_content="second", calls=[_tool_call(1, None, '"https://example.com"}')]),
            _chunk(calls=[_tool_call(0, "search", "invalid json")], finish_reason="tool_calls"),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=4, completion_tokens_details=SimpleNamespace(reasoning_tokens=2)
                ),
            ),
        ]
        client, stream = _make_client(chunks)
        messages = [
            {
                "role": "assistant",
                "content": "previous",
                "reasoning_content": "hidden",
                "usage": USAGE,
                "finish_reason": "stop",
            }
        ]
        params = replace(PARAMS, enable_thinking=False, extra_body={"top_k": 7})
        content, calls, reasoning, usage, finish_reason = await model.call_model(
            asyncio.Semaphore(1), client, params, messages
        )
        self.assertEqual((content, reasoning, usage, finish_reason), ("answer", "firstsecond", USAGE, "tool_calls"))
        self.assertEqual(
            calls,
            [
                {"id": "call_missing_0", "name": "search", "arguments": {"input": "invalid json"}},
                {"id": "call_missing_1", "name": "fetch", "arguments": {"url": "https://example.com"}},
            ],
        )
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"], [{"role": "assistant", "content": "previous"}])
        self.assertEqual(request["extra_body"], {"top_k": 7, "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertEqual(messages[0]["reasoning_content"], "hidden")
        self.assertEqual(messages[0]["finish_reason"], "stop")
        stream.close.assert_awaited_once()

    async def test_zero_top_k_omits_extra_body_and_keeps_tool_schema(self):
        client, stream = _make_client([])
        tools = [{"type": "function", "function": {"name": "search"}}]
        result = await model.call_model(asyncio.Semaphore(1), client, PARAMS, [], tools)
        self.assertEqual(result, ("", [], "", None, None))
        request = client.chat.completions.create.call_args.kwargs
        self.assertNotIn("extra_body", request)
        self.assertEqual(request["tools"], tools)
        stream.close.assert_awaited_once()

    async def test_failed_stream_is_closed_and_error_propagates(self):
        client, stream = _make_client([], error=RuntimeError("stream failed"))
        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            await model.call_model(asyncio.Semaphore(1), client, PARAMS, [])
        stream.close.assert_awaited_once()


class QATests(unittest.IsolatedAsyncioTestCase):
    async def test_calls_model_once_and_captures_answer_and_metadata(self):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with (
            patch.object(harness, "call_model", new_callable=AsyncMock) as call,
            patch.object(harness.time, "perf_counter", side_effect=[10.0, 11.23456]),
        ):
            call.return_value = ("  answer\n", [], "reason", USAGE, "stop")
            result = await harness.qa(ctx, "question", system_prompt="system")

        self.assertIsInstance(result, harness.AgentResult)
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.latency_sec, 1.235)
        self.assertEqual((result.final_content, result.finish_reasons, result.sys_error), ("answer", {"stop": 1}, None))
        self.assertEqual(result.usage, USAGE)
        self.assertEqual(
            result.messages,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "  answer\n",
                    "reasoning_content": "reason",
                    "usage": USAGE,
                    "finish_reason": "stop",
                },
            ],
        )
        call.assert_awaited_once_with(ctx.sem, ctx.client_llm, ctx.params, result.messages)

    async def test_empty_answer_keeps_usage_and_assistant_finish_reason(self):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with patch.object(harness, "call_model", new_callable=AsyncMock) as call:
            call.return_value = (" \n", [], "reason", USAGE, "length")
            result = await harness.qa(ctx, "question", system_prompt="system")
        self.assertEqual(result.final_content, "")
        self.assertEqual(result.usage, USAGE)
        self.assertEqual(result.finish_reasons, {"length": 1})
        self.assertEqual(result.steps, 1)
        self.assertIsNone(result.sys_error)
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.messages[-1]["finish_reason"], "length")
        call.assert_awaited_once()

    async def test_missing_usage_and_reasoning_remain_optional(self):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with patch.object(harness, "call_model", new_callable=AsyncMock, return_value=("answer", [], "", None, None)):
            result = await harness.qa(ctx, "question", system_prompt="system")
        self.assertEqual(result.usage, dict.fromkeys(USAGE, 0))
        self.assertEqual(result.finish_reasons, {"unknown": 1})
        self.assertEqual(
            result.messages[-1],
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "",
                "usage": None,
                "finish_reason": "unknown",
            },
        )

    async def test_request_error_is_recorded_without_a_fallback_request(self):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with patch.object(
            harness, "call_model", new_callable=AsyncMock, side_effect=RuntimeError("request failed")
        ) as call:
            result = await harness.qa(ctx, "question", system_prompt="system")
        self.assertIsNone(result.final_content)
        self.assertEqual(result.finish_reasons, {})
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.sys_error, "RuntimeError: request failed")
        self.assertEqual(result.usage, dict.fromkeys(USAGE, 0))
        self.assertEqual(len(result.messages), 2)
        call.assert_awaited_once()

    async def test_task_cancellation_propagates(self):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with (
            patch.object(harness, "call_model", new_callable=AsyncMock, side_effect=asyncio.CancelledError),
            self.assertRaises(asyncio.CancelledError),
        ):
            await harness.qa(ctx, "question", system_prompt="system")


class SystemRecordTests(unittest.TestCase):
    def test_serializes_agent_result_and_preserves_fractional_score(self):
        messages = [{"role": "assistant", "content": "raw answer", "finish_reason": "stop"}]
        result = harness.AgentResult(messages, "raw answer", dict(USAGE), {"stop": 1}, 1, None, 1.235)
        score_result = scoring.ScoreResult("answer", "prediction", 0.375, "normalized")
        record = runner.make_record({"id": 7}, result, score_result, subject="Algebra")
        self.assertIsInstance(record, runner.SystemRecord)
        self.assertIs(record.agent_result, result)
        self.assertIs(record.score_result, score_result)
        self.assertEqual(record.id, "7")
        self.assertEqual(record.agent_result.latency_sec, 1.235)
        self.assertEqual(record.agent_result.usage, USAGE)
        self.assertEqual(record.agent_result.messages, messages)
        self.assertEqual(record.metadata, {"subject": "Algebra"})
        payload = json.loads(json.dumps(asdict(record)))
        self.assertEqual(set(payload), {"id", "agent_result", "score_result", "metadata"})
        self.assertEqual(payload["score_result"]["reference_answer"], "answer")
        self.assertEqual(payload["score_result"]["extracted_prediction"], "prediction")
        self.assertEqual(payload["score_result"]["score"], 0.375)
        self.assertIsInstance(payload["score_result"]["score"], float)

    def test_scoring_error_stays_with_score_without_mutating_agent(self):
        score_result = scoring.ScoreResult("answer", "answer", 1.0, "judge", sys_error="RuntimeError: judge failed")
        record = runner.make_record({"id": "1"}, _agent_result("answer"), score_result)
        self.assertIsNone(record.agent_result.sys_error)
        self.assertEqual(record.score_result.sys_error, "RuntimeError: judge failed")
        self.assertEqual(record.score_result.score, 0.0)
        self.assertEqual(score_result.score, 1.0)
        self.assertEqual(runner.build_summary("test", [record], "model")["sys_errors"], 1)

    def test_summary_uses_nested_scores_and_execution_errors(self):
        good = runner.make_record(
            {"id": "1"},
            _agent_result("answer"),
            scoring.ScoreResult("answer", "answer", 0.375, "normalized"),
        )
        failed = runner.make_record(
            {"id": "2"},
            _agent_result("answer", sys_error="request failed"),
            scoring.ScoreResult("answer", "answer", 1.0, "normalized"),
        )
        summary = runner.build_summary("test", [good, failed], "model")
        self.assertEqual(
            summary,
            {"benchmark": "test", "model": "model", "num_tasks": 2, "correct": 0.375, "score": 0.19, "sys_errors": 1},
        )

    def test_judge_usage_and_messages_stay_with_the_score(self):
        judge_messages = [
            {"role": "assistant", "content": "correct: yes", "usage": {"prompt_tokens": 999}, "finish_reason": "stop"}
        ]
        record = runner.make_record(
            {"id": "1"},
            _agent_result("answer"),
            scoring.ScoreResult("answer", "answer", 1.0, "judge", judge_messages),
        )
        self.assertEqual(record.agent_result.usage, USAGE)
        self.assertEqual(record.agent_result.messages, [])
        self.assertEqual(record.score_result.messages, judge_messages)


class GAIAAnswerTests(unittest.IsolatedAsyncioTestCase):
    def test_parses_string_answer_and_keeps_leading_zeros(self):
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": " 00321 "}'), "321", method="normalized").extracted_prediction,
            "00321",
        )
        self.assertEqual(
            scoring.verify(
                _agent_result('{"answer": "Paris; 42"}'), "Paris;42", method="normalized"
            ).extracted_prediction,
            "Paris; 42",
        )

    def test_parses_fenced_json_and_numeric_answers(self):
        self.assertEqual(
            scoring.verify(
                _agent_result('```json\n{"answer": "Paris"}\n```'), "Paris", method="normalized"
            ).extracted_prediction,
            "Paris",
        )
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": 42}'), "42", method="normalized").extracted_prediction, "42"
        )
        self.assertEqual(
            scoring.verify(_agent_result('{"answer": 1.25}'), "1.25", method="normalized").extracted_prediction, "1.25"
        )

    def test_rejects_missing_or_invalid_answer_fields(self):
        for content in (
            "Paris",
            "",
            '{"answer":',
            '{"other": "Paris"}',
            '{"answer": true}',
            '{"answer": null}',
            '{"answer": ["Paris"]}',
            '{"answer": {"city": "Paris"}}',
        ):
            with self.subTest(content=content):
                self.assertIsNone(
                    scoring.verify(_agent_result(content), "Paris", method="normalized").extracted_prediction
                )

    async def test_wrong_or_malformed_answers_do_not_trigger_model_extraction(self):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        for content, prediction in (('{"answer": "London"}', "London"), ("The answer is Paris", None)):
            with self.subTest(content=content):
                result = harness.AgentResult([], content, dict(USAGE), {"stop": 1}, 1, None)
                with (
                    patch.object(harness, "react", new_callable=AsyncMock, return_value=result),
                    patch.object(harness, "call_model", new_callable=AsyncMock) as call,
                    patch.object(model, "call_model", new_callable=AsyncMock) as direct_call,
                ):
                    record = await gaia.solve_task(
                        {"id": "1", "task_id": "task", "question": "Capital?", "gold": "Paris", "level": 1},
                        ctx,
                        tools=[],
                        args=argparse.Namespace(max_steps=16, task_timeout=3600),
                    )
                self.assertEqual(record.score_result.extracted_prediction, prediction)
                self.assertEqual(record.score_result.score, 0.0)
                self.assertIsNone(record.agent_result.sys_error)
                self.assertEqual(result.messages, [])
                call.assert_not_awaited()
                direct_call.assert_not_awaited()


class ReactFormatTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_tool_empty_and_final_turns_and_sums_usage(self):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        tool_run = AsyncMock(return_value="observation")
        tool = harness.Tool(harness.SEARCH_SCHEMA, "query", tool_run)
        calls = [{"id": "search_1", "name": "search", "arguments": {"query": "capital"}}]
        with patch.object(harness, "call_model", new_callable=AsyncMock) as call:
            call.side_effect = [
                ("", calls, "search reasoning", USAGE, "tool_calls"),
                ("", [], "empty reasoning", USAGE, "length"),
                ('{"answer": "Paris"}', [], "", USAGE, "stop"),
            ]
            result = await harness.react(
                ctx, "Capital?", tools=[tool], system_prompt=gaia.SYSTEM_PROMPT, max_steps=2, task_timeout=1
            )
        self.assertIsInstance(result, harness.AgentResult)
        self.assertIsNone(result.sys_error)
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.usage, {key: value * 3 for key, value in USAGE.items()})
        self.assertEqual(result.finish_reasons, {"tool_calls": 1, "length": 1, "stop": 1})
        assistants = [message for message in result.messages if message["role"] == "assistant"]
        self.assertEqual([message["finish_reason"] for message in assistants], ["tool_calls", "length", "stop"])
        self.assertEqual(assistants[1]["reasoning_content"], "empty reasoning")
        self.assertEqual(assistants[0]["tool_calls"][0]["function"]["arguments"], '{"query": "capital"}')
        tool_run.assert_awaited_once_with("capital")
        self.assertIsNone(call.call_args.kwargs.get("tools"))

    async def test_missing_metadata_uses_unknown_finish_reason_without_losing_turn(self):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with patch.object(harness, "call_model", new_callable=AsyncMock, return_value=("answer", [], "", None, None)):
            result = await harness.react(
                ctx, "Question?", tools=[], system_prompt="system", max_steps=1, task_timeout=1
            )
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.finish_reasons, {"unknown": 1})
        self.assertEqual(result.messages[-1]["finish_reason"], "unknown")
        self.assertEqual(result.usage, dict.fromkeys(USAGE, 0))

    async def test_final_request_preserves_the_system_output_format(self):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with (
            patch.object(harness, "call_model", new_callable=AsyncMock) as call,
            patch.object(harness.time, "perf_counter", side_effect=[20.0, 22.34567]),
        ):
            call.return_value = ('{"answer": "Paris"}', [], "", USAGE, "stop")
            result = await harness.react(
                ctx,
                "Capital?",
                tools=[],
                system_prompt=gaia.SYSTEM_PROMPT,
                max_steps=0,
                task_timeout=1,
            )
        self.assertEqual(result.final_content, '{"answer": "Paris"}')
        self.assertEqual(result.latency_sec, 2.346)
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.messages[-1]["finish_reason"], "stop")
        self.assertIsNone(result.sys_error)
        self.assertIn("format required by the system prompt", result.messages[-2]["content"])
        call.assert_awaited_once()


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_stream_is_closed_and_error_is_recorded(self):
        client, stream = _make_client([], error=RuntimeError("judge stream failed"))
        ctx = model.Context(client, JUDGE_PARAMS, asyncio.Semaphore(1))
        result = await scoring.judge(ctx, _agent_result("prediction"), "gold", question="question")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.sys_error, "RuntimeError: judge stream failed")
        self.assertEqual(len(result.messages), 1)
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
        ctx = model.Context(client, JUDGE_PARAMS, asyncio.Semaphore(1))
        with patch.object(scoring, "call_model", wraps=model.call_model) as call:
            result = await scoring.judge(ctx, _agent_result("The capital is Paris."), "Paris", question="Capital?")
        self.assertIsInstance(result, scoring.ScoreResult)
        self.assertEqual(
            (result.reference_answer, result.extracted_prediction, result.score, result.method),
            ("Paris", "Paris", 1.0, "judge"),
        )
        self.assertIsInstance(result.score, float)
        self.assertIsNone(result.sys_error)
        self.assertEqual(
            result.messages[-1],
            {
                "role": "assistant",
                "content": content,
                "reasoning_content": "judge reasoning",
                "usage": USAGE,
                "finish_reason": "stop",
            },
        )
        call.assert_awaited_once_with(ctx.sem, client, ctx.params, result.messages)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["messages"],
            [
                {
                    "role": "user",
                    "content": scoring.JUDGE_PROMPT.format(
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
        ctx = model.Context(client, JUDGE_PARAMS, asyncio.Semaphore(1))
        result = await scoring.judge(ctx, _agent_result("prediction"), "gold", question="question")
        self.assertEqual(result.score, 0.0)
        self.assertIn("ValueError: judge returned no verdict", result.sys_error or "")
        self.assertEqual(result.messages[-1]["content"], "no verdict")
        self.assertEqual(result.messages[-1]["finish_reason"], "length")
        stream.close.assert_awaited_once()

    async def test_no_answer_extracted_is_distinct_from_raw_prediction(self):
        client, _ = _make_client([_chunk(content="extracted_final_answer: None\nreasoning: No answer.\ncorrect: no")])
        ctx = model.Context(client, JUDGE_PARAMS, asyncio.Semaphore(1))
        result = await scoring.judge(ctx, _agent_result("I do not know."), "gold", question="question")
        self.assertEqual(result.score, 0.0)
        self.assertIsNone(result.extracted_prediction)
        self.assertEqual(result.messages[-1]["finish_reason"], "unknown")

    async def test_agent_failure_does_not_start_judging(self):
        client, _ = _make_client([])
        ctx = model.Context(client, JUDGE_PARAMS, asyncio.Semaphore(1))
        result = await scoring.judge(
            ctx, _agent_result("partial", sys_error="task_timeout"), "gold", question="question"
        )
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.messages, [])
        client.chat.completions.create.assert_not_awaited()

    async def test_missing_final_content_does_not_start_judging(self):
        client, _ = _make_client([])
        ctx = model.Context(client, JUDGE_PARAMS, asyncio.Semaphore(1))
        result = await scoring.judge(ctx, _agent_result(None), "gold", question="question")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.messages, [])
        client.chat.completions.create.assert_not_awaited()

    async def test_task_cancellation_propagates(self):
        ctx = model.Context(MagicMock(), JUDGE_PARAMS, asyncio.Semaphore(1))
        with (
            patch.object(scoring, "call_model", new_callable=AsyncMock, side_effect=asyncio.CancelledError),
            self.assertRaises(asyncio.CancelledError),
        ):
            await scoring.judge(ctx, _agent_result("prediction"), "gold", question="question")


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_model_params_limits_tasks_and_persists_predictions(self):
        parser = argparse.ArgumentParser()
        model.add_model_args(parser)
        args = parser.parse_args(["--model", "test-model", "--top-k", "7", "--concurrency", "4"])
        client = MagicMock(close=AsyncMock())
        active = 0
        peak = 0
        contexts = []

        async def solve_task(task, ctx):
            nonlocal active, peak
            contexts.append(ctx)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return _make_record(task["id"])

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(runner, "build_llm_client", return_value=client),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            path = Path(temp) / "predictions.json"
            records = await runner.run(
                args, [{"id": str(i)} for i in range(4)], solve_task, path, "test", max_active_tasks=2
            )
            self.assertEqual(json.loads(path.read_text()), [asdict(record) for record in records])
            self.assertFalse(path.with_suffix(".json.tmp").exists())
        self.assertEqual(peak, 2)
        self.assertEqual({record.id for record in records}, {"0", "1", "2", "3"})
        self.assertTrue(all(ctx.params.extra_body == {"top_k": 7} for ctx in contexts))
        client.close.assert_awaited_once()

    async def test_failed_task_cancels_other_tasks_and_closes_client(self):
        parser = argparse.ArgumentParser()
        model.add_model_args(parser)
        args = parser.parse_args(["--concurrency", "2"])
        client = MagicMock(close=AsyncMock())
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def solve_task(task, ctx):
            if task["id"] == "fail":
                await started.wait()
                raise RuntimeError("task failed")
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(runner, "build_llm_client", return_value=client),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "task failed"),
        ):
            await runner.run(args, [{"id": "fail"}, {"id": "wait"}], solve_task, Path(temp) / "out.json", "test")
        self.assertTrue(cancelled.is_set())
        client.close.assert_awaited_once()

    async def test_rejects_invalid_concurrency_before_creating_client(self):
        with patch.object(runner, "build_llm_client") as build:
            with self.assertRaisesRegex(ValueError, "concurrency must be positive"):
                await runner.run(argparse.Namespace(concurrency=0), [], AsyncMock(), Path("unused.json"), "test")
            with self.assertRaisesRegex(ValueError, "max_active_tasks must be positive"):
                await runner.run(
                    argparse.Namespace(concurrency=1), [], AsyncMock(), Path("unused.json"), "test", max_active_tasks=0
                )
        build.assert_not_called()


class RunnerResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_are_shared_by_tasks_and_closed_after_completion(self):
        await self._assert_resources(use_tools=True)

    async def test_tools_and_judge_are_shared_and_closed_after_completion(self):
        await self._assert_resources(use_tools=True, use_judge=True)

    async def test_judge_can_be_used_without_tools(self):
        await self._assert_resources(use_judge=True)

    async def test_task_failure_closes_tools_and_both_model_clients(self):
        await self._assert_resources(use_tools=True, use_judge=True, failure="task")

    async def test_tool_setup_failure_closes_every_acquired_resource(self):
        await self._assert_resources(use_tools=True, use_judge=True, failure="tools")

    async def test_judge_setup_failure_closes_primary_model_client(self):
        await self._assert_resources(use_tools=True, use_judge=True, failure="judge")

    async def _assert_resources(self, *, use_tools=False, use_judge=False, failure=None):
        parser = argparse.ArgumentParser()
        with patch.dict(os.environ, TEST_ENV, clear=True):
            model.add_model_args(parser, include_judge=use_judge)
            harness.add_tool_args(parser)
            harness.add_paradigm_args(parser)
        args = parser.parse_args([])
        client = MagicMock(close=AsyncMock())
        judge = MagicMock(close=AsyncMock())
        tool_scope = MagicMock()
        tool_clients = (MagicMock(), MagicMock())
        tool_scope.__aenter__.return_value = tool_clients
        tool_scope.__aexit__.return_value = False
        tools = [MagicMock()]

        async def solve_task(task, ctx, **kwargs):
            client.close.assert_not_awaited()
            judge.close.assert_not_awaited()
            tool_scope.__aexit__.assert_not_awaited()
            if failure == "task":
                raise RuntimeError("task failed")
            return _make_record(task["id"])

        solver = AsyncMock(side_effect=solve_task)
        clients = [client, RuntimeError("judge failed") if failure == "judge" else judge]
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(runner, "build_llm_client", side_effect=clients) as build_client,
            patch.object(harness, "tool_clients", return_value=tool_scope) as open_tools,
            patch.object(harness, "make_tools", return_value=tools) as make_tools,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            if failure == "tools":
                make_tools.side_effect = RuntimeError("tools failed")
            with self.assertRaisesRegex(RuntimeError, f"{failure} failed") if failure else contextlib.nullcontext():
                await runner.run(
                    args,
                    [{"id": "1"}, {"id": "2"}],
                    solver,
                    Path(temp) / "predictions.json",
                    "test",
                    use_tools=use_tools,
                    use_judge=use_judge,
                )
            self.assertEqual(build_client.call_count, 2 if use_judge else 1)
            if use_judge:
                self.assertEqual(
                    build_client.call_args.kwargs,
                    {
                        "api_key": "judge-secret",
                        "base_url": TEST_ENV["JUDGE_BASE_URL"],
                        "timeout": args.timeout,
                    },
                )
            if use_tools and failure != "judge":
                open_tools.assert_called_once_with()
                make_tools.assert_called_once_with(*tool_clients, args)
                tool_scope.__aexit__.assert_awaited_once()
            else:
                open_tools.assert_not_called()
                make_tools.assert_not_called()
            if failure in ("tools", "judge"):
                solver.assert_not_awaited()
            else:
                for call in solver.await_args_list:
                    self.assertIs(call.args[1].client_llm, client)
                    expected = {"args": args}
                    if use_tools:
                        expected["tools"] = tools
                    if use_judge:
                        judge_ctx = call.kwargs["judge_ctx"]
                        self.assertIs(judge_ctx.client_llm, judge)
                        self.assertIs(judge_ctx.sem, call.args[1].sem)
                        self.assertEqual(judge_ctx.params, JUDGE_PARAMS)
                        expected["judge_ctx"] = judge_ctx
                    self.assertEqual(call.kwargs, expected)
        client.close.assert_awaited_once()
        if use_judge and failure != "judge":
            judge.close.assert_awaited_once()
        else:
            judge.close.assert_not_awaited()


class BenchmarkScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_aime_response_records_usage_reasoning_and_answer(self):
        await self._assert_single_turn(aime, {"id": "1", "problem": "1+1?", "gold": 2}, '{"answer": 2}', 2)

    async def test_gpqa_response_records_usage_reasoning_and_choice(self):
        task = {"id": "1", "question": "Choose A", "choices": dict.fromkeys("ABCD", "choice"), "gold": "A"}
        await self._assert_single_turn(gpqa, task, '{"answer": "A"}', "A")

    async def test_math_response_records_usage_reasoning_and_symbolic_answer(self):
        task = {"id": "1", "problem": "1+1?", "gold": "2", "subject": "Algebra", "level": 1}
        await self._assert_single_turn(math, task, r"\boxed{2}", "2")

    async def test_browsecomp_records_shared_judge_score_and_errors(self):
        task = {"id": "1", "question": "Capital?", "gold": "Paris", "topic": "Geography"}
        await self._assert_judged_task(browsecomp, task)

    async def test_webwalkerqa_records_shared_judge_score_and_errors(self):
        task = {"id": "1", "question": "Capital?", "gold": "Paris", "type": "single_source", "difficulty": "easy"}
        await self._assert_judged_task(webwalkerqa, task)

    async def test_gaia_json_answer_preserves_raw_content_and_usage_counts(self):
        content = ' {"answer": "Paris"} '
        messages = [{"role": "assistant", "content": content, "usage": USAGE}]
        result = harness.AgentResult(messages, content, dict(USAGE), {"stop": 1}, 1, None)
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with (
            patch.object(harness, "react", new_callable=AsyncMock, return_value=result),
            patch.object(harness, "call_model", new_callable=AsyncMock) as call,
            patch.object(model, "call_model", new_callable=AsyncMock) as direct_call,
        ):
            record = await gaia.solve_task(
                {"id": "1", "task_id": "task", "question": "Capital of France?", "gold": "Paris", "level": 1},
                ctx,
                tools=[],
                args=argparse.Namespace(max_steps=16, task_timeout=3600),
            )
        self.assertEqual((record.score_result.score, record.score_result.extracted_prediction), (1.0, "Paris"))
        self.assertIsInstance(record.score_result.score, float)
        self.assertEqual(record.agent_result.messages, messages)
        self.assertEqual(record.agent_result.messages[-1]["content"], content)
        self.assertEqual(record.agent_result.finish_reasons, {"stop": 1})
        self.assertEqual({key: record.agent_result.usage[key] for key in USAGE}, USAGE)
        call.assert_not_awaited()
        direct_call.assert_not_awaited()

    async def _assert_judged_task(self, module, task):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        result = harness.AgentResult([], "Paris", dict(USAGE), {"stop": 1}, 1, None)
        client, stream = _make_client([_chunk(content="correct: yes")])
        args = argparse.Namespace(max_steps=16, task_timeout=3600)
        judge_ctx = model.Context(client, JUDGE_PARAMS, ctx.sem)
        with patch.object(harness, "react", new_callable=AsyncMock, return_value=result):
            record = await module.solve_task(task, ctx, tools=[], judge_ctx=judge_ctx, args=args)
            self.assertEqual((record.score_result.extracted_prediction, record.score_result.score), ("Paris", 1.0))
            self.assertEqual(record.score_result.messages[-1]["content"], "correct: yes")
            self.assertIsNone(record.agent_result.sys_error)
            stream.close.assert_awaited_once()
            client.chat.completions.create.side_effect = RuntimeError("judge unavailable")
            record = await module.solve_task(task, ctx, tools=[], judge_ctx=judge_ctx, args=args)
            self.assertEqual(record.score_result.score, 0.0)
            self.assertIsNone(record.agent_result.sys_error)
            self.assertEqual(record.score_result.sys_error, "RuntimeError: judge unavailable")
            result.sys_error = "task_timeout (3600s)"
            client.chat.completions.create.reset_mock()
            record = await module.solve_task(task, ctx, tools=[], judge_ctx=judge_ctx, args=args)
            self.assertEqual(record.score_result.score, 0.0)
            self.assertEqual(record.agent_result.sys_error, result.sys_error)
            client.chat.completions.create.assert_not_awaited()

    async def _assert_single_turn(self, module, task, content, extracted):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with patch.object(harness, "call_model", new_callable=AsyncMock) as call:
            call.return_value = (f" {content} ", [], "reason", USAGE, "stop")
            record = await module.solve_task(task, ctx)
            self.assertEqual((record.score_result.score, record.score_result.extracted_prediction), (1.0, extracted))
            self.assertIsInstance(record.score_result.score, float)
            self.assertEqual(record.agent_result.finish_reasons, {"stop": 1})
            self.assertEqual(record.agent_result.steps, 1)
            self.assertEqual(
                record.agent_result.messages[-1],
                {
                    "role": "assistant",
                    "content": f" {content} ",
                    "reasoning_content": "reason",
                    "usage": USAGE,
                    "finish_reason": "stop",
                },
            )
            self.assertEqual({key: record.agent_result.usage[key] for key in USAGE}, USAGE)
            call.side_effect = RuntimeError("request failed")
            record = await module.solve_task(task, ctx)
            self.assertEqual(record.score_result.score, 0)
            self.assertEqual(record.agent_result.sys_error, "RuntimeError: request failed")
            self.assertEqual(len(record.agent_result.messages), 2)


def _agent_result(final_content, *, sys_error=None):
    return harness.AgentResult([], final_content, dict(USAGE), {"stop": 1}, 1, sys_error)


def _make_record(task_id, **metadata):
    return runner.make_record(
        {"id": task_id},
        _agent_result("answer"),
        scoring.ScoreResult("answer", "answer", 1.0, "integer"),
        **metadata,
    )


def _run_main(module, flags):
    tool_flags = (
        ["--search-provider", "deepseek", "--fetch-provider", "deepseek"]
        if module in (browsecomp, gaia, webwalkerqa)
        else []
    )
    client = MagicMock(close=AsyncMock())
    record = _make_record("1", topic="test", level=1, subject="Algebra", type="single_source", difficulty="easy")
    with (
        tempfile.TemporaryDirectory() as temp,
        patch.dict(os.environ, TEST_ENV, clear=True),
        patch.object(module, "load_dotenv"),
        patch("sys.argv", [module.__name__, *flags, *tool_flags]),
        patch.object(module, "load_tasks", return_value=[{"id": "1"}]),
        patch.object(module, "solve_task", new_callable=AsyncMock, return_value=record),
        patch.object(runner, "prepare_run_dir", return_value=Path(temp)),
        patch.object(model, "build_llm_client", return_value=client),
        patch.object(runner, "build_llm_client", return_value=client),
        patch.object(runner, "Context", wraps=model.Context) as context,
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        module.main()
    return context.call_args_list[0].kwargs["params"]


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
