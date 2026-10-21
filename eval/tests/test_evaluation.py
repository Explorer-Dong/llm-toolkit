"""Offline contracts for shared model options, response capture, and benchmark scoring."""

import argparse
import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from eval import aime, browsecomp, gaia, gpqa, math, webwalkerqa
from eval.core import harness, model, runner

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
        harness.add_agent_args(parser)
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
        harness.add_agent_args(parser)
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
        messages = [{"role": "assistant", "content": "previous", "reasoning_content": "hidden", "usage": USAGE}]
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


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_model_params_limits_tasks_and_persists_predictions(self):
        parser = argparse.ArgumentParser()
        model.add_model_args(parser)
        args = parser.parse_args(["--model", "test-model", "--top-k", "7", "--concurrency", "4"])
        client = MagicMock(close=AsyncMock())
        active = 0
        peak = 0
        contexts = []

        async def solve(task, ctx):
            nonlocal active, peak
            contexts.append(ctx)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return {"id": task["id"], "score": 1, "sys_error": None}

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(runner, "build_llm_client", return_value=client),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            path = Path(temp) / "predictions.json"
            records = await runner.run(
                args, [{"id": str(i)} for i in range(4)], solve, path, "test", max_active_tasks=2
            )
            self.assertEqual(json.loads(path.read_text()), records)
            self.assertFalse(path.with_suffix(".json.tmp").exists())
        self.assertEqual(peak, 2)
        self.assertEqual({record["id"] for record in records}, {"0", "1", "2", "3"})
        self.assertTrue(all(ctx.params.extra_body == {"top_k": 7} for ctx in contexts))
        client.close.assert_awaited_once()

    async def test_failed_task_cancels_other_tasks_and_closes_client(self):
        parser = argparse.ArgumentParser()
        model.add_model_args(parser)
        args = parser.parse_args(["--concurrency", "2"])
        client = MagicMock(close=AsyncMock())
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def solve(task, ctx):
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
            await runner.run(args, [{"id": "fail"}, {"id": "wait"}], solve, Path(temp) / "out.json", "test")
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


class BenchmarkScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_aime_response_records_usage_reasoning_and_answer(self):
        await self._assert_single_turn(aime, {"id": "1", "problem": "1+1?", "gold": 2}, '{"answer": 2}', 2)

    async def test_gpqa_response_records_usage_reasoning_and_choice(self):
        task = {"id": "1", "question": "Choose A", "choices": dict.fromkeys("ABCD", "choice"), "gold": "A"}
        await self._assert_single_turn(gpqa, task, '{"answer": "A"}', "A")

    async def test_math_response_records_usage_reasoning_and_symbolic_answer(self):
        task = {"id": "1", "problem": "1+1?", "gold": "2", "subject": "Algebra", "level": 1}
        await self._assert_single_turn(math, task, r"\boxed{2}", "2")

    async def test_browsecomp_judge_formats_prompt_and_reads_yes(self):
        client, stream = _make_client([_chunk(content="correct: YES")])
        result = await browsecomp.scorer(asyncio.Semaphore(1), client, "test-judge", "question", "answer", "gold")
        self.assertEqual(result, (True, "correct: YES"))
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["messages"],
            [
                {
                    "role": "user",
                    "content": browsecomp.JUDGE_PROMPT.format(
                        question="question", response="answer", correct_answer="gold"
                    ),
                }
            ],
        )
        self.assertEqual((request["model"], request["temperature"], request["max_tokens"]), ("test-judge", 0, 4096))
        self.assertEqual(request["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        stream.close.assert_awaited_once()

    async def test_webwalkerqa_judge_reads_no_and_rejects_missing_verdict(self):
        client, stream = _make_client([_chunk(content="correct: no")])
        result = await webwalkerqa.scorer(asyncio.Semaphore(1), client, "test-judge", "q", "p", "g")
        self.assertEqual(result, (False, "correct: no"))
        stream.close.assert_awaited_once()
        client, stream = _make_client([_chunk(content="no verdict")])
        with self.assertRaisesRegex(ValueError, "judge returned no verdict"):
            await webwalkerqa.scorer(asyncio.Semaphore(1), client, "test-judge", "q", "p", "g")
        stream.close.assert_awaited_once()

    async def test_gaia_extraction_preserves_raw_content_and_usage_counts(self):
        result = SimpleNamespace(
            answer="The answer is Paris",
            sys_error=None,
            messages=[],
            usage=dict.fromkeys(USAGE, 0),
            finish_reasons={},
            steps=1,
        )
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with (
            patch.object(gaia.harness, "react", new_callable=AsyncMock, return_value=result),
            patch.object(
                model, "call_model", new_callable=AsyncMock, return_value=(" Paris ", [], "reason", USAGE, "stop")
            ),
        ):
            record = await gaia.solve(
                {"id": "1", "task_id": "task", "question": "Capital of France?", "gold": "Paris", "level": 1},
                ctx,
                tools=[],
                args=argparse.Namespace(max_steps=16, task_timeout=3600),
            )
        self.assertEqual((record["score"], record["extracted_prediction"]), (1, "Paris"))
        self.assertEqual(record["messages"][-1]["content"], " Paris ")
        self.assertEqual(record["finish_reasons"], {"stop": 1})
        self.assertEqual({key: record[key] for key in USAGE}, USAGE)

    async def _assert_single_turn(self, module, task, content, extracted):
        ctx = model.Context(client_llm=MagicMock(), params=PARAMS, sem=asyncio.Semaphore(1))
        with patch.object(model, "call_model", new_callable=AsyncMock) as call:
            call.return_value = (f" {content} ", [], "reason", USAGE, "stop")
            record = await module.solve(task, ctx)
            self.assertEqual((record["score"], record["extracted_prediction"]), (1, extracted))
            self.assertEqual(
                record["messages"][-1],
                {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "reason",
                    "usage": USAGE,
                },
            )
            self.assertEqual({key: record[key] for key in USAGE}, USAGE)
            call.side_effect = RuntimeError("request failed")
            record = await module.solve(task, ctx)
            self.assertEqual(record["score"], 0)
            self.assertEqual(record["sys_error"], "RuntimeError: request failed")
            self.assertEqual(len(record["messages"]), 2)


def _run_main(module, flags):
    tool_flags = ["--search-provider", "deepseek", "--fetch-provider", "deepseek"] if hasattr(module, "amain") else []
    client = MagicMock(close=AsyncMock())
    with (
        tempfile.TemporaryDirectory() as temp,
        patch.dict(os.environ, TEST_ENV, clear=True),
        patch.object(module, "load_dotenv"),
        patch("sys.argv", [module.__name__, *flags, *tool_flags]),
        patch.object(module, "load_data", return_value=[]),
        patch.object(runner, "prepare_run_dir", return_value=Path(temp)),
        patch.object(model, "build_llm_client", return_value=client),
        patch.object(runner, "build_llm_client", return_value=client),
        patch.object(runner, "Context", wraps=model.Context) as context,
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        module.main()
    return context.call_args.kwargs["params"]


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
