"""Lossless streamed messages, request filtering, and interrupted model calls."""

import argparse
import asyncio
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from eval import model

__all__ = []
PARAMS = model.ModelParams("test", 0, 0.95, 1024, 42)
USAGE = {"prompt_tokens": 10, "completion_tokens": 4, "reasoning_tokens": 2}


class ModelTests(unittest.IsolatedAsyncioTestCase):
    def test_thinking_defaults_and_generation_flag_overrides(self):
        parser = argparse.ArgumentParser()
        model.add_model_args(parser)
        defaults = model.build_model_params(parser.parse_args(["--model", "test"]))
        self.assertTrue(defaults.enable_thinking)
        self.assertEqual(defaults.reasoning_effort, "medium")
        self.assertTrue(defaults.include_reasoning)
        selected = model.build_model_params(
            parser.parse_args(
                [
                    "--model",
                    "test",
                    "--no-enable-thinking",
                    "--reasoning-effort",
                    "high",
                ]
            )
        )
        self.assertFalse(selected.enable_thinking)
        self.assertEqual(selected.reasoning_effort, "high")

    async def test_standalone_context_owns_a_real_client(self):
        ctx = model.make_context("https://model.example/v1", "key", PARAMS, 2, timeout=1)
        try:
            self.assertEqual(str(ctx.client.base_url), "https://model.example/v1/")
            self.assertEqual(ctx.semaphore._value, 2)
        finally:
            await ctx.client.close()
        with self.assertRaises(ValueError):
            model.make_context(None, "key", PARAMS, 0)

    async def test_stream_reassembles_content_reasoning_usage_and_tool_fragments(self):
        first = SimpleNamespace(index=1, id="call_", function=SimpleNamespace(name="fe", arguments='{ "url":'))
        second = SimpleNamespace(index=1, id="1", function=SimpleNamespace(name="tch", arguments=' "example" }'))
        other = SimpleNamespace(index=0, id=None, function=SimpleNamespace(name="search", arguments="invalid"))
        chunks = [
            _chunk("answer", "first", [first]),
            _chunk(reasoning="second", calls=[second, other], finish="tool_calls"),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=4, completion_tokens_details=SimpleNamespace(reasoning_tokens=2)
                ),
            ),
        ]
        ctx, stream = _context(chunks)
        ctx = replace(
            ctx,
            params=replace(PARAMS, enable_thinking=False, include_reasoning=False, extra_body={"top_k": 7}),
        )
        messages = [
            {
                "role": "assistant",
                "content": "old",
                "reasoning_content": "hidden",
                "usage": USAGE,
                "finish_reason": "stop",
                "incomplete": True,
                "error": "log",
                "latency_sec": 1.5,
                "queue_sec": 0.5,
            },
            {
                "role": "tool",
                "content": "page",
                "tool_call_id": "old",
                "is_error": True,
                "tool_result": {"x": 1},
                "error": "log",
                "latency_sec": 0.4,
                "tool_name": "fetch",
            },
        ]
        turn = await model.call_model(ctx, messages)
        self.assertEqual(
            (turn.content, turn.reasoning, turn.usage, turn.finish_reason),
            ("answer", "firstsecond", USAGE, "tool_calls"),
        )
        self.assertEqual(turn.tool_calls[0]["id"], "call_missing_0")
        self.assertEqual(
            turn.tool_calls[1],
            {"id": "call_1", "type": "function", "function": {"name": "fetch", "arguments": '{ "url": "example" }'}},
        )
        request = ctx.client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["messages"],
            [{"role": "assistant", "content": "old"}, {"role": "tool", "content": "page", "tool_call_id": "old"}],
        )
        self.assertEqual(request["extra_body"], {"top_k": 7, "chat_template_kwargs": {"enable_thinking": False}})
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertTrue(messages[0]["incomplete"])
        stream.close.assert_awaited_once()

    async def test_tools_and_omitted_generation_options(self):
        ctx, _ = _context([])
        tools = [{"type": "function", "function": {"name": "search"}}]
        messages = [{"role": "assistant", "content": "old", "reasoning_content": "kept", "usage": USAGE}]
        turn = await model.call_model(ctx, messages, tools)
        self.assertEqual(replace(turn, latency_sec=None, queue_sec=None), model.AssistantTurn())
        request = ctx.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"], [{"role": "assistant", "content": "old", "reasoning_content": "kept"}])
        self.assertEqual(request["extra_body"], {"chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual(request["reasoning_effort"], "medium")
        self.assertEqual(request["tools"], tools)
        await model.call_model(ctx, [], [])
        self.assertNotIn("tools", ctx.client.chat.completions.create.call_args.kwargs)

    async def test_reasoning_effort_override_reaches_model_requests(self):
        ctx, _ = _context([])
        ctx = replace(ctx, params=replace(PARAMS, reasoning_effort="xhigh"))
        await model.call_model(ctx, [])
        request = ctx.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["reasoning_effort"], "xhigh")
        self.assertEqual(request["extra_body"]["chat_template_kwargs"], {"enable_thinking": True})

    async def test_failed_stream_returns_partial_response_and_closes(self):
        ctx, stream = _context([_chunk("partial", "thought")], error=RuntimeError("failed"))
        turn = await model.call_model(ctx, [])
        self.assertEqual((turn.content, turn.reasoning, str(turn.error)), ("partial", "thought", "failed"))
        self.assertTrue(turn.message()["incomplete"])
        stream.close.assert_awaited_once()

    async def test_creation_error_and_cancellation_are_returned_for_outer_handling(self):
        ctx, _ = _context([])
        ctx.client.chat.completions.create.side_effect = RuntimeError("create failed")
        turn = await model.call_model(ctx, [])
        self.assertEqual(str(turn.error), "create failed")
        ctx = replace(ctx, semaphore=asyncio.Semaphore(0))
        future = asyncio.create_task(model.call_model(ctx, []))
        await asyncio.sleep(0)
        future.cancel()
        interrupted = await future
        self.assertIsInstance(interrupted.error, asyncio.CancelledError)
        self.assertEqual(interrupted.latency_sec, interrupted.queue_sec)

    async def test_timing_separates_model_queue_and_retains_reported_cached_tokens(self):
        ctx, _ = _context(
            [
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=4,
                        completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
                        prompt_tokens_details=SimpleNamespace(cached_tokens=7),
                    ),
                ),
            ]
        )
        ctx = replace(ctx, semaphore=asyncio.Semaphore(0))
        pending = asyncio.create_task(model.call_model(ctx, []))
        await asyncio.sleep(0.01)
        ctx.semaphore.release()
        turn = await pending
        assert turn.queue_sec is not None and turn.latency_sec is not None
        self.assertGreaterEqual(turn.queue_sec, 0.005)
        self.assertGreaterEqual(turn.latency_sec, turn.queue_sec)
        self.assertEqual(turn.usage, {**USAGE, "cached_tokens": 7})
        self.assertEqual(turn.message()["latency_sec"], turn.latency_sec)


def _chunk(content=None, reasoning=None, calls=(), finish=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=calls)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


def _context(chunks, error=None):
    async def responses():
        for chunk in chunks:
            yield chunk
        if error is not None:
            raise error

    stream = MagicMock(close=AsyncMock())
    stream.__aiter__.side_effect = responses
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=stream)
    return model.Context(client, PARAMS, asyncio.Semaphore(1)), stream


if __name__ == "__main__":
    unittest.main()
