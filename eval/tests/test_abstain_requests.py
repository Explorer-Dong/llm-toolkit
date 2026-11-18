"""Abstention generation defaults and serialized model requests."""

import asyncio
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncOpenAI

from eval.agent import Trajectory
from eval.benchmarks._common.metrics import ScoreResult
from eval.benchmarks.agent_abstain import rollout as aa
from eval.benchmarks.agent_abstain import scoring as aas
from eval.benchmarks.webshop_abstain import rollout as ws
from eval.benchmarks.webshop_abstain import scoring as wss
from eval.model.openai import Context, ModelParams, call_model
from eval.model.utils import build_model_params

__all__ = []
MESSAGES = [{"role": "user", "content": "test"}]
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]


class AbstainRequestTests(unittest.IsolatedAsyncioTestCase):
    def test_generation_defaults_and_explicit_overrides(self):
        for module, tokens in ((aa, 16384), (ws, 4096)):
            with self.subTest(benchmark=module.__name__):
                args = module.parse_args(["--model", "Qwen3.5-9B"])
                self.assertEqual((args.max_turns, args.temperature, args.max_tokens), (30, 0, tokens))
                self.assertTrue(args.enable_thinking)
                self.assertEqual(args.reasoning_effort, "medium")
                self.assertEqual(args.concurrency, 1 if module is aa else 4)
                self.assertIsNone(args.top_p)
                self.assertIsNone(args.seed)
                self.assertEqual((args.timeout, args.max_retries), (600, 2) if module is aa else (90, 0))
                selected = module.parse_args(
                    ["--model", "Qwen3.8-27B", "--reasoning-effort", "xhigh", "--max-tokens", "32768"]
                )
                self.assertEqual((selected.reasoning_effort, selected.max_tokens), ("xhigh", 32768))

    async def test_agent_rollout_wire_request(self):
        args = aa.parse_args(["--model", "Qwen3.5-9B"])
        ctx, requests = _context(build_model_params(args))
        messages = [
            *MESSAGES,
            {"role": "assistant", "content": "", "reasoning_content": "previous thought", "usage": {}},
        ]
        try:
            turn = await call_model(ctx, messages, TOOLS)
            self.assertIsNone(turn.error)
            self.assertEqual(
                requests,
                [
                    {
                        "model": args.model,
                        "messages": [
                            *MESSAGES,
                            {"role": "assistant", "content": "", "reasoning_content": "previous thought"},
                        ],
                        "temperature": 0,
                        "max_completion_tokens": 16384,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "reasoning_effort": "medium",
                        "store": False,
                        "chat_template_kwargs": {"enable_thinking": True},
                        "tools": TOOLS,
                    }
                ],
            )
        finally:
            await ctx.client.close()

    async def test_webshop_rollout_wire_request_and_reasoning_history(self):
        args = ws.parse_args(["--model", "Qwen3.5-9B"])
        ctx, requests = _context(build_model_params(args))
        messages = [
            *MESSAGES,
            {"role": "assistant", "content": "", "reasoning_content": "previous thought", "usage": {}},
        ]
        try:
            turn = await call_model(ctx, messages, TOOLS)
            self.assertIsNone(turn.error)
            self.assertEqual((turn.content, turn.reasoning), ("done", "thought"))
            self.assertEqual(
                requests[0],
                {
                    "model": args.model,
                    "messages": [
                        *MESSAGES,
                        {"role": "assistant", "content": "", "reasoning_content": "previous thought"},
                    ],
                    "temperature": 0,
                    "max_tokens": 4096,
                    "reasoning_effort": "medium",
                    "chat_template_kwargs": {"enable_thinking": True},
                    "tools": TOOLS,
                },
            )
            await call_model(ctx, messages)
            self.assertNotIn("tools", requests[-1])
            self.assertEqual({k: v for k, v in requests[0].items() if k != "tools"}, requests[-1])
        finally:
            await ctx.client.close()

    async def test_agent_judge_wire_request(self):
        await self._check_judge(aas, 16384, 120)

    async def test_webshop_judge_wire_request(self):
        await self._check_judge(wss, 8192, 180)

    async def test_webshop_nonstream_tool_calls_and_token_usage(self):
        params = build_model_params(ws.parse_args(["--model", "test"]))
        calls = [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"keywords":"jeans"}'}}
        ]
        ctx, _ = _context(params, tool_calls=calls)
        try:
            turn = await call_model(ctx, MESSAGES, TOOLS)
            self.assertIsNone(turn.error)
            self.assertEqual(turn.tool_calls, calls)
            self.assertEqual(turn.finish_reason, "tool_calls")
            self.assertEqual(turn.usage, {"prompt_tokens": 3, "completion_tokens": 2, "reasoning_tokens": 0})
        finally:
            await ctx.client.close()

    async def test_webshop_request_attempts_do_not_multiply_sdk_retries(self):
        args = ws.parse_args(["--model", "test"])
        ctx, requests = _context(build_model_params(args), status=500)
        ctx = replace(ctx, request_attempts=args.request_attempts, retry_delay=args.retry_delay)
        try:
            with patch("eval.model.openai.asyncio.sleep", new_callable=AsyncMock) as sleep:
                turn = await call_model(ctx, MESSAGES)
            self.assertIsNotNone(turn.error)
            self.assertEqual(len(requests), 3)
            self.assertEqual([call.args for call in sleep.await_args_list], [(1,), (1,)])
        finally:
            await ctx.client.close()

    async def test_agent_judge_retries_transient_errors_only(self):
        sample = aa.load_tasks(aa.parse_args(["--model", "test", "--limit", "1"]))[0]
        for status, attempts, delays in ((500, 4, [1, 2, 4]), (400, 1, [])):
            with self.subTest(status=status):
                ctx, requests = _context(ModelParams("judge", 0, None, 16384, None), status=status)
                try:
                    with patch.object(aas.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                        result = await aas.score(
                            sample,
                            Trajectory(sample["id"], final_answer="done"),
                            {"execution_log": []},
                            {},
                            contexts=[ctx],
                        )
                    self.assertIsNotNone(result.sys_error)
                    self.assertEqual(len(requests), attempts)
                    self.assertEqual([call.args[0] for call in sleep.await_args_list], delays)
                finally:
                    await ctx.client.close()

    async def _check_judge(self, module, tokens, timeout):
        args = module.parse_args(["unused-run", "--judge-model", "deepseek-flash"])
        self.assertTrue(args.enable_thinking)
        self.assertEqual(args.reasoning_effort, "medium")
        requests = []
        contexts = []

        def make_context(base_url, api_key, params, concurrency, **options):
            self.assertEqual(options, {"timeout": timeout, "max_retries": 0})
            ctx, bodies = _context(params)
            contexts.append(ctx)
            requests.append(bodies)
            return ctx

        async def run_score_call(sample, *_args, **kwargs):
            ctx = kwargs["ctx"] if "ctx" in kwargs else kwargs["contexts"][0]
            turn = await call_model(ctx, MESSAGES)
            self.assertIsNone(turn.error)
            return ScoreResult(sample["gold"], sample["gold"], 1.0, "test")

        sample = {"id": "1", "gold": "correct_act", "meta": {}}
        with (
            patch.object(module, "make_context", side_effect=make_context),
            patch.object(
                module,
                "init_score_dir",
                return_value=(
                    Path("unused-score"),
                    [sample],
                    {"1": Trajectory("1", final_answer="done")},
                    {"1": None},
                    {"model": "test"},
                    {},
                    None,
                    {},
                    None,
                ),
            ),
            patch.object(module, "score", side_effect=run_score_call),
            patch.object(module, "write_json"),
            patch.object(module, "finalize_score_dir", return_value={}),
        ):
            await module._run_score(args)
        self.assertEqual(
            requests[0],
            [
                {
                    "model": "deepseek-flash",
                    "messages": MESSAGES,
                    "temperature": 0,
                    "max_tokens": tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "medium",
                }
            ],
        )


def _context(params, *, status=200, tool_calls=()):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "unavailable", "type": "server_error"}})
        if body.get("stream"):
            chunk = {"id": "test", "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
            return httpx.Response(
                200,
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None if tool_calls else "done",
                            "reasoning_content": "thought",
                            "tool_calls": list(tool_calls),
                        },
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    client = AsyncOpenAI(
        api_key="test",
        base_url="https://model.example/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),  # ty: ignore[invalid-argument-type]
    )
    return Context(client, params, asyncio.Semaphore(1)), requests
