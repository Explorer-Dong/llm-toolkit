"""ReAct budgets, trajectory completeness, cancellation, and persistence contracts."""

import asyncio
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from eval import agent
from eval.agent import Task, Trajectory, run_episode
from eval.harness.tool import Connection, ToolBox, ToolResponse
from eval.harness.utils import load_trajectory, save_trajectory
from eval.model.openai import AssistantTurn, Context, ModelParams

__all__ = []
PARAMS = ModelParams("model", 0, 1, 1024, 42)
USAGE = {"prompt_tokens": 10, "completion_tokens": 4, "reasoning_tokens": 2}
SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the internet. Returns a JSON list of results, each with title, url, and snippet.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query."}},
            "required": ["query"],
        },
    },
}


class ReactTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "trajectory.json"
        self.client = MagicMock(close=AsyncMock())
        self.ctx = Context(self.client, PARAMS, asyncio.Semaphore(1))
        self.task = Task("sample", "system", "question", max_turns=3, meta={"tag": "test"})
        self.connection = MagicMock(spec=Connection, name="common", schemas=[SEARCH_SCHEMA])
        self.connection.name = "common"
        self.connection.call = AsyncMock(return_value=ToolResponse("page"))
        self.tools = ToolBox((self.connection,))

    async def test_invalid_model_responses_are_recorded_without_console_errors(self):
        for reason, content in (("stop", ""), ("length", "partial"), ("max_tokens", "partial")):
            with (
                self.subTest(reason=reason),
                self.assertNoLogs("eval.harness", level="WARNING"),
                patch.object(
                    agent,
                    "call_model",
                    AsyncMock(return_value=AssistantTurn(content=content, finish_reason=reason)),
                ),
            ):
                result = await run_episode(replace(self.task, strict_response=True), self.ctx, ToolBox(), out=self.path)
            self.assertEqual(result.episode_status, "invalid_response")
            self.assertEqual(result.finish_reason, reason)
            self.assertIn("ModelResponseError", result.error or "")
            self.assertEqual(load_trajectory(self.path), result)

    async def test_tool_calls_after_final_turn_are_recorded_without_console_errors(self):
        turn = _reply("", calls=[_tool_call("search", '{"query":"first"}')])
        with (
            self.assertNoLogs("eval.harness", level="WARNING"),
            patch.object(agent, "call_model", AsyncMock(side_effect=[turn, turn])),
        ):
            result = await run_episode(replace(self.task, max_turns=1, strict_response=True), self.ctx, self.tools)
        self.assertIn("ModelResponseError", result.error or "")
        self.assertEqual(self.connection.call.await_count, 1)

    async def test_system_failures_are_logged_and_persisted(self):
        turn = AssistantTurn(error=RuntimeError("Endpoint unavailable"))
        with (
            self.assertLogs("eval.agent", level="ERROR") as logs,
            patch.object(agent, "call_model", AsyncMock(return_value=turn)),
        ):
            result = await run_episode(self.task, self.ctx, ToolBox(), out=self.path)
        self.assertIn("Endpoint unavailable", "\n".join(logs.output))
        self.assertEqual(result.error, "RuntimeError: Endpoint unavailable")
        self.assertEqual(load_trajectory(self.path), result)

    async def test_endpoint_finish_reason_and_local_status_remain_separate(self):
        for reason in ("stop", None, "provider_custom_value"):
            turn = AssistantTurn(content="answer", finish_reason=reason)
            with self.subTest(reason=reason), patch.object(agent, "call_model", AsyncMock(return_value=turn)):
                result = await run_episode(self.task, self.ctx, ToolBox(), out=self.path)
            saved = json.loads(self.path.read_text())
            self.assertNotIn("stop_reason", saved)
            self.assertEqual(saved["episode_status"], "final_answer")
            self.assertEqual(saved["finish_reason"], reason)
            self.assertEqual(saved["messages"][-1]["finish_reason"], reason)
            self.assertEqual(result.finish_reasons, {} if reason is None else {reason: 1})

    async def test_legacy_stop_reason_records_remain_readable(self):
        legacy = asdict(Trajectory("legacy", messages=[{"role": "assistant", "finish_reason": "stop"}]))
        legacy["stop_reason"] = legacy.pop("episode_status")
        legacy.pop("finish_reason")
        self.path.write_text(json.dumps(legacy))
        original = self.path.read_bytes()
        loaded = load_trajectory(self.path)
        self.assertEqual(loaded.episode_status, "max_turns")
        self.assertEqual(loaded.finish_reason, "stop")
        self.assertEqual(self.path.read_bytes(), original)

    async def test_single_turn_qa_uses_the_same_loop_and_persists_all_metadata(self):
        with patch.object(agent, "call_model", new_callable=AsyncMock, return_value=_reply("  answer\n")) as call:
            result = await run_episode(replace(self.task, max_turns=1), self.ctx, ToolBox(), out=self.path)
        self.assertEqual((result.final_answer, result.turns, result.episode_status), ("answer", 1, "final_answer"))
        self.assertEqual(result.usage, USAGE)
        self.assertEqual(result.finish_reasons, {"stop": 1})
        self.assertEqual(result.messages[-1]["content"], "  answer\n")
        self.assertEqual(result.messages[-1]["reasoning_content"], "reason")
        self.assertEqual(load_trajectory(self.path), result)
        self.assertEqual(json.loads(self.path.read_text()), asdict(result))
        self.assertEqual(call.call_args.args[2], [])
        self.client.close.assert_not_awaited()
        self.assertNotIn("score", asdict(result))
        self.assertNotIn("reference_answer", asdict(result))

    async def test_tool_round_trip_keeps_exact_arguments_and_raw_mcp_result(self):
        raw = {"content": [{"type": "text", "text": "page"}], "structuredContent": {"source": "url"}, "isError": False}
        self.connection.call.return_value = ToolResponse("page", raw=raw)
        calls = [_tool_call("search", '{ "query": "capital", "extra": [1, 2] }')]
        with patch.object(agent, "call_model", new_callable=AsyncMock) as call:
            call.side_effect = [_reply("", calls=calls, finish="tool_calls"), _reply("Paris")]
            result = await run_episode(self.task, self.ctx, self.tools, out=self.path)
        self.assertEqual(result.messages[2]["tool_calls"], calls)
        self.assertEqual(result.messages[3]["tool_result"], raw)
        self.assertEqual(result.messages[3]["tool_name"], "search")
        self.assertGreaterEqual(result.messages[3]["latency_sec"], 0)
        self.assertEqual(result.usage, {key: 2 * value for key, value in USAGE.items()})
        self.assertEqual(result.tools, [SEARCH_SCHEMA])
        self.assertEqual(result.final_answer, "Paris")
        self.connection.call.assert_awaited_once_with("search", {"query": "capital", "extra": [1, 2]})

    async def test_exhausted_turns_get_one_final_request_without_tools(self):
        with patch.object(agent, "call_model", new_callable=AsyncMock) as call:
            search = _reply("", calls=[_tool_call("search", '{"query":"q"}')], finish="tool_calls")
            call.side_effect = [search, search, _reply('{"answer":"Paris"}')]
            result = await run_episode(replace(self.task, max_turns=2), self.ctx, self.tools, out=self.path)
        self.assertEqual(call.await_count, 3)
        self.assertEqual(
            [request.args[2] for request in call.await_args_list], [[SEARCH_SCHEMA], [SEARCH_SCHEMA], None]
        )
        self.assertEqual(self.connection.call.await_count, 2)
        self.assertEqual((result.turns, result.episode_status, result.error), (3, "final_answer", None))
        self.assertEqual(result.final_answer, '{"answer":"Paris"}')
        self.assertIn("No steps remain", result.messages[-2]["content"])
        self.assertEqual(result.usage, {key: value * 3 for key, value in USAGE.items()})
        self.assertEqual(result.finish_reasons, {"tool_calls": 2, "stop": 1})
        self.assertEqual(load_trajectory(self.path), result)

    async def test_unknown_answer_call_can_finish_in_the_final_request(self):
        with patch.object(agent, "call_model", new_callable=AsyncMock) as call:
            call.side_effect = [
                _reply("", calls=[_tool_call("answer", '{"answer":"cloak"}')], finish="tool_calls"),
                _reply('{"answer":"cloak"}'),
            ]
            result = await run_episode(replace(self.task, max_turns=1), self.ctx, self.tools, out=self.path)
        self.assertEqual(result.messages[3]["content"], "Unknown tool 'answer'.")
        self.assertEqual((result.final_answer, result.turns, result.error), ('{"answer":"cloak"}', 2, None))
        self.assertIsNone(call.await_args_list[-1].args[2])
        self.connection.call.assert_not_awaited()

    async def test_empty_reply_consumes_a_turn_and_retries_within_budget(self):
        with patch.object(agent, "call_model", new_callable=AsyncMock) as call:
            call.side_effect = [_reply(""), _reply("answer")]
            result = await run_episode(self.task, self.ctx, ToolBox(), out=self.path)
        self.assertEqual(result.turns, 2)
        self.assertIn("empty", result.messages[3]["content"])
        self.assertEqual(result.final_answer, "answer")

    async def test_empty_final_response_stops_after_one_final_request(self):
        with patch.object(agent, "call_model", new_callable=AsyncMock, return_value=_reply("")) as call:
            result = await run_episode(replace(self.task, max_turns=1), self.ctx, ToolBox(), out=self.path)
        self.assertEqual(call.await_count, 2)
        self.assertEqual(result.turns, 2)
        self.assertEqual(result.episode_status, "max_turns")
        self.assertIsNone(result.final_answer)
        self.assertIsNone(call.await_args_list[-1].args[2])

    async def test_final_request_never_executes_tool_calls(self):
        search = _reply("", calls=[_tool_call("search", '{"query":"q"}')], finish="tool_calls")
        with patch.object(agent, "call_model", new_callable=AsyncMock, side_effect=[search, search]) as call:
            result = await run_episode(replace(self.task, max_turns=1), self.ctx, self.tools, out=self.path)
        self.assertEqual((call.await_count, self.connection.call.await_count), (2, 1))
        self.assertEqual(result.messages[-1]["tool_calls"], search.tool_calls)
        self.assertEqual(result.episode_status, "max_turns")
        self.assertIsNone(result.final_answer)

    async def test_final_request_failure_preserves_partial_output_and_usage(self):
        partial = _reply("partial")
        partial.error = RuntimeError("final request failed")
        with patch.object(agent, "call_model", new_callable=AsyncMock, side_effect=[_reply(""), partial]) as call:
            result = await run_episode(replace(self.task, max_turns=1), self.ctx, ToolBox(), out=self.path)
        self.assertEqual(call.await_count, 2)
        self.assertEqual(result.error, "RuntimeError: final request failed")
        self.assertEqual(result.episode_status, "error")
        self.assertEqual(result.messages[-1]["content"], "partial")
        self.assertTrue(result.messages[-1]["incomplete"])
        self.assertEqual(result.usage, {key: value * 2 for key, value in USAGE.items()})
        self.assertEqual(load_trajectory(self.path), result)

    async def test_malformed_arguments_are_preserved_and_reported_without_executing(self):
        for arguments in ('{"query":', "[]", "42"):
            with self.subTest(arguments=arguments), patch.object(agent, "call_model", new_callable=AsyncMock) as call:
                call.side_effect = [_reply("", calls=[_tool_call("search", arguments)]), _reply("answer")]
                result = await run_episode(self.task, self.ctx, self.tools, out=self.path)
                self.assertEqual(result.messages[2]["tool_calls"][0]["function"]["arguments"], arguments)
                self.assertTrue(result.messages[3]["is_error"])
                self.assertIn("JSON", result.messages[3]["content"])
                self.assertEqual(result.final_answer, "answer")
        self.connection.call.assert_not_awaited()

    async def test_unknown_tool_is_an_observation_and_the_loop_continues(self):
        with patch.object(agent, "call_model", new_callable=AsyncMock) as call:
            call.side_effect = [_reply("", calls=[_tool_call("missing", "{}")]), _reply("answer")]
            result = await run_episode(self.task, self.ctx, self.tools, out=self.path)
        self.assertEqual(result.messages[3]["content"], "Unknown tool 'missing'.")
        self.assertEqual(result.final_answer, "answer")
        self.connection.call.assert_not_awaited()

    async def test_multiple_calls_keep_order_and_recoverable_errors(self):
        self.connection.call.side_effect = [ToolResponse("unavailable", is_error=True), ToolResponse("page")]
        calls = [_tool_call("search", '{"query":"first"}', "one"), _tool_call("search", '{"query":"second"}', "two")]
        with patch.object(agent, "call_model", new_callable=AsyncMock) as call:
            call.side_effect = [_reply("", calls=calls), _reply("answer")]
            result = await run_episode(self.task, self.ctx, self.tools, out=self.path)
        observations = [message for message in result.messages if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in observations], ["one", "two"])
        self.assertEqual([message["is_error"] for message in observations], [True, False])
        self.assertIsNone(result.error)

    async def test_fatal_tool_error_is_saved_and_stops_before_another_model_request(self):
        self.connection.call.return_value = ToolResponse("", is_error=True, fatal="HTTP 402: balance")
        with patch.object(agent, "call_model", new_callable=AsyncMock) as call:
            call.return_value = _reply("", calls=[_tool_call("search", '{"query":"q"}')])
            result = await run_episode(self.task, self.ctx, self.tools, out=self.path)
        call.assert_awaited_once()
        self.assertIn("HTTP 402", result.error or "")
        self.assertEqual(result.episode_status, "error")
        self.assertEqual(result.messages[-1]["content"], "")
        self.assertIn("402", result.messages[-1]["error"])
        self.assertEqual(load_trajectory(self.path), result)

    async def test_environment_interface_failure_is_persisted(self):

        class BrokenEnvironment(ToolBox):
            @property
            def schemas(self):
                raise RuntimeError("discovery failed")

            async def call_tool(self, name, arguments):
                raise AssertionError("unreachable")

        result = await run_episode(self.task, self.ctx, BrokenEnvironment(), out=self.path)
        self.assertEqual(result.error, "RuntimeError: discovery failed")
        self.assertEqual(result.turns, 0)
        self.assertEqual(load_trajectory(self.path), result)

    async def test_task_timeout_includes_waiting_for_the_model_semaphore(self):
        ctx = replace(self.ctx, semaphore=asyncio.Semaphore(0))
        result = await run_episode(replace(self.task, timeout=0.01), ctx, ToolBox(), out=self.path)
        self.assertEqual(result.episode_status, "timeout")
        self.assertIn("task_timeout", result.error or "")
        self.assertTrue(result.messages[-1]["incomplete"])
        self.client.chat.completions.create.assert_not_called()

    async def test_task_timeout_during_a_tool_call_saves_the_attempt(self):

        async def wait_forever(*args):
            await asyncio.Event().wait()

        self.connection.call.side_effect = wait_forever
        with patch.object(agent, "call_model", new_callable=AsyncMock) as call:
            call.return_value = _reply("", calls=[_tool_call("search", '{"query":"q"}')])
            result = await run_episode(replace(self.task, timeout=0.01), self.ctx, self.tools, out=self.path)
        self.assertEqual(result.episode_status, "timeout")
        self.assertEqual(result.messages[-1]["role"], "tool")
        self.assertIn("CancelledError", result.messages[-1]["error"])
        self.assertGreaterEqual(result.messages[-1]["latency_sec"], 0.005)
        self.assertEqual(load_trajectory(self.path), result)

    async def test_final_request_is_within_the_episode_timeout(self):
        entered = asyncio.Event()

        async def reply(*args):
            if not entered.is_set():
                entered.set()
                return _reply("")
            await asyncio.Event().wait()

        with patch.object(agent, "call_model", new_callable=AsyncMock, side_effect=reply) as call:
            result = await run_episode(
                replace(self.task, max_turns=1, timeout=0.01), self.ctx, ToolBox(), out=self.path
            )
        self.assertEqual(call.await_count, 2)
        self.assertEqual(result.episode_status, "timeout")
        self.assertIn("task_timeout", result.error or "")
        self.assertEqual(load_trajectory(self.path), result)

    async def test_failed_stream_keeps_partial_content_reasoning_and_usage(self):
        stream = MagicMock(close=AsyncMock())

        async def chunks():
            yield _chunk("partial", "reason")
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=4, completion_tokens_details=SimpleNamespace(reasoning_tokens=2)
                ),
            )
            raise RuntimeError("stream failed")

        stream.__aiter__.side_effect = chunks
        self.ctx.client.chat.completions.create = AsyncMock(return_value=stream)
        result = await run_episode(self.task, self.ctx, ToolBox(), out=self.path)
        self.assertEqual(result.error, "RuntimeError: stream failed")
        self.assertEqual(result.messages[-1]["content"], "partial")
        self.assertEqual(result.messages[-1]["reasoning_content"], "reason")
        self.assertTrue(result.messages[-1]["incomplete"])
        self.assertEqual(result.usage, USAGE)
        self.assertEqual(load_trajectory(self.path), result)
        stream.close.assert_awaited_once()

    async def test_cancellation_propagates_after_saving_partial_stream(self):
        started = asyncio.Event()
        stream = MagicMock(close=AsyncMock())

        async def chunks():
            yield _chunk("partial", "reason")
            started.set()
            await asyncio.Event().wait()

        stream.__aiter__.side_effect = chunks
        self.ctx.client.chat.completions.create = AsyncMock(return_value=stream)
        future = asyncio.create_task(run_episode(self.task, self.ctx, ToolBox(), out=self.path))
        await started.wait()
        future.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await future
        saved = load_trajectory(self.path)
        self.assertEqual((saved.episode_status, saved.error), ("cancelled", "CancelledError"))
        self.assertEqual(saved.messages[-1]["content"], "partial")
        stream.close.assert_awaited_once()
        self.client.close.assert_not_awaited()

    async def test_storage_failure_propagates_to_eval(self):
        with (
            patch.object(agent, "call_model", new_callable=AsyncMock, return_value=_reply("answer")),
            patch.object(agent, "save_trajectory", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            await run_episode(self.task, self.ctx, ToolBox(), out=self.path)


class SchemaTests(unittest.TestCase):
    def test_rejects_invalid_task_budgets_and_empty_ids(self):
        cases: list[dict[str, Any]] = [{"max_turns": 0}, {"max_turns": -1}, {"timeout": -1}, {"id": ""}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                parameters: dict[str, Any] = {"id": "id", "system": "s", "prompt": "q"} | kwargs
                Task(**parameters)

    def test_atomic_storage_preserves_existing_file_when_publish_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.json"
            original = Trajectory("sample", final_answer="old")
            save_trajectory(original, path)
            with patch("eval.harness.utils.os.replace", side_effect=OSError("failed")), self.assertRaises(OSError):
                save_trajectory(Trajectory("sample", final_answer="new"), path)
            self.assertEqual(load_trajectory(path), original)
            self.assertEqual(list(Path(temporary).iterdir()), [path])


def _reply(content, *, calls=None, finish="stop"):
    return AssistantTurn(content, "reason", calls or [], dict(USAGE), finish)


def _tool_call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _chunk(content, reasoning):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=[])
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=None)])


if __name__ == "__main__":
    unittest.main()
