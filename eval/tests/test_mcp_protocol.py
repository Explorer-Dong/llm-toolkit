"""Audit every benchmark entrypoint through real MCP sessions and fake model turns."""

import asyncio
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mcp import ClientSession

from eval import agent
from eval.benchmarks._common.storage import PACKAGE_DIR
from eval.harness import tool as harness_tool
from eval.harness.env import ExecResult
from eval.harness.utils import load_trajectory
from eval.model.openai import AssistantTurn

__all__ = []
BENCHMARKS = {
    "aime",
    "gpqa",
    "math",
    "gaia",
    "browsecomp",
    "webwalkerqa",
    "swe_bench",
    "agent_abstain",
    "webshop_abstain",
}


class BenchmarkMCPTests(unittest.TestCase):
    def test_protocol_audit_covers_every_benchmark(self):
        directory = PACKAGE_DIR / "benchmarks"
        self.assertEqual({path.parent.name for path in directory.glob("*/rollout.py")}, BENCHMARKS)

    def test_question_answering_entrypoints_expose_no_tools(self):
        for benchmark in ("aime", "gpqa", "math"):
            with self.subTest(benchmark=benchmark):
                trace, requests = _run_benchmark(self, benchmark, [])
                self.assertEqual(trace.tools, [])
                self.assertEqual(requests, [])

    def test_browsecomp_default_tools_use_mcp(self):
        _check_web_tools(self, "browsecomp")

    def test_gaia_default_tools_use_mcp(self):
        _check_web_tools(self, "gaia")

    def test_webwalkerqa_default_tools_use_mcp(self):
        _check_web_tools(self, "webwalkerqa")

    def test_swe_bench_default_shell_and_files_use_mcp(self):
        bench = importlib.import_module("eval.benchmarks.swe_bench.rollout")
        task = bench.load_tasks(bench.parse_args(["--model", "test", "--limit", "1"]))[0]
        with (
            patch.object(bench, "prepare", AsyncMock(return_value={task.raw["image"]: "fixture-image"})),
            patch.object(bench.SWEEnv, "setup", AsyncMock()),
            patch.object(bench.SWEEnv, "teardown", AsyncMock()),
            patch.object(bench.SWEEnv, "exec", AsyncMock(return_value=ExecResult("fixture patch", "", 0))),
            patch.object(bench.SWEEnv, "read_file", AsyncMock(return_value="fixture source")),
        ):
            trace, _ = _run_benchmark(
                self,
                "swe_bench",
                [_call("shell", {"cmd": "git status"}), _call("read_file", {"path": "source.py"})],
            )
        self.assertEqual(
            {tool["function"]["name"] for tool in trace.tools}, {"shell", "read_file", "write_file", "list_dir"}
        )

    def test_agent_abstain_custom_tool_uses_mcp_and_records_execution(self):
        bench = importlib.import_module("eval.benchmarks.agent_abstain.rollout")
        task = bench.load_tasks(bench.parse_args(["--model", "test", "--limit", "1"]))[0]
        step = task.raw["task_yaml"]["execution_dag"]["nodes"][0]
        trace, requests = _run_benchmark(self, "agent_abstain", [_call(step["tool"].replace(".", "_"), step["params"])])
        self.assertEqual(requests[-1]["params"]["name"], step["tool"])
        self.assertEqual(requests[-1]["params"]["arguments"], step["params"])
        self.assertTrue(trace.tools)

    def test_webshop_abstain_search_and_click_use_mcp_and_keep_purchase_receipts(self):
        bench = importlib.import_module("eval.benchmarks.webshop_abstain.rollout")
        shop = MagicMock(observation="Fixture products", session="fixture-session")
        shop.get_available_actions.return_value = {"has_search_bar": True, "clickables": ["Buy Now"]}
        shop.step.side_effect = [("Results", 0, False, {}), ("Purchased", 1, True, {})]

        async def setup(env):
            env.environment = shop

        with patch.object(bench.WebShopEnv, "setup", setup):
            trace, _ = _run_benchmark(
                self,
                "webshop_abstain",
                [_call("search", {"keywords": "fixture product"}), _call("click", {"value": "Buy Now"})],
                confirm_calls=[_call("click", {"value": "Buy Now"})],
            )
        self.assertEqual(trace.tools, bench.FIRST_TURN_TOOLS)
        self.assertEqual(
            [call.args for call in shop.step.call_args_list], [("search[fixture product]",), ("click[Buy Now]",)]
        )
        outputs = [message for message in trace.messages if message["role"] == "tool"]
        self.assertTrue(outputs[1].get("intercepted"))
        self.assertIn("Executed: True. Reward: 1. Done: True.", outputs[-1]["content"])
        shop.close.assert_called_once()


def _check_web_tools(test, benchmark):
    with patch.object(harness_tool, "_execute_tool", AsyncMock(return_value=("Fixture web content", False))):
        trace, _ = _run_benchmark(
            test, benchmark, [_call("search", {"query": "fixture"}), _call("fetch", {"url": "https://example.com"})]
        )
    test.assertEqual({tool["function"]["name"] for tool in trace.tools}, {"search", "fetch"})


def _run_benchmark(test, benchmark, calls, *, confirm_calls=()):
    bench = importlib.import_module(f"eval.benchmarks.{benchmark}.rollout")
    requests = []
    send_request = ClientSession.send_request

    async def record_request(session, request, *args, **kwargs):
        requests.append(request.root.model_dump(mode="json", by_alias=True, exclude_none=True))
        return await send_request(session, request, *args, **kwargs)

    client = MagicMock(close=AsyncMock())

    async def close_client(*_args):
        await client.close()

    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(side_effect=close_client)
    turns = [AssistantTurn(content="Completed", finish_reason="stop")]
    if confirm_calls:
        turns.insert(
            0,
            AssistantTurn(tool_calls=[dict(call, id=f"confirm_{index}") for index, call in enumerate(confirm_calls)]),
        )
    if calls:
        turns.insert(0, AssistantTurn(tool_calls=[dict(call, id=f"call_{index}") for index, call in enumerate(calls)]))
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.object(ClientSession, "send_request", record_request),
        patch.object(bench, "make_client", return_value=client),
        patch.object(agent, "call_model", AsyncMock(side_effect=turns)) as model,
    ):
        output = Path(directory) / "run"
        run_args = bench.parse_args(["--model", "test", "--limit", "1", "--output-dir", str(output)])
        asyncio.run(bench.rollout(run_args, bench.load_tasks(run_args)))
        trace = load_trajectory(next((output / "trajectories").glob("*.json")))
        test.assertIsNone(trace.error)
        expected_turns = 1 + int(bool(calls)) + int(bool(confirm_calls))
        test.assertEqual(model.await_count, expected_turns)
        if benchmark == "agent_abstain":
            evidence = json.loads(next((output / "environment_outputs").glob("*/environment_result.json")).read_text())
            test.assertEqual(len(evidence["data"]["execution_log"]), len(calls))
            test.assertTrue(all(entry["success"] for entry in evidence["data"]["execution_log"]))
    client.close.assert_awaited_once()
    test.assertEqual(
        [request["method"] for request in requests],
        ["initialize", "tools/list", *["tools/call" for _ in calls]] if calls else [],
    )
    outputs = [message for message in trace.messages if message["role"] == "tool" and not message.get("intercepted")]
    test.assertEqual(len(outputs), len(calls))
    for output in outputs:
        test.assertFalse(output["is_error"])
        test.assertEqual(output["tool_result"]["content"][0]["text"], output["content"])
    return trace, requests


def _call(name, arguments):
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


if __name__ == "__main__":
    unittest.main()
