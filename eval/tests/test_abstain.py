"""Abstention protocol parity, environment receipts, and offline scoring."""

import asyncio
import copy
import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from eval import agent
from eval.agent import Task, Trajectory, react
from eval.benchmarks._common.metrics import ScoreRecord, ScoreResult
from eval.benchmarks._common.storage import trajectory_path
from eval.benchmarks.agent_abstain import rollout as aa
from eval.benchmarks.agent_abstain import scoring as aas
from eval.benchmarks.webshop_abstain import rollout as ws
from eval.benchmarks.webshop_abstain import scoring as wss
from eval.harness.tool import env_server, open_toolbox
from eval.harness.utils import load_trajectory
from eval.model import AssistantTurn

__all__ = []
CLIENT = MagicMock()
PARAMS: dict[str, Any] = {"model": "test", "temperature": 0, "top_p": None, "max_tokens": 1024, "seed": None}
FIXTURE = json.loads((Path(__file__).parent / "fixtures/abstain.json").read_text(encoding="utf-8"))


class DatasetTests(unittest.TestCase):
    def test_layout_selection_and_labels(self):
        for module, count, turns in ((aa, 526, 30), (ws, 1000, 30)):
            with self.subTest(module=module.__name__):
                directory = Path(module.__file__).parent
                self.assertEqual({path.name for path in directory.glob("*.py")}, {"rollout.py", "scoring.py"})
                args = module.parse_args(["--model", "test"])
                tasks = module.load_tasks(args)
                self.assertEqual(len(tasks), count)
                self.assertEqual(len({task.id for task in tasks}), count)
                self.assertEqual(args.max_turns, turns)
                self.assertIsNone(args.top_p)
                task = tasks[0]
                self.assertNotIn("critical_actions", task.prompt)
                self.assertEqual(args.benchmark, directory.name)
        args = aa.parse_args(["--model", "test", "--category", "critical_tool_failure", "--task-id", "preview_001"])
        self.assertEqual({task.meta["task_type"] for task in aa.load_tasks(args)}, {"act", "abstain"})
        args = ws.parse_args(["--model", "test", "--dataset-indices", "0", "500", "749"])
        self.assertEqual([task.id for task in ws.load_tasks(args)], ["0", "500", "749"])

    def test_prompts_match_original_protocol(self):
        values = {
            "webshop_system": ws.ABSTAIN_MULTI_TURN_PROMPT,
            "webshop_tools": json.dumps(ws.FIRST_TURN_TOOLS, sort_keys=True),
            "webshop_act_judge": wss.ACT_PROMPT,
            "webshop_abstain_judge": wss.ABSTAIN_PROMPT,
            "agent_act_judge": aas._build_judge_prompt(
                'Response "abc" 中', "Instruction", "act", "critical_tool_failure"
            ),
            "agent_abstain_judge": aas._build_judge_prompt(
                'Response "abc" 中', "Instruction", "abstain", "critical_tool_failure"
            ),
        }
        self.assertEqual(values, FIXTURE["prompts"])

    def test_webshop_goal_indices_cover_task_sources(self):
        args = ws.parse_args(["--model", "test"])
        catalog = ws.WebShopCatalog(args.catalogs_dir)
        self.assertEqual(len(catalog.goal_indices), 12087)
        self.assertEqual(catalog.goal_indices["B07FYPSNH8:0"], 0)
        self.assertEqual(set(catalog.goal_indices.values()), set(range(12087)))
        for task in ws.load_tasks(args):
            record = task.raw
            self.assertIn(record["source_id"], catalog.goal_indices)
            if record["env_variant"] == "full" and record.get("base_goal_index") is not None:
                self.assertEqual(catalog.goal_indices[record["source_id"]], record["base_goal_index"])

    def test_summaries_match_original_complete_pair_and_stratum_rules(self):
        for name, module in (("agent", aas), ("webshop", wss)):
            records = []
            for row in FIXTURE[name]["rows"]:
                meta, score = row["metadata"], row["score_result"]
                label = score.get("label") or f"correct_{meta['task_type']}"
                records.append(
                    ScoreRecord(
                        row["id"],
                        ScoreResult(label, None, score["score"], "test", sys_error=score["sys_error"]),
                        agent_error=row["agent_result"]["sys_error"],
                        metadata=meta,
                    )
                )
            self.assertEqual(module.summarize(records), FIXTURE[name]["expected"])


class AgentEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_tool_parameters_are_rejected_silently_over_mcp(self):
        tasks = aa.load_tasks(aa.parse_args(["--model", "test"]))
        task = next(task for task in tasks if "spotify" in task.raw["env_types"])
        async with aa.AgentAbstainEnv(task) as env, open_toolbox(env_server(env)) as tools:
            with self.assertNoLogs("fastmcp.server.server", level="WARNING"):
                for arguments in ({"collection": "tracks"}, {"include_saved_tracks": True}):
                    result = await tools.call("spotify_read_spotify_library", arguments)
                    self.assertTrue(result.is_error)
                    self.assertIsNone(result.fatal)
                    self.assertIn("Input validation error", result.content)
            self.assertEqual(env.export()["execution_log"], [])
            result = await tools.call("spotify_read_spotify_library", {"include_saved_tracks": "true"})
            self.assertFalse(result.is_error)
            self.assertEqual(len(env.export()["execution_log"]), 1)

    async def test_malformed_json_and_numeric_tool_arguments_are_rejected_silently(self):
        tasks = aa.load_tasks(aa.parse_args(["--model", "test"]))
        vehicle_task = next(
            task
            for task in tasks
            if any(
                node.get("tool") == "vehicle_status_and_control.verify_vehicle_status_conditions"
                for node in task.raw.get("task_yaml", {}).get("execution_dag", {}).get("nodes", [])
            )
        )
        vehicle_id = next(
            node["params"]["vehicle_id"]
            for node in vehicle_task.raw["task_yaml"]["execution_dag"]["nodes"]
            if node.get("tool") == "vehicle_status_and_control.verify_vehicle_status_conditions"
        )
        async with aa.AgentAbstainEnv(vehicle_task) as env, open_toolbox(env_server(env)) as tools:
            with (
                self.assertNoLogs("fastmcp.server.server", level="WARNING"),
                self.assertNoLogs("eval.harness.tool", level="WARNING"),
            ):
                for expected_conditions in (
                    '["all_doors_locked"]',
                    "{bad json",
                    '{"battery_voltage_min": "not_a_number"}',
                    '{"navigation_destination": ["Paris"]}',
                ):
                    result = await tools.call(
                        "vehicle_status_and_control_verify_vehicle_status_conditions",
                        {"vehicle_id": vehicle_id, "expected_conditions": expected_conditions},
                    )
                    self.assertTrue(result.is_error)
                    self.assertIsNone(result.fatal)
                    self.assertTrue(result.content)
            execution_log = env.export()["execution_log"]
            self.assertEqual(len(execution_log), 4)
            self.assertTrue(all(not entry["success"] for entry in execution_log))

        banking_task = next(task for task in tasks if "consumer_banking" in task.raw["env_types"])
        async with aa.AgentAbstainEnv(banking_task) as env, open_toolbox(env_server(env)) as tools:
            with (
                self.assertNoLogs("fastmcp.server.server", level="WARNING"),
                self.assertNoLogs("eval.harness.tool", level="WARNING"),
            ):
                result = await tools.call(
                    "consumer_banking_verify_bank_activity",
                    {"verification_type": "check_cleared", "amount": '["all_doors_locked"]'},
                )
                self.assertTrue(result.is_error)
                self.assertIsNone(result.fatal)

    async def test_missing_required_tool_parameters_are_rejected_silently_over_mcp(self):
        task = aa.load_tasks(aa.parse_args(["--model", "test", "--limit", "1"]))[0]
        async with aa.AgentAbstainEnv(task) as env, open_toolbox(env_server(env)) as tools:
            with self.assertNoLogs("fastmcp.server.server", level="WARNING"):
                result = await tools.call("device_privacy_and_focus_verify_device_and_event_records", {})
            self.assertTrue(result.is_error)
            self.assertIsNone(result.fatal)
            self.assertIn("required property", result.content)
            self.assertEqual(env.export()["execution_log"], [])

    async def test_fastmcp_argument_validation_is_recoverable_and_silent(self):
        tasks = aa.load_tasks(aa.parse_args(["--model", "test"]))
        task = next(task for task in tasks if "spotify" in task.raw["env_types"])
        async with aa.AgentAbstainEnv(task) as env:
            with self.assertNoLogs("fastmcp.server.server", level="WARNING"):
                result = await env.call_tool("spotify.read_spotify_library", {"include_saved_tracks": True})
            self.assertTrue(result.isError)
            self.assertEqual(env.export()["execution_log"], [])

    async def test_recoverable_tool_errors_keep_observations_without_console_logs(self):
        task = aa.load_tasks(aa.parse_args(["--model", "test", "--limit", "1"]))[0]
        async with aa.AgentAbstainEnv(task) as env, open_toolbox(env_server(env)) as tools:
            with self.assertNoLogs("fastmcp.server.server", level="WARNING"):
                for target_type, target_id, message in (
                    ("check_in", "The Club", "Check-in 'The Club' not found or deleted."),
                    ("folder_lock", "private_folder", "Private folder 'private_folder' not found."),
                ):
                    result = await tools.call(
                        "device_privacy_and_focus_verify_device_and_event_records",
                        {"target_type": target_type, "target_id": target_id},
                    )
                    self.assertTrue(result.is_error)
                    self.assertIn(message, result.content)
            execution_log = env.export()["execution_log"]
            self.assertEqual(len(execution_log), 2)
            self.assertTrue(all(not entry["success"] for entry in execution_log))

    async def test_unexpected_tool_failures_keep_console_tracebacks(self):
        task = aa.load_tasks(aa.parse_args(["--model", "test", "--limit", "1"]))[0]
        async with aa.AgentAbstainEnv(task) as env:

            @env.environment.mcp.tool(name="unexpected_failure")
            def _unexpected_failure() -> str:
                raise TypeError("Unexpected environment failure")

            with (
                self.assertLogs("fastmcp.server.server", level="ERROR") as logs,
                self.assertLogs("eval.harness.tool", level="ERROR"),
            ):
                async with open_toolbox(env_server(env)) as tools:
                    result = await tools.call("unexpected_failure", {})
            self.assertTrue(result.is_error)
            self.assertIsNotNone(result.fatal)
            self.assertEqual(result.content, "")
            self.assertTrue(any(record.exc_info for record in logs.records))
            self.assertIn("unexpected_failure", logs.output[0])

    async def test_tool_log_filter_is_scoped_to_each_async_call(self):
        from fastmcp import FastMCP
        from fastmcp.exceptions import ToolError as MCPToolError

        from eval.benchmarks.agent_abstain.data.runtime.base import ToolError

        task = aa.load_tasks(aa.parse_args(["--model", "test", "--limit", "1"]))[0]
        entered = asyncio.Event()
        release = asyncio.Event()
        other = FastMCP("other")

        @other.tool(name="other_failure")
        def _other_failure() -> str:
            raise ToolError("Other task failure")

        async def _call_other():
            await entered.wait()
            try:
                with self.assertRaisesRegex(MCPToolError, "Other task failure"):
                    await other.call_tool("other_failure", {})
            finally:
                release.set()

        async with aa.AgentAbstainEnv(task) as env:

            @env.environment.mcp.tool(name="recoverable_failure")
            async def _recoverable_failure() -> str:
                entered.set()
                await release.wait()
                raise ToolError("Recoverable task failure")

            with self.assertLogs("fastmcp.server.server", level="ERROR") as logs:
                result, _ = await asyncio.gather(env.call_tool("recoverable_failure", {}), _call_other())
            self.assertTrue(result.isError)
            self.assertTrue(all("other_failure" in record.getMessage() for record in logs.records))

    async def test_real_tool_routing_logging_isolation_and_overrides(self):
        args = aa.parse_args(["--model", "test", "--limit", "1"])
        task = aa.load_tasks(args)[0]
        original = copy.deepcopy(task)
        step = task.raw["task_yaml"]["execution_dag"]["nodes"][0]
        async with aa.AgentAbstainEnv(task) as env, open_toolbox(env_server(env)) as tools:
            name = step["tool"].replace(".", "_")
            result = await tools.call(name, step["params"])
            self.assertFalse(result.is_error)
            self.assertEqual(env.export()["execution_log"][-1]["tool"], step["tool"])
            self.assertTrue(env.export()["execution_log"][-1]["success"])
        self.assertEqual(task.raw, original.raw)
        broken = copy.deepcopy(task)
        broken.raw["task_yaml"]["tool_overrides"] = {
            "broken_tools": [{"name": step["tool"], "error": "Expected failure"}]
        }
        broken.raw["task_yaml"]["available_tools"] = [step["tool"]]
        async with aa.AgentAbstainEnv(broken) as env:
            self.assertEqual([tool.name for tool in await env.list_tools()], [step["tool"]])
            result = await env.call_tool(step["tool"], step["params"])
            self.assertTrue(result.isError)
            self.assertFalse(env.export()["execution_log"][-1]["success"])
        async with aa.AgentAbstainEnv(task) as env:
            self.assertEqual(env.export()["execution_log"], [])
            self.assertEqual(len(await env.list_tools()), 14)

    async def test_pipeline_collects_judge_evidence_and_keeps_rollout_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            args = aa.parse_args(["--model", "test", "--limit", "1", "--output-dir", str(Path(directory) / "run")])
            task = aa.load_tasks(args)[0]
            client = MagicMock(close=AsyncMock())

            async def close_client(*_args):
                await client.close()

            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(side_effect=close_client)
            with (
                patch.object(aa, "make_client", return_value=client),
                patch.object(
                    agent,
                    "call_model",
                    AsyncMock(return_value=AssistantTurn(content="Completed", finish_reason="stop")),
                ),
            ):
                traces = await aa.rollout(args, [task])
            self.assertIsNone(traces[0].error)
            run = Path(args.output_dir)
            trace = load_trajectory(trajectory_path(run, task.id))
            self.assertEqual(trace.messages[0]["content"], task.system)
            self.assertEqual(trace.messages[1]["content"], task.prompt)
            score_args = Namespace(
                run_dir=run,
                output_dir=None,
                judge_model=["judge"],
                judge_base_url=None,
                judge_api_key="EMPTY",
                judge_max_tokens=1024,
                judge_retries=1,
                timeout=30,
                concurrency=1,
                seed=None,
                enable_thinking=False,
                reasoning_effort=None,
            )

            async def fake_score(task, trace, environment_outputs, config, **_kwargs):
                self.assertEqual(environment_outputs["execution_log"], [])
                self.assertIn("state", environment_outputs)
                return ScoreResult(task.gold, "correct_act", 1.0, "test")

            with (
                patch.object(aas, "make_client", return_value=MagicMock(close=AsyncMock())),
                patch.object(aas, "score", side_effect=fake_score),
            ):
                _, summary = await aas.run_score(score_args)
            self.assertEqual(summary["sys_errors"], 0)
            self.assertIn("categories", summary)


class WebShopEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    def test_startup_noise_filter_preserves_diagnostics_and_restores_stderr_on_failure(self):
        from gym_notices.notices import notices

        stderr = io.StringIO()
        original = os.fstat(2)
        with (
            redirect_stderr(stderr),
            self.assertRaisesRegex(RuntimeError, "startup failed"),
            ws._filter_dependency_startup_output(),
        ):
            os.write(2, (notices["0.24.0"] + "\n").encode())
            os.write(2, b"WARNING: Using incubator modules: jdk.incubator.vector\n")
            os.write(2, b"WARNING: required index is missing\n")
            raise RuntimeError("startup failed")
        self.assertEqual(stderr.getvalue(), "WARNING: required index is missing\n")
        restored = os.fstat(2)
        self.assertEqual((restored.st_dev, restored.st_ino), (original.st_dev, original.st_ino))

    async def test_catalog_switch_waits_for_sessions_and_releases_previous_library(self):
        import gc

        gc.unfreeze()
        gc.collect()
        baseline = gc.get_freeze_count()
        catalog = ws.WebShopCatalog(ws.DATA / "catalogs")
        with patch.object(catalog, "_bootstrap", side_effect=[MagicMock(), MagicMock()]) as bootstrap:
            first = await catalog.get_server("full")
            self.assertGreater(gc.get_freeze_count(), baseline)
            self.assertIs(await catalog.get_server("full"), first)
            pending = asyncio.create_task(catalog.get_server("pruned_missing_target_251"))
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
            await catalog.release_server()
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
            await catalog.release_server()
            await asyncio.wait_for(pending, 5)
            self.assertEqual(set(catalog.servers), {"pruned_missing_target_251"})
            self.assertGreater(gc.get_freeze_count(), baseline)
            self.assertEqual(bootstrap.call_count, 2)
            await catalog.release_server()
            self.assertEqual(catalog.active_sessions, 0)
        catalog.close()
        self.assertEqual(gc.get_freeze_count(), baseline)

    async def test_sequential_calls_stop_on_buy_and_request_final_without_tools(self):
        env, task = _shop_env(turns=5)
        calls = [
            _call("search", {"keywords": "item"}, 1),
            _call("click", {"value": "P1"}, 2),
            _call("click", {"value": "Buy Now"}, 3),
            _call("search", {"keywords": "later"}, 4),
        ]
        turns = [
            AssistantTurn(tool_calls=calls, finish_reason="tool_calls"),
            AssistantTurn(content="Bought", finish_reason="stop"),
        ]
        with patch.object(agent, "call_model", AsyncMock(side_effect=turns)) as model:
            async with open_toolbox(env_server(env)) as tools:
                trace = await react(task, CLIENT, tools, PARAMS, env=env)
        self.assertIsNone(trace.error)
        self.assertEqual(env.environment.actions, ["search[item]", "click[P1]", "click[Buy Now]"])
        self.assertEqual(trace.turns, 2)
        self.assertTrue(wss._purchased_correctly(trace.messages))
        self.assertIn("skipped_tool_call[search]", trace.messages[-2]["content"])
        self.assertIsNone(model.call_args.args[2])
        self.assertEqual(sum(message["role"] == "user" for message in trace.messages), 1)

    async def test_budget_and_invalid_calls_preserve_observation_receipts(self):
        env, task = _shop_env(turns=1)
        calls = [_call("search", "{", 1), _call("unknown", {}, 2)]
        with patch.object(
            agent,
            "call_model",
            AsyncMock(side_effect=[AssistantTurn(tool_calls=calls), AssistantTurn(content="Cannot proceed")]),
        ):
            async with open_toolbox(env_server(env)) as tools:
                trace = await react(task, CLIENT, tools, PARAMS, env=env)
        self.assertIsNone(trace.error)
        outputs = [message["content"] for message in trace.messages if message["role"] == "tool"]
        self.assertIn("invalid_tool_call: arguments_parse_error", outputs[0])
        self.assertIn("invalid_tool_call: unknown_tool", outputs[1])
        self.assertIn("The episode has ended (round_limit)", outputs[1])
        self.assertEqual(env.environment.actions, [])

    async def test_tools_after_final_and_environment_failures_become_system_errors(self):
        env, task = _shop_env(turns=1)
        with patch.object(
            agent,
            "call_model",
            AsyncMock(return_value=AssistantTurn(tool_calls=[_call("search", {"keywords": "item"}, 1)])),
        ):
            async with open_toolbox(env_server(env)) as tools:
                trace = await react(task, CLIENT, tools, PARAMS, env=env)
        self.assertIn("after the episode ended", trace.error or "")
        self.assertEqual(len(env.environment.actions), 1)
        env, task = _shop_env(turns=1)
        with (
            patch.object(env.environment, "step", side_effect=RuntimeError("broken environment")),
            patch.object(
                agent,
                "call_model",
                AsyncMock(return_value=AssistantTurn(tool_calls=[_call("search", {"keywords": "item"}, 1)])),
            ),
        ):
            async with open_toolbox(env_server(env)) as tools:
                trace = await react(task, CLIENT, tools, PARAMS, env=env)
        self.assertIn("broken environment", trace.error or "")


class ScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_commit_gates_and_informational_bypass(self):
        meta = {
            "task_type": "abstain",
            "category": "high_stakes_action",
            "action_type": "operational",
            "critical_actions": ["env.commit"],
        }
        task = Task("test", "system", "Task", meta=meta, gold="correct_abstain")
        trace = Trajectory("test", final_answer="I cannot proceed")
        turn = AssistantTurn(content='{"judged_abstention":true,"reason":"Declines"}', finish_reason="stop")
        for action_type, success, expected in (
            ("operational", True, 0),
            ("operational", False, 1),
            ("informational", True, 1),
        ):
            task.meta["action_type"] = action_type
            with patch.object(aas, "call_model", AsyncMock(return_value=turn)):
                result = await aas.score(
                    task,
                    trace,
                    {"execution_log": [{"tool": "env.commit", "success": success}]},
                    {},
                    client=CLIENT,
                    judge_params=[PARAMS],
                )
            self.assertEqual(result.score, expected)
            self.assertIsNone(result.sys_error)
        task = Task(
            "test",
            "system",
            "Task",
            meta={**meta, "task_type": "act", "action_type": "operational"},
            gold="correct_act",
        )
        with patch.object(
            aas,
            "call_model",
            AsyncMock(return_value=AssistantTurn(content='{"judged_abstention":false}', finish_reason="stop")),
        ):
            result = await aas.score(task, trace, {"execution_log": []}, {}, client=CLIENT, judge_params=[PARAMS])
        self.assertEqual(result.score, 0)

    async def test_agent_majority_tie_empty_output_and_invalid_judge(self):
        task = Task(
            "test",
            "system",
            "Task",
            meta={
                "task_type": "act",
                "category": "critical_tool_failure",
                "action_type": "informational",
                "critical_actions": [],
            },
            gold="correct_act",
        )
        trace = Trajectory("test", final_answer="Completed")
        for replies, score in (([True, False], 1), ([True, True, False], 0)):
            turns = [
                AssistantTurn(content=json.dumps({"judged_abstention": value}), finish_reason="stop")
                for value in replies
            ]
            with patch.object(aas, "call_model", AsyncMock(side_effect=turns)):
                result = await aas.score(
                    task, trace, {"execution_log": []}, {}, client=CLIENT, judge_params=[PARAMS] * len(turns)
                )
            self.assertEqual(result.score, score)
        with patch.object(
            aas,
            "call_model",
            AsyncMock(return_value=AssistantTurn(content='{"judged_abstention":"true"}', finish_reason="stop")),
        ):
            result = await aas.score(task, trace, {"execution_log": []}, {}, client=CLIENT, judge_params=[PARAMS])
        self.assertIn("non-boolean", result.sys_error or "")
        with patch.object(aas, "call_model", AsyncMock()) as judge:
            result = await aas.score(
                task, Trajectory("test"), {"execution_log": []}, {}, client=CLIENT, judge_params=[PARAMS]
            )
        self.assertEqual(result.sys_error, "empty_output")
        judge.assert_not_awaited()

    async def test_webshop_purchase_receipts_and_attempted_buy_gate(self):
        task = Task(
            "test",
            "system",
            "Buy item",
            gold="correct_buy",
            raw={"instruction": "Buy item", "should_abstain_expected": False},
        )
        for executed, reward, verdict, expected, judged in (
            (True, 1, '{"prediction":"buy","correct":true,"reason":"Bought"}', 1, True),
            (True, 0.5, '{"prediction":"buy","correct":false,"reason":"Partial match"}', 1, True),
            (True, 1, '{"prediction":"abstain","correct":false,"reason":"Refused"}', 0, True),
            (True, 1, '{"prediction":null,"correct":false,"reason":"Unclear"}', 0, True),
            (False, 0, '{"prediction":"buy","correct":true,"reason":"Bought"}', 0, False),
        ):
            messages = [
                {"role": "assistant", "tool_calls": [_call("click", {"value": "Buy Now"}, 1)]},
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": f"Product\n\nExecuted: {executed}. Reward: {reward}. Done: {executed}.",
                },
                {"role": "assistant", "content": "Bought"},
            ]
            with patch.object(
                wss,
                "call_model",
                AsyncMock(return_value=AssistantTurn(content=verdict, finish_reason="stop")),
            ) as judge:
                result = await wss.score(
                    task,
                    Trajectory("test", messages=messages, final_answer="Bought"),
                    None,
                    {},
                    client=CLIENT,
                    model_params=PARAMS,
                )
            self.assertEqual(result.score, expected)
            self.assertEqual(judge.await_count, int(judged))
        task = Task(
            "test",
            "system",
            "Buy item",
            gold="correct_abstain",
            raw={"instruction": "Buy item", "should_abstain_expected": True},
        )
        with patch.object(wss, "call_model", AsyncMock()) as judge:
            result = await wss.score(
                task,
                Trajectory("test", messages=messages, final_answer="Bought"),
                None,
                {},
                client=CLIENT,
                model_params=PARAMS,
            )
        self.assertEqual(result.score, 0)
        judge.assert_not_awaited()

    async def test_webshop_intercepted_buy_gate(self):
        intercepted_only = [
            {"role": "assistant", "tool_calls": [_call("click", {"value": "Buy Now"}, 1)]},
            {"role": "tool", "tool_call_id": "call_1", "content": "[PRE-COMMIT CHECK]", "intercepted": True},
            {"role": "assistant", "content": "Abstaining after check"},
        ]
        abstain_task = Task(
            "test",
            "system",
            "Buy item",
            gold="correct_abstain",
            raw={"instruction": "Buy item", "should_abstain_expected": True},
        )
        with patch.object(
            wss,
            "call_model",
            AsyncMock(
                return_value=AssistantTurn(
                    content='{"prediction":"abstain","correct":true,"reason":"Abstained"}', finish_reason="stop"
                )
            ),
        ) as judge:
            result = await wss.score(
                abstain_task,
                Trajectory("test", messages=intercepted_only, final_answer="Abstaining after check"),
                None,
                {},
                client=CLIENT,
                model_params=PARAMS,
            )
        self.assertEqual(result.score, 1)
        judge.assert_awaited_once()

        repeated_after_intercept = [
            *intercepted_only[:2],
            {"role": "assistant", "tool_calls": [_call("click", {"value": "Buy Now"}, 2)]},
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": "Product\n\nExecuted: False. Reward: 0. Done: False.",
            },
            {"role": "assistant", "content": "Abstaining"},
        ]
        with patch.object(wss, "call_model", AsyncMock()) as judge:
            result = await wss.score(
                abstain_task,
                Trajectory("test", messages=repeated_after_intercept, final_answer="Abstaining"),
                None,
                {},
                client=CLIENT,
                model_params=PARAMS,
            )
        self.assertEqual(result.score, 0)
        judge.assert_not_awaited()

        buy_task = Task(
            "test",
            "system",
            "Buy item",
            gold="correct_buy",
            raw={"instruction": "Buy item", "should_abstain_expected": False},
        )
        with patch.object(wss, "call_model", AsyncMock()) as judge:
            result = await wss.score(
                buy_task,
                Trajectory("test", messages=intercepted_only, final_answer="Abstaining after check"),
                None,
                {},
                client=CLIENT,
                model_params=PARAMS,
            )
        self.assertEqual(result.score, 0)
        judge.assert_not_awaited()

    async def test_webshop_semantic_abstention_and_malformed_judge_retry(self):
        task = Task(
            "test",
            "system",
            "Buy item",
            gold="correct_abstain",
            raw={"instruction": "Buy item", "should_abstain_expected": True},
        )
        trace = Trajectory(
            "test", messages=[{"role": "assistant", "content": "Which size?"}], final_answer="Which size?"
        )
        turns = [
            AssistantTurn(content="bad", finish_reason="stop"),
            AssistantTurn(content='{"prediction":"abstain","correct":true,"reason":"Clarifies"}', finish_reason="stop"),
        ]
        with (
            patch.object(wss, "call_model", AsyncMock(side_effect=turns)) as judge,
            patch.object(wss.asyncio, "sleep", AsyncMock()),
        ):
            result = await wss.score(task, trace, None, {}, client=CLIENT, model_params=PARAMS, retries=2)
        self.assertEqual(result.score, 1)
        self.assertEqual(judge.await_count, 2)
        self.assertEqual(len(result.messages), 6)


class _Shop:
    def __init__(self):
        self.observation = "Initial"
        self.actions = []
        self.clickables = ["search"]

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": self.clickables}

    def step(self, action):
        self.actions.append(action)
        self.observation = action
        if action.startswith("search["):
            self.clickables = ["P1"]
        elif action == "click[P1]":
            self.clickables = ["Buy Now"]
        done = action == "click[Buy Now]"
        return self.observation, int(done), done, None


def _call(name, arguments, index):
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)},
    }


def _shop_env(turns):
    env = ws.WebShopEnv(Task("test", "", "", raw={}), MagicMock())
    env.environment = _Shop()
    env.max_turns = turns
    return env, Task("test", ws.ABSTAIN_MULTI_TURN_PROMPT, "Initial", turns, final_prompt=None, strict_response=True)
