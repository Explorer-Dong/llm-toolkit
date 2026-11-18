"""Each benchmark loads local assets, builds isolated agent input, and scores offline."""

import asyncio
import contextlib
import importlib
import io
import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.server.fastmcp import FastMCP

from eval import agent, model
from eval.agent import Task
from eval.benchmarks._common import cli
from eval.model import build_model_params

__all__ = []
BENCHMARKS = ("aime", "gpqa", "math", "gaia", "browsecomp", "webwalkerqa")
WEB = ("gaia", "browsecomp", "webwalkerqa")
TEST_ENV = {
    "MODEL_NAME": "test",
    "SERPER_API_KEY": "serper-secret",
    "DS_TOOL_KEY": "ds-secret",
    "DS_TOOL_URL": "https://tools.example",
    "CRAWL4AI_TOKEN": "crawler-secret",
    "CRAWL4AI_URL": "https://crawler.example",
}


class BenchmarkTests(unittest.TestCase):
    def test_each_benchmark_loads_self_contained_data_and_filters(self):
        for name in BENCHMARKS:
            with self.subTest(benchmark=name):
                bench = importlib.import_module(f"eval.benchmarks.{name}.rollout")
                args = _args(bench, ["--limit", "1"])
                task = bench.load_tasks(args)[0]
                self.assertEqual(task.max_turns, 16 if name in WEB else 1)
                self.assertNotIn("gold", str(task))
                self.assertEqual(Path(bench.DATA).parent.name, name)
                self.assertTrue(any(Path(bench.DATA).glob("*.json")))
                if name == "math":
                    args.subject, args.level = task.meta["subject"], task.meta["level"]
                elif name == "gaia":
                    args.level = task.meta["level"]
                elif name == "browsecomp":
                    args.topic = task.meta["topic"]
                elif name == "webwalkerqa":
                    args.type, args.difficulty = task.meta["type"], task.meta["difficulty"]
                self.assertEqual(bench.load_tasks(args)[0].id, task.id)
        bench = importlib.import_module("eval.benchmarks.aime.rollout")
        for year in (2024, 2025, 2026):
            self.assertTrue(bench.load_tasks(_args(bench, ["--year", str(year)])))

    def test_generation_defaults_turn_budget_and_concurrency_controls(self):
        for name in (*BENCHMARKS, "swe_bench"):
            bench = importlib.import_module(f"eval.benchmarks.{name}.rollout")
            args = _args(bench)
            expected_concurrency = 6 if name in ("browsecomp", "webwalkerqa") else (2 if name == "swe_bench" else 4)
            self.assertEqual(args.concurrency, expected_concurrency)
            self.assertFalse(hasattr(args, "judge_model"))
            self.assertTrue(args.enable_thinking)
            self.assertEqual(args.reasoning_effort, "medium")
            selected = _args(bench, ["--no-enable-thinking", "--reasoning-effort", "high"])
            params = build_model_params(selected)
            self.assertFalse(params["enable_thinking"])
            self.assertEqual(params["reasoning_effort"], "high")
            if name not in WEB and name != "swe_bench":
                self.assertEqual((args.max_turns, args.max_tokens, args.temperature, args.top_k), (1, 81920, 1, 20))
            args = _args(
                bench,
                [
                    "--max-turns",
                    "3",
                    "--top-k",
                    "0",
                    "--task-timeout",
                    "0",
                    "--concurrency",
                    "5",
                ],
            )
            self.assertEqual(
                (args.max_turns, args.top_k, args.task_timeout, args.concurrency),
                (3, 0, 0, 5),
            )

    def test_common_provider_combinations_credentials_and_default_shared_server(self):
        for name, search, fetch in itertools.product(WEB, ("serper", "deepseek"), ("crawl4ai", "deepseek")):
            bench = importlib.import_module(f"eval.benchmarks.{name}.rollout")
            args = _args(bench, ["--search-provider", search, "--fetch-provider", fetch])
            self.assertEqual((args.search_provider, args.fetch_provider), (search, fetch))
            self.assertEqual((args.ds_tool_key, args.serper_api_key), ("ds-secret", "serper-secret"))
            with (
                patch.dict(os.environ, TEST_ENV, clear=True),
                patch.object(sys, "argv", ["rollout", name, "--search-provider", search, "--fetch-provider", fetch]),
                patch.object(bench, "load_dotenv"),
                patch.object(bench, "load_tasks", return_value=[]),
                patch.object(bench, "common_server", wraps=bench.common_server) as server,
                patch.object(bench, "open_toolbox", wraps=bench.open_toolbox),
                patch.object(bench, "rollout", new_callable=AsyncMock, return_value=[]),
            ):
                self.assertEqual(cli.rollout_main(), 0)
            with tempfile.TemporaryDirectory() as tmp:
                run_args = _args(
                    bench,
                    ["--search-provider", search, "--fetch-provider", fetch, "--output-dir", str(Path(tmp) / "run")],
                )
                with (
                    patch.object(bench, "make_client", return_value=_mock_client()),
                    patch.object(bench, "common_server", return_value=FastMCP("common")) as server,
                    patch.object(agent, "call_model", return_value=model.AssistantTurn("answer")),
                ):
                    asyncio.run(bench.rollout(run_args, [Task("1", "s", "Q", gold="A")]))
                config = server.call_args.args[0]
                self.assertEqual((config.search_provider, config.fetch_provider), (search, fetch))
        for name in WEB:
            with self.subTest(benchmark=name):
                args = _args(importlib.import_module(f"eval.benchmarks.{name}.rollout"))
                self.assertEqual((args.search_provider, args.fetch_provider), ("deepseek", "deepseek"))
        args = _args(importlib.import_module("eval.benchmarks.gaia.rollout"), ["--ds-tool-key", "override"])
        self.assertEqual(args.ds_tool_key, "override")

    def test_invalid_limits_providers_and_names_are_rejected(self):
        for flags in (
            ["--limit", "0"],
            ["--concurrency", "0"],
            ["--max-turns", "0"],
            ["--task-timeout", "-1"],
            ["--search-results", "51"],
            ["--search-provider", "invalid"],
            ["--fetch-provider", "invalid"],
        ):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                _args(importlib.import_module("eval.benchmarks.gaia.rollout"), flags)
        for entry, name in itertools.product((cli.rollout_main, cli.scoring_main), ("../aime", "a.b", "AIME")):
            with (
                self.subTest(entry=entry.__name__, benchmark=name),
                patch.object(sys, "argv", [entry.__name__, name]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                entry()

    def test_benchmark_modules_and_stage_entrypoint_help(self):
        for name, stage in itertools.product((*BENCHMARKS, "swe_bench"), ("rollout", "scoring")):
            with self.subTest(benchmark=name, stage=stage):
                result = subprocess.run(
                    [sys.executable, "-m", f"eval.benchmarks.{name}.{stage}", "--help"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--concurrency", result.stdout)
                if stage == "scoring":
                    self.assertEqual("--judge-model" in result.stdout, name in ("browsecomp", "webwalkerqa"))
        for script in ("rollout", "scoring"):
            result = subprocess.run(
                [str(Path(sys.executable).parent / script), "aime", "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--concurrency", result.stdout)


class PipelineSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_six_benchmarks_generate_then_score_from_original_datasets(self):
        for name in BENCHMARKS:
            with self.subTest(benchmark=name), tempfile.TemporaryDirectory() as tmp:
                bench = importlib.import_module(f"eval.benchmarks.{name}.rollout")
                grader = importlib.import_module(f"eval.benchmarks.{name}.scoring")
                args = _args(bench, ["--limit", "1", "--output-dir", str(Path(tmp) / "run")])
                task = bench.load_tasks(args)[0]
                content = str(task.gold) if name in ("browsecomp", "webwalkerqa") else json.dumps({"answer": task.gold})
                if name == "math":
                    content = rf"\boxed{{{task.gold}}}"
                with contextlib.ExitStack() as stack:
                    rollout_client = stack.enter_context(
                        patch.object(bench, "make_client", return_value=_mock_client())
                    )
                    call = stack.enter_context(
                        patch.object(agent, "call_model", return_value=model.AssistantTurn(content))
                    )
                    if name in WEB:
                        stack.enter_context(patch.object(bench, "common_server", return_value=FastMCP("common")))
                    traces = await bench.rollout(args, [task])
                rollout_client.assert_called_once()
                self.assertEqual(call.await_count, 1)
                self.assertIsNone(traces[0].error)
                run_dir = Path(args.output_dir)
                self.assertEqual({p.name for p in run_dir.iterdir()}, {"config.json", "trajectories"})
                config = json.loads((run_dir / "config.json").read_text())
                self.assertEqual(config["sample_ids"], [task.id])
                before = {str(p): p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
                judge_client = MagicMock(close=AsyncMock())
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            bench,
                            "make_client",
                            side_effect=AssertionError("Rollout called during scoring"),
                        )
                    )
                    if name in ("browsecomp", "webwalkerqa"):
                        build = stack.enter_context(patch.object(grader, "make_client", return_value=judge_client))
                        stack.enter_context(patch.object(grader, "load_dotenv"))
                        judge_call = stack.enter_context(
                            patch.object(grader, "call_model", return_value=model.AssistantTurn("correct: yes"))
                        )
                    if name == "math":
                        # Math-Verify uses signal-based timeouts on the process's main thread.
                        scored = await asyncio.to_thread(
                            subprocess.run,
                            [sys.executable, "-m", "eval.benchmarks.math.scoring", str(run_dir)],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(scored.returncode, 0, scored.stderr)
                    else:
                        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        stack.enter_context(patch.object(sys, "argv", ["scoring", name, str(run_dir)]))
                        self.assertEqual(await asyncio.to_thread(cli.scoring_main), 0)
                output = next((run_dir / "scores").iterdir())
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual((summary["num_tasks"], summary["correct"], summary["sys_errors"]), (1, 1, 0))
                for path, contents in before.items():
                    self.assertEqual(Path(path).read_bytes(), contents)
                self.assertEqual(
                    {p.name for p in output.iterdir()}, {"config.json", "trajectories.json", "summary.json"}
                )
                score_config = json.loads((output / "config.json").read_text())
                self.assertEqual(score_config["source_run_dir"], str(run_dir))
                self.assertNotIn("judge_api_key", score_config)
                if name in ("browsecomp", "webwalkerqa"):
                    build.assert_called_once()
                    self.assertTrue(judge_call.call_args.kwargs["enable_thinking"])
                    self.assertEqual(judge_call.call_args.kwargs["reasoning_effort"], "medium")
                    judge_client.close.assert_awaited_once()


def _mock_client():
    client = MagicMock(close=AsyncMock())

    async def close_client(*_args):
        await client.close()

    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(side_effect=close_client)
    return client


def _args(bench, flags=()):
    with patch.dict(os.environ, TEST_ENV, clear=True), patch.object(bench, "load_dotenv"):
        return bench.parse_args(flags)


if __name__ == "__main__":
    unittest.main()
