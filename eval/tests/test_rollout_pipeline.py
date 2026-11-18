"""Resource scopes, two concurrency limits, collection, and failure retention."""

import argparse
import asyncio
import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.server.fastmcp import FastMCP

from eval import agent
from eval.agent import Task, Trajectory
from eval.benchmarks._common import resume
from eval.benchmarks._common.storage import save_environment_result, trajectory_path
from eval.benchmarks._common.tasks import run_tasks
from eval.benchmarks.aime import rollout as aime_rollout
from eval.benchmarks.gaia import rollout as gaia_rollout
from eval.benchmarks.swe_bench import rollout as swe_rollout
from eval.harness.env import Env, WorkspaceEnv
from eval.harness.utils import encode_id, load_trajectory, save_trajectory
from eval.model import AssistantTurn

__all__ = []


class RolloutPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name) / "run"
        rows = [
            {
                "id": str(index),
                "problem": f"problem-{index}",
                "problem_statement": f"problem-{index}",
                "question": f"question-{index}",
                "image": f"img-{index}",
                "base_commit": "a" * 40,
                "gold": "secret",
                "meta": {},
            }
            for index in range(4)
        ]
        self.dataset = Path(self.temporary.name) / "dataset.json"
        self.dataset.write_text(json.dumps(rows))
        self.tasks = [
            Task(
                str(index),
                "system",
                f"problem-{index}",
                gold="secret",
                raw={"image": f"img-{index}", "base_commit": "a" * 40},
            )
            for index in range(4)
        ]
        self.args = argparse.Namespace(
            benchmark="aime",
            dataset_file=str(self.dataset),
            model="test",
            base_url=None,
            api_key="secret",
            timeout=1,
            concurrency=2,
            temperature=0,
            top_p=1,
            top_k=0,
            max_tokens=1024,
            seed=42,
            enable_thinking=True,
            reasoning_effort="medium",
            output_dir=str(self.run_dir),
            max_turns=1,
            task_timeout=30,
            search_provider="deepseek",
            fetch_provider="deepseek",
            search_results=8,
            observation_chars=12000,
            ds_tool_url="",
            ds_tool_key="",
            serper_api_key="",
            crawl4ai_url="",
            crawl4ai_token="",
        )
        self.client = MagicMock(close=AsyncMock())

        async def close_client(*_args):
            await self.client.close()

        self.client.__aenter__ = AsyncMock(return_value=self.client)
        self.client.__aexit__ = AsyncMock(side_effect=close_client)

    async def test_duplicate_task_ids_are_rejected_before_runtime_or_output(self):
        with self.assertRaises(ValueError):
            resume.init_rollout_dir(self.args, [self.tasks[0], self.tasks[0]])
        self.assertFalse(self.run_dir.exists())

    async def test_default_rollout_directories_handle_second_level_timestamp_collisions(self):
        self.args.output_dir = None
        output = Path(self.temporary.name) / "outputs"
        with (
            patch.object(resume, "OUTPUT_DIR", output),
            patch.object(resume, "datetime") as clock,
            patch.object(aime_rollout, "make_client", return_value=self.client),
            patch.object(agent, "call_model", return_value=AssistantTurn("answer")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            clock.now.return_value = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
            await aime_rollout.rollout(self.args, self.tasks)
            await aime_rollout.rollout(self.args, self.tasks)
        directories = sorted((output / "aime").iterdir())
        self.assertEqual(len(directories), 2)
        self.assertRegex(directories[0].name, r"^test_\d{8}-\d{6}$")
        self.assertEqual(directories[1].name, directories[0].name + "-1")
        config = json.loads((directories[0] / "config.json").read_text())
        self.assertGreater(config["wall_time_sec"], 0)
        trace = load_trajectory(next((directories[0] / "trajectories").glob("*.json")))
        assert trace.runtime_latency_sec is not None
        self.assertGreaterEqual(trace.runtime_latency_sec, trace.latency_sec - 0.001)

    async def test_envs_are_per_sample_and_collect_runs_before_teardown(self):
        self.args.benchmark = "swe_bench"
        events = []
        environments = []
        active = peak = 0

        class SampleEnv(WorkspaceEnv):
            def __init__(self, image, base_commit):
                super().__init__()
                self.image = image
                self.base_commit = base_commit

            async def setup(env):
                nonlocal active, peak
                await super().setup()
                active += 1
                peak = max(peak, active)
                events.append(("open", env.workdir))

            async def teardown(env):
                nonlocal active
                events.append(("close", env.workdir))
                await super().teardown()
                active -= 1

        async def collect(task, trace, env, directory):
            self.assertIsNotNone(env.workdir)
            self.assertTrue(env.workdir.exists())
            events.append(("collect", env.workdir))
            return {"final": trace.final_answer}

        def make_env(image, base_commit):
            env = SampleEnv(image, base_commit)
            environments.append(env)
            return env

        async def answer(*_args, **_kwargs):
            await asyncio.sleep(0.002)
            return AssistantTurn("answer")

        with (
            patch.object(swe_rollout, "prepare", AsyncMock(return_value={f"img-{i}": f"digest-{i}" for i in range(4)})),
            patch.object(swe_rollout, "SWEEnv", side_effect=make_env),
            patch.object(swe_rollout, "collect", side_effect=collect),
            patch.object(swe_rollout, "make_client", return_value=self.client),
            patch.object(agent, "call_model", side_effect=answer),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            traces = await swe_rollout.rollout(self.args, self.tasks)
        self.assertEqual([t.task_id for t in traces], ["0", "1", "2", "3"])
        self.assertEqual((peak, active), (2, 0))
        self.assertEqual(len({id(env) for env in environments}), 4)
        for i, trace in enumerate(traces):
            self.assertEqual(load_trajectory(trajectory_path(self.run_dir, str(i))), trace)
            environment_result = json.loads(
                ((self.run_dir / "environment_outputs" / encode_id(str(i))) / "environment_result.json").read_text()
            )
            self.assertEqual(environment_result["data"], {"final": "answer"})
        for kind, path in events:
            if kind == "open":
                self.assertLess(events.index(("collect", path)), events.index(("close", path)))
        self.assertEqual(self.client.close.await_count, 1)
        self.assertNotIn("secret", (self.run_dir / "config.json").read_text())
        self.assertFalse((self.run_dir / "scores").exists())
        self.assertFalse((self.run_dir / "samples.json").exists())

    async def test_task_concurrency_bounds_active_environments_and_requests(self):
        self.args.benchmark = "swe_bench"
        self.args.concurrency = 3
        entered = peak_envs = active_models = peak_models = 0

        class ActiveEnv(Env):
            def __init__(self, *_args):
                pass

            async def setup(env):
                nonlocal entered, peak_envs
                entered += 1
                peak_envs = max(peak_envs, entered)

            async def teardown(env):
                nonlocal entered
                entered -= 1

        async def request(**kwargs):
            nonlocal active_models, peak_models
            active_models += 1
            peak_models = max(peak_models, active_models)

            async def chunks():
                nonlocal active_models
                await asyncio.sleep(0.01)
                active_models -= 1
                if False:
                    yield None

            stream = MagicMock(close=AsyncMock())
            stream.__aiter__.side_effect = chunks
            return stream

        self.client.chat.completions.create = AsyncMock(side_effect=request)
        with (
            patch.object(swe_rollout, "prepare", AsyncMock(return_value={f"img-{i}": f"digest-{i}" for i in range(4)})),
            patch.object(swe_rollout, "SWEEnv", side_effect=ActiveEnv),
            patch.object(swe_rollout, "collect", AsyncMock(return_value={})),
            patch.object(swe_rollout, "make_client", return_value=self.client),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await swe_rollout.rollout(self.args, self.tasks)
        self.assertEqual((peak_envs, peak_models, entered), (3, 3, 0))

    async def test_shared_mcp_lifespan_opens_once_for_all_tasks(self):
        self.args.benchmark = "gaia"
        opened = closed = 0

        @asynccontextmanager
        async def lifespan(server):
            nonlocal opened, closed
            opened += 1
            try:
                yield None
            finally:
                closed += 1

        server = FastMCP("shared", lifespan=lifespan)

        @server.tool()
        async def ping(value: str) -> str:
            return value

        async def reply(client, messages, tools, **_params):
            self.assertEqual((opened, closed), (1, 0))
            self.assertEqual(tools[0]["function"]["name"], "ping")
            return AssistantTurn("answer")

        with (
            patch.object(gaia_rollout, "make_client", return_value=self.client),
            patch.object(gaia_rollout, "common_server", return_value=server),
            patch.object(agent, "call_model", side_effect=reply),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            traces = await gaia_rollout.rollout(self.args, self.tasks)
        self.assertEqual((opened, closed, len(traces)), (1, 1, 4))

    async def test_setup_and_teardown_errors_are_saved_per_task(self):
        self.args.benchmark = "swe_bench"
        for failure in ("setup", "teardown"):
            self.args.output_dir = str(Path(self.temporary.name) / failure)
            closed = []

            class Broken(Env):
                failure_phase = failure
                closed_environments = closed

                def __init__(self, *_args):
                    pass

                async def setup(env):
                    if env.failure_phase == "setup":
                        raise RuntimeError("setup failed")

                async def teardown(env):
                    env.closed_environments.append(True)
                    if env.failure_phase == "teardown":
                        raise RuntimeError("teardown failed")

            with (
                self.subTest(failure=failure),
                patch.object(
                    swe_rollout, "prepare", AsyncMock(return_value={f"img-{i}": f"digest-{i}" for i in range(4)})
                ),
                patch.object(swe_rollout, "SWEEnv", side_effect=Broken),
                patch.object(swe_rollout, "collect", AsyncMock(return_value={})),
                patch.object(swe_rollout, "make_client", return_value=self.client),
                patch.object(agent, "call_model", return_value=AssistantTurn("answer")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                traces = await swe_rollout.rollout(self.args, self.tasks)
            self.assertEqual(len(closed), 4)
            self.assertTrue(all(t.episode_status == "environment_error" for t in traces))
            self.assertTrue(all(failure in (t.error or "") for t in traces))
            if failure == "teardown":
                self.assertTrue(all(t.final_answer == "answer" for t in traces))

    async def test_collection_failure_preserves_trace_and_cleans_env(self):
        self.args.benchmark = "swe_bench"
        environments = []

        def make_env(*_args):
            result = WorkspaceEnv()
            environments.append(result)
            return result

        with (
            patch.object(swe_rollout, "prepare", AsyncMock(return_value={f"img-{i}": f"digest-{i}" for i in range(4)})),
            patch.object(swe_rollout, "SWEEnv", side_effect=make_env),
            patch.object(swe_rollout, "collect", AsyncMock(side_effect=ValueError("evidence failed"))),
            patch.object(swe_rollout, "make_client", return_value=self.client),
            patch.object(agent, "call_model", return_value=AssistantTurn("answer")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            traces = await swe_rollout.rollout(self.args, self.tasks)
        self.assertTrue(all(e.workdir is None for e in environments))
        self.assertTrue(all(t.episode_status == "environment_error" and t.final_answer == "answer" for t in traces))
        environment_result = json.loads(
            ((self.run_dir / "environment_outputs" / "0") / "environment_result.json").read_text()
        )
        self.assertIn("evidence failed", environment_result["error"])

    async def test_cancellation_drops_partial_outputs_and_closes_admitted_envs(self):
        self.args.benchmark = "swe_bench"
        active = asyncio.Event()
        environments = []
        collected = []

        def make_env(*_args):
            result = WorkspaceEnv()
            environments.append(result)
            return result

        async def wait(*_args, **_kwargs):
            active.set()
            await asyncio.Event().wait()

        async def collect(task, trace, env, directory):
            collected.append(task.id)
            return {"cancelled": True}

        with (
            patch.object(swe_rollout, "prepare", AsyncMock(return_value={f"img-{i}": f"digest-{i}" for i in range(4)})),
            patch.object(swe_rollout, "SWEEnv", side_effect=make_env),
            patch.object(swe_rollout, "collect", side_effect=collect),
            patch.object(swe_rollout, "make_client", return_value=self.client),
            patch.object(agent, "call_model", side_effect=wait),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            future = asyncio.create_task(swe_rollout.rollout(self.args, self.tasks))
            await active.wait()
            future.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await future
        self.assertTrue(all(e.workdir is None for e in environments))
        self.assertEqual(collected, [])
        self.assertEqual(
            list((self.run_dir / "trajectories").glob("*.json")) if (self.run_dir / "trajectories").exists() else [], []
        )
        self.assertEqual(
            list((self.run_dir / "environment_outputs").iterdir())
            if (self.run_dir / "environment_outputs").exists()
            else [],
            [],
        )
        self.assertEqual(self.client.close.await_count, 1)

    async def test_cancellation_during_collection_drops_partial_outputs(self):
        self.args.benchmark = "swe_bench"
        started = asyncio.Event()
        envs = []

        def make(*_args):
            envs.append(WorkspaceEnv())
            return envs[-1]

        async def collect(task, trace, env, directory):
            (directory / "patch.diff").write_text("partial diff")
            started.set()
            await asyncio.Event().wait()

        with (
            patch.object(swe_rollout, "prepare", AsyncMock(return_value={f"img-{i}": f"digest-{i}" for i in range(4)})),
            patch.object(swe_rollout, "SWEEnv", side_effect=make),
            patch.object(swe_rollout, "collect", side_effect=collect),
            patch.object(swe_rollout, "make_client", return_value=self.client),
            patch.object(agent, "call_model", return_value=AssistantTurn("answer")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            future = asyncio.create_task(swe_rollout.rollout(self.args, self.tasks))
            await started.wait()
            future.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await future
        self.assertTrue(all(env.workdir is None for env in envs))
        self.assertEqual(list((self.run_dir / "trajectories").glob("*.json")), [])
        self.assertEqual(list((self.run_dir / "environment_outputs").iterdir()), [])

    async def test_cancelled_rollout_keeps_finished_and_drops_interrupted_for_resume(self):
        self.args.concurrency = 2
        first_done = asyncio.Event()
        second_started = asyncio.Event()

        async def reply(client, messages, tools, **_params):
            if "question-0" in messages[1]["content"]:
                first_done.set()
                return AssistantTurn("done-0")
            await first_done.wait()
            second_started.set()
            await asyncio.Event().wait()

        tasks = [Task(str(i), "system", f"question-{i}", gold=i) for i in range(4)]
        with (
            patch.object(aime_rollout, "make_client", return_value=self.client),
            patch.object(agent, "call_model", side_effect=reply),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            future = asyncio.create_task(aime_rollout.rollout(self.args, tasks))
            await second_started.wait()
            future.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await future

        saved_files = [p.name for p in (self.run_dir / "trajectories").glob("*.json")]
        self.assertEqual(saved_files, ["0.json"])

        resumed_ids = []

        async def resume_reply(client, messages, tools, **_params):
            resumed_ids.append(messages[1]["content"])
            return AssistantTurn("resumed")

        self.args.output_dir = None
        self.args.resume = self.run_dir
        with (
            patch.object(aime_rollout, "make_client", return_value=self.client),
            patch.object(agent, "call_model", side_effect=resume_reply),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            traces = await aime_rollout.rollout(self.args, tasks)
        self.assertEqual(sorted(resumed_ids), ["question-1", "question-2", "question-3"])
        self.assertEqual([t.final_answer for t in traces], ["done-0", "resumed", "resumed", "resumed"])

    async def test_storage_error_propagates_after_env_cleanup(self):
        self.args.benchmark = "swe_bench"
        envs = []

        def make(*_args):
            envs.append(WorkspaceEnv())
            return envs[-1]

        with (
            patch.object(swe_rollout, "prepare", AsyncMock(return_value={f"img-{i}": f"digest-{i}" for i in range(4)})),
            patch.object(swe_rollout, "SWEEnv", side_effect=make),
            patch.object(swe_rollout, "collect", AsyncMock(return_value={})),
            patch.object(swe_rollout, "make_client", return_value=self.client),
            patch.object(swe_rollout, "save_trajectory", side_effect=OSError("disk full")),
            patch.object(agent, "call_model", return_value=AssistantTurn("answer")),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            await swe_rollout.rollout(self.args, self.tasks)
        self.assertTrue(all(env.workdir is None for env in envs))
        self.assertEqual(self.client.close.await_count, 1)

    async def test_resume_skips_completed_and_reruns_incomplete_tasks(self):
        self.args.benchmark = "swe_bench"
        seen_ids = []

        async def collect(task, trace, env, directory):
            (directory / "evidence.txt").write_text(f"evidence-{task.id}")
            return {"sample_id": task.id}

        async def reply(client, messages, tools, **_params):
            seen_ids.append(messages[1]["content"])
            return AssistantTurn("resumed")

        self.run_dir.mkdir()
        (self.run_dir / "config.json").write_text(
            json.dumps(
                {
                    "benchmark": "swe_bench",
                    "model": "test",
                    "dataset_file": str(self.dataset),
                    "sample_ids": [task.id for task in self.tasks],
                    "collect_environment_outputs": True,
                    "wall_time_sec": 1.5,
                }
            )
        )
        # 0: complete trajectory + complete environment outputs
        save_trajectory(Trajectory("0", final_answer="original-0"), trajectory_path(self.run_dir, "0"))
        env0 = self.run_dir / "environment_outputs" / "0"
        env0.mkdir(parents=True)
        (env0 / "evidence.txt").write_text("evidence-0")
        save_environment_result(env0, "0", {"sample_id": "0"}, None)
        # 1: cancelled trajectory + error in environment_result.json
        save_trajectory(
            Trajectory("1", episode_status="cancelled", error="CancelledError"),
            trajectory_path(self.run_dir, "1"),
        )
        env1 = self.run_dir / "environment_outputs" / "1"
        env1.mkdir(parents=True)
        save_environment_result(env1, "1", None, "Cancelled before collection")
        # 2: trajectory has no error, but environment output is incomplete (missing file)
        save_trajectory(Trajectory("2", final_answer="stale-2"), trajectory_path(self.run_dir, "2"))
        env2 = self.run_dir / "environment_outputs" / "2"
        env2.mkdir(parents=True)
        save_environment_result(env2, "2", {"sample_id": "2"}, "collection crashed")
        # 3: missing trajectory file completely

        self.args.output_dir = None
        self.args.resume = self.run_dir
        with (
            patch.object(swe_rollout, "prepare", AsyncMock(return_value={f"img-{i}": f"digest-{i}" for i in range(4)})),
            patch.object(swe_rollout, "SWEEnv", side_effect=lambda *_args: WorkspaceEnv()),
            patch.object(swe_rollout, "collect", side_effect=collect),
            patch.object(swe_rollout, "make_client", return_value=self.client),
            patch.object(agent, "call_model", side_effect=reply),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            traces = await swe_rollout.rollout(self.args, self.tasks)

        self.assertEqual([t.task_id for t in traces], ["0", "1", "2", "3"])
        self.assertEqual(traces[0].final_answer, "original-0")
        self.assertEqual(sorted(seen_ids), ["problem-1", "problem-2", "problem-3"])
        for task_id in ("1", "2", "3"):
            loaded = load_trajectory(trajectory_path(self.run_dir, task_id))
            self.assertEqual(loaded.final_answer, "resumed")
            self.assertIsNone(loaded.error)
            result_path = self.run_dir / "environment_outputs" / task_id / "environment_result.json"
            result = json.loads(result_path.read_text())
            self.assertIsNone(result["error"])
            self.assertEqual(result["files"], ["evidence.txt"])
        config = json.loads((self.run_dir / "config.json").read_text())
        self.assertGreater(config["wall_time_sec"], 1.5)
        self.assertNotIn("resume", config)

    async def test_resume_all_complete_skips_model_and_server_setup(self):
        with (
            patch.object(gaia_rollout, "make_client", return_value=self.client),
            patch.object(gaia_rollout, "common_server", return_value=FastMCP("common")),
            patch.object(agent, "call_model", return_value=AssistantTurn("answer")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.args.benchmark = "gaia"
            initial = await gaia_rollout.rollout(self.args, self.tasks)

        self.args.output_dir = None
        self.args.resume = self.run_dir
        with (
            patch.object(gaia_rollout, "make_client", side_effect=AssertionError("should not open client")),
            patch.object(gaia_rollout, "open_toolbox", side_effect=AssertionError("should not open toolbox")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            resumed = await gaia_rollout.rollout(self.args, self.tasks)
        self.assertEqual(resumed, initial)

    async def test_resume_rejects_mismatched_config_and_output_dir(self):
        with (
            patch.object(aime_rollout, "make_client", return_value=self.client),
            patch.object(agent, "call_model", return_value=AssistantTurn("answer")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await aime_rollout.rollout(self.args, self.tasks)

        with self.assertRaisesRegex(ValueError, "combine"):
            self.args.resume = self.run_dir
            await aime_rollout.rollout(self.args, self.tasks)

        self.args.output_dir = None
        for field, value, pattern in (
            ("benchmark", "other", "Expected"),
            ("model", "other-model", "Model"),
            ("temperature", 0.5, "differs from saved config"),
            ("max_turns", 5, "differs from saved config"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, pattern):
                bad = argparse.Namespace(**(vars(self.args) | {field: value}))
                resume.init_rollout_dir(bad, self.tasks)

        with self.assertRaisesRegex(ValueError, "sample ids"):
            resume.init_rollout_dir(self.args, self.tasks[:-1])

    def test_benchmark_parse_args_resume_restores_and_validates_config(self):
        benchmarks = (
            "aime",
            "gpqa",
            "math",
            "gaia",
            "browsecomp",
            "webwalkerqa",
            "swe_bench",
            "agent_abstain",
            "webshop_abstain",
        )
        for name in benchmarks:
            with self.subTest(benchmark=name), tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "run"
                run_dir.mkdir()
                bench = importlib.import_module(f"eval.benchmarks.{name}.rollout")
                with (
                    patch.dict(os.environ, {"MODEL_NAME": "saved-model"}, clear=True),
                    patch.object(bench, "load_dotenv"),
                ):
                    initial_flags = ["--limit", "1", "--concurrency", "3"]
                    if name == "aime":
                        initial_flags.extend(["--year", "2025"])
                    initial = bench.parse_args(initial_flags)
                package = Path(resume.__file__).resolve().parents[2]
                data_file = Path(initial.dataset_file).resolve()
                saved = {key: value for key, value in vars(initial).items() if key != "resume"}
                relative_data = (
                    str(data_file.relative_to(package)) if data_file.is_relative_to(package) else str(data_file)
                )
                saved.update(
                    dataset_file=relative_data,
                    sample_ids=["1"],
                    collect_environment_outputs=False,
                )
                (run_dir / "config.json").write_text(json.dumps(saved, default=str))

                with patch.dict(os.environ, {}, clear=True), patch.object(bench, "load_dotenv"):
                    resumed = bench.parse_args(
                        [
                            "--resume",
                            str(run_dir),
                            "--concurrency",
                            "5",
                            "--timeout",
                            "120",
                            "--base-url",
                            "https://override.example/v1",
                        ]
                    )
                self.assertEqual(resumed.model, "saved-model")
                self.assertEqual(resumed.limit, 1)
                self.assertEqual((resumed.concurrency, resumed.timeout), (5, 120.0))
                self.assertEqual(resumed.base_url, "https://override.example/v1")
                self.assertEqual(Path(resumed.dataset_file).resolve(), data_file)
                if name == "aime":
                    self.assertEqual(resumed.year, 2025)
                if name == "webshop_abstain":
                    self.assertIsInstance(resumed.catalogs_dir, Path)

                for bad_flags in (
                    ["--resume", str(run_dir), "--limit", "2"],
                    ["--resume", str(run_dir), "--model", "other-model"],
                    ["--resume", str(run_dir), "--temperature", "0.3"],
                    ["--resume", str(run_dir), "--max-tokens", "128"],
                    ["--resume", str(run_dir), "--max-turns", "99"],
                    ["--resume", str(run_dir), "--no-enable-thinking"],
                    ["--resume", str(run_dir), "--output-dir", str(Path(tmp) / "other")],
                ):
                    with (
                        patch.dict(os.environ, {}, clear=True),
                        patch.object(bench, "load_dotenv"),
                        contextlib.redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        bench.parse_args(bad_flags)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_and_out_of_order_results_and_invalid_limit(self):
        async def solve(value):
            await asyncio.sleep((3 - value) * 0.001)
            return value * 2

        self.assertEqual(await run_tasks([], solve, concurrency=1, desc="empty"), [])
        self.assertEqual(await run_tasks([0, 1, 2], solve, concurrency=3, desc="order"), [0, 2, 4])
        with self.assertRaises(ValueError):
            await run_tasks([], solve, concurrency=0, desc="invalid")

    async def test_failure_cancels_and_drains_admitted_tasks(self):
        started = asyncio.Event()
        closed = []

        async def solve(value):
            if value == 0:
                await started.wait()
                raise RuntimeError("failed")
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                closed.append(value)

        with self.assertRaisesRegex(RuntimeError, "failed"):
            await run_tasks([0, 1, 2], solve, concurrency=2, desc="fail")
        self.assertEqual(closed, [1])


if __name__ == "__main__":
    unittest.main()
