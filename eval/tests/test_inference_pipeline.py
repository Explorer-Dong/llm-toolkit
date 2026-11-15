"""Resource scopes, two concurrency limits, collection, and failure retention."""

import argparse
import asyncio
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from harness import Env, Task, WorkspaceEnv, agent, load_trajectory, model
from harness.utils import encode_id
from mcp.server.fastmcp import FastMCP

from eval import inference as eval_inference
from eval.utils import run_tasks, trajectory_path

__all__ = []
PARAMS = model.ModelParams("test", 0, 1, 1024, 42)


class InferencePipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name) / "run"
        self.samples: list[dict[str, Any]] = [{"id": str(index), "gold": "secret", "meta": {}} for index in range(4)]
        self.tasks = [Task(sample["id"], "system", "question") for sample in self.samples]
        self.runtime = {}
        self.dataset = Path(self.temporary.name) / "dataset.json"
        self.dataset.write_text(json.dumps(self.samples))
        self.args = argparse.Namespace(
            benchmark="fixture",
            dataset_file=str(self.dataset),
            model="test",
            base_url=None,
            api_key="secret",
            timeout=1,
            concurrency=2,
            sample_concurrency=2,
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
        )
        self.client = MagicMock(close=AsyncMock())
        self.ctx = model.Context(self.client, PARAMS, asyncio.Semaphore(2))

    async def test_sample_task_mismatches_are_rejected_before_runtime_or_output(self):
        cases = [
            (self.samples, self.tasks[:-1]),
            (self.samples, [Task("wrong", "system", "question"), *self.tasks[1:]]),
            ([self.samples[0], self.samples[0]], [self.tasks[0], self.tasks[0]]),
        ]
        for samples, tasks in cases:
            with (
                self.subTest(ids=[task.id for task in tasks]),
                patch.object(eval_inference, "make_context") as context,
                self.assertRaises(ValueError),
            ):
                await eval_inference.run_inference(self.args, samples, tasks)
            context.assert_not_called()
            self.assertFalse(self.run_dir.exists())

    async def test_default_inference_directories_handle_second_level_timestamp_collisions(self):
        self.args.output_dir = None
        output = Path(self.temporary.name) / "outputs"
        with (
            patch.object(eval_inference, "OUTPUT_DIR", output),
            patch.object(eval_inference, "datetime") as clock,
            patch.object(eval_inference, "make_context", return_value=self.ctx),
            patch.object(agent, "call_model", return_value=model.AssistantTurn("answer")),
        ):
            clock.now.return_value = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
            await eval_inference.run_inference(self.args, self.samples, self.tasks)
            await eval_inference.run_inference(self.args, self.samples, self.tasks)
        directories = sorted((output / "fixture").iterdir())
        self.assertEqual(len(directories), 2)
        self.assertRegex(directories[0].name, r"^test_\d{8}-\d{6}$")
        self.assertEqual(directories[1].name, directories[0].name + "-1")
        config = json.loads((directories[0] / "config.json").read_text())
        self.assertGreater(config["wall_time_sec"], 0)
        trace = load_trajectory(next((directories[0] / "trajectories").glob("*.json")))
        assert trace.runtime_latency_sec is not None
        self.assertGreaterEqual(trace.runtime_latency_sec, trace.latency_sec - 0.001)

    async def test_envs_are_per_sample_and_collect_runs_before_teardown(self):
        events = []
        environments = []
        active = peak = 0

        class SampleEnv(WorkspaceEnv):
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

        def make_env(sample):
            env = SampleEnv()
            environments.append(env)
            return env

        async def collect(sample, trace, env, directory):
            self.assertIsNotNone(env.workdir)
            self.assertTrue(env.workdir.exists())
            events.append(("collect", env.workdir))
            return {"final": trace.final_answer}

        async def answer(*args):
            await asyncio.sleep(0.002)
            return model.AssistantTurn("answer")

        with (
            patch.object(eval_inference, "make_context", return_value=self.ctx),
            patch.dict(self.runtime, {"make_env": make_env}),
            patch.dict(self.runtime, {"collect": collect}),
            patch.object(agent, "call_model", side_effect=answer),
        ):
            traces = await eval_inference.run_inference(self.args, self.samples, self.tasks, **self.runtime)
        self.assertEqual([t.task_id for t in traces], ["0", "1", "2", "3"])
        self.assertEqual((peak, active), (2, 0))
        self.assertEqual(len({id(env) for env in environments}), 4)
        for i, trace in enumerate(traces):
            self.assertEqual(load_trajectory(trajectory_path(self.run_dir, str(i))), trace)
            manifest = json.loads(((self.run_dir / "artifacts" / encode_id(str(i))) / "manifest.json").read_text())
            self.assertEqual(manifest["data"], {"final": "answer"})
        for kind, path in events:
            if kind == "open":
                self.assertLess(events.index(("collect", path)), events.index(("close", path)))
        self.assertEqual(self.client.close.await_count, 1)
        self.assertNotIn("secret", (self.run_dir / "config.json").read_text())
        self.assertFalse((self.run_dir / "scores").exists())
        self.assertFalse((self.run_dir / "samples.json").exists())

    async def test_active_sample_and_model_concurrency_are_independent(self):
        self.args.sample_concurrency = 3
        entered = peak_envs = active_models = peak_models = 0

        class ActiveEnv(Env):
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

        self.ctx.client.chat.completions.create = AsyncMock(side_effect=request)
        with (
            patch.object(eval_inference, "make_context", return_value=self.ctx),
            patch.dict(self.runtime, {"make_env": lambda s: ActiveEnv()}),
        ):
            await eval_inference.run_inference(self.args, self.samples, self.tasks, **self.runtime)
        self.assertEqual((peak_envs, peak_models, entered), (3, 2, 0))

    async def test_shared_mcp_lifespan_opens_once_for_all_samples(self):
        from contextlib import asynccontextmanager

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

        async def reply(ctx, messages, tools):
            self.assertEqual((opened, closed), (1, 0))
            self.assertEqual(tools[0]["function"]["name"], "ping")
            return model.AssistantTurn("answer")

        with (
            patch.object(eval_inference, "make_context", return_value=self.ctx),
            patch.dict(self.runtime, {"shared_servers": [server]}),
            patch.object(agent, "call_model", side_effect=reply),
        ):
            traces = await eval_inference.run_inference(self.args, self.samples, self.tasks, **self.runtime)
        self.assertEqual((opened, closed, len(traces)), (1, 1, 4))

    async def test_setup_and_teardown_errors_are_saved_per_sample(self):
        for failure in ("setup", "teardown"):
            self.args.output_dir = str(Path(self.temporary.name) / failure)
            closed = []

            class Broken(Env):
                failure_phase = failure
                closed_environments = closed

                async def setup(env):
                    if env.failure_phase == "setup":
                        raise RuntimeError("setup failed")

                async def teardown(env):
                    env.closed_environments.append(True)
                    if env.failure_phase == "teardown":
                        raise RuntimeError("teardown failed")

            with (
                self.subTest(failure=failure),
                patch.object(eval_inference, "make_context", return_value=self.ctx),
                patch.dict(self.runtime, {"make_env": lambda s: Broken()}),
                patch.object(agent, "call_model", return_value=model.AssistantTurn("answer")),
            ):
                traces = await eval_inference.run_inference(self.args, self.samples, self.tasks, **self.runtime)
            self.assertEqual(len(closed), 4)
            self.assertTrue(all(t.stop_reason == "environment_error" for t in traces))
            self.assertTrue(all(failure in (t.error or "") for t in traces))
            if failure == "teardown":
                self.assertTrue(all(t.final_answer == "answer" for t in traces))

    async def test_collection_failure_preserves_trace_and_cleans_env(self):
        environments = []

        def env(sample):
            result = WorkspaceEnv()
            environments.append(result)
            return result

        with (
            patch.object(eval_inference, "make_context", return_value=self.ctx),
            patch.dict(self.runtime, {"make_env": env}),
            patch.dict(self.runtime, {"collect": AsyncMock(side_effect=ValueError("evidence failed"))}),
            patch.object(agent, "call_model", return_value=model.AssistantTurn("answer")),
        ):
            traces = await eval_inference.run_inference(self.args, self.samples, self.tasks, **self.runtime)
        self.assertTrue(all(e.workdir is None for e in environments))
        self.assertTrue(all(t.stop_reason == "environment_error" and t.final_answer == "answer" for t in traces))
        manifest = json.loads(((self.run_dir / "artifacts" / "0") / "manifest.json").read_text())
        self.assertIn("evidence failed", manifest["error"])

    async def test_cancellation_collects_partial_evidence_and_closes_admitted_envs(self):
        active = asyncio.Event()
        environments = []
        collected = []

        def env(sample):
            result = WorkspaceEnv()
            environments.append(result)
            return result

        async def wait(*args):
            active.set()
            await asyncio.Event().wait()

        async def collect(sample, trace, env, directory):
            self.assertIsNotNone(env.workdir)
            collected.append(sample["id"])
            return {"cancelled": True}

        with (
            patch.object(eval_inference, "make_context", return_value=self.ctx),
            patch.dict(self.runtime, {"make_env": env}),
            patch.dict(self.runtime, {"collect": collect}),
            patch.object(agent, "call_model", side_effect=wait),
        ):
            future = asyncio.create_task(
                eval_inference.run_inference(self.args, self.samples, self.tasks, **self.runtime)
            )
            await active.wait()
            future.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await future
        self.assertTrue(all(e.workdir is None for e in environments))
        self.assertEqual(len(collected), len(environments))
        for sample_id in collected:
            trace = load_trajectory(trajectory_path(self.run_dir, sample_id))
            self.assertEqual(trace.stop_reason, "cancelled")
        self.assertEqual(self.client.close.await_count, 1)

    async def test_cancellation_during_collection_marks_partial_evidence(self):
        started = asyncio.Event()
        envs = []

        def make(sample):
            envs.append(WorkspaceEnv())
            return envs[-1]

        async def collect(sample, trace, env, directory):
            started.set()
            await asyncio.Event().wait()

        with (
            patch.object(eval_inference, "make_context", return_value=self.ctx),
            patch.dict(self.runtime, {"make_env": make}),
            patch.dict(self.runtime, {"collect": collect}),
            patch.object(agent, "call_model", return_value=model.AssistantTurn("answer")),
        ):
            future = asyncio.create_task(
                eval_inference.run_inference(self.args, self.samples, self.tasks, **self.runtime)
            )
            await started.wait()
            future.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await future
        self.assertTrue(all(env.workdir is None for env in envs))
        manifests = list((self.run_dir / "artifacts").glob("*/manifest.json"))
        self.assertEqual(len(manifests), len(envs))
        self.assertTrue(all("CancelledError" in json.loads(p.read_text())["error"] for p in manifests))

    async def test_storage_error_propagates_after_env_cleanup(self):
        envs = []

        def make(sample):
            envs.append(WorkspaceEnv())
            return envs[-1]

        with (
            patch.object(eval_inference, "make_context", return_value=self.ctx),
            patch.dict(self.runtime, {"make_env": make}),
            patch("harness.main.save_trajectory", side_effect=OSError("disk full")),
            patch.object(agent, "call_model", return_value=model.AssistantTurn("answer")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            await eval_inference.run_inference(self.args, self.samples, self.tasks, **self.runtime)
        self.assertTrue(all(env.workdir is None for env in envs))
        self.assertEqual(self.client.close.await_count, 1)


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
