"""SWE-bench selection, reference isolation, patch evidence, and fresh grading."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from eval.agent import Trajectory
from eval.benchmarks._common.storage import PACKAGE_DIR, save_environment_result, trajectory_path
from eval.benchmarks.swe_bench import rollout as swe
from eval.benchmarks.swe_bench import scoring as swe_scoring
from eval.harness.env import DockerEnv, ExecResult
from eval.harness.utils import encode_id, save_trajectory

__all__ = []


class SWEbenchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = swe.parse_args(["--model", "fixture", "--subset", "easy", "--limit", "2"])
        self.samples = swe.load_tasks(self.args)

    def test_easy_subset_has_two_stable_instances_and_two_concurrency_limits(self):
        self.assertEqual([s["id"] for s in self.samples], ["astropy__astropy-14309", "astropy__astropy-14995"])
        self.assertEqual((self.args.concurrency, self.args.sample_concurrency), (2, 2))
        for sample in self.samples:
            self.assertEqual(sample["difficulty"], "<15 min fix")
            task = swe.build_task(sample, self.args)
            altered = dict(
                sample,
                patch="GOLD_PATCH_SENTINEL",
                test_patch="TEST_PATCH_SENTINEL",
                hints_text="HINT_SENTINEL",
                FAIL_TO_PASS=["TEST_NAME_SENTINEL"],
            )
            self.assertEqual(task, swe.build_task(altered, self.args))
            self.assertEqual(task.prompt, sample["problem_statement"])
            self.assertEqual(task.max_turns, 40)

    def test_instance_filter_and_unknown_instance_validation(self):
        args = swe.parse_args(["--model", "fixture", "--instance-id", self.samples[1]["id"]])
        self.assertEqual([s["id"] for s in swe.load_tasks(args)], [self.samples[1]["id"]])
        args.instance_id = ["unknown"]
        with self.assertRaisesRegex(ValueError, "Unknown"):
            swe.load_tasks(args)

    async def test_prepare_resolves_each_image_once_and_env_uses_digest(self):
        with patch.object(
            swe, "_prepare_image", new_callable=AsyncMock, side_effect=["repo@sha256:first", "repo@sha256:second"]
        ) as prepare:
            images = await swe.prepare([*self.samples, self.samples[0]], concurrency=2)
        self.assertEqual(prepare.await_count, 2)
        env = swe.SWEEnv(images[self.samples[0]["image"]], self.samples[0]["base_commit"])
        self.assertIsInstance(env, DockerEnv)
        self.assertEqual((env.image, env.workdir), ("repo@sha256:first", "/testbed"))
        self.assertIsNone(env.container)

    async def test_environment_resets_repository_and_activates_python_before_commands(self):
        env = swe.SWEEnv("image", self.samples[0]["base_commit"])
        with (
            patch.object(DockerEnv, "setup", new_callable=AsyncMock) as opened,
            patch.object(DockerEnv, "exec", new_callable=AsyncMock, return_value=ExecResult("", "", 0)) as execute,
        ):
            await env.setup()
            await env.exec("python check.py", timeout=10)
        opened.assert_awaited_once()
        self.assertIn("git reset --hard " + self.samples[0]["base_commit"], execute.call_args_list[0].args[0])
        self.assertIn("conda activate testbed", execute.call_args_list[0].args[0])
        self.assertIn("conda activate testbed && python check.py", execute.call_args_list[1].args[0])
        self.assertEqual(execute.call_args_list[1].kwargs["timeout"], 10)

    async def test_collect_captures_patch_without_persisting_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "environment_outputs" / encode_id(self.samples[0]["id"])
            directory.mkdir(parents=True)
            env = MagicMock(image="repo@sha256:first", container="rollout")
            env.exec = AsyncMock(return_value=ExecResult("model patch\n", "", 0))
            data = await swe.collect(self.samples[0], Trajectory(self.samples[0]["id"]), env, directory)
            self.assertEqual((directory / "patch.diff").read_text(), "model patch\n")
            self.assertEqual({p.name for p in directory.iterdir()}, {"patch.diff"})
            self.assertEqual(data["rollout_container"], "rollout")
            self.assertIn(self.samples[0]["base_commit"], env.exec.call_args.args[0])

    async def test_offline_score_applies_saved_patch_in_a_fresh_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = self.samples[0]
            directory = Path(tmp) / "environment_outputs" / encode_id(sample["id"])
            directory.mkdir(parents=True)
            (directory / "patch.diff").write_text("saved model patch")
            fresh = MagicMock(container="fresh-scoring")
            fresh.write_file = AsyncMock()
            log = (
                ">>>>> Start Test Output\n"
                + "\n".join(f"PASSED {name}" for name in sample["FAIL_TO_PASS"] + sample["PASS_TO_PASS"])
                + "\n>>>>> End Test Output\n"
            )
            fresh.exec = AsyncMock(side_effect=[ExecResult("applied", "", 0), ExecResult(log, "", 0)])
            scope = MagicMock()
            scope.__aenter__ = AsyncMock(return_value=fresh)
            scope.__aexit__ = AsyncMock(return_value=False)
            with patch.object(swe_scoring, "SWEEnv", return_value=scope) as make:
                result = await swe_scoring.score(
                    sample,
                    Trajectory(sample["id"]),
                    {"patch_file": "patch.diff", "image": "repo@sha256:original", "rollout_container": "old"},
                    {"model": "fixture"},
                    args=swe_scoring.parse_args([str(Path(tmp))]),
                )
            make.assert_called_once_with("repo@sha256:original", sample["base_commit"])
            self.assertEqual(fresh.write_file.call_args_list[0].args, (".git/harness.patch", "saved model patch"))
            self.assertEqual(fresh.write_file.call_args_list[1].args, (".git/harness-eval.sh", sample["eval_script"]))
            self.assertEqual((result.score, result.sys_error), (1, None))
            self.assertEqual(result.messages[0]["scoring_container"], "fresh-scoring")
            scope.__aexit__.assert_awaited_once()
            self.assertEqual((directory / "patch.diff").read_text(), "saved model patch")

    def test_resume_prepares_only_unfinished_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            data_file = Path(self.args.dataset_file).resolve()
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "benchmark": "swe_bench",
                        "model": "fixture",
                        "subset": "easy",
                        "limit": 2,
                        "dataset_file": str(data_file.relative_to(PACKAGE_DIR)),
                        "sample_ids": [s["id"] for s in self.samples],
                        "collect_environment_outputs": True,
                    }
                )
            )
            done_id = self.samples[0]["id"]
            save_trajectory(Trajectory(done_id, final_answer="fixed"), trajectory_path(run_dir, done_id))
            env_dir = run_dir / "environment_outputs" / encode_id(done_id)
            env_dir.mkdir(parents=True)
            (env_dir / "patch.diff").write_text("diff")
            save_environment_result(env_dir, done_id, {"patch_file": "patch.diff"}, None)

            seen_images = []

            async def fake_prepare(pending, *, concurrency):
                seen_images.extend(sample["id"] for sample in pending)
                return {sample["image"]: f"digest:{sample['id']}" for sample in pending}

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(swe, "load_dotenv"),
                patch.object(swe, "prepare", side_effect=fake_prepare),
                patch.object(swe, "make_context", side_effect=RuntimeError("stop after prepare")),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "stop after prepare"),
            ):
                swe.main(["--resume", str(run_dir)])
            self.assertEqual(seen_images, [self.samples[1]["id"]])


if __name__ == "__main__":
    unittest.main()
