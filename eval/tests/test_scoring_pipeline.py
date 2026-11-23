"""Independent scoring, saved task identities, and environment outputs."""

import argparse
import ast
import contextlib
import importlib
import io
import json
import os
import shutil
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from eval.agent import Task, Trajectory
from eval.benchmarks._common import resume
from eval.benchmarks._common.metrics import ScoreRecord, ScoreResult, build_summary, collect_costs
from eval.benchmarks._common.storage import save_environment_result, trajectory_path
from eval.benchmarks.aime import scoring as aime
from eval.benchmarks.browsecomp import scoring as browsecomp
from eval.benchmarks.gpqa import scoring as gpqa
from eval.harness.utils import save_trajectory
from eval.model import AssistantTurn

__all__ = []
REPO = Path(__file__).resolve().parents[2]


class ScoreEnvironmentOutputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name) / "run"
        self.run_dir.mkdir()
        _write_run(self.run_dir)

    async def test_previous_output_names_remain_readable_without_rewriting_rollout(self):
        config_path = self.run_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["collect_artifacts"] = config.pop("collect_environment_outputs")
        config_path.write_text(json.dumps(config))
        directory = self.run_dir / "environment_outputs"
        (directory / "1/environment_result.json").rename(directory / "1/manifest.json")
        directory.rename(self.run_dir / "artifacts")
        original = _rollout_bytes(self.run_dir)
        _, summary = await _run_score(_score_args(self.run_dir))
        self.assertEqual((summary["score"], summary["sys_errors"]), (1.0, 0))
        self.assertEqual(_rollout_bytes(self.run_dir), original)

    async def test_repeated_scoring_keeps_previous_results_and_rollout(self):
        original = _rollout_bytes(self.run_dir)
        with patch.object(resume, "datetime") as clock:
            clock.now.return_value = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
            first, _ = await _run_score(_score_args(self.run_dir))
            first_bytes = {path.name: path.read_bytes() for path in first.iterdir()}
            second, _ = await _run_score(_score_args(self.run_dir))
        self.assertNotEqual(first, second)
        self.assertRegex(first.name, r"^\d{8}-\d{6}$")
        self.assertEqual(second.name, first.name + "-1")
        self.assertEqual({path.name: path.read_bytes() for path in first.iterdir()}, first_bytes)
        self.assertEqual(_rollout_bytes(self.run_dir), original)

    async def test_benchmark_owns_grading_policy_independently_of_rollout_options(self):
        config_path = self.run_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["scoring_method"] = "old_policy"
        config_path.write_text(json.dumps(config))
        original = _rollout_bytes(self.run_dir)
        _, summary = await _run_score(_score_args(self.run_dir))
        self.assertEqual((summary["benchmark"], summary["score"], summary["sys_errors"]), ("aime", 1, 0))
        self.assertEqual(_rollout_bytes(self.run_dir), original)

    async def test_deepseek_flash_requests_configured_name_and_persists_mapped_name(self):
        _write_run(self.run_dir, benchmark="browsecomp")
        client = MagicMock(close=AsyncMock())
        with (
            patch.object(browsecomp, "load_tasks", side_effect=_load_tasks),
            patch.object(browsecomp, "make_client", return_value=client),
            patch.object(browsecomp, "call_model", return_value=AssistantTurn("correct: yes")) as call,
        ):
            output, _ = await browsecomp.run_score(_score_args(self.run_dir, judge_model="deepseek-flash"))
        self.assertEqual(call.call_args.kwargs["model"], "deepseek-flash")
        self.assertEqual(json.loads((output / "config.json").read_text())["judge_model"], "deepseek-v4.1-flash-1017")

    async def test_wrong_benchmark_is_rejected_before_grading(self):
        evaluate = AsyncMock()
        with (
            patch.object(gpqa, "load_tasks", side_effect=_load_tasks),
            patch.object(gpqa, "score", evaluate),
            self.assertRaisesRegex(ValueError, "Expected a"),
        ):
            await gpqa.run_score(_score_args(self.run_dir))
        evaluate.assert_not_awaited()
        self.assertFalse((self.run_dir / "scores").exists())

    async def test_missing_trajectory_is_explicit_before_judge_creation(self):
        _write_run(self.run_dir, benchmark="browsecomp")
        trajectory_path(self.run_dir, "1").unlink()
        with (
            patch.object(browsecomp, "load_tasks", side_effect=_load_tasks),
            patch.object(browsecomp, "make_client") as build,
            self.assertRaisesRegex(ValueError, "Missing trajectory"),
        ):
            await browsecomp.run_score(_score_args(self.run_dir))
        build.assert_not_called()
        self.assertFalse((self.run_dir / "scores").exists())

    async def test_scoring_loads_labels_from_the_configured_dataset(self):
        (self.run_dir.parent / "dataset.json").write_text('[{"id":"1","gold":43}]')
        _, summary = await _run_score(_score_args(self.run_dir))
        self.assertEqual((summary["score"], summary["sys_errors"]), (0.0, 0))

    async def test_sample_selection_must_match_saved_ids(self):
        config_path = self.run_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["sample_ids"] = ["other"]
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "selection"):
            await _run_score(_score_args(self.run_dir))

    async def test_mismatched_trajectory_id_is_rejected(self):
        save_trajectory(Trajectory("other"), trajectory_path(self.run_dir, "1"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            await _run_score(_score_args(self.run_dir))

    async def test_duplicate_samples_and_extra_trajectories_are_rejected(self):
        with (
            patch.object(aime, "load_tasks", side_effect=lambda args: _load_tasks(args) * 2),
            self.assertRaisesRegex(ValueError, "unique"),
        ):
            await aime.run_score(_score_args(self.run_dir))
        save_trajectory(Trajectory("extra"), trajectory_path(self.run_dir, "extra"))
        with self.assertRaisesRegex(ValueError, "saved sample ids"):
            await _run_score(_score_args(self.run_dir))

    async def test_scoring_cannot_overwrite_rollout_environment_outputs(self):
        original = _rollout_bytes(self.run_dir)
        for output in (self.run_dir, self.run_dir.parent, self.run_dir / "trajectories/new"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "separate"):
                await _run_score(_score_args(self.run_dir, output_dir=output))
        self.assertEqual(_rollout_bytes(self.run_dir), original)

    async def test_judge_failure_is_scoring_only_and_keeps_agent_trace(self):
        _write_run(self.run_dir, benchmark="browsecomp")
        original = _rollout_bytes(self.run_dir)
        client = MagicMock(close=AsyncMock())
        with (
            patch.object(browsecomp, "load_tasks", side_effect=_load_tasks),
            patch.object(browsecomp, "make_client", return_value=client),
            patch.object(
                browsecomp, "call_model", new_callable=AsyncMock, side_effect=RuntimeError("judge unavailable")
            ),
            self.assertLogs("eval.benchmarks.browsecomp.scoring", level="ERROR") as logs,
        ):
            output, summary = await browsecomp.run_score(_score_args(self.run_dir))
        record = json.loads((output / "trajectories.json").read_text())[0]
        self.assertIsNone(record["agent_error"])
        self.assertEqual(record["score_result"]["sys_error"], "RuntimeError: judge unavailable")
        self.assertIn("judge unavailable", "\n".join(logs.output))
        self.assertEqual((summary["agent_errors"], summary["scoring_errors"], summary["score"]), (0, 1, 0.0))
        self.assertEqual(_rollout_bytes(self.run_dir), original)
        client.close.assert_awaited_once()

    async def test_failed_agent_keeps_denominator_and_separate_error(self):
        save_trajectory(
            Trajectory("1", final_answer='{"answer":42}', error="task_timeout"),
            trajectory_path(self.run_dir, "1"),
        )
        evaluate = AsyncMock()
        output, summary = await _run_score(_score_args(self.run_dir), evaluate=evaluate)
        evaluate.assert_not_awaited()
        record = json.loads((output / "trajectories.json").read_text())[0]
        self.assertEqual((record["agent_error"], record["score_result"]["score"]), ("task_timeout", 0.0))
        self.assertIsNone(record["score_result"]["sys_error"])
        self.assertIsInstance(record["score_result"]["score"], float)
        self.assertEqual((summary["num_tasks"], summary["agent_errors"], summary["scoring_errors"]), (1, 1, 0))

    async def test_missing_environment_output_files_are_rejected_before_scoring(self):
        directory = self.run_dir / "environment_outputs/1"
        (directory / "evidence.txt").write_text("original")
        save_environment_result(directory, "1", {"file": "evidence.txt"}, None)
        evaluate = AsyncMock()
        (directory / "evidence.txt").unlink()
        with self.assertRaisesRegex(ValueError, "environment_result"):
            await _run_score(_score_args(self.run_dir), evaluate=evaluate)
        evaluate.assert_not_awaited()
        self.assertFalse((self.run_dir / "scores").exists())

    async def test_incomplete_collection_requires_an_execution_error(self):
        save_environment_result(self.run_dir / "environment_outputs/1", "1", None, "collection failed")
        with self.assertRaisesRegex(ValueError, "incomplete collection"):
            await _run_score(_score_args(self.run_dir))

    async def test_hybrid_policy_combines_rule_and_judge_in_one_scoring_stage(self):
        model_params = {"model": "judge", "temperature": 0, "top_p": 1, "max_tokens": 1024, "seed": 42}
        orig_aime_score = aime.score

        async def hybrid(task, trace, environment_outputs, config):
            rule = await orig_aime_score(task, trace, environment_outputs, config)
            judged = await browsecomp.score(
                task, trace, environment_outputs, config, client=MagicMock(), model_params=model_params
            )
            judged.method = "hybrid"
            judged.score = 0.4 * rule.score + 0.6 * judged.score
            return judged

        with patch.object(browsecomp, "call_model", return_value=AssistantTurn("correct: no")):
            output, summary = await _run_score(_score_args(self.run_dir), evaluate=hybrid)
        record = json.loads((output / "trajectories.json").read_text())[0]["score_result"]
        self.assertEqual((summary["score"], record["method"]), (0.4, "hybrid"))
        self.assertEqual(len(record["messages"]), 2)

    async def test_resume_scoring_skips_succeeded_and_retries_errors_and_missing(self):
        dataset = self.run_dir.parent / "dataset.json"
        rows = [
            {"id": "1", "question": "Q1", "gold": 1, "meta": {}},
            {"id": "2", "question": "Q2", "gold": 2, "meta": {}},
            {"id": "3", "question": "Q3", "gold": 3, "meta": {}},
        ]
        dataset.write_text(json.dumps(rows))
        (self.run_dir / "config.json").write_text(
            json.dumps(
                {
                    "benchmark": "aime",
                    "model": "model",
                    "dataset_file": str(dataset),
                    "sample_ids": ["1", "2", "3"],
                    "collect_environment_outputs": False,
                }
            )
        )
        for row in rows:
            task_id = str(row["id"])
            save_trajectory(
                Trajectory(task_id, final_answer=json.dumps({"answer": row["gold"]})),
                trajectory_path(self.run_dir, task_id),
            )
        shutil.rmtree(self.run_dir / "environment_outputs", ignore_errors=True)

        seen_first = []

        async def first_eval(task, trace, env_out, config):
            seen_first.append(task.id)
            if task.id == "2":
                raise RuntimeError("transient judge failure")
            return ScoreResult(task.gold, task.gold, 1.0, "integer")

        with contextlib.redirect_stdout(io.StringIO()):
            score_dir, first_summary = await _run_score(_score_args(self.run_dir), evaluate=first_eval)
        self.assertEqual(seen_first, ["1", "2", "3"])
        self.assertEqual(first_summary["scoring_errors"], 1)

        # Simulate task "3" missing from trajectories.json (as if interrupted mid-scoring)
        saved_records = json.loads((score_dir / "trajectories.json").read_text())
        (score_dir / "trajectories.json").write_text(json.dumps([r for r in saved_records if r["task_id"] != "3"]))

        seen_resume = []

        async def resume_eval(task, trace, env_out, config):
            seen_resume.append(task.id)
            return ScoreResult(task.gold, task.gold, 1.0, "integer")

        resume_args = _score_args(None, resume=score_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            resumed_dir, resumed_summary = await _run_score(resume_args, evaluate=resume_eval)

        self.assertEqual(resumed_dir, score_dir)
        self.assertEqual(sorted(seen_resume), ["2", "3"])
        final_records = json.loads((score_dir / "trajectories.json").read_text())
        self.assertEqual([r["task_id"] for r in final_records], ["1", "2", "3"])
        self.assertEqual((resumed_summary["score"], resumed_summary["scoring_errors"]), (1.0, 0))
        self.assertGreater(
            resumed_summary["costs"]["scoring"]["time_sec"]["wall"],
            first_summary["costs"]["scoring"]["time_sec"]["wall"],
        )

        # Resuming an already-complete score directory invokes evaluate zero times
        no_eval = AsyncMock()
        with contextlib.redirect_stdout(io.StringIO()):
            await _run_score(_score_args(None, resume=score_dir), evaluate=no_eval)
        no_eval.assert_not_awaited()

    async def test_resume_scoring_rejects_mismatched_configs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            score_dir, _ = await _run_score(_score_args(self.run_dir))

        with self.assertRaisesRegex(ValueError, "combine"):
            await _run_score(_score_args(self.run_dir, resume=score_dir, output_dir=self.run_dir / "scores/other"))
        with self.assertRaisesRegex(ValueError, "score run"):
            await gpqa.run_score(_score_args(None, resume=score_dir))
        with self.assertRaisesRegex(ValueError, "scoring method"):
            resume.init_score_dir(_score_args(None, resume=score_dir), _load_tasks, benchmark="aime", method="other")
        with self.assertRaisesRegex(ValueError, "Source run"):
            await _run_score(_score_args(self.run_dir.parent / "other-run", resume=score_dir))
        with self.assertRaisesRegex(ValueError, "differs from saved config"):
            await _run_score(_score_args(None, resume=score_dir, judge_model="other-judge"))
        with self.assertRaisesRegex(ValueError, "differs from saved config"):
            await _run_score(_score_args(None, resume=score_dir, seed=99))

    def test_benchmark_parse_args_resume_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            score_dir = run_dir / "scores" / "20261118-120000"
            score_dir.mkdir(parents=True)
            for name in ("aime", "browsecomp", "agent_abstain", "swe_bench"):
                with self.subTest(benchmark=name):
                    scorer = importlib.import_module(f"eval.benchmarks.{name}.scoring")
                    saved = {
                        "source_run_dir": str(run_dir.resolve()),
                        "benchmark": name,
                        "scoring_method": "m",
                        "concurrency": 3,
                    }
                    if name in ("browsecomp", "agent_abstain"):
                        saved["judge_model"] = ["saved-judge"] if name == "agent_abstain" else "saved-judge"
                        saved["judge_max_tokens"] = 4096
                        saved["enable_thinking"] = False
                        saved["reasoning_effort"] = "low"
                        saved["timeout"] = 120.0
                    if name == "swe_bench":
                        saved["test_timeout"] = 900.0
                    (score_dir / "config.json").write_text(json.dumps(saved))

                    override_flags = ["--resume", str(score_dir), "--concurrency", "5"]
                    if name in ("browsecomp", "agent_abstain"):
                        override_flags.extend(["--timeout", "240", "--judge-base-url", "https://judge.new/v1"])
                    with patch.dict(os.environ, {}, clear=True), patch.object(scorer, "load_dotenv", create=True):
                        resumed = scorer.parse_args(override_flags)
                    self.assertEqual(Path(resumed.run_dir).resolve(), run_dir.resolve())
                    self.assertEqual(resumed.concurrency, 5)
                    if name in ("browsecomp", "agent_abstain"):
                        self.assertEqual(resumed.judge_model, saved["judge_model"])
                        self.assertEqual(resumed.timeout, 240.0)
                        self.assertEqual(resumed.judge_base_url, "https://judge.new/v1")
                        self.assertFalse(resumed.enable_thinking)
                    if name == "swe_bench":
                        self.assertEqual(resumed.test_timeout, 900.0)

                    if name == "browsecomp":
                        for bad_flags in (
                            ["--resume", str(score_dir), "--judge-model", "other-judge"],
                            ["--resume", str(score_dir), "--judge-max-tokens", "2048"],
                            ["--resume", str(score_dir), "--reasoning-effort", "high"],
                        ):
                            with (
                                patch.dict(os.environ, {}, clear=True),
                                patch.object(scorer, "load_dotenv"),
                                contextlib.redirect_stderr(io.StringIO()),
                                self.assertRaises(SystemExit),
                            ):
                                scorer.parse_args(bad_flags)
                    if name == "swe_bench":
                        with (
                            patch.dict(os.environ, {}, clear=True),
                            patch.object(scorer, "load_dotenv", create=True),
                            contextlib.redirect_stderr(io.StringIO()),
                            self.assertRaises(SystemExit),
                        ):
                            scorer.parse_args(["--resume", str(score_dir), "--test-timeout", "300"])
                    with (
                        patch.dict(os.environ, {}, clear=True),
                        patch.object(scorer, "load_dotenv", create=True),
                        contextlib.redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        scorer.parse_args([str(Path(tmp) / "wrong-run"), "--resume", str(score_dir)])

    def test_summary_preserves_fractional_scores_and_separate_tool_errors(self):
        records = [
            ScoreRecord("1", ScoreResult("a", "a", 0.375, "judge"), tool_errors=2, metadata={"level": 1}),
            ScoreRecord("2", ScoreResult("a", None, 0.0, "judge"), agent_error="failed", metadata={"level": 2}),
        ]
        summary = build_summary("test", records, "model", group_by=("level",))
        self.assertEqual((summary["correct"], summary["score"], summary["num_tasks"]), (0.375, 0.1875, 2))
        self.assertEqual((summary["sys_errors"], summary["tool_errors"]), (1, 2))
        self.assertEqual(summary["by_level"]["1"]["score"], 0.375)


class CostSummaryTests(unittest.TestCase):
    def test_phase_totals_means_tool_breakdown_and_failed_sample_costs(self):
        first = [
            {
                "role": "assistant",
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "reasoning_tokens": 2, "cached_tokens": 3},
                "latency_sec": 2.0,
                "queue_sec": 0.5,
                "tool_calls": [{"id": "one", "function": {"name": "search"}}],
            },
            {"role": "tool", "tool_call_id": "one", "latency_sec": 1.0},
            {
                "role": "assistant",
                "usage": {"prompt_tokens": 20, "completion_tokens": 6, "reasoning_tokens": 2, "cached_tokens": 4},
                "latency_sec": 3.0,
                "queue_sec": 0.5,
                "tool_calls": [{"id": "two", "function": {"name": "search"}}],
            },
            {"role": "tool", "tool_call_id": "two", "latency_sec": 2.0, "is_error": True},
        ]
        second = [
            {
                "role": "assistant",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "reasoning_tokens": 1},
                "latency_sec": 4.0,
                "queue_sec": 1.0,
            },
            {"role": "tool", "tool_name": "fetch", "latency_sec": 1.0},
        ]
        judge = [
            {
                "role": "assistant",
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "reasoning_tokens": 1, "cached_tokens": 1},
                "latency_sec": 3.0,
                "queue_sec": 0.5,
            }
        ]
        records = [
            ScoreRecord(
                "one",
                ScoreResult("a", "a", 1.0, "judge"),
                metadata={"level": 1},
                costs={
                    "rollout": collect_costs(first, sample_sec=12, episode_sec=10),
                    "scoring": collect_costs(judge, sample_sec=4, episode_sec=4),
                },
            ),
            ScoreRecord(
                "two",
                ScoreResult("b", None, 0.0, "judge"),
                agent_error="failed",
                metadata={"level": 2},
                costs={
                    "rollout": collect_costs(second, sample_sec=6, episode_sec=5),
                    "scoring": collect_costs([], sample_sec=0, episode_sec=0),
                },
            ),
        ]
        summary = build_summary(
            "fixture", records, "model", group_by=("level",), rollout_wall_sec=12, scoring_wall_sec=4
        )
        rollout, scoring, total = (summary["costs"][phase] for phase in ("rollout", "scoring", "total"))
        self.assertEqual(
            (
                rollout["tokens"]["prompt_tokens"],
                rollout["tokens"]["completion_tokens"],
                rollout["tokens"]["total_tokens"],
            ),
            (35, 12, 47),
        )
        self.assertEqual(
            (total["tokens"]["total_tokens"], total["tokens"]["reasoning_tokens"], total["tokens"]["cached_tokens"]),
            (54, 6, 8),
        )
        self.assertEqual(total["tokens"]["mean_per_task"]["total_tokens"], 27)
        self.assertEqual(total["tokens"]["mean_per_model_call"]["total_tokens"], 13.5)
        self.assertEqual((total["calls"]["model"], total["calls"]["tool"], total["calls"]["tool_errors"]), (4, 3, 1))
        self.assertEqual(total["calls"]["mean_per_task"]["model"], 2)
        self.assertEqual((total["time_sec"]["sample"], total["time_sec"]["wall"]), (22, 16))
        self.assertEqual(
            (total["time_sec"]["model"], total["time_sec"]["model_queue"], total["time_sec"]["tool"]), (9.5, 2.5, 4)
        )
        self.assertEqual(total["time_sec"]["mean_per_model_call"], 2.375)
        self.assertAlmostEqual(total["time_sec"]["mean_per_tool_call"], 4 / 3)
        self.assertEqual((scoring["calls"]["model"], scoring["calls"]["tool"]), (1, 0))
        self.assertEqual(
            total["by_tool"]["search"],
            {"calls": 2, "errors": 1, "time_sec": 3, "mean_calls_per_task": 1, "mean_time_sec_per_call": 1.5},
        )
        self.assertEqual(summary["by_level"]["2"]["costs"]["rollout"]["tokens"]["total_tokens"], 7)

    def test_missing_historical_timing_stays_unknown_and_usage_coverage_is_explicit(self):
        cost = collect_costs(
            [{"role": "assistant", "usage": None}, {"role": "tool", "tool_name": "search"}],
            sample_sec=None,
            episode_sec=5,
        )
        record = ScoreRecord("one", ScoreResult("a", "a", 1.0, "rule"), costs={"rollout": cost})
        summary = build_summary("fixture", [record], "model")["costs"]["rollout"]
        self.assertEqual((summary["calls"]["model"], summary["calls"]["model_with_usage"]), (1, 0))
        self.assertIsNone(summary["time_sec"]["sample"])
        self.assertIsNone(summary["time_sec"]["model"])
        self.assertIsNone(summary["time_sec"]["mean_per_tool_call"])
        self.assertIsNone(summary["time_sec"]["wall"])


class DependencyBoundaryTests(unittest.TestCase):
    def test_internal_modules_use_relative_imports_without_eval_prefix(self):
        root = REPO / "eval/src/eval"
        for path in root.rglob("*.py"):
            if "data" in path.relative_to(root).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    module = node.module or ""
                    self.assertFalse(module == "eval" or module.startswith("eval."), str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name == "eval" or alias.name.startswith("eval."), str(path))

    def test_model_harness_agent_and_common_boundaries(self):
        root = REPO / "eval/src/eval"
        for path in (root / "model").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level >= 2:
                    self.fail(f"{path} imports outside model/: {node.module}")
        for path in (root / "harness").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level >= 2:
                    module = node.module or ""
                    self.assertFalse(module.startswith("benchmarks"), str(path))
        for node in ast.walk(ast.parse((root / "agent.py").read_text())):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                self.assertFalse(module.startswith("benchmarks"), "agent.py must not import benchmarks")
        for path in (root / "benchmarks/_common").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level >= 3:
                    module = node.module or ""
                    self.assertFalse(module.startswith(("model", "harness.env", "harness.tool")), str(path))
                    if module == "agent":
                        self.assertFalse(any(alias.name == "react" for alias in node.names), str(path))

    def test_benchmark_imports_stay_inside_their_own_directory_or_common(self):
        root = REPO / "eval/src/eval/benchmarks"
        for path in root.glob("*/*.py"):
            own = path.parent.name
            if own == "_common":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level == 2:
                    target = (node.module or "").split(".", 1)[0]
                    self.assertIn(target, {"_common", own}, str(path))

    def test_agent_uses_only_the_borrowed_environment_interface(self):
        tree = ast.parse((REPO / "eval/src/eval/agent.py").read_text())
        self.assertFalse(
            any(
                isinstance(node, ast.AsyncWith)
                and any(
                    isinstance(item.context_expr, ast.Name) and item.context_expr.id == "environment"
                    for item in node.items
                )
                for node in ast.walk(tree)
            )
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute) and node.attr in ("connect", "close", "setup", "teardown")
                for node in ast.walk(tree)
            )
        )


async def _run_score(args, *, evaluate=aime.score):
    with patch.object(aime, "load_tasks", side_effect=_load_tasks), patch.object(aime, "score", evaluate):
        return await aime.run_score(args)


def _load_tasks(args):
    return [
        Task(str(row["id"]), "system", str(row.get("question", "")), gold=row["gold"], meta=row.get("meta", {}))
        for row in json.loads(Path(args.dataset_file).read_text())
    ]


def _score_args(run_dir, **extra):
    return argparse.Namespace(
        **(
            {
                "run_dir": run_dir,
                "output_dir": None,
                "concurrency": 2,
                "judge_model": "judge",
                "judge_api_key": "secret",
                "judge_base_url": "https://judge.example/v1",
                "judge_max_tokens": 4096,
                "enable_thinking": True,
                "reasoning_effort": "medium",
                "timeout": 1,
                "seed": 42,
            }
            | extra
        )
    )


def _rollout_bytes(run_dir):
    return {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file() and not path.is_relative_to(run_dir / "scores")
    }


def _write_run(run_dir, *, benchmark="aime"):
    dataset = run_dir.parent / "dataset.json"
    dataset.write_text(json.dumps([{"id": "1", "question": "Q", "gold": 42, "meta": {}}]))
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "model": "model",
                "dataset_file": str(dataset),
                "sample_ids": ["1"],
                "collect_environment_outputs": True,
            }
        )
    )
    save_trajectory(Trajectory("1", final_answer='{"answer":42}'), trajectory_path(run_dir, "1"))
    save_environment_result(run_dir / "environment_outputs/1", "1", None, None)


if __name__ == "__main__":
    unittest.main()
