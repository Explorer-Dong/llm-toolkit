"""Independent scoring, dataset provenance, and immutable inference evidence."""

import argparse
import ast
import asyncio
import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from harness import Trajectory, save_trajectory
from harness.model import AssistantTurn, Context, ModelParams

from eval import scoring as eval_scoring
from eval.benchmarks.aime import scoring as aime
from eval.benchmarks.browsecomp import scoring as browsecomp
from eval.inference import _save_artifacts
from eval.utils import trajectory_path

__all__ = []
REPO = Path(__file__).resolve().parents[2]


class ScoreArtifactTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name) / "run"
        self.run_dir.mkdir()
        _write_run(self.run_dir)

    async def test_repeated_scoring_keeps_previous_results_and_inference(self):
        original = _inference_bytes(self.run_dir)
        with patch.object(eval_scoring, "datetime") as clock:
            clock.now.return_value = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
            first, _ = await _run_score(_score_args(self.run_dir))
            first_bytes = {path.name: path.read_bytes() for path in first.iterdir()}
            second, _ = await _run_score(_score_args(self.run_dir))
        self.assertNotEqual(first, second)
        self.assertRegex(first.name, r"^\d{8}-\d{6}$")
        self.assertEqual(second.name, first.name + "-1")
        self.assertEqual({path.name: path.read_bytes() for path in first.iterdir()}, first_bytes)
        self.assertEqual(_inference_bytes(self.run_dir), original)

    async def test_benchmark_owns_grading_policy_independently_of_inference_options(self):
        config_path = self.run_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["scoring_method"] = "old_policy"
        config_path.write_text(json.dumps(config))
        original = _inference_bytes(self.run_dir)
        _, summary = await _run_score(_score_args(self.run_dir))
        self.assertEqual((summary["benchmark"], summary["score"], summary["sys_errors"]), ("aime", 1, 0))
        self.assertEqual(_inference_bytes(self.run_dir), original)

    async def test_wrong_benchmark_is_rejected_before_grading(self):
        evaluate = AsyncMock()
        with self.assertRaisesRegex(ValueError, "Expected a"):
            await _run_score(_score_args(self.run_dir), evaluate=evaluate, benchmark="gpqa")
        evaluate.assert_not_awaited()
        self.assertFalse((self.run_dir / "scores").exists())

    async def test_missing_trajectory_is_explicit_before_judge_creation(self):
        _write_run(self.run_dir, benchmark="browsecomp")
        trajectory_path(self.run_dir, "1").unlink()
        with (
            patch.object(browsecomp, "load_tasks", side_effect=_load_tasks),
            patch.object(browsecomp, "make_context") as build,
            self.assertRaisesRegex(ValueError, "Missing trajectory"),
        ):
            await browsecomp._run_score(_score_args(self.run_dir))
        build.assert_not_called()
        self.assertFalse((self.run_dir / "scores").exists())

    async def test_dataset_change_is_rejected_before_grading(self):
        (self.run_dir.parent / "dataset.json").write_text('[{"id":"1","gold":43}]')
        evaluate = AsyncMock()
        with self.assertRaisesRegex(ValueError, "Dataset checksum"):
            await _run_score(_score_args(self.run_dir), evaluate=evaluate)
        evaluate.assert_not_awaited()
        self.assertFalse((self.run_dir / "scores").exists())

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
        with self.assertRaisesRegex(ValueError, "unique"):
            await eval_scoring.run_score(
                _score_args(self.run_dir),
                lambda args: _load_tasks(args) * 2,
                aime.score,
                benchmark="aime",
                method="integer",
            )
        save_trajectory(Trajectory("extra"), trajectory_path(self.run_dir, "extra"))
        with self.assertRaisesRegex(ValueError, "saved sample ids"):
            await _run_score(_score_args(self.run_dir))

    async def test_scoring_cannot_overwrite_inference_artifacts(self):
        original = _inference_bytes(self.run_dir)
        for output in (self.run_dir, self.run_dir.parent, self.run_dir / "trajectories/new"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "separate"):
                await _run_score(_score_args(self.run_dir, output_dir=output))
        self.assertEqual(_inference_bytes(self.run_dir), original)

    async def test_judge_failure_is_scoring_only_and_keeps_agent_trace(self):
        _write_run(self.run_dir, benchmark="browsecomp")
        original = _inference_bytes(self.run_dir)
        client = MagicMock(close=AsyncMock())
        ctx = Context(client, ModelParams("judge", 0, 1, 1024, 42), asyncio.Semaphore(1))
        with (
            patch.object(browsecomp, "load_tasks", side_effect=_load_tasks),
            patch.object(browsecomp, "make_context", return_value=ctx),
            patch.object(
                browsecomp, "call_model", new_callable=AsyncMock, side_effect=RuntimeError("judge unavailable")
            ),
        ):
            output, summary = await browsecomp._run_score(_score_args(self.run_dir))
        record = json.loads((output / "trajectories.json").read_text())[0]
        self.assertIsNone(record["agent_error"])
        self.assertEqual(record["score_result"]["sys_error"], "RuntimeError: judge unavailable")
        self.assertEqual((summary["agent_errors"], summary["scoring_errors"], summary["score"]), (0, 1, 0.0))
        self.assertEqual(_inference_bytes(self.run_dir), original)
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

    async def test_artifact_tampering_and_missing_files_are_rejected_before_scoring(self):
        directory = self.run_dir / "artifacts/1"
        (directory / "evidence.txt").write_text("original")
        _save_artifacts(directory, "1", {"file": "evidence.txt"}, None)
        (directory / "evidence.txt").write_text("tampered")
        evaluate = AsyncMock()
        with self.assertRaisesRegex(ValueError, "checksum"):
            await _run_score(_score_args(self.run_dir), evaluate=evaluate)
        evaluate.assert_not_awaited()
        self.assertFalse((self.run_dir / "scores").exists())
        (directory / "evidence.txt").unlink()
        with self.assertRaisesRegex(ValueError, "manifest"):
            await _run_score(_score_args(self.run_dir))

    async def test_incomplete_collection_requires_an_execution_error(self):
        _save_artifacts(self.run_dir / "artifacts/1", "1", None, "collection failed")
        with self.assertRaisesRegex(ValueError, "incomplete collection"):
            await _run_score(_score_args(self.run_dir))

    async def test_hybrid_policy_combines_rule_and_judge_in_one_scoring_stage(self):
        ctx = Context(MagicMock(), ModelParams("judge", 0, 1, 1024, 42), asyncio.Semaphore(1))

        async def hybrid(sample, trace, artifacts, config):
            rule = await aime.score(sample, trace, artifacts, config)
            judged = await browsecomp.score(sample, trace, artifacts, config, ctx=ctx)
            judged.method = "hybrid"
            judged.score = 0.4 * rule.score + 0.6 * judged.score
            return judged

        with patch.object(browsecomp, "call_model", return_value=AssistantTurn("correct: no")):
            output, summary = await eval_scoring.run_score(
                _score_args(self.run_dir), _load_tasks, hybrid, benchmark="aime", method="hybrid"
            )
        record = json.loads((output / "trajectories.json").read_text())[0]["score_result"]
        self.assertEqual((summary["score"], record["method"]), (0.4, "hybrid"))
        self.assertEqual(len(record["messages"]), 2)

    def test_summary_preserves_fractional_scores_and_separate_tool_errors(self):
        records = [
            eval_scoring.ScoreRecord(
                "1", eval_scoring.ScoreResult("a", "a", 0.375, "judge"), tool_errors=2, metadata={"level": 1}
            ),
            eval_scoring.ScoreRecord(
                "2", eval_scoring.ScoreResult("a", None, 0.0, "judge"), agent_error="failed", metadata={"level": 2}
            ),
        ]
        summary = eval_scoring.build_summary("test", records, "model", group_by=("level",))
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
            eval_scoring.ScoreRecord(
                "one",
                eval_scoring.ScoreResult("a", "a", 1.0, "judge"),
                metadata={"level": 1},
                costs={
                    "inference": eval_scoring._collect_costs(first, sample_sec=12, episode_sec=10),
                    "scoring": eval_scoring._collect_costs(judge, sample_sec=4, episode_sec=4),
                },
            ),
            eval_scoring.ScoreRecord(
                "two",
                eval_scoring.ScoreResult("b", None, 0.0, "judge"),
                agent_error="failed",
                metadata={"level": 2},
                costs={
                    "inference": eval_scoring._collect_costs(second, sample_sec=6, episode_sec=5),
                    "scoring": eval_scoring._collect_costs([], sample_sec=0, episode_sec=0),
                },
            ),
        ]
        summary = eval_scoring.build_summary(
            "fixture", records, "model", group_by=("level",), inference_wall_sec=12, scoring_wall_sec=4
        )
        inference, scoring, total = (summary["costs"][phase] for phase in ("inference", "scoring", "total"))
        self.assertEqual(
            (
                inference["tokens"]["prompt_tokens"],
                inference["tokens"]["completion_tokens"],
                inference["tokens"]["total_tokens"],
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
        self.assertEqual(summary["by_level"]["2"]["costs"]["inference"]["tokens"]["total_tokens"], 7)

    def test_missing_historical_timing_stays_unknown_and_usage_coverage_is_explicit(self):
        cost = eval_scoring._collect_costs(
            [{"role": "assistant", "usage": None}, {"role": "tool", "tool_name": "search"}],
            sample_sec=None,
            episode_sec=5,
        )
        record = eval_scoring.ScoreRecord(
            "one", eval_scoring.ScoreResult("a", "a", 1.0, "rule"), costs={"inference": cost}
        )
        summary = eval_scoring.build_summary("fixture", [record], "model")["costs"]["inference"]
        self.assertEqual((summary["calls"]["model"], summary["calls"]["model_with_usage"]), (1, 0))
        self.assertIsNone(summary["time_sec"]["sample"])
        self.assertIsNone(summary["time_sec"]["model"])
        self.assertIsNone(summary["time_sec"]["mean_per_tool_call"])
        self.assertIsNone(summary["time_sec"]["wall"])


class DependencyBoundaryTests(unittest.TestCase):
    def test_harness_imports_are_independent_of_eval(self):
        for path in (REPO / "harness/src/harness").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("eval"), str(path))
                elif isinstance(node, ast.Import):
                    self.assertTrue(all(not alias.name.startswith("eval") for alias in node.names), str(path))

    def test_benchmark_imports_stay_inside_their_own_directory(self):
        for path in (REPO / "eval/src/eval/benchmarks").glob("*/*.py"):
            own_package = f"eval.benchmarks.{path.parent.name}"
            for node in ast.walk(ast.parse(path.read_text())):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                for name in names:
                    if name.startswith("eval.benchmarks."):
                        self.assertTrue(name.startswith(own_package + "."), str(path))

    def test_agent_uses_only_the_borrowed_environment_interface(self):
        tree = ast.parse((REPO / "harness/src/harness/agent.py").read_text())
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


async def _run_score(args, *, evaluate=aime.score, benchmark="aime"):
    return await eval_scoring.run_score(args, _load_tasks, evaluate, benchmark=benchmark, method="integer")


def _load_tasks(args):
    return json.loads(Path(args.dataset_file).read_text())


def _score_args(run_dir, **extra):
    return argparse.Namespace(
        **(
            {
                "run_dir": run_dir,
                "output_dir": None,
                "concurrency": 2,
                "sample_concurrency": 2,
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


def _inference_bytes(run_dir):
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
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "sample_ids": ["1"],
                "collect_artifacts": True,
            }
        )
    )
    save_trajectory(Trajectory("1", final_answer='{"answer":42}'), trajectory_path(run_dir, "1"))
    _save_artifacts(run_dir / "artifacts/1", "1", None, None)


if __name__ == "__main__":
    unittest.main()
