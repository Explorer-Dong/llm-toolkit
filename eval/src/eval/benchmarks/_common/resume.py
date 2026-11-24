"""CLI option parsing, resume validation, and run directory initialization helpers."""

import argparse
import json
import re
import time
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ... import OUTPUT_DIR
from ...agent import Task, Trajectory
from ...harness.utils import encode_id, write_json
from .metrics import ScoreRecord, build_summary, load_score_record
from .storage import (
    PACKAGE_DIR,
    create_directory,
    load_completed_task,
    load_rollout_artifacts,
    resolve_dataset_file,
)

__all__ = [
    "SECRET_KEYS",
    "parse_rollout_args",
    "parse_score_args",
    "filter_pending_tasks",
    "restore_resume_args",
    "validate_resume_config",
    "init_rollout_dir",
    "finalize_rollout_dir",
    "init_score_dir",
    "finalize_score_dir",
]

SECRET_KEYS = {"api_key", "serper_api_key", "crawl4ai_token", "ds_tool_key", "judge_api_key"}
MODEL_NAMES = {"deepseek-flash": "deepseek-v4.1-flash-1017"}
_API_MODEL_NAMES = {persisted: api for api, persisted in MODEL_NAMES.items()}
ROLLOUT_OVERRIDABLE = frozenset(
    {
        "base_url",
        "api_key",
        "timeout",
        "concurrency",
        "max_retries",
        "request_attempts",
        "retry_delay",
        "ds_tool_url",
        "ds_tool_key",
        "serper_api_key",
        "crawl4ai_url",
        "crawl4ai_token",
        "java_home",
    }
)
ROLLOUT_IGNORED = frozenset(
    {
        "resume",
        "output_dir",
        "sample_concurrency",
        "sample_ids",
        "collect_environment_outputs",
        "wall_time_sec",
    }
)
SCORING_OVERRIDABLE = frozenset(
    {
        "judge_base_url",
        "judge_api_key",
        "timeout",
        "concurrency",
        "judge_retries",
    }
)
SCORING_IGNORED = frozenset(
    {
        "resume",
        "output_dir",
        "sample_concurrency",
        "run_dir",
        "source_run_dir",
        "scoring_method",
        "group_by",
        "dataset_file",
        "wall_time_sec",
        "judge_prompt",
        "act_prompt",
        "abstain_prompt",
        "scoring_description",
    }
)


def parse_rollout_args(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser.add_argument("--limit", type=int, help="run the first N matching samples")
    parser.add_argument("--output-dir", help="new rollout directory; defaults to outputs/<benchmark>/<run>")
    parser.add_argument("--resume", type=Path, help="existing rollout directory to resume")
    args = parser.parse_args(argv)
    if args.resume is not None:
        if args.output_dir is not None:
            parser.error("--resume cannot be combined with --output-dir")
        resume_dir = Path(args.resume).resolve()
        config_path = resume_dir / "config.json"
        if not config_path.is_file():
            parser.error(f"Missing config.json in resume directory: {resume_dir}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        expected_benchmark = getattr(args, "benchmark", None)
        if expected_benchmark is not None and saved.get("benchmark") != expected_benchmark:
            parser.error(f"Expected a {expected_benchmark!r} run, got {saved.get('benchmark')!r}")
        restore_resume_args(
            parser,
            args,
            argv,
            saved,
            package_dir=PACKAGE_DIR,
            overridable=ROLLOUT_OVERRIDABLE,
            ignored=ROLLOUT_IGNORED,
        )
        args.resume = resume_dir
    if not args.model:
        parser.error("--model or MODEL_NAME is required")
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_turns < 1 or args.task_timeout < 0:
        parser.error("max_turns must be positive and task_timeout must be nonnegative")
    return args


def parse_score_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    concurrency: int = 1,
) -> argparse.Namespace:
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="existing score directory to resume")
    parser.add_argument("--concurrency", type=int, default=concurrency, help="maximum concurrent tasks")
    args = parser.parse_args(argv)
    if args.resume is not None:
        if args.output_dir is not None:
            parser.error("--resume cannot be combined with --output-dir")
        resume_dir = Path(args.resume).resolve()
        config_path = resume_dir / "config.json"
        if not config_path.is_file():
            parser.error(f"Missing config.json in resume directory: {resume_dir}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        source_run_dir = saved.get("source_run_dir")
        if not source_run_dir:
            parser.error(f"Missing source_run_dir in resume config: {config_path}")
        if args.run_dir is not None and Path(args.run_dir).resolve() != Path(source_run_dir).resolve():
            parser.error("run_dir does not match source_run_dir in resume config")
        args.run_dir = Path(source_run_dir)
        restore_resume_args(
            parser,
            args,
            argv,
            saved,
            package_dir=PACKAGE_DIR,
            overridable=SCORING_OVERRIDABLE,
            ignored=SCORING_IGNORED,
        )
        args.resume = resume_dir
    elif args.run_dir is None:
        parser.error("run_dir is required unless --resume is specified")
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    return args


def filter_pending_tasks(args: argparse.Namespace, tasks: list[Task], *, collect_enabled: bool = False) -> list[Task]:
    """Return tasks that do not yet have a completed output in the resumed run directory."""
    resume_dir = getattr(args, "resume", None)
    if resume_dir is None:
        return tasks
    run_dir = Path(resume_dir).resolve()
    return [task for task in tasks if load_completed_task(run_dir, task.id, collect_enabled=collect_enabled) is None]


def restore_resume_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    saved: dict[str, Any],
    *,
    package_dir: Path,
    overridable: AbstractSet[str],
    ignored: AbstractSet[str],
) -> dict[str, Any]:
    """Restore saved run parameters onto args while rejecting non-system overrides."""
    explicit = _explicit_cli_args(parser, argv)
    path_dests = {action.dest for action in parser._actions if action.type is Path}
    option_names = {action.dest: action.option_strings[0] for action in parser._actions if action.option_strings}
    for key, saved_value in saved.items():
        if key in ignored or key in SECRET_KEYS or key not in vars(args):
            continue
        if key in overridable:
            if key not in explicit:
                restored = Path(saved_value) if key in path_dests and saved_value is not None else saved_value
                setattr(args, key, restored)
            continue
        is_path = key in path_dests
        if key in explicit and not _resume_values_match(
            key, explicit[key], saved_value, package_dir=package_dir, is_path=is_path
        ):
            parser.error(f"{option_names.get(key, key)} cannot differ from saved config on --resume")
        if key == "dataset_file" and saved_value is not None:
            setattr(args, key, str(resolve_dataset_file(saved_value, package_dir)))
        elif is_path and saved_value is not None:
            setattr(args, key, Path(saved_value))
        elif key in {"model", "judge_model"}:
            setattr(args, key, explicit[key] if key in explicit else _api_model(saved_value))
        else:
            setattr(args, key, saved_value)
    return explicit


def validate_resume_config(
    args: argparse.Namespace,
    saved: dict[str, Any],
    *,
    package_dir: Path,
    overridable: AbstractSet[str],
    ignored: AbstractSet[str],
) -> None:
    """Ensure non-system parameters on args match the saved config when resuming."""
    for key, saved_value in saved.items():
        if key in ignored or key in overridable or key in SECRET_KEYS or key not in vars(args):
            continue
        current = getattr(args, key)
        if not _resume_values_match(
            key, current, saved_value, package_dir=package_dir, is_path=isinstance(current, Path)
        ):
            raise ValueError(f"Resume {key} differs from saved config: {current!r} != {saved_value!r}")


def init_rollout_dir(
    args: argparse.Namespace,
    tasks: list[Task],
    *,
    collect_enabled: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Any] | None, dict[str, Trajectory], list[Task]]:
    if args.concurrency < 1:
        raise ValueError("Concurrency must be positive")
    resume_dir = getattr(args, "resume", None)
    if resume_dir is not None and getattr(args, "output_dir", None) is not None:
        raise ValueError("Cannot combine --resume with --output-dir")
    ids = [task.id for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Task ids must be unique")
    for task_id in ids:
        encode_id(task_id)
    data_file = Path(args.dataset_file).resolve()
    saved_config: dict[str, Any] | None = None
    if resume_dir is not None:
        run_dir = Path(resume_dir).resolve()
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise ValueError(f"Missing config.json in resume directory: {run_dir}")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config.get("benchmark") != args.benchmark:
            raise ValueError(f"Expected a {args.benchmark!r} run, got {saved_config.get('benchmark')!r}")
        if saved_config.get("sample_ids") != ids:
            raise ValueError("Dataset selection does not match the rollout sample ids")
        if bool(saved_config.get("collect_environment_outputs")) != collect_enabled:
            raise ValueError("Collection mode does not match the saved rollout config")
        if saved_config.get("model") != MODEL_NAMES.get(args.model, args.model):
            raise ValueError(f"Model {args.model!r} does not match saved model {saved_config.get('model')!r}")
        if (
            "dataset_file" in saved_config
            and resolve_dataset_file(saved_config["dataset_file"], PACKAGE_DIR) != data_file
        ):
            raise ValueError("Dataset file does not match the saved rollout config")
        validate_resume_config(
            args,
            saved_config,
            package_dir=PACKAGE_DIR,
            overridable=ROLLOUT_OVERRIDABLE,
            ignored=ROLLOUT_IGNORED,
        )
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        model = re.sub(r"[^A-Za-z0-9_.-]+", "-", MODEL_NAMES.get(args.model, args.model)).lower()
        run_dir = (
            Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR / args.benchmark / f"{model}_{timestamp}"
        )
        if args.output_dir:
            run_dir.mkdir(parents=True, exist_ok=False)
        else:
            run_dir = create_directory(run_dir)
    config = {key: value for key, value in vars(args).items() if key not in SECRET_KEYS and key != "resume"}
    config["model"] = MODEL_NAMES.get(args.model, args.model)
    if saved_config is not None and "output_dir" in saved_config:
        config["output_dir"] = saved_config["output_dir"]
    config.update(
        dataset_file=(
            str(data_file.relative_to(PACKAGE_DIR)) if data_file.is_relative_to(PACKAGE_DIR) else str(data_file)
        ),
        sample_ids=ids,
        collect_environment_outputs=collect_enabled,
    )
    completed: dict[str, Trajectory] = {}
    pending: list[Task] = []
    if resume_dir is not None:
        for task in tasks:
            existing_trace = load_completed_task(run_dir, task.id, collect_enabled=collect_enabled)
            if existing_trace is not None:
                completed[task.id] = existing_trace
            else:
                pending.append(task)
        print(f"Resuming {run_dir}: skipping {len(completed)} completed, running {len(pending)} remaining")
    else:
        pending = list(tasks)
    write_json(run_dir / "config.json", config)
    return run_dir, config, saved_config, completed, pending


def finalize_rollout_dir(
    run_dir: Path,
    config: dict[str, Any],
    saved_config: dict[str, Any] | None,
    *,
    started: float,
    resumed_count: int,
    benchmark: str,
    traces: list[Trajectory],
) -> None:
    elapsed = time.perf_counter() - started
    if saved_config is None or not resumed_count:
        config["wall_time_sec"] = round(elapsed, 6)
    elif saved_config.get("wall_time_sec") is not None:
        config["wall_time_sec"] = round(saved_config["wall_time_sec"] + elapsed, 6)
    else:
        config["wall_time_sec"] = None
    write_json(run_dir / "config.json", config)
    errors = sum(
        trace.error is not None and trace.episode_status not in {"invalid_response", "context_length_exceeded"}
        for trace in traces
    )
    print(f"Saved {len(traces)} trajectories; system errors: {errors}; output: {run_dir}")
    print(f"Score separately: scoring {benchmark} {run_dir}")


def init_score_dir(
    args: argparse.Namespace,
    load_tasks: Callable[[argparse.Namespace], list[Task]],
    *,
    benchmark: str,
    method: str,
    group_by: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> tuple[
    Path,
    list[Task],
    dict[str, Trajectory],
    dict[str, dict | None],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, ScoreRecord],
    float | None,
]:
    resume_dir = getattr(args, "resume", None)
    if resume_dir is not None and getattr(args, "output_dir", None) is not None:
        raise ValueError("Cannot combine --resume with --output-dir")
    saved_score_config: dict[str, Any] | None = None
    resume_output: Path | None = None
    if resume_dir is not None:
        resume_output = Path(resume_dir).resolve()
        score_config_path = resume_output / "config.json"
        if not score_config_path.is_file():
            raise ValueError(f"Missing config.json in resume directory: {resume_output}")
        saved_score_config = json.loads(score_config_path.read_text(encoding="utf-8"))
        if saved_score_config.get("benchmark") != benchmark:
            raise ValueError(f"Expected a {benchmark!r} score run, got {saved_score_config.get('benchmark')!r}")
        if saved_score_config.get("scoring_method") != method:
            raise ValueError(f"Expected scoring method {method!r}, got {saved_score_config.get('scoring_method')!r}")
        saved_run_dir = saved_score_config.get("source_run_dir")
        if not saved_run_dir:
            raise ValueError(f"Missing source_run_dir in resume config: {score_config_path}")
        if getattr(args, "run_dir", None) is None:
            args.run_dir = Path(saved_run_dir)
    run_dir = Path(args.run_dir).resolve()
    if saved_score_config is not None:
        if Path(saved_score_config["source_run_dir"]).resolve() != run_dir:
            raise ValueError("Source run directory does not match the saved score config")
        validate_resume_config(
            args,
            saved_score_config,
            package_dir=PACKAGE_DIR,
            overridable=SCORING_OVERRIDABLE,
            ignored=SCORING_IGNORED,
        )
    rollout_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if rollout_config["benchmark"] != benchmark:
        raise ValueError(f"Expected a {benchmark!r} run, got {rollout_config['benchmark']!r}")
    if args.concurrency < 1:
        raise ValueError("Concurrency must be positive")
    data_file = resolve_dataset_file(rollout_config["dataset_file"], PACKAGE_DIR)
    tasks = load_tasks(argparse.Namespace(**(rollout_config | {"dataset_file": str(data_file)})))
    ids = [task.id for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Task ids must be unique")
    if ids != rollout_config["sample_ids"]:
        raise ValueError("Dataset selection does not match the rollout task ids")
    trajectories, environment_outputs = load_rollout_artifacts(run_dir, ids, rollout_config)

    if resume_output is not None:
        output = resume_output
    elif args.output_dir:
        output = Path(args.output_dir).resolve()
    else:
        output = run_dir / "scores" / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if run_dir.is_relative_to(output) or (
        output.is_relative_to(run_dir) and not output.is_relative_to(run_dir / "scores")
    ):
        raise ValueError("Score output must be separate from rollout outputs")
    if data_file.is_relative_to(output):
        raise ValueError("Score output must be separate from the dataset")
    if resume_output is None:
        if args.output_dir:
            output.mkdir(parents=True, exist_ok=False)
        else:
            output = create_directory(output)

    kept: dict[str, ScoreRecord] = {}
    if resume_output is not None:
        trajectories_file = output / "trajectories.json"
        if trajectories_file.is_file():
            raw_records = json.loads(trajectories_file.read_text(encoding="utf-8"))
            if not isinstance(raw_records, list):
                raise ValueError(f"Saved trajectories must be a JSON list: {trajectories_file}")
            for raw in raw_records:
                record = load_score_record(raw)
                if (
                    record.task_id in trajectories
                    and record.score_result.sys_error is None
                    and record.agent_error == trajectories[record.task_id].error
                ):
                    kept[record.task_id] = record
        pending_count = sum(1 for task in tasks if task.id not in kept)
        print(f"Resuming scoring in {output}: skipping {len(kept)} completed, running {pending_count} remaining")

    prev_wall: float | None = None
    if saved_score_config is not None:
        prev_wall = saved_score_config.get("wall_time_sec")
        if prev_wall is None and (output / "summary.json").is_file():
            prev_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            prev_wall = prev_summary.get("costs", {}).get("scoring", {}).get("time_sec", {}).get("wall")

    score_config = {key: value for key, value in vars(args).items() if key not in SECRET_KEYS and key != "resume"}
    if "judge_model" in score_config:
        score_config["judge_model"] = _canonical_model(score_config["judge_model"])
    if saved_score_config is not None and "output_dir" in saved_score_config:
        score_config["output_dir"] = saved_score_config["output_dir"]
    score_config.update(
        source_run_dir=str(run_dir),
        benchmark=benchmark,
        scoring_method=method,
        group_by=group_by,
        dataset_file=str(data_file),
    )
    score_config.update(metadata or {})
    write_json(output / "config.json", score_config)
    write_json(output / "trajectories.json", [asdict(r) for r in kept.values()])
    return (
        output,
        tasks,
        trajectories,
        environment_outputs,
        rollout_config,
        score_config,
        saved_score_config,
        kept,
        prev_wall,
    )


def finalize_score_dir(
    output: Path,
    score_config: dict[str, Any],
    saved_score_config: dict[str, Any] | None,
    kept: dict[str, ScoreRecord],
    prev_wall: float | None,
    started: float,
    ordered: list[ScoreRecord] | None,
    *,
    benchmark: str,
    rollout_config: dict[str, Any],
    group_by: tuple[str, ...] = (),
    summarize: Callable[[list[ScoreRecord]], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    elapsed = time.perf_counter() - started
    if saved_score_config is None or not kept:
        scoring_wall_sec: float | None = elapsed
    elif prev_wall is not None:
        scoring_wall_sec = prev_wall + elapsed
    else:
        scoring_wall_sec = None
    score_config["wall_time_sec"] = round(scoring_wall_sec, 6) if scoring_wall_sec is not None else None
    write_json(output / "config.json", score_config)
    if ordered is None:
        return None
    write_json(output / "trajectories.json", [asdict(r) for r in ordered])
    summary = build_summary(
        benchmark,
        ordered,
        MODEL_NAMES.get(rollout_config["model"], rollout_config["model"]),
        group_by=group_by,
        rollout_wall_sec=rollout_config.get("wall_time_sec"),
        scoring_wall_sec=scoring_wall_sec,
    )
    if summarize is not None:
        summary.update(summarize(ordered))
    write_json(output / "summary.json", summary)
    return summary


def _resume_values_match(
    key: str,
    current: Any,
    saved_value: Any,
    *,
    package_dir: Path,
    is_path: bool,
) -> bool:
    if key in {"model", "judge_model"}:
        return _canonical_model(current) == _canonical_model(saved_value)
    if key == "dataset_file" and saved_value is not None and current is not None:
        return Path(current).resolve() == resolve_dataset_file(saved_value, package_dir)
    if is_path and saved_value is not None and current is not None:
        return Path(current).resolve() == Path(saved_value).resolve()
    return current == saved_value


def _canonical_model(value: Any) -> Any:
    if isinstance(value, list):
        return [MODEL_NAMES.get(item, item) for item in value]
    if isinstance(value, str):
        return MODEL_NAMES.get(value, value)
    return value


def _api_model(value: Any) -> Any:
    if isinstance(value, list):
        return [_API_MODEL_NAMES.get(item, item) for item in value]
    if isinstance(value, str):
        return _API_MODEL_NAMES.get(value, value)
    return value


def _explicit_cli_args(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> dict[str, Any]:
    saved_action_defaults = [(action, action.default) for action in parser._actions]
    saved_parser_defaults = dict(parser._defaults)
    try:
        for action in parser._actions:
            action.default = argparse.SUPPRESS
        parser._defaults.clear()
        return vars(parser.parse_args(argv))
    finally:
        for action, default in saved_action_defaults:
            action.default = default
        parser._defaults.clear()
        parser._defaults.update(saved_parser_defaults)
