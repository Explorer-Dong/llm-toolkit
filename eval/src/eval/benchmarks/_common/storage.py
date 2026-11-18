"""Output directory, dataset path, trajectory, and environment artifact storage helpers."""

import json
import shutil
from pathlib import Path
from typing import Any

from ...agent import Trajectory
from ...harness.utils import encode_id, load_trajectory, write_json

__all__ = [
    "PACKAGE_DIR",
    "trajectory_path",
    "create_directory",
    "resolve_dataset_file",
    "load_completed_sample",
    "prepare_environment_dir",
    "save_environment_result",
    "load_rollout_artifacts",
]

PACKAGE_DIR = Path(__file__).resolve().parents[2]


def trajectory_path(run_dir: Path, task_id: str) -> Path:
    return run_dir / "trajectories" / f"{encode_id(task_id)}.json"


def create_directory(directory: Path) -> Path:
    candidate = directory
    index = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            candidate = directory.with_name(f"{directory.name}-{index}")
            index += 1
        else:
            return candidate


def resolve_dataset_file(value: str | Path, package_dir: Path = PACKAGE_DIR) -> Path:
    """Resolve a dataset path stored either relative to the package root or absolute."""
    path = Path(value)
    if not path.is_absolute():
        path = package_dir / path
    return path.resolve()


def load_completed_sample(run_dir: Path, task_id: str, *, collect_enabled: bool = False) -> Trajectory | None:
    path = trajectory_path(run_dir, task_id)
    if not path.is_file():
        return None
    try:
        trace = load_trajectory(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if trace.task_id != task_id or trace.error is not None:
        return None
    if not collect_enabled:
        return trace
    environment_dir = run_dir / "environment_outputs" / encode_id(task_id)
    environment_result_path = environment_dir / "environment_result.json"
    if environment_dir.is_symlink() or not environment_dir.is_dir() or not environment_result_path.is_file():
        return None
    try:
        environment_result = json.loads(environment_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(environment_result, dict)
        or environment_result.get("task_id") != task_id
        or environment_result.get("error") is not None
    ):
        return None
    actual_files = set()
    for file in environment_dir.rglob("*"):
        if file.is_symlink():
            return None
        if file.is_file() and file != environment_result_path:
            actual_files.add(str(file.relative_to(environment_dir)))
    recorded_files = environment_result.get("files")
    if not isinstance(recorded_files, list) or actual_files != set(recorded_files):
        return None
    return trace


def prepare_environment_dir(run_dir: Path, task_id: str) -> Path:
    environment_dir = run_dir / "environment_outputs" / encode_id(task_id)
    if environment_dir.is_symlink():
        environment_dir.unlink()
    elif environment_dir.exists():
        shutil.rmtree(environment_dir)
    environment_dir.mkdir(parents=True, exist_ok=False)
    return environment_dir


def save_environment_result(directory: Path, task_id: str, data: dict | None, error: str | None) -> None:
    files = []
    for file in sorted(directory.rglob("*")):
        if file.name == "environment_result.json" and file.parent == directory:
            continue
        if file.is_symlink():
            raise ValueError("Collected evidence must contain regular files and directories")
        if file.is_file():
            files.append(str(file.relative_to(directory)))
    write_json(
        directory / "environment_result.json", {"task_id": task_id, "data": data, "files": files, "error": error}
    )


def load_rollout_artifacts(
    run_dir: Path, ids: list[str], config: dict[str, Any]
) -> tuple[dict[str, Trajectory], dict[str, dict | None]]:
    """Load and verify saved rollout trajectories and environment outputs against sample IDs."""
    trajectories: dict[str, Trajectory] = {}
    environment_outputs: dict[str, dict | None] = {}
    collect_environment_outputs = config.get("collect_environment_outputs", config.get("collect_artifacts", False))
    legacy_layout = "collect_environment_outputs" not in config and "collect_artifacts" in config
    environment_root = run_dir / ("artifacts" if legacy_layout else "environment_outputs")
    result_filename = "manifest.json" if legacy_layout else "environment_result.json"
    for task_id in ids:
        path = trajectory_path(run_dir, task_id)
        if not path.is_file():
            raise ValueError(f"Missing trajectory for sample {task_id!r}")
        trace = load_trajectory(path)
        if trace.task_id != task_id:
            raise ValueError(f"Trajectory id does not match sample {task_id!r}")
        trajectories[task_id] = trace
        environment_outputs[task_id] = None
        if not collect_environment_outputs:
            continue
        directory = environment_root / encode_id(task_id)
        if directory.is_symlink():
            raise ValueError("Environment output directories must be regular directories")
        environment_result_path = directory / result_filename
        environment_result = json.loads(environment_result_path.read_text(encoding="utf-8"))
        if environment_result["task_id"] != task_id:
            raise ValueError(f"Environment output id does not match sample {task_id!r}")
        if environment_result["error"] and trace.error is None:
            raise ValueError(f"Sample {task_id!r} has incomplete collection")
        actual_files = set()
        for file in directory.rglob("*"):
            if file.is_symlink():
                raise ValueError("Environment output files must stay inside their sample directory")
            if file.is_file() and file != environment_result_path:
                actual_files.add(str(file.relative_to(directory)))
        if actual_files != set(environment_result["files"]):
            raise ValueError(f"Environment output files do not match {result_filename} for {task_id!r}")
        environment_outputs[task_id] = environment_result["data"]
    if set((run_dir / "trajectories").glob("*.json")) != {trajectory_path(run_dir, i) for i in ids}:
        raise ValueError("Trajectory files do not match the saved sample ids")
    expected_environment_outputs = (
        {environment_root / encode_id(i) for i in ids} if collect_environment_outputs else set()
    )
    actual_environment_outputs = set(environment_root.iterdir()) if environment_root.exists() else set()
    if actual_environment_outputs != expected_environment_outputs:
        raise ValueError("Environment output directories do not match the saved sample ids")
    return trajectories, environment_outputs
