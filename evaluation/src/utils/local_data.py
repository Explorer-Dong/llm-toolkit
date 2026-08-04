from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import DownloadConfig, load_dataset
import huggingface_hub.constants as hf_constants


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_local_or_hf_dataset(
    repo_id: str,
    local_name: str,
    split: str,
    local_files_only: bool,
) -> Any:
    local_dir = DATA_DIR / local_name
    if _has_downloaded_files(local_dir):
        return _load_local_dataset(local_dir, split)

    download_config = DownloadConfig(local_files_only=local_files_only, max_retries=0)
    old_hf_offline = hf_constants.HF_HUB_OFFLINE
    if local_files_only:
        hf_constants.HF_HUB_OFFLINE = True
    try:
        try:
            return load_dataset(repo_id, split=split, download_config=download_config)
        except Exception as first_error:
            fallback_splits = ["test", "train", "validation"]
            for fallback_split in fallback_splits:
                if fallback_split == split:
                    continue
                try:
                    return load_dataset(repo_id, split=fallback_split, download_config=download_config)
                except Exception:
                    continue
            try:
                dataset_dict = load_dataset(repo_id, download_config=download_config)
            except Exception:
                raise first_error
            return _select_split(dataset_dict, split, first_error)
    finally:
        hf_constants.HF_HUB_OFFLINE = old_hf_offline


def _has_downloaded_files(local_dir: Path) -> bool:
    if not local_dir.exists():
        return False
    ignored_names = {".cache", ".git"}
    return any(path.name not in ignored_names for path in local_dir.iterdir())


def _load_local_dataset(local_dir: Path, split: str) -> Any:
    try:
        return load_dataset(str(local_dir), split=split)
    except Exception as first_error:
        try:
            dataset_dict = load_dataset(str(local_dir))
            return _select_split(dataset_dict, split, first_error)
        except Exception:
            pass

        parquet_files = sorted(str(path) for path in local_dir.rglob("*.parquet"))
        if parquet_files:
            dataset_dict = load_dataset("parquet", data_files=_group_data_files(parquet_files))
            return _select_split(dataset_dict, split, first_error)

        json_files = sorted(str(path) for path in local_dir.rglob("*.jsonl")) + sorted(
            str(path) for path in local_dir.rglob("*.json")
        )
        if json_files:
            dataset_dict = load_dataset("json", data_files=_group_data_files(json_files))
            return _select_split(dataset_dict, split, first_error)

        csv_files = sorted(str(path) for path in local_dir.rglob("*.csv"))
        if csv_files:
            dataset_dict = load_dataset("csv", data_files=_group_data_files(csv_files))
            return _select_split(dataset_dict, split, first_error)

        raise first_error


def _group_data_files(paths: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for path in paths:
        name = Path(path).name
        split = name.split("-")[0] if "-" in name else Path(path).stem
        if split not in {"train", "test", "validation", "dev"}:
            split = "train"
        grouped.setdefault(split, []).append(path)
    return grouped


def _select_split(dataset_dict: Any, split: str, first_error: Exception) -> Any:
    if split in dataset_dict:
        return dataset_dict[split]
    for fallback_split in ["test", "train", "validation", "dev"]:
        if fallback_split in dataset_dict:
            return dataset_dict[fallback_split]
    raise first_error
