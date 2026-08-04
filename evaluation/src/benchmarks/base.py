from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.utils.call_llm import LLMClient
from src.utils.schema import Prediction, RunConfig, Task


class Benchmark(ABC):
    name: str

    @abstractmethod
    def load_tasks(self, split: str, limit: int | None) -> list[Task]:
        ...

    @abstractmethod
    def run_one(
        self,
        task: Task,
        client: LLMClient,
        config: RunConfig,
    ) -> Prediction:
        ...

    @abstractmethod
    def score(
        self,
        predictions: list[Prediction],
        output_dir: Path,
    ) -> dict[str, Any]:
        ...
