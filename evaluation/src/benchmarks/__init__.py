from __future__ import annotations

from typing import Any

from src.benchmarks.aime26 import AIME26Benchmark
from src.benchmarks.base import Benchmark
from src.benchmarks.gpqa_diamond import GPQADiamondBenchmark

BENCHMARKS = {
    "gpqa_diamond": GPQADiamondBenchmark,
    "aime26": AIME26Benchmark,
}

def get_benchmark(name: str, settings: dict[str, Any] | None = None) -> Benchmark:
    benchmark_class = BENCHMARKS.get(name)
    if benchmark_class is None:
        valid = ", ".join(sorted(BENCHMARKS))
        raise ValueError(f"Unknown benchmark {name!r}. Valid benchmarks: {valid}")
    return benchmark_class()  # type: ignore[call-arg]
