"""CLI dispatchers for benchmark rollout and scoring commands."""

import argparse
import asyncio
import importlib
import json
import re
import sys

__all__ = ["rollout_main", "scoring_main"]


def rollout_main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark rollout.")
    parser.add_argument("benchmark", help="directory name under benchmarks/, e.g. aime")
    args = parser.parse_args(sys.argv[1:2])
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.benchmark):
        parser.error("benchmark names must use snake_case")
    module = importlib.import_module(f"..{args.benchmark}.rollout", package=__package__)
    bench_args = module.parse_args(sys.argv[2:])
    asyncio.run(module.rollout(bench_args, module.load_tasks(bench_args)))
    return 0


def scoring_main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark scoring.")
    parser.add_argument("benchmark", help="directory name under benchmarks/, e.g. aime")
    args = parser.parse_args(sys.argv[1:2])
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.benchmark):
        parser.error("benchmark names must use snake_case")
    module = importlib.import_module(f"..{args.benchmark}.scoring", package=__package__)
    score_args = module.parse_args(sys.argv[2:])
    output, summary = asyncio.run(module.run_score(score_args))
    print(f"Scores: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
