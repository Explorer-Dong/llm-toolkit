"""CLI dispatchers for benchmark rollout and scoring commands."""

import argparse
import importlib
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
    return module.main(sys.argv[2:])


def scoring_main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark scoring.")
    parser.add_argument("benchmark", help="directory name under benchmarks/, e.g. aime")
    args = parser.parse_args(sys.argv[1:2])
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.benchmark):
        parser.error("benchmark names must use snake_case")
    module = importlib.import_module(f"..{args.benchmark}.scoring", package=__package__)
    return module.main(sys.argv[2:])
