"""Benchmark execution and independent offline scoring."""

from pathlib import Path

__all__ = ["OUTPUT_DIR", "ROOT"]

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).parent.parent.name == "src" else Path.cwd()
OUTPUT_DIR = ROOT / "outputs"
