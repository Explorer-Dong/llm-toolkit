"""Model client, CLI options, and request helpers."""

import argparse
import os
from typing import Any

from .openai import AssistantTurn, call_model, make_client

__all__ = [
    "AssistantTurn",
    "make_client",
    "call_model",
    "add_model_args",
    "build_model_params",
]


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    max_tokens: int = 8192,
    timeout: float = 900.0,
) -> None:
    """Register model connection and generation options."""
    parser.add_argument("--model", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=temperature, help="sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling cutoff")
    parser.add_argument("--top-k", type=int, default=top_k, help="0 = do not send top_k")
    parser.add_argument("--max-tokens", type=int, default=max_tokens, help="completion token limit per model request")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed for reproducible runs")
    parser.add_argument("--timeout", type=float, default=timeout, help="per model request timeout in seconds")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable model thinking (default: enabled)",
    )
    parser.add_argument(
        "--reasoning-effort", default="medium", metavar="LEVEL", help="reasoning effort (default: medium)"
    )


def build_model_params(args: argparse.Namespace) -> dict[str, Any]:
    """Build keyword arguments for call_model from the shared model options."""
    return {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "extra_body": {**getattr(args, "extra_body", {}), **({"top_k": args.top_k} if args.top_k else {})} or None,
        "reasoning_effort": args.reasoning_effort,
        "stream": getattr(args, "stream", True),
        "max_tokens_field": getattr(args, "max_tokens_field", "max_tokens"),
        "include_reasoning": getattr(args, "include_reasoning", True),
    }
