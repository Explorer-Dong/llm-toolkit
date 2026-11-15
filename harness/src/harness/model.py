"""OpenAI-compatible model requests, streaming responses, and per-run concurrency."""

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import AsyncOpenAI

__all__ = [
    "AssistantTurn",
    "Context",
    "ModelParams",
    "add_model_args",
    "add_thinking_args",
    "build_model_params",
    "call_model",
    "make_context",
]

_LLM_MAX_RETRIES = 3
_LOG_ONLY_KEYS = (
    "reasoning_content",
    "usage",
    "finish_reason",
    "tool_result",
    "is_error",
    "error",
    "incomplete",
    "latency_sec",
    "queue_sec",
    "tool_name",
)


@dataclass(frozen=True)
class ModelParams:
    """Static fields of one chat-completion request; messages and tools vary per call."""

    model: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    enable_thinking: bool = True
    extra_body: dict[str, Any] | None = None
    reasoning_effort: str = "medium"


@dataclass(frozen=True)
class Context:
    """Per-run resources shared by every task solve."""

    client: AsyncOpenAI
    params: ModelParams
    semaphore: asyncio.Semaphore


@dataclass
class AssistantTurn:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    error: BaseException | None = None
    latency_sec: float | None = None
    queue_sec: float | None = None

    def message(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": "assistant",
            "content": self.content or None,
            "reasoning_content": self.reasoning,
            "usage": self.usage,
            "finish_reason": self.finish_reason or "unknown",
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.error is not None:
            result.update(incomplete=True, error=f"{type(self.error).__name__}: {self.error}")
        if self.latency_sec is not None:
            result.update(latency_sec=self.latency_sec, queue_sec=self.queue_sec)
        return result


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    max_tokens: int = 8192,
    timeout: float = 900.0,
    concurrency: int = 4,
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
    add_thinking_args(parser)
    parser.add_argument("--concurrency", type=int, default=concurrency, help="maximum concurrent model requests")


def add_thinking_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        default=True,
        help="disable model thinking (enabled by default)",
    )
    parser.add_argument(
        "--reasoning-effort", default="medium", metavar="LEVEL", help="reasoning effort (default: medium)"
    )


def build_model_params(args: argparse.Namespace) -> ModelParams:
    """Build request parameters from the shared model options."""
    return ModelParams(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        enable_thinking=args.enable_thinking,
        extra_body={"top_k": args.top_k} if args.top_k else None,
        reasoning_effort=args.reasoning_effort,
    )


def make_context(
    base_url: str | None,
    api_key: str,
    params: ModelParams,
    concurrency: int = 4,
    *,
    timeout: float = 900,
) -> Context:
    if concurrency < 1:
        raise ValueError("Model concurrency must be positive")
    return Context(
        _make_model_client(api_key=api_key, base_url=base_url, timeout=timeout), params, asyncio.Semaphore(concurrency)
    )


async def call_model(
    ctx: Context,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> AssistantTurn:
    """Capture a streamed response, including partial output and its interruption."""
    started = time.perf_counter()
    admitted = None
    params = ctx.params
    extra_body = dict(params.extra_body or {})
    extra_body["chat_template_kwargs"] = {"enable_thinking": params.enable_thinking}
    request: dict[str, Any] = {
        "model": params.model,
        "messages": [{k: v for k, v in m.items() if k not in _LOG_ONLY_KEYS} for m in messages],
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": params.seed,
        "extra_body": extra_body,
    }
    if params.enable_thinking:
        request["reasoning_effort"] = params.reasoning_effort
    if tools:
        request["tools"] = tools
    turn = AssistantTurn()
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    stream = None
    try:
        async with ctx.semaphore:
            admitted = time.perf_counter()
            stream = await ctx.client.chat.completions.create(**request)
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    details = getattr(usage, "completion_tokens_details", None)
                    turn.usage = {
                        "prompt_tokens": usage.prompt_tokens or 0,
                        "completion_tokens": usage.completion_tokens or 0,
                        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
                    }
                    prompt_details = getattr(usage, "prompt_tokens_details", None)
                    cached_tokens = getattr(prompt_details, "cached_tokens", None)
                    if cached_tokens is not None:
                        turn.usage["cached_tokens"] = cached_tokens
                for choice in chunk.choices or ():
                    if choice.finish_reason:
                        turn.finish_reason = choice.finish_reason
                    delta = choice.delta
                    if getattr(delta, "content", None):
                        content.append(delta.content)
                    thought = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                    if thought:
                        reasoning.append(thought)
                    for call in getattr(delta, "tool_calls", None) or ():
                        index = call.index if call.index is not None else len(calls)
                        slot = calls.setdefault(index, {"id": "", "name": "", "arguments": []})
                        if call.id:
                            slot["id"] += call.id
                        if call.function and call.function.name:
                            slot["name"] += call.function.name
                        if call.function and call.function.arguments:
                            slot["arguments"].append(call.function.arguments)
    except BaseException as exc:
        turn.error = exc
    finally:
        turn.content = "".join(content)
        turn.reasoning = "".join(reasoning)
        turn.tool_calls = [
            {
                "id": calls[i]["id"] or f"call_missing_{i}",
                "type": "function",
                "function": {"name": calls[i]["name"], "arguments": "".join(calls[i]["arguments"])},
            }
            for i in sorted(calls)
        ]
        if stream is not None:
            try:
                await stream.close()
            except BaseException as exc:
                if turn.error is None:
                    turn.error = exc
        finished = time.perf_counter()
        turn.latency_sec = round(finished - started, 6)
        turn.queue_sec = round((admitted if admitted is not None else finished) - started, 6)
    return turn


def _make_model_client(*, api_key: str, base_url: str | None, timeout: float) -> AsyncOpenAI:
    # openai vendors httpx as httpx2 and types http_client against it; the plain httpx
    # client is interface-compatible at runtime. trust_env=False keeps local/LAN
    # endpoints away from ambient proxies
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=_LLM_MAX_RETRIES,
        http_client=httpx.AsyncClient(trust_env=False),  # ty: ignore[invalid-argument-type]
    )
