"""Model call primitives: request assembly, streaming parse, and per-call capture."""

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI

__all__ = ["Context", "ModelParams", "add_model_args", "build_llm_client", "build_model_params", "call_model"]

_LLM_MAX_RETRIES = 3
_LOG_ONLY_KEYS = ("reasoning_content", "usage")


@dataclass(frozen=True)
class ModelParams:
    """Static fields of one chat-completion request; messages and tools vary per call."""

    model: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    enable_thinking: bool | None = None  # None = leave the chat template default
    extra_body: dict[str, Any] | None = None


@dataclass(frozen=True)
class Context:
    """Per-run resources shared by every task solve."""

    client_llm: AsyncOpenAI
    params: ModelParams
    sem: asyncio.Semaphore


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    max_tokens: int = 8192,
    timeout: float = 900.0,
    concurrency: int = 4,
    include_judge: bool = False,
) -> None:
    """Register model connection and generation options, plus optional judge connection options."""
    parser.add_argument("--model", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=temperature, help="sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling cutoff")
    parser.add_argument("--top-k", type=int, default=top_k, help="0 = do not send top_k")
    parser.add_argument("--max-tokens", type=int, default=max_tokens, help="completion token limit per model request")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed for reproducible runs")
    parser.add_argument("--timeout", type=float, default=timeout, help="per model request timeout in seconds")
    parser.add_argument("--enable-thinking", action="store_true", help="enable the model's thinking mode")
    parser.add_argument("--concurrency", type=int, default=concurrency, help="maximum concurrent model requests")
    if include_judge:
        parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o"), help="judge model for scoring")
        parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL"), help="base URL for the judger")
        parser.add_argument("--judge-api-key", default=os.getenv("JUDGE_API_KEY"), help="API key for the judger")


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
    )


def build_llm_client(*, api_key: str, base_url: str | None, timeout: float) -> AsyncOpenAI:
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


async def call_model(
    request_sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    params: ModelParams,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]], str, dict[str, int] | None, str | None]:
    "Stream one chat completion; return (content, tool_calls, reasoning, usage, finish_reason)."

    # reasoning_content and usage are log-only extensions; keep them out of the wire request
    extra_body = dict(params.extra_body) if params.extra_body else {}
    if params.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": params.enable_thinking}
    request: dict[str, Any] = {
        "model": params.model,
        "messages": [
            {key: value for key, value in message.items() if key not in _LOG_ONLY_KEYS} for message in messages
        ],
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": params.seed,
    }
    if extra_body:
        request["extra_body"] = extra_body
    if tools is not None:
        request["tools"] = tools
    async with request_sem:
        stream = await client.chat.completions.create(**request)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        call_usage: dict[str, int] | None = None
        try:
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    details = getattr(usage, "completion_tokens_details", None)
                    call_usage = {
                        "prompt_tokens": usage.prompt_tokens or 0,
                        "completion_tokens": usage.completion_tokens or 0,
                        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
                    }
                for choice in chunk.choices or ():
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if getattr(delta, "content", None):
                        content_parts.append(delta.content)
                    # some servers stream reasoning as `reasoning`, others as `reasoning_content`
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    for call in getattr(delta, "tool_calls", None) or ():
                        index = call.index if call.index is not None else len(calls)
                        slot = calls.setdefault(index, {"id": None, "name": None, "arguments": []})
                        if call.id:
                            slot["id"] = call.id
                        if call.function and call.function.name:
                            slot["name"] = call.function.name
                        if call.function and call.function.arguments:
                            slot["arguments"].append(call.function.arguments)
        finally:
            await stream.close()
    parsed_calls = []
    for index in sorted(calls):
        slot = calls[index]
        raw = "".join(slot["arguments"])
        try:
            arguments = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            arguments = {"input": raw}
        if not isinstance(arguments, dict):
            arguments = {"input": arguments}
        parsed_calls.append({"id": slot["id"] or f"call_missing_{index}", "name": slot["name"], "arguments": arguments})
    return "".join(content_parts), parsed_calls, "".join(reasoning_parts), call_usage, finish_reason
