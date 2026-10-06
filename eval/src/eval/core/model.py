"""Model call primitives: request assembly, streaming parse, and per-call capture."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI

LLM_MAX_RETRIES = 3
LOG_ONLY_KEYS = ("reasoning_content", "usage")


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


def build_llm_client(*, api_key: str, base_url: str | None, timeout: float) -> AsyncOpenAI:
    # openai vendors httpx as httpx2 and types http_client against it; the plain httpx
    # client is interface-compatible at runtime. trust_env=False keeps local/LAN
    # endpoints away from ambient proxies
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=LLM_MAX_RETRIES,
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
            {key: value for key, value in message.items() if key not in LOG_ONLY_KEYS} for message in messages
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
