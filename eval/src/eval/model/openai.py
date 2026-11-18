"""OpenAI-compatible model requests and streaming responses."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import AsyncOpenAI

__all__ = [
    "AssistantTurn",
    "make_client",
    "call_model",
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
            "finish_reason": self.finish_reason,
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.error is not None:
            result.update(incomplete=True, error=f"{type(self.error).__name__}: {self.error}")
        if self.latency_sec is not None:
            result.update(latency_sec=self.latency_sec, queue_sec=self.queue_sec)
        return result


def make_client(
    base_url: str | None = None,
    api_key: str = "EMPTY",
    *,
    timeout: float = 900,
    max_retries: int = _LLM_MAX_RETRIES,
) -> AsyncOpenAI:
    """Create a shared async OpenAI client with an isolated httpx connection pool."""
    if max_retries < 0:
        raise ValueError("Invalid model client max_retries")
    # openai vendors httpx as httpx2 and types http_client against it; the plain httpx
    # client is interface-compatible at runtime. trust_env=False keeps local/LAN
    # endpoints away from ambient proxies
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        http_client=httpx.AsyncClient(trust_env=False),  # ty: ignore[invalid-argument-type]
    )


async def call_model(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    model: str,
    temperature: float = 0.0,
    top_p: float | None = 0.95,
    top_k: int = 0,
    max_tokens: int = 8192,
    seed: int | None = None,
    enable_thinking: bool | None = True,
    extra_body: dict[str, Any] | None = None,
    reasoning_effort: str | None = "medium",
    stream: bool | None = True,
    max_tokens_field: str = "max_tokens",
    include_reasoning: bool = True,
    request_attempts: int = 1,
    retry_delay: float = 1.0,
) -> AssistantTurn:
    """Capture a model response using a shared client and request retry budget."""
    if request_attempts < 1 or retry_delay < 0:
        raise ValueError("Invalid model request retry limits")
    resolved_extra_body = dict(extra_body or {})
    if top_k:
        resolved_extra_body["top_k"] = top_k
    if enable_thinking is not None:
        resolved_extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    omitted_keys = set(_LOG_ONLY_KEYS)
    if include_reasoning:
        omitted_keys.remove("reasoning_content")
    request: dict[str, Any] = {
        "model": model,
        "messages": [{k: v for k, v in m.items() if k not in omitted_keys} for m in messages],
        "temperature": temperature,
        max_tokens_field: max_tokens,
    }
    if top_p is not None:
        request["top_p"] = top_p
    if seed is not None:
        request["seed"] = seed
    if stream is not None:
        request["stream"] = stream
    if stream:
        request["stream_options"] = {"include_usage": True}
    if resolved_extra_body:
        request["extra_body"] = resolved_extra_body
    if enable_thinking is not False and reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    if tools:
        request["tools"] = tools

    started = time.perf_counter()
    for attempt in range(request_attempts):
        turn = AssistantTurn()
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        response_stream = None
        try:
            response = await client.chat.completions.create(**request)
            if not stream:
                if not response.choices:
                    raise ValueError("Model response has no choices")
                choice = response.choices[0]
                message = choice.message
                content.append(message.content or "")
                reasoning.append(
                    getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
                )
                turn.finish_reason = choice.finish_reason
                turn.usage = _token_usage(response.usage)
                for index, call in enumerate(message.tool_calls or ()):
                    calls[index] = {
                        "id": call.id or "",
                        "name": call.function.name,
                        "arguments": [call.function.arguments],
                    }
            else:
                response_stream = response
            if response_stream is not None:
                async for chunk in response_stream:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        turn.usage = _token_usage(usage)
                    for choice in chunk.choices or ():
                        if choice.finish_reason is not None:
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
            if response_stream is not None:
                try:
                    await response_stream.close()
                except BaseException as exc:
                    if turn.error is None:
                        turn.error = exc
        if not isinstance(turn.error, Exception) or attempt + 1 == request_attempts:
            turn.latency_sec = round(time.perf_counter() - started, 6)
            turn.queue_sec = 0.0
            return turn
        await asyncio.sleep(retry_delay)
    raise ValueError("Model request attempts must be positive")


def _token_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    result = {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
    }
    cached_tokens = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
    if cached_tokens is not None:
        result["cached_tokens"] = cached_tokens
    return result
