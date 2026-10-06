"""Agent harness: tool definition, web tools, and the ReAct loop."""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import httpx

from eval.core.model import Context, call_model

TOOL_TIMEOUT_SEC = 60
WEB_MAX_RETRIES = 3
RETRY_BASE_DELAY = 3

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Web search. Returns a JSON list of results, each with title, url, and snippet.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Precise web search query."},
            },
            "required": ["query"],
        },
    },
}

READ_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Fetch a web page and return its readable text (truncated).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http(s) URL to fetch."},
            },
            "required": ["url"],
        },
    },
}


class _QuotaExhausted(Exception):
    """Serper rejected the key for quota/auth reasons; rotate the pool instead of retrying."""


@dataclass(frozen=True)
class Tool:
    """One agent tool: its function schema, primary argument key, and executor."""

    schema: dict[str, Any]
    key: str
    run: Callable[[str], Awaitable[str]]
    timeout: float = TOOL_TIMEOUT_SEC


@dataclass
class ReactResult:
    """Everything one ReAct episode produced; scoring and the record stay with the benchmark."""

    messages: list[dict[str, Any]]
    answer: str | None
    usage: dict[str, int]
    finish_reasons: dict[str, int]
    steps: int
    sys_error: str | None


@asynccontextmanager
async def web_clients() -> AsyncIterator[tuple[httpx.AsyncClient, httpx.AsyncClient]]:
    """HTTP clients for the web tools; crawl4ai is local/LAN so it must bypass ambient proxies."""
    async with (
        httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"User-Agent": "eval-harness/1.0"},
            follow_redirects=True,
        ) as client_search,
        httpx.AsyncClient(trust_env=False, follow_redirects=True) as client_read,
    ):
        yield client_search, client_read


def clean_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + (" ..." if len(text) > limit else "")


def make_search(client: httpx.AsyncClient, api_key: str, limit: int, clean_limit: int) -> Tool:
    "Build the search tool over a comma-separated serper key pool, rotated round-robin."
    keys = [key.strip() for key in api_key.split(",") if key.strip()]
    balance_cache: dict[str, int | None] = {}  # None = not probed yet; 0 = exhausted
    index = 0

    async def run(query: str) -> str:
        nonlocal index
        if not keys:
            return "Tool error: no serper API key configured."

        async def post(key: str) -> httpx.Response:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": key},
                json={"q": query, "num": limit},
            )
            if response.status_code in (401, 402, 403):
                raise _QuotaExhausted(response.status_code)
            response.raise_for_status()
            return response

        async def balance(key: str) -> int | None:
            "Remaining credits for one key from the free /account endpoint; None if unknown."
            try:
                response = await client.get("https://google.serper.dev/account", headers={"X-API-KEY": key})
                response.raise_for_status()
                return int(response.json().get("balance") or 0)
            except Exception:  # noqa: BLE001 - a failed probe must not fail the search
                return None

        for _ in range(len(keys)):
            key = keys[index]
            index = (index + 1) % len(keys)
            left = balance_cache.get(key)
            if left == 0:
                continue
            if left is None:
                # first use of this key: check its quota before spending a search on it
                left = await balance(key)
                if left is not None and left <= 0:
                    balance_cache[key] = 0
                    continue
                balance_cache[key] = left
            try:
                response = await _fetch_with_retry(partial(post, key))
            except _QuotaExhausted:
                balance_cache[key] = 0
                continue
            if isinstance(left, int) and left > 0:
                balance_cache[key] = left - 1
            results = []
            for result in response.json().get("organic", []):
                url = result.get("link", "")
                if not url.startswith(("http://", "https://")):
                    continue
                results.append(
                    {
                        "title": clean_text(result.get("title", ""), 300),
                        "url": url,
                        "snippet": clean_text(result.get("snippet", ""), 700),
                    }
                )
            if not results:
                return "No search results were returned. Try a different query."
            return clean_text(json.dumps(results, ensure_ascii=False, indent=2), clean_limit)

        return "Tool error: every serper API key is out of credits."

    return Tool(SEARCH_SCHEMA, "query", run)


def make_read(client: httpx.AsyncClient, server: str, token: str, clean_limit: int) -> Tool:
    headers = {"Authorization": f"Bearer {token}"}

    async def run(url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return "The read tool only accepts an http(s) URL."

        async def post() -> httpx.Response:
            response = await client.post(
                f"{server.rstrip('/')}/md",
                json={"url": url},
                headers=headers,
                # crawl4ai renders the page server-side before answering; this only needs
                # to outlive the tool timeout, which is the real bound
                timeout=httpx.Timeout(TOOL_TIMEOUT_SEC + 10, connect=10.0),
            )
            response.raise_for_status()
            return response

        try:
            response = await _fetch_with_retry(post)
        except httpx.ConnectError:
            return f"Tool error: crawl4ai server unreachable at {server}."
        payload = response.json()
        if payload.get("success") is False:
            return f"The crawler failed on this URL: {payload.get('error') or 'unknown error'}"
        markdown = payload.get("markdown") or ""
        if isinstance(markdown, dict):
            markdown = markdown.get("fit_markdown") or markdown.get("raw_markdown") or ""
        text = clean_text(markdown, clean_limit)
        return text or "The page contained no readable text."

    return Tool(READ_SCHEMA, "url", run)


def fold(
    usage: dict[str, int],
    finish_reasons: dict[str, int],
    call_usage: dict[str, int] | None,
    finish_reason: str | None,
) -> None:
    "Accumulate one model call's usage into episode totals."
    if call_usage:
        for key in usage:
            usage[key] += call_usage[key]
    name = finish_reason or "unknown"
    finish_reasons[name] = finish_reasons.get(name, 0) + 1


async def react(
    ctx: Context,
    question: str,
    *,
    tools: list[Tool],
    system_prompt: str,
    max_steps: int,
    task_timeout: int,
) -> ReactResult:
    "Run one ReAct episode to a final answer; tools, prompts and budgets come from the caller."

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    finish_reasons: dict[str, int] = {}
    steps = 0
    answer: str | None = None
    sys_error: str | None = None
    started = time.perf_counter()
    deadline = started + task_timeout if task_timeout else None
    timed_out = False
    schemas = [tool.schema for tool in tools]

    try:
        for _step in range(1, max_steps + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break
            content, calls, reasoning, call_usage, finish_reason = await call_model(
                ctx.sem, ctx.client_llm, ctx.params, messages, tools=schemas or None
            )
            steps += 1
            fold(usage, finish_reasons, call_usage, finish_reason)

            if calls:
                message: dict[str, Any] = {"role": "assistant", "content": content or None}
                if reasoning:
                    message["reasoning_content"] = reasoning
                message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                        },
                    }
                    for call in calls
                ]
                if call_usage:
                    message["usage"] = call_usage
                messages.append(message)
                for call in calls:
                    tool = next((t for t in tools if t.schema["function"]["name"] == call["name"]), None)
                    if tool is None:
                        observation = f"Unknown tool {call['name']!r}."
                    else:
                        observation = await _execute(tool, call["arguments"])
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
                continue

            if content:
                answer = content
                message = {"role": "assistant", "content": content}
                if reasoning:
                    message["reasoning_content"] = reasoning
                if call_usage:
                    message["usage"] = call_usage
                messages.append(message)
                break

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your reply was empty. Call one of the tools if you need more information, "
                        "or reply with only the final answer now."
                    ),
                }
            )

        if answer is None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "No steps remain. Do not call any more tools. Reply now with only the final answer value "
                        "itself — no reasoning, explanation, or any other text."
                    ),
                }
            )
            content, _, reasoning, call_usage, finish_reason = await call_model(
                ctx.sem, ctx.client_llm, ctx.params, messages
            )
            fold(usage, finish_reasons, call_usage, finish_reason)
            answer = content.strip() or None
            if content:
                message = {"role": "assistant", "content": content}
                if reasoning:
                    message["reasoning_content"] = reasoning
                if call_usage:
                    message["usage"] = call_usage
                messages.append(message)
            if answer is None:
                sys_error = f"task_timeout ({task_timeout}s)" if timed_out else f"max_steps_exceeded ({max_steps})"

    except Exception as exc:
        sys_error = f"{type(exc).__name__}: {exc}"

    return ReactResult(messages, answer, usage, finish_reasons, steps, sys_error)


async def _execute(tool: Tool, arguments: dict[str, Any]) -> str:
    "Run one tool call; failures become observations, never exceptions."
    value = str(arguments.get(tool.key, arguments.get("input", "")))
    try:
        return await asyncio.wait_for(tool.run(value), tool.timeout)
    except TimeoutError:
        return f"Tool error: the call was cancelled after exceeding {tool.timeout}s."
    except Exception as exc:  # noqa: BLE001 - tool failures become observations
        return f"Tool error: {type(exc).__name__}: {exc}"


async def _fetch_with_retry(fetch: Callable[[], Awaitable[httpx.Response]]) -> httpx.Response:
    "Run an HTTP call, retrying transient failures with exponential backoff."

    last: Exception | None = None
    for attempt in range(WEB_MAX_RETRIES):
        if attempt:
            await asyncio.sleep(RETRY_BASE_DELAY**attempt)
        try:
            return await fetch()
        except httpx.HTTPError as exc:
            last = exc
    raise last
