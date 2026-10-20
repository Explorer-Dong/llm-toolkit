"""Agent harness: Tool use and the ReAct loop."""

import argparse
import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from eval.core.model import Context, call_model
from eval.core.utils import clean_text, record_model_call

# Globals

__all__ = [
    "FETCH_SCHEMA",
    "SEARCH_SCHEMA",
    "ReactResult",
    "Tool",
    "add_tool_args",
    "make_tools",
    "react",
    "tool_clients",
    "validate_tool_args",
]

_TOOL_TIMEOUT_SEC = 60
_TOOL_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 3
_HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0, pool=5.0)
_HTTP_LIMITS = httpx.Limits(max_connections=8, max_keepalive_connections=4, keepalive_expiry=10.0)
_RETRYABLE_STATUS_CODES = frozenset({408, 429, *range(500, 600)})
_logger = logging.getLogger(__name__)


SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the internet. Returns a JSON list of results, each with title, url, and snippet.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
            "required": ["query"],
        },
    },
}

FETCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch",
        "description": "Fetch a page and return its readable text (truncated).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http(s) URL to fetch."},
            },
            "required": ["url"],
        },
    },
}

# Classes


@dataclass(frozen=True)
class Tool:
    """One agent tool: its function schema, primary argument key, and executor."""

    schema: dict[str, Any]
    key: str
    run: Callable[[str], Awaitable[str]]
    timeout: float = _TOOL_TIMEOUT_SEC


@dataclass
class ReactResult:
    """Everything one ReAct episode produced; scoring and the record stay with the benchmark."""

    messages: list[dict[str, Any]]
    answer: str | None
    usage: dict[str, int]
    finish_reasons: dict[str, int]
    steps: int
    sys_error: str | None


# Public functions


@asynccontextmanager
async def tool_clients() -> AsyncIterator[tuple[httpx.AsyncClient, httpx.AsyncClient]]:
    """HTTP clients for the built-in tools; both endpoints should avoid ambient proxy settings."""
    async with (
        httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT,
            limits=_HTTP_LIMITS,
            headers={"User-Agent": "eval-harness/1.0"},
            follow_redirects=True,
            trust_env=False,
        ) as client_search,
        httpx.AsyncClient(trust_env=False, follow_redirects=True) as client_fetch,
    ):
        yield client_search, client_fetch


def add_tool_args(parser: argparse.ArgumentParser) -> None:
    """Select search and fetch backends independently; keep the existing defaults."""
    parser.add_argument("--search-provider", choices=("serper", "deepseek"), default="serper")
    parser.add_argument("--fetch-provider", choices=("crawl4ai", "deepseek"), default="crawl4ai")
    parser.add_argument("--serper-api-key", default=os.getenv("SERPER_API_KEY"))
    parser.add_argument("--crawl4ai-url", default=os.getenv("CRAWL4AI_URL"))
    parser.add_argument("--crawl4ai-token", default=os.getenv("CRAWL4AI_TOKEN"))
    parser.add_argument(
        "--ds-tool-url",
        default=os.getenv("DS_TOOL_URL"),
        help="DeepSeek tools base URL ending in /v1/tools",
    )
    parser.add_argument("--ds-tool-key", default=os.getenv("DS_TOOL_KEY"), help="DeepSeek tools API key")


def validate_tool_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Require credentials only for the backends selected for this run."""
    required = {}
    if args.search_provider == "serper":
        required["SERPER_API_KEY / --serper-api-key"] = args.serper_api_key
    if args.fetch_provider == "crawl4ai":
        required["CRAWL4AI_URL / --crawl4ai-url"] = args.crawl4ai_url
        required["CRAWL4AI_TOKEN / --crawl4ai-token"] = args.crawl4ai_token
    if "deepseek" in (args.search_provider, args.fetch_provider):
        required["DS_TOOL_URL / --ds-tool-url"] = args.ds_tool_url
        required["DS_TOOL_KEY / --ds-tool-key"] = args.ds_tool_key
    for name, value in required.items():
        if not value or not value.strip():
            parser.error(f"the selected tool providers require {name}")
    if args.search_provider == "deepseek" and not 1 <= args.search_results <= 50:
        parser.error("DeepSeek --search-results must be between 1 and 50")


def make_tools(
    client_search: httpx.AsyncClient, client_fetch: httpx.AsyncClient, args: argparse.Namespace
) -> list[Tool]:
    """Build the same search/fetch interface for each benchmark."""
    search = (
        _make_deepseek_search(
            client_search, args.ds_tool_url, args.ds_tool_key, args.search_results, args.observation_chars
        )
        if args.search_provider == "deepseek"
        else _make_serper_search(client_search, args.serper_api_key, args.search_results, args.observation_chars)
    )
    fetch = (
        _make_deepseek_fetch(client_fetch, args.ds_tool_url, args.ds_tool_key, args.observation_chars)
        if args.fetch_provider == "deepseek"
        else _make_crawl4ai_fetch(client_fetch, args.crawl4ai_url, args.crawl4ai_token, args.observation_chars)
    )
    return [search, fetch]


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
    schemas = [tool.schema for tool in tools]
    timeout_scope = asyncio.timeout(task_timeout or None)

    try:
        async with timeout_scope:
            for _step in range(1, max_steps + 1):
                content, calls, reasoning, call_usage, finish_reason = await call_model(
                    ctx.sem, ctx.client_llm, ctx.params, messages, tools=schemas or None
                )
                steps += 1
                record_model_call(usage, finish_reasons, call_usage, finish_reason)

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
                            observation = await _execute_tool(tool, call["arguments"])
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
                record_model_call(usage, finish_reasons, call_usage, finish_reason)
                answer = content.strip() or None
                if content:
                    message = {"role": "assistant", "content": content}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                    if call_usage:
                        message["usage"] = call_usage
                    messages.append(message)
                if answer is None:
                    sys_error = f"max_steps_exceeded ({max_steps})"

    except TimeoutError as exc:
        sys_error = f"task_timeout ({task_timeout}s)" if timeout_scope.expired() else f"{type(exc).__name__}: {exc}"

    except Exception as exc:
        sys_error = f"{type(exc).__name__}: {exc}"

    return ReactResult(messages, answer, usage, finish_reasons, steps, sys_error)


# Private functions


def _make_serper_search(client: httpx.AsyncClient, api_key: str, limit: int, clean_limit: int) -> Tool:
    "Build the search tool over a comma-separated serper key pool, rotated round-robin."
    keys = [key.strip() for key in api_key.split(",") if key.strip()]
    balance_cache: dict[str, int | None] = {}  # None = not probed yet; 0 = exhausted
    index = 0

    async def run(query: str) -> str:
        nonlocal index
        if not keys:
            return "Tool error: no serper API key configured."

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
                response = await _http_request_with_retry(
                    client,
                    "POST",
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": key},
                    json={"q": query, "num": limit},
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (401, 402, 403):
                    raise
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


def _make_crawl4ai_fetch(client: httpx.AsyncClient, server: str, token: str, clean_limit: int) -> Tool:
    """Build the fetch tool using a Crawl4AI server."""
    headers = {"Authorization": f"Bearer {token}"}

    async def run(url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return "The fetch tool only accepts an http(s) URL."

        try:
            response = await _http_request_with_retry(
                client,
                "POST",
                f"{server.rstrip('/')}/md",
                json={"url": url},
                headers=headers,
                # crawl4ai renders the page server-side before answering; this only needs
                # to outlive the tool timeout, which is the real bound
                timeout=httpx.Timeout(_TOOL_TIMEOUT_SEC + 10, connect=10.0),
            )
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

    return Tool(FETCH_SCHEMA, "url", run)


def _make_deepseek_search(client: httpx.AsyncClient, server: str, api_key: str, limit: int, clean_limit: int) -> Tool:
    """Expose DeepSeek search results through the shared search tool."""
    endpoint = f"{server.rstrip('/')}/search"

    async def run(query: str) -> str:
        if not query.strip() or len(query) > 128:
            return "Tool error: DeepSeek search query must contain 1 to 128 characters and not be blank."
        response = await _http_request_with_retry(
            client,
            "POST",
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "count": limit},
            timeout=_HTTP_TIMEOUT,
        )
        payload = response.json()
        results = []
        for result in payload["results"]:
            url = result.get("url", "")
            if not url.startswith(("http://", "https://")):
                continue
            results.append(
                {
                    "title": clean_text(result.get("title", ""), 300),
                    "url": url,
                    "snippet": clean_text(result.get("summary") or result.get("snippet", ""), 1000),
                }
            )
        if not results:
            return "No search results were returned. Try a different query."
        return clean_text(json.dumps(results, ensure_ascii=False, indent=2), clean_limit)

    return Tool(SEARCH_SCHEMA, "query", run)


def _make_deepseek_fetch(client: httpx.AsyncClient, server: str, api_key: str, clean_limit: int) -> Tool:
    """Fetch Markdown content from the DeepSeek tools API."""
    endpoint = f"{server.rstrip('/')}/fetch"

    async def run(url: str) -> str:
        parsed = httpx.URL(url)
        if parsed.scheme not in ("http", "https") or not parsed.host or parsed.userinfo or len(url) > 8192:
            return "The fetch tool requires an http(s) URL with a host, no credentials, and at most 8192 characters."
        response = await _http_request_with_retry(
            client,
            "POST",
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"url": url},
            timeout=_HTTP_TIMEOUT,
        )
        payload = response.json()
        return clean_text(payload["content"], clean_limit) or "The page contained no readable text."

    return Tool(FETCH_SCHEMA, "url", run)


async def _http_request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **request_kwargs: Any,
) -> httpx.Response:
    "Make an HTTP request, retrying only transient transport and upstream failures."

    last: Exception | None = None
    for attempt in range(_TOOL_MAX_RETRIES):
        if attempt:
            await asyncio.sleep(_RETRY_BASE_DELAY**attempt)
        try:
            response = await client.request(method, url, **request_kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRYABLE_STATUS_CODES:
                raise
            last = exc
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last = exc
        if attempt == _TOOL_MAX_RETRIES - 1:
            _logger.warning(
                "tool request failed after %d attempts: %s: %s", _TOOL_MAX_RETRIES, type(last).__name__, last
            )
    raise last


async def _execute_tool(tool: Tool, arguments: dict[str, Any]) -> str:
    "Run one tool call; failures become observations, never exceptions."
    value = str(arguments.get(tool.key, arguments.get("input", "")))
    try:
        return await asyncio.wait_for(tool.run(value), tool.timeout)
    except TimeoutError:
        return f"Tool error: the call was cancelled after exceeding {tool.timeout}s."
    except Exception as exc:  # noqa: BLE001 - tool failures become observations
        return f"Tool error: {type(exc).__name__}: {exc}"
