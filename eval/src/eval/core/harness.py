"""Agent = Model + Harness. This is harness for constraint."""

import argparse
import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from eval.core.model import Context, call_model
from eval.core.utils import clean_text

__all__ = [
    "FETCH_SCHEMA",
    "SEARCH_SCHEMA",
    "AgentResult",
    "Tool",
    "add_tool_args",
    "add_paradigm_args",
    "make_tools",
    "qa",
    "react",
    "tool_clients",
]

_TOOL_TIMEOUT_SEC = 60
_TOOL_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 3
_HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0, pool=5.0)
_HTTP_LIMITS = httpx.Limits(max_connections=8, max_keepalive_connections=4, keepalive_expiry=10.0)
_RETRYABLE_STATUS_CODES = frozenset({408, 429, *range(500, 600)})
# DeepSeek Search / Fetch API error descriptions:
# https://api-docs.deepseek.com/guides/search/#errors
_DEEPSEEK_OBSERVABLE_ERRORS = {
    403: "The Fetch request was rejected by the security policy.",
    404: (
        "Fetch could not retrieve usable page content, for example due to an unavailable URL, "
        "a URL blocked by site policy, an error from the target site, an unsupported content type, "
        "or a page with no extractable Markdown content."
    ),
    413: "The request body is too large.",
}
_DEVELOPER_HTTP_ERRORS = {
    400: "Invalid request headers or body. Check the request format and field values.",
    401: "Missing or invalid API key. Check the tool credentials.",
    402: "Insufficient account balance. Top up the tool account before retrying.",
}
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


@dataclass(frozen=True)
class Tool:
    """One agent tool: its function schema, primary argument key, and executor."""

    schema: dict[str, Any]
    key: str
    run: Callable[[str], Awaitable[str]]
    timeout: float = _TOOL_TIMEOUT_SEC
    provider: str = ""


@dataclass
class AgentResult:
    """One agent episode and its call metadata; scoring stays with the benchmark."""

    messages: list[dict[str, Any]]
    final_content: str | None
    usage: dict[str, int]
    finish_reasons: dict[str, int]
    steps: int
    sys_error: str | None
    latency_sec: float = 0.0


def add_tool_args(parser: argparse.ArgumentParser) -> None:
    """Select search and fetch backends independently; keep the existing defaults."""
    parser.add_argument("--search-provider", choices=("serper", "deepseek"), default="serper")
    parser.add_argument("--fetch-provider", choices=("crawl4ai", "deepseek"), default="deepseek")
    parser.add_argument("--serper-api-key", default=os.getenv("SERPER_API_KEY", ""))
    parser.add_argument("--crawl4ai-url", default=os.getenv("CRAWL4AI_URL", ""))
    parser.add_argument("--crawl4ai-token", default=os.getenv("CRAWL4AI_TOKEN", ""))
    parser.add_argument("--ds-tool-url", default=os.getenv("DS_TOOL_URL", ""), help="DeepSeek tools base URL")
    parser.add_argument("--ds-tool-key", default=os.getenv("DS_TOOL_KEY", ""), help="DeepSeek tools API key")


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


def add_paradigm_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared agent budgets and observation limits."""
    parser.add_argument("--max-steps", type=int, default=16, help="maximum tool-use turns per task")
    parser.add_argument("--search-results", type=int, choices=range(1, 51), metavar="N", default=8, help="1~50")
    parser.add_argument("--observation-chars", type=int, default=12000, help="observation text truncation limit")
    parser.add_argument("--task-timeout", type=int, default=3600, help="per-task time budget in seconds")


async def qa(
    ctx: Context,
    question: str,
    *,
    system_prompt: str,
) -> AgentResult:
    """Harness with QA paradigm."""

    started = time.perf_counter()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    final_content: str | None = None
    finish_reasons: dict[str, int] = {}
    steps = 0
    sys_error: str | None = None
    try:
        content, _, reasoning_content, call_usage, finish_reason = await call_model(
            ctx.sem, ctx.client_llm, ctx.params, messages
        )
        steps += 1
        if call_usage:
            for key in usage:
                usage[key] += call_usage[key]
        finish_reason = finish_reason or "unknown"
        finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
        final_content = content.strip()
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning_content,
                "usage": call_usage,
                "finish_reason": finish_reason,
            }
        )
    except Exception as exc:
        sys_error = f"{type(exc).__name__}: {exc}"

    return AgentResult(
        messages, final_content, usage, finish_reasons, steps, sys_error, round(time.perf_counter() - started, 3)
    )


async def react(
    ctx: Context,
    question: str,
    *,
    tools: list[Tool],
    system_prompt: str,
    max_steps: int,
    task_timeout: int,
) -> AgentResult:
    "Harness with ReAct paradigm."

    started = time.perf_counter()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    finish_reasons: dict[str, int] = {}
    steps = 0
    final_content: str | None = None
    sys_error: str | None = None
    schemas = [tool.schema for tool in tools]
    timeout_scope = asyncio.timeout(task_timeout or None)

    try:
        async with timeout_scope:
            for _step in range(1, max_steps + 1):
                content, calls, reasoning_content, call_usage, finish_reason = await call_model(
                    ctx.sem, ctx.client_llm, ctx.params, messages, tools=schemas or None
                )
                steps += 1
                if call_usage:
                    for key in usage:
                        usage[key] += call_usage[key]
                finish_reason = finish_reason or "unknown"
                finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1

                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": content or None,
                    "finish_reason": finish_reason,
                }
                if reasoning_content:
                    message["reasoning_content"] = reasoning_content
                if call_usage:
                    message["usage"] = call_usage

                if calls:
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
                messages.append(message)

                if calls:
                    for call in calls:
                        tool = next((t for t in tools if t.schema["function"]["name"] == call["name"]), None)
                        if tool is None:
                            observation = f"Unknown tool {call['name']!r}."
                        else:
                            observation = await _execute_tool(tool, call["arguments"])
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
                    continue

                if content:
                    final_content = content
                    break

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your reply was empty. Call one of the tools if you need more information, "
                            "or reply with the final answer in the format required by the system prompt."
                        ),
                    }
                )

            if final_content is None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "No steps remain. Do not call any more tools. Reply now with only the final answer "
                            "in the format required by the system prompt."
                        ),
                    }
                )
                content, _, reasoning_content, call_usage, finish_reason = await call_model(
                    ctx.sem, ctx.client_llm, ctx.params, messages
                )
                steps += 1
                if call_usage:
                    for key in usage:
                        usage[key] += call_usage[key]
                finish_reason = finish_reason or "unknown"
                finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
                final_content = content.strip() or None
                message = {"role": "assistant", "content": content, "finish_reason": finish_reason}
                if reasoning_content:
                    message["reasoning_content"] = reasoning_content
                if call_usage:
                    message["usage"] = call_usage
                messages.append(message)
                if final_content is None:
                    sys_error = f"max_steps_exceeded ({max_steps})"

    except TimeoutError as exc:
        sys_error = f"task_timeout ({task_timeout}s)" if timeout_scope.expired() else f"{type(exc).__name__}: {exc}"

    except Exception as exc:
        sys_error = f"{type(exc).__name__}: {exc}"

    return AgentResult(
        messages, final_content, usage, finish_reasons, steps, sys_error, round(time.perf_counter() - started, 3)
    )


async def _execute_tool(tool: Tool, arguments: dict[str, Any]) -> str:
    """Return actionable observations; report developer errors and stop the episode."""
    value = str(arguments.get(tool.key, arguments.get("input", "")))
    try:
        return await asyncio.wait_for(tool.run(value), tool.timeout)
    except TimeoutError:
        return f"The tool call timed out after {tool.timeout}s. Try another source."
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if tool.provider == "deepseek" and status in _DEEPSEEK_OBSERVABLE_ERRORS:
            return _DEEPSEEK_OBSERVABLE_ERRORS[status]
        if status in _RETRYABLE_STATUS_CODES:
            return "The tool service is temporarily unavailable after retries. Try another source."
        detail = exc.response.text
        for header in ("Authorization", "X-API-KEY"):
            secret = exc.request.headers.get(header, "").removeprefix("Bearer ").strip()
            if secret:
                detail = detail.replace(secret, "[redacted]")
        _logger.error(
            "%s %s failed (HTTP %d): %s Server response: %s",
            tool.provider or "Tool",
            tool.schema["function"]["name"],
            status,
            _DEVELOPER_HTTP_ERRORS.get(status, exc.response.reason_phrase),
            clean_text(detail, 2000),
        )
        raise
    except (httpx.TimeoutException, httpx.NetworkError):
        return "The tool service could not be reached after retries. Try another source."
    except Exception:
        _logger.exception("%s %s failed", tool.provider or "Tool", tool.schema["function"]["name"])
        raise


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

    return Tool(SEARCH_SCHEMA, "query", run, provider="serper")


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

    return Tool(FETCH_SCHEMA, "url", run, provider="crawl4ai")


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

    return Tool(SEARCH_SCHEMA, "query", run, provider="deepseek")


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

    return Tool(FETCH_SCHEMA, "url", run, provider="deepseek")


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
