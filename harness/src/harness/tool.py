"""Common tools, MCP connections, and sample-scoped runtime adapters."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import re
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, fields, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, PaginatedRequestParams, TextContent

from harness.env import Env
from harness.utils import clean_text

__all__ = [
    "Connection",
    "HttpSpec",
    "StdioSpec",
    "ToolBox",
    "ToolConfig",
    "ToolResponse",
    "add_tool_args",
    "common_server",
    "connect",
    "env_server",
    "open_toolbox",
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


@dataclass(frozen=True)
class ToolConfig:
    search_provider: str = "deepseek"
    fetch_provider: str = "deepseek"
    search_results: int = 8
    observation_chars: int = 12000
    ds_tool_url: str = field(default_factory=lambda: os.getenv("DS_TOOL_URL", ""))
    ds_tool_key: str = field(default_factory=lambda: os.getenv("DS_TOOL_KEY", ""))
    serper_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    crawl4ai_url: str = field(default_factory=lambda: os.getenv("CRAWL4AI_URL", ""))
    crawl4ai_token: str = field(default_factory=lambda: os.getenv("CRAWL4AI_TOKEN", ""))

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ToolConfig:
        return cls(**{item.name: getattr(args, item.name) for item in fields(cls)})


@dataclass(frozen=True)
class StdioSpec:
    cmd: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | Path | None = None
    name: str = "stdio"
    timeout: float = 70


@dataclass(frozen=True)
class HttpSpec:
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    name: str = "http"
    timeout: float = 70


@dataclass
class ToolResponse:
    content: str
    is_error: bool = False
    raw: dict[str, Any] | None = None
    fatal: str | None = None


@dataclass(frozen=True)
class _ProviderTool:
    """A configured provider operation and its execution budget."""

    name: str
    argument: str
    run: Callable[[str], Awaitable[str]]
    timeout: float = _TOOL_TIMEOUT_SEC
    provider: str = ""


class Connection:
    """An initialized MCP session borrowed for the lifetime of connect()."""

    def __init__(self, session: ClientSession, name: str, schemas: list[dict[str, Any]], timeout: float) -> None:
        self.session = session
        self.name = name
        self.schemas = schemas
        self.timeout = timedelta(seconds=timeout)

    async def call(self, name: str, args: dict[str, Any]) -> ToolResponse:
        result = await self.session.call_tool(name, args, read_timeout_seconds=self.timeout)
        raw = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        content = "\n".join(
            block["text"] if block["type"] == "text" else json.dumps(block, ensure_ascii=False)
            for block in raw["content"]
        )
        if not content and result.structuredContent is not None:
            content = json.dumps(result.structuredContent, ensure_ascii=False)
        fatal = (raw.get("_meta") or {}).get("harness", {}).get("fatal")
        return ToolResponse(content, result.isError, raw, fatal)


class ToolBox:
    """Immutable routing over shared and sample-local MCP connections."""

    def __init__(self, connections: tuple[Connection, ...] = ()) -> None:
        self._connections = connections
        self._schemas: list[dict[str, Any]] = []
        self._routes: dict[str, tuple[Connection, str]] = {}
        counts = Counter(tool["function"]["name"] for connection in connections for tool in connection.schemas)
        reserved = {name for name, count in counts.items() if count == 1}
        for connection in connections:
            for schema in connection.schemas:
                original = schema["function"]["name"]
                base = f"{connection.name}__{original}" if counts[original] > 1 else original
                base = re.sub(r"[^A-Za-z0-9_-]", "_", base)[:64] or "tool"
                name = base
                index = 2
                while name in self._routes or (counts[original] > 1 and name in reserved):
                    suffix = f"_{index}"
                    name = base[: 64 - len(suffix)] + suffix
                    index += 1
                item = copy.deepcopy(schema)
                item["function"]["name"] = name
                self._schemas.append(item)
                self._routes[name] = connection, original

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._schemas)

    async def call(self, name: str, args: dict[str, Any]) -> ToolResponse:
        route = self._routes.get(name)
        if route is None:
            return ToolResponse(f"Unknown tool {name!r}.", is_error=True)
        connection, original = route
        return await connection.call(original, args)

    @asynccontextmanager
    async def scoped(self, *specs: StdioSpec | HttpSpec | FastMCP) -> AsyncGenerator[ToolBox, None]:
        async with AsyncExitStack() as resources:
            local = [await resources.enter_async_context(connect(spec)) for spec in specs]
            yield ToolBox((*self._connections, *local))


def add_tool_args(parser: argparse.ArgumentParser) -> None:
    """Configure common search and fetch tools independently."""
    defaults = ToolConfig()
    parser.add_argument("--search-provider", choices=("deepseek", "serper"), default=defaults.search_provider)
    parser.add_argument("--fetch-provider", choices=("deepseek", "crawl4ai"), default=defaults.fetch_provider)
    parser.add_argument(
        "--search-results", type=int, choices=range(1, 51), metavar="N", default=defaults.search_results
    )
    parser.add_argument("--observation-chars", type=int, default=defaults.observation_chars)
    parser.add_argument("--ds-tool-url", default=defaults.ds_tool_url, help="DeepSeek tools base URL")
    parser.add_argument("--ds-tool-key", default=defaults.ds_tool_key, help="DeepSeek tools API key")
    parser.add_argument("--serper-api-key", default=defaults.serper_api_key)
    parser.add_argument("--crawl4ai-url", default=defaults.crawl4ai_url)
    parser.add_argument("--crawl4ai-token", default=defaults.crawl4ai_token)


@asynccontextmanager
async def connect(spec: StdioSpec | HttpSpec | FastMCP) -> AsyncGenerator[Connection, None]:
    timeout = 70 if isinstance(spec, FastMCP) else spec.timeout
    if timeout <= 0:
        raise ValueError("MCP timeout must be positive")
    caller_error: BaseException | None = None
    async with AsyncExitStack() as resources:
        if isinstance(spec, FastMCP):
            session = await resources.enter_async_context(
                create_connected_server_and_client_session(spec, read_timeout_seconds=timedelta(seconds=timeout))
            )
            name = spec.name
        else:
            if isinstance(spec, StdioSpec):
                read, write = await resources.enter_async_context(
                    stdio_client(StdioServerParameters(command=spec.cmd, args=spec.args, env=spec.env, cwd=spec.cwd))
                )
            else:
                client = await resources.enter_async_context(
                    httpx.AsyncClient(headers=spec.headers, trust_env=False, timeout=timeout)
                )
                read, write, _ = await resources.enter_async_context(
                    streamable_http_client(spec.url, http_client=client)
                )
            session = await resources.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=timeout))
            )
            await session.initialize()
            name = spec.name
        schemas: list[dict[str, Any]] = []
        cursor = None
        while True:
            page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
            schemas.extend(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in page.tools
            )
            if not page.nextCursor:
                break
            cursor = page.nextCursor
        if len({s["function"]["name"] for s in schemas}) != len(schemas):
            raise ValueError(f"MCP server {name!r} returned duplicate tool names")
        try:
            yield Connection(session, name, schemas, timeout)
        except BaseException as exc:
            # Close AnyIO transports before rethrowing the caller error, preserving its type.
            caller_error = exc
    if caller_error is not None:
        raise caller_error


@asynccontextmanager
async def open_toolbox(*specs: StdioSpec | HttpSpec | FastMCP) -> AsyncGenerator[ToolBox, None]:
    async with ToolBox().scoped(*specs) as tools:
        yield tools


def common_server(config: ToolConfig | None = None, *, search: str | None = None, fetch: str | None = None) -> FastMCP:
    """Expose common search and fetch tools through an in-memory or external MCP session."""
    config = config or ToolConfig()
    config = replace(
        config, search_provider=search or config.search_provider, fetch_provider=fetch or config.fetch_provider
    )
    if config.search_provider not in {"deepseek", "serper"} or config.fetch_provider not in {"deepseek", "crawl4ai"}:
        raise ValueError("Unknown common tool provider")
    if config.observation_chars < 1 or not 1 <= config.search_results <= 50:
        raise ValueError("Invalid observation or search result limit")
    handlers: dict[str, _ProviderTool] = {}

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncGenerator[None, None]:
        async with (
            httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT,
                limits=_HTTP_LIMITS,
                headers={"User-Agent": "llm-toolkit-harness/1.0"},
                follow_redirects=True,
                trust_env=False,
            ) as search_client,
            httpx.AsyncClient(trust_env=False, follow_redirects=True) as fetch_client,
        ):
            handlers.update(
                {provider.name: provider for provider in _make_common_tools(search_client, fetch_client, config)}
            )
            try:
                yield None
            finally:
                handlers.clear()

    server = FastMCP("common", lifespan=lifespan, log_level="WARNING")

    async def invoke(name: str, arguments: dict[str, Any]) -> CallToolResult:
        provider = handlers[name]
        try:
            content, is_error = await _execute_tool(provider, arguments)
            return _mcp_result(content, is_error=is_error)
        except Exception as exc:
            detail = (
                f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
            )
            return _mcp_result("", is_error=True, fatal=f"{provider.provider} {name} failed: {detail}")

    @server.tool(name="search", structured_output=False)
    async def search_tool(query: str) -> CallToolResult:
        """Search the internet. Returns results with title, url, and snippet."""
        return await invoke("search", {"query": query})

    @server.tool(name="fetch", structured_output=False)
    async def fetch_tool(url: str) -> CallToolResult:
        """Fetch an http(s) page and return its readable text, truncated."""
        return await invoke("fetch", {"url": url})

    return server


def env_server(env: Env) -> FastMCP:
    server = FastMCP("env", log_level="WARNING")

    if "exec" in env.caps:

        @server.tool(structured_output=False)
        async def shell(cmd: str, timeout: float = 60) -> CallToolResult:
            """Run a shell command inside the sample environment."""
            try:
                result = await env.exec(cmd, timeout=max(0.01, min(timeout, 60)))
                return _mcp_result(
                    json.dumps({"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code}),
                    is_error=result.exit_code != 0,
                )
            except (OSError, ValueError, TimeoutError) as exc:
                return _mcp_result(str(exc) or "Command timed out", is_error=True)
            except Exception as exc:
                return _mcp_result("", is_error=True, fatal=f"Environment tool failed: {type(exc).__name__}: {exc}")

    if "files" in env.caps:

        @server.tool(structured_output=False)
        async def read_file(path: str) -> CallToolResult:
            """Read a UTF-8 file inside the sample workspace."""
            return await _call_env_tool(env.read_file(path))

        @server.tool(structured_output=False)
        async def write_file(path: str, content: str) -> CallToolResult:
            """Write a UTF-8 file inside the sample workspace."""
            return await _call_env_tool(env.write_file(path, content))

        @server.tool(structured_output=False)
        async def list_dir(path: str = ".") -> CallToolResult:
            """List entries inside the sample workspace."""
            return await _call_env_tool(env.list_dir(path))

    if "browser" in env.caps:
        from playwright.async_api import Error as BrowserError

        page = getattr(env, "page", None)
        if page is None:
            raise RuntimeError("Browser capability requires an open page")

        @server.tool(structured_output=False)
        async def browser_goto(url: str) -> CallToolResult:
            """Navigate the sample browser to a URL."""
            try:
                await page.goto(url)
            except BrowserError as exc:
                return _mcp_result(str(exc), is_error=True)
            return await _call_env_tool(page.locator("body").inner_text(), recoverable=(BrowserError,))

        @server.tool(structured_output=False)
        async def browser_click(selector: str) -> CallToolResult:
            """Click the first element matching a CSS selector."""
            return await _call_env_tool(page.locator(selector).first.click(), recoverable=(BrowserError,))

        @server.tool(structured_output=False)
        async def browser_fill(selector: str, value: str) -> CallToolResult:
            """Fill an input matching a CSS selector."""
            return await _call_env_tool(page.locator(selector).first.fill(value), recoverable=(BrowserError,))

        @server.tool(structured_output=False)
        async def browser_text() -> CallToolResult:
            """Read the visible text of the current page."""
            return await _call_env_tool(page.locator("body").inner_text(), recoverable=(BrowserError,))

    return server


def _make_common_tools(
    search_client: httpx.AsyncClient, fetch_client: httpx.AsyncClient, config: ToolConfig
) -> list[_ProviderTool]:
    """Select provider implementations behind the common search/fetch interface."""
    search = (
        _make_deepseek_search(
            search_client, config.ds_tool_url, config.ds_tool_key, config.search_results, config.observation_chars
        )
        if config.search_provider == "deepseek"
        else _make_serper_search(search_client, config.serper_api_key, config.search_results, config.observation_chars)
    )
    fetch = (
        _make_deepseek_fetch(fetch_client, config.ds_tool_url, config.ds_tool_key, config.observation_chars)
        if config.fetch_provider == "deepseek"
        else _make_crawl4ai_fetch(fetch_client, config.crawl4ai_url, config.crawl4ai_token, config.observation_chars)
    )
    return [search, fetch]


async def _execute_tool(tool: _ProviderTool, arguments: dict[str, Any]) -> tuple[str, bool]:
    """Return actionable observations; report developer errors and stop the episode."""
    value = str(arguments.get(tool.argument, arguments.get("input", "")))
    try:
        content = await asyncio.wait_for(tool.run(value), tool.timeout)
        return content, content.startswith(("Tool error:", "The crawler failed"))
    except TimeoutError:
        return f"The tool call timed out after {tool.timeout}s. Try another source.", True
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if tool.provider == "deepseek" and status in _DEEPSEEK_OBSERVABLE_ERRORS:
            return _DEEPSEEK_OBSERVABLE_ERRORS[status], True
        if status in _RETRYABLE_STATUS_CODES:
            return "The tool service is temporarily unavailable after retries. Try another source.", True
        detail = exc.response.text
        for header in ("Authorization", "X-API-KEY"):
            secret = exc.request.headers.get(header, "").removeprefix("Bearer ").strip()
            if secret:
                detail = detail.replace(secret, "[redacted]")
        _logger.error(
            "%s %s failed (HTTP %d): %s Server response: %s",
            tool.provider or "Tool",
            tool.name,
            status,
            _DEVELOPER_HTTP_ERRORS.get(status, exc.response.reason_phrase),
            clean_text(detail, 2000),
        )
        raise
    except (httpx.TimeoutException, httpx.NetworkError):
        return "The tool service could not be reached after retries. Try another source.", True
    except Exception:
        _logger.exception("%s %s failed", tool.provider or "Tool", tool.name)
        raise


def _make_deepseek_search(
    client: httpx.AsyncClient, base_url: str, api_key: str, search_results: int, observation_chars: int
) -> _ProviderTool:
    """Expose DeepSeek search results through the shared search tool."""
    endpoint = f"{base_url.rstrip('/')}/search"

    async def run(query: str) -> str:
        if not query.strip() or len(query) > 128:
            return "Tool error: DeepSeek search query must contain 1 to 128 characters and not be blank."
        response = await _request_with_retry(
            client,
            "POST",
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "count": search_results},
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
        return clean_text(json.dumps(results, ensure_ascii=False, indent=2), observation_chars)

    return _ProviderTool("search", "query", run, provider="deepseek")


def _make_deepseek_fetch(
    client: httpx.AsyncClient, base_url: str, api_key: str, observation_chars: int
) -> _ProviderTool:
    """Fetch Markdown content from the DeepSeek tools API."""
    endpoint = f"{base_url.rstrip('/')}/fetch"

    async def run(url: str) -> str:
        parsed = httpx.URL(url)
        if parsed.scheme not in ("http", "https") or not parsed.host or parsed.userinfo or len(url) > 8192:
            return "The fetch tool requires an http(s) URL with a host, no credentials, and at most 8192 characters."
        response = await _request_with_retry(
            client,
            "POST",
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"url": url},
            timeout=_HTTP_TIMEOUT,
        )
        payload = response.json()
        return clean_text(payload["content"], observation_chars) or "The page contained no readable text."

    return _ProviderTool("fetch", "url", run, provider="deepseek")


def _make_serper_search(
    client: httpx.AsyncClient, api_key: str, search_results: int, observation_chars: int
) -> _ProviderTool:
    "Build the search tool over a comma-separated serper key pool, rotated round-robin."
    keys = [key.strip() for key in api_key.split(",") if key.strip()]
    balance_cache: dict[str, int | None] = {}  # None = not probed yet; 0 = exhausted
    key_index = 0

    async def _get_balance(key: str) -> int | None:
        "Remaining credits for one key from the free /account endpoint; None if unknown."
        try:
            response = await client.get("https://google.serper.dev/account", headers={"X-API-KEY": key})
            response.raise_for_status()
            return int(response.json().get("balance") or 0)
        except Exception:  # noqa: BLE001 - a failed probe must not fail the search
            return None

    async def run(query: str) -> str:
        nonlocal key_index
        if not keys:
            return "Tool error: no serper API key configured."

        for _ in range(len(keys)):
            key = keys[key_index]
            key_index = (key_index + 1) % len(keys)
            balance = balance_cache.get(key)
            if balance == 0:
                continue
            if balance is None:
                # first use of this key: check its quota before spending a search on it
                balance = await _get_balance(key)
                if balance is not None and balance <= 0:
                    balance_cache[key] = 0
                    continue
                balance_cache[key] = balance
            try:
                response = await _request_with_retry(
                    client,
                    "POST",
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": key},
                    json={"q": query, "num": search_results},
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (401, 402, 403):
                    raise
                balance_cache[key] = 0
                continue
            if isinstance(balance, int) and balance > 0:
                balance_cache[key] = balance - 1
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
            return clean_text(json.dumps(results, ensure_ascii=False, indent=2), observation_chars)

        return "Tool error: every serper API key is out of credits."

    return _ProviderTool("search", "query", run, provider="serper")


def _make_crawl4ai_fetch(client: httpx.AsyncClient, base_url: str, token: str, observation_chars: int) -> _ProviderTool:
    """Build the fetch tool using a Crawl4AI server."""
    headers = {"Authorization": f"Bearer {token}"}

    async def run(url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return "The fetch tool only accepts an http(s) URL."

        try:
            response = await _request_with_retry(
                client,
                "POST",
                f"{base_url.rstrip('/')}/md",
                json={"url": url},
                headers=headers,
                # crawl4ai renders the page server-side before answering; this only needs
                # to outlive the tool timeout, which is the real bound
                timeout=httpx.Timeout(_TOOL_TIMEOUT_SEC + 10, connect=10.0),
            )
        except httpx.ConnectError:
            return f"Tool error: crawl4ai server unreachable at {base_url}."
        payload = response.json()
        if payload.get("success") is False:
            return f"The crawler failed on this URL: {payload.get('error') or 'unknown error'}"
        markdown = payload.get("markdown") or ""
        if isinstance(markdown, dict):
            markdown = markdown.get("fit_markdown") or markdown.get("raw_markdown") or ""
        text = clean_text(markdown, observation_chars)
        return text or "The page contained no readable text."

    return _ProviderTool("fetch", "url", run, provider="crawl4ai")


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **request_kwargs: Any,
) -> httpx.Response:
    "Make an HTTP request, retrying only transient transport and upstream failures."

    last_error: Exception | None = None
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
            last_error = exc
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
        if attempt == _TOOL_MAX_RETRIES - 1:
            _logger.warning(
                "tool request failed after %d attempts: %s: %s",
                _TOOL_MAX_RETRIES,
                type(last_error).__name__,
                last_error,
            )
    raise last_error


async def _call_env_tool(call: Awaitable[Any], *, recoverable: tuple[type[Exception], ...] = ()) -> CallToolResult:
    try:
        value = await call
        return _mcp_result(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    except (OSError, ValueError, TimeoutError) as exc:
        return _mcp_result(str(exc), is_error=True)
    except Exception as exc:
        if isinstance(exc, recoverable):
            return _mcp_result(str(exc), is_error=True)
        return _mcp_result("", is_error=True, fatal=f"Environment tool failed: {type(exc).__name__}: {exc}")


def _mcp_result(content: str, *, is_error: bool = False, fatal: str | None = None) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=content)] if content else [],
        isError=is_error,
        _meta={"harness": {"fatal": fatal}} if fatal else None,
    )
