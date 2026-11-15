"""Real stdio/Streamable HTTP integration plus MCP pagination contracts."""

import asyncio
import json
import os
import socket
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mcp import types

from harness import tool as tool_module
from harness.tool import HttpSpec, StdioSpec, connect

__all__ = []

FIXTURE = Path(__file__).parent / "fixtures/mcp_server.py"


class MCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_discovers_schemas_and_preserves_structured_results(self):
        config = StdioSpec(cmd=sys.executable, args=[str(FIXTURE)])
        async with connect(config) as connection:
            self.assertIn("increment", {tool["function"]["name"] for tool in connection.schemas})
            first, second = await asyncio.gather(
                connection.call("increment", {"delta": 1}), connection.call("increment", {"delta": 2})
            )
            assert first.raw is not None and second.raw is not None
            self.assertFalse(first.is_error)
            counts = sorted([first.raw["structuredContent"]["count"], second.raw["structuredContent"]["count"]])
            self.assertEqual(counts[-1], 3)
            pid = first.raw["structuredContent"]["pid"]
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_stdio_preserves_recoverable_fatal_and_nontext_results(self):
        async with connect(StdioSpec(cmd=sys.executable, args=[str(FIXTURE)])) as connection:
            recoverable = await connection.call("recoverable", {})
            fatal = await connection.call("fatal", {})
            rich = await connection.call("rich", {})
        self.assertEqual(
            (recoverable.content, recoverable.is_error, recoverable.fatal), ("page unavailable", True, None)
        )
        self.assertEqual((fatal.content, fatal.is_error, fatal.fatal), ("", True, "HTTP 402: balance"))
        assert rich.raw is not None
        self.assertEqual(rich.raw["content"][0]["data"], "YWJj")
        self.assertEqual(rich.raw["structuredContent"], {"image": "test"})
        self.assertIn("YWJj", rich.content)

    async def test_streamable_http_round_trip_closes_the_connection_scope(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(FIXTURE),
            "--http-port",
            str(port),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(10):
                while True:
                    if process.returncode is not None:
                        self.fail(f"HTTP MCP fixture exited with {process.returncode}")
                    try:
                        _, writer = await asyncio.open_connection("127.0.0.1", port)
                    except OSError:
                        await asyncio.sleep(0.02)
                        continue
                    writer.close()
                    await writer.wait_closed()
                    break
            async with connect(HttpSpec(url=f"http://127.0.0.1:{port}/mcp")) as connection:
                response = await connection.call("increment", {"delta": 7})
                assert response.raw is not None
                self.assertEqual(response.raw["structuredContent"]["count"], 7)
                self.assertFalse(response.is_error)
        finally:
            process.terminate()
            await process.wait()


class MCPContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_pagination_discovers_all_tools_before_yielding(self):
        session = MagicMock(initialize=AsyncMock())
        session.list_tools = AsyncMock(
            side_effect=[
                types.ListToolsResult(
                    tools=[types.Tool(name="one", inputSchema={"type": "object"})], nextCursor="next"
                ),
                types.ListToolsResult(tools=[types.Tool(name="two", inputSchema={"type": "object"})]),
            ]
        )
        scope = MagicMock()
        scope.__aenter__ = AsyncMock(return_value=session)
        scope.__aexit__ = AsyncMock(return_value=False)

        @asynccontextmanager
        async def transport(*args, **kwargs):
            yield MagicMock(), MagicMock()

        with (
            patch.object(tool_module, "stdio_client", transport),
            patch.object(tool_module, "ClientSession", return_value=scope),
        ):
            async with connect(StdioSpec(cmd="server")) as connection:
                self.assertEqual([tool["function"]["name"] for tool in connection.schemas], ["one", "two"])
                self.assertEqual(session.list_tools.await_count, 2)
                self.assertEqual(session.list_tools.await_args_list[1].kwargs["params"].cursor, "next")
        scope.__aexit__.assert_awaited_once()

    async def test_initialization_failure_closes_acquired_transports(self):
        closed = []

        @asynccontextmanager
        async def transport(*args, **kwargs):
            try:
                yield MagicMock(), MagicMock()
            finally:
                closed.append(True)

        session = MagicMock(initialize=AsyncMock(side_effect=RuntimeError("init failed")))
        scope = MagicMock()
        scope.__aenter__ = AsyncMock(return_value=session)
        scope.__aexit__ = AsyncMock(return_value=False)
        with (
            patch.object(tool_module, "stdio_client", transport),
            patch.object(tool_module, "ClientSession", return_value=scope),
            self.assertRaisesRegex(RuntimeError, "init failed"),
        ):
            async with connect(StdioSpec(cmd="server")):
                self.fail("initialization should fail before yielding")
        self.assertEqual(closed, [True])
        scope.__aexit__.assert_awaited_once()

    async def test_connection_requires_a_positive_timeout(self):
        with self.assertRaises(ValueError):
            async with connect(StdioSpec(cmd="server", timeout=0)):
                self.fail("Invalid timeout")


class ToolBoxScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_session_supports_concurrent_children_and_local_name_collisions(self):
        from mcp.server.fastmcp import FastMCP

        from harness.tool import open_toolbox

        shared = FastMCP("web")

        @shared.tool()
        async def search(query: str) -> str:
            await asyncio.sleep(0.001)
            return f"shared:{query}"

        async with open_toolbox(shared) as parent:
            original = parent.schemas

            async def one(index):
                local = FastMCP(f"sample_{index}")

                @local.tool(name="search")
                async def local_search(query: str) -> str:
                    return f"{index}:{query}"

                async with parent.scoped(local) as child:
                    names = [s["function"]["name"] for s in child.schemas]
                    self.assertEqual(names, ["web__search", f"sample_{index}__search"])
                    a, b = await asyncio.gather(
                        child.call(names[0], {"query": str(index)}), child.call(names[1], {"query": "local"})
                    )
                    self.assertEqual((a.content, b.content), (f"shared:{index}", f"{index}:local"))
                    self.assertEqual(parent.schemas, original)

            await asyncio.gather(*(one(i) for i in range(4)))
            self.assertEqual((await parent.call("search", {"query": "after"})).content, "shared:after")
            self.assertTrue((await parent.call("unknown", {})).is_error)

    async def test_normalized_prefix_collisions_and_long_names_remain_unique(self):
        from harness.tool import Connection, ToolBox

        tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
        connections = []
        for name in ("same space", "same_space", "s" * 100):
            c = MagicMock(spec=Connection, schemas=tools)
            c.name = name
            connections.append(c)
        box = ToolBox(tuple(connections))
        names = [s["function"]["name"] for s in box.schemas]
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(len(n) <= 64 for n in names))
        box.schemas[0]["function"]["name"] = "mutated"
        self.assertEqual([s["function"]["name"] for s in box.schemas], names)


class BuiltInServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_common_tools_use_deepseek_through_mcp(self):
        import httpx

        from harness.tool import common_server, open_toolbox

        requests = []

        def respond(request):
            requests.append(request)
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.headers["Authorization"], "Bearer ds-secret")
            if request.url.path == "/v1/tools/search":
                self.assertEqual(json.loads(request.content), {"query": "example", "count": 8})
                return httpx.Response(
                    200, json={"results": [{"title": "Example", "url": "https://example.com", "summary": "summary"}]}
                )
            self.assertEqual(request.url.path, "/v1/tools/fetch")
            self.assertEqual(json.loads(request.content), {"url": "https://example.com"})
            return httpx.Response(200, json={"content": "page content"})

        client_class = httpx.AsyncClient

        def make_client(**kwargs):
            return client_class(transport=httpx.MockTransport(respond), **kwargs)

        with (
            patch.dict(
                os.environ, {"DS_TOOL_URL": "https://ds.example/v1/tools", "DS_TOOL_KEY": "ds-secret"}, clear=True
            ),
            patch.object(httpx, "AsyncClient", side_effect=make_client),
        ):
            async with open_toolbox(common_server()) as tools:
                search = await tools.call("search", {"query": "example"})
                fetch = await tools.call("fetch", {"url": "https://example.com"})
        self.assertFalse(search.is_error)
        self.assertIn("summary", search.content)
        self.assertEqual(fetch.content, "page content")
        self.assertIsNone(fetch.fatal)
        self.assertEqual([request.url.path for request in requests], ["/v1/tools/search", "/v1/tools/fetch"])

    async def test_common_mcp_preserves_recoverable_and_fatal_error_channels(self):
        import httpx

        from harness import tool

        async with tool.open_toolbox(tool.common_server()) as tools:
            with patch.object(tool, "_execute_tool", return_value=("Page unavailable", True)):
                response = await tools.call("fetch", {"url": "https://example.com"})
            self.assertEqual((response.content, response.is_error, response.fatal), ("Page unavailable", True, None))
            request = httpx.Request("POST", "https://private.example/tools", headers={"Authorization": "Bearer secret"})
            error = httpx.HTTPStatusError(
                "private diagnostic secret", request=request, response=httpx.Response(402, request=request)
            )
            with patch.object(tool, "_execute_tool", side_effect=error):
                response = await tools.call("fetch", {"url": "https://example.com"})
            self.assertEqual(response.content, "")
            self.assertIn("HTTP 402", response.fatal or "")
            self.assertNotIn("secret", str(response.raw))

    async def test_unexpected_environment_errors_are_fatal(self):
        from harness import Env
        from harness.tool import env_server, open_toolbox

        class Broken(Env):
            caps = frozenset({"files", "exec"})

            async def read_file(self, path):
                raise KeyError("implementation failed")

            async def exec(self, cmd, timeout=60):
                raise RuntimeError("implementation failed")

        async with Broken() as env, open_toolbox(env_server(env)) as tools:
            for name, args in (("read_file", {"path": "x"}), ("shell", {"cmd": "x"})):
                response = await tools.call(name, args)
                self.assertTrue(response.is_error)
                self.assertIsNotNone(response.fatal)
                self.assertEqual(response.content, "")

    async def test_builtin_common_stdio_cli_discovers_the_same_tools(self):
        from harness.tool import open_toolbox

        async with open_toolbox(
            StdioSpec(cmd=str(Path(sys.executable).with_name("harness")), args=["serve-common"])
        ) as tools:
            self.assertEqual({s["function"]["name"] for s in tools.schemas}, {"search", "fetch"})


if __name__ == "__main__":
    unittest.main()
