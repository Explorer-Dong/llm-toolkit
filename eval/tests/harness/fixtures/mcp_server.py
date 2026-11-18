"""Deterministic MCP server used by real stdio/HTTP integration tests."""

import argparse
import asyncio
import os

from mcp import types
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

parser = argparse.ArgumentParser()
parser.add_argument("--http-port", type=int)
args = parser.parse_args()
server = FastMCP("test-tools", port=args.http_port or 8000, log_level="ERROR")
counter = 0


class CounterResult(BaseModel):
    count: int
    pid: int
    sample: str


@server.tool()
def increment(delta: int = 1) -> CounterResult:
    global counter
    counter += delta
    return CounterResult(count=counter, pid=os.getpid(), sample=os.getenv("TEST_SAMPLE", ""))


@server.tool(structured_output=False)
def recoverable() -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text="page unavailable")], isError=True)


@server.tool(structured_output=False)
def fatal() -> types.CallToolResult:
    return types.CallToolResult(content=[], isError=True, _meta={"harness": {"fatal": "HTTP 402: balance"}})


@server.tool(structured_output=False)
def rich() -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.ImageContent(type="image", data="YWJj", mimeType="image/png")],
        structuredContent={"image": "test"},
    )


@server.tool()
async def pause(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return "finished"


server.run(transport="streamable-http" if args.http_port else "stdio")
