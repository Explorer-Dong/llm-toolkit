"""Run real smoke checks for the configured search and fetch providers."""

import argparse
import asyncio
import json
import os
import sys
from argparse import Namespace

from dotenv import load_dotenv

from eval import ROOT
from eval.core import harness

# Globals

TOOL_ENV = {
    "serper-search": ("SERPER_API_KEY",),
    "deepseek-search": ("DS_TOOL_URL", "DS_TOOL_KEY"),
    "crawl4ai-fetch": ("CRAWL4AI_URL", "CRAWL4AI_TOKEN"),
    "deepseek-fetch": ("DS_TOOL_URL", "DS_TOOL_KEY"),
}
ALL_TOOLS = tuple(TOOL_ENV)


# Public functions


def parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Smoke-test real search and fetch tool APIs")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--tool", choices=ALL_TOOLS, action="append", help="tool to test; may be repeated")
    selection.add_argument("--all", action="store_true", help="test every configured provider")
    parser.add_argument("--query", default="OpenAI official website", help="query sent by search tools")
    parser.add_argument("--url", default="https://example.com/", help="URL fetched by fetch tools")
    return parser.parse_args()


async def run_tool(name: str, query: str, url: str) -> str:
    missing = [key for key in TOOL_ENV[name] if not os.getenv(key, "").strip()]
    if missing:
        raise ValueError(f"missing configuration: {', '.join(missing)}")

    search_provider = "deepseek" if name == "deepseek-search" else "serper"
    fetch_provider = "deepseek" if name == "deepseek-fetch" else "crawl4ai"
    args = Namespace(
        search_provider=search_provider,
        fetch_provider=fetch_provider,
        serper_api_key=os.getenv("SERPER_API_KEY", ""),
        crawl4ai_url=os.getenv("CRAWL4AI_URL", ""),
        crawl4ai_token=os.getenv("CRAWL4AI_TOKEN", ""),
        ds_tool_url=os.getenv("DS_TOOL_URL", ""),
        ds_tool_key=os.getenv("DS_TOOL_KEY", ""),
        search_results=4,
        observation_chars=12000,
    )

    async with harness.tool_clients() as (client_search, client_fetch):
        tools = harness.make_tools(client_search, client_fetch, args)
        is_search = name.endswith("-search")
        tool = tools[0] if is_search else tools[1]
        result = await asyncio.wait_for(tool.run(query if is_search else url), timeout=tool.timeout + 10)

    if result.startswith(("Tool error:", "No search results", "The crawler failed")):
        raise RuntimeError(result)
    if is_search:
        results = json.loads(result)
        if not isinstance(results, list) or not results:
            raise RuntimeError("search returned no usable results")
        return f"{len(results)} results"
    if not result.strip() or result.startswith("The page contained no readable text"):
        raise RuntimeError("fetch returned no readable content")
    return f"{len(result)} characters"


def _redact(message: str) -> str:
    secrets = []
    for key in ("SERPER_API_KEY", "DS_TOOL_KEY", "CRAWL4AI_TOKEN"):
        secrets.extend(part.strip() for part in os.getenv(key, "").split(",") if part.strip())
    for secret in secrets:
        message = message.replace(secret, "[redacted]")
    return message


async def _run_selected(names: list[str], query: str, url: str) -> int:
    failed = False
    for name in names:
        try:
            summary = await run_tool(name, query, url)
        except Exception as exc:  # noqa: BLE001 - report each provider and continue through --all
            failed = True
            print(f"FAIL {name}: {_redact(f'{type(exc).__name__}: {exc}')}")
        else:
            print(f"PASS {name}: {summary}")
    return int(failed)


def main() -> int:
    args = parse_args()
    names = list(ALL_TOOLS) if args.all else args.tool
    return asyncio.run(_run_selected(names, args.query, args.url))


if __name__ == "__main__":
    sys.exit(main())
