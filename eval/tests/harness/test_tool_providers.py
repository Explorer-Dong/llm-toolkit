"""Provider request, retry, and error contracts using fake HTTP transports."""

import argparse
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from eval.harness import add_harness_args
from eval.harness import tool as tool_module

__all__ = []
TEST_ENV = {
    "DS_TOOL_URL": "https://ds.example/v1/tools/",
    "DS_TOOL_KEY": "ds-secret",
    "SERPER_API_KEY": "serper-secret",
    "CRAWL4AI_URL": "https://crawler.example",
    "CRAWL4AI_TOKEN": "crawler-secret",
}


class SerperSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_credentials_and_exhausted_credits_are_logged(self):
        for api_key, balance in (("", None), ("serper-secret", 0)):
            with self.subTest(api_key_configured=bool(api_key)):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request, balance=balance: httpx.Response(200, json={"balance": balance})
                    )
                ) as client:
                    search = tool_module._make_serper_search(client, api_key, 8, 100)
                    with self.assertLogs(tool_module._logger, level="ERROR"):
                        result = await tool_module._execute_tool(search, {"query": "example"})
                self.assertTrue(result[1])

    async def test_blank_query_is_silent_even_without_credentials(self):
        async with httpx.AsyncClient() as client:
            search = tool_module._make_serper_search(client, "", 8, 100)
            with self.assertNoLogs(tool_module._logger, level="WARNING"):
                result = await tool_module._execute_tool(search, {"query": " "})
        self.assertTrue(result[1])

    async def test_sends_authenticated_search_and_formats_results(self):
        requests = []

        def respond(request):
            requests.append(request)
            self.assertEqual(request.headers["X-API-KEY"], "serper-secret")
            if request.url.path == "/account":
                self.assertEqual(request.method, "GET")
                return httpx.Response(200, json={"balance": 10})
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/search")
            self.assertEqual(json.loads(request.content), {"q": "example", "num": 8})
            return httpx.Response(
                200,
                json={
                    "organic": [
                        {
                            "title": "Example",
                            "link": "https://example.com",
                            "snippet": "summary",
                        },
                        {"title": "Invalid", "link": "javascript:alert(1)", "snippet": "ignored"},
                    ]
                },
            )

        args = _parse_args(flags=["--search-provider", "serper", "--fetch-provider", "deepseek"])
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            tools = tool_module._make_common_tools(client, client, args)
            search = tools[0]
            result = json.loads(await search.run("example"))

        self.assertEqual([request.url.path for request in requests], ["/account", "/search"])
        self.assertEqual(result, [{"title": "Example", "url": "https://example.com", "snippet": "summary"}])


class Crawl4AIFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_authenticated_fetch_and_returns_readable_markdown(self):
        requests = []

        def respond(request):
            requests.append(request)
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/md")
            self.assertEqual(request.headers["Authorization"], "Bearer crawler-secret")
            self.assertEqual(json.loads(request.content), {"url": "https://example.com"})
            return httpx.Response(
                200,
                json={"success": True, "markdown": {"fit_markdown": "page content", "raw_markdown": "fallback"}},
            )

        args = _parse_args(flags=["--search-provider", "deepseek", "--fetch-provider", "crawl4ai"])
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            tools = tool_module._make_common_tools(client, client, args)
            result = await tools[1].run("https://example.com")

        self.assertEqual(len(requests), 1)
        self.assertEqual(result, "page content")

    async def test_rejects_invalid_url_and_reports_crawler_failure(self):
        requests = []

        def respond(request):
            requests.append(request)
            self.assertEqual(request.method, "POST")
            return httpx.Response(200, json={"success": False, "error": "render failed"})

        args = _parse_args(flags=["--search-provider", "deepseek", "--fetch-provider", "crawl4ai"])
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            fetch = tool_module._make_common_tools(client, client, args)[1]
            invalid_result = await fetch.run("file:///tmp/page")
            failure_result = await fetch.run("https://example.com")

        self.assertIn("only accepts an http(s) URL", invalid_result)
        self.assertEqual(failure_result, "The crawler failed on this URL: render failed")
        self.assertEqual(len(requests), 1)


class DeepSeekSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_to_tools_search_and_formats_results(self):
        requests = []

        def respond(request):
            requests.append(request)
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/v1/tools/search")
            self.assertEqual(request.headers["Authorization"], "Bearer ds-secret")
            self.assertEqual(json.loads(request.content), {"query": "example", "count": 8})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                            "summary": "summary",
                            "snippet": "short",
                        }
                    ]
                },
            )

        args = _parse_args(flags=["--search-provider", "deepseek", "--fetch-provider", "crawl4ai"])
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            search = tool_module._make_common_tools(client, client, args)[0]
            result = json.loads(await search.run("example"))

        self.assertEqual(len(requests), 1)
        self.assertEqual(result, [{"title": "Example", "url": "https://example.com", "snippet": "summary"}])

    async def test_rejects_invalid_query_and_handles_empty_results(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"results": []})

        args = _parse_args(flags=["--search-provider", "deepseek", "--fetch-provider", "crawl4ai"])
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            search = tool_module._make_common_tools(client, client, args)[0]
            for query in ("", "   ", "汉" * 129):
                self.assertIn("Tool error", await search.run(query))
            self.assertIn("No search results", await search.run("example"))

        self.assertEqual(len(requests), 1)


class DeepSeekFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_to_tools_fetch_with_or_without_trailing_slash(self):
        requests = []

        def respond(request):
            requests.append(request)
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/v1/tools/fetch")
            self.assertEqual(request.headers["Authorization"], "Bearer ds-secret")
            self.assertEqual(json.loads(request.content), {"url": "https://example.com"})
            return httpx.Response(200, json={"title": "Example", "content": "page content"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            for base_url in ("https://ds.example/v1/tools", "https://ds.example/v1/tools/"):
                args = _parse_args(flags=["--search-provider", "serper", "--fetch-provider", "deepseek"])
                args.ds_tool_url = base_url
                fetch = tool_module._make_common_tools(client, client, args)[1]
                result = await fetch.run("https://example.com")
                self.assertEqual(result, "page content")

        self.assertEqual(len(requests), 2)

    async def test_rejects_invalid_urls_and_truncates_content(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"content": "abcdefgh"})

        args = _parse_args(
            flags=["--search-provider", "serper", "--fetch-provider", "deepseek", "--observation-chars", "4"]
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            fetch = tool_module._make_common_tools(client, client, args)[1]
            for url in (
                "file:///tmp/test",
                "https://",
                "https://user:pass@example.com",
                "https://example.com/" + "x" * 8192,
            ):
                self.assertIn("fetch tool requires", await fetch.run(url))
            self.assertEqual(await fetch.run("https://example.com"), "abcd ...")

        self.assertEqual(len(requests), 1)


class HttpRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_http_failures_and_returns_actionable_observation(self):
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                attempts = []

                def respond(request, attempts=attempts, status=status):
                    attempts.append(request)
                    return httpx.Response(status, json={"error": {"message": "failure"}})

                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                    fetch = tool_module._make_deepseek_fetch(client, TEST_ENV["DS_TOOL_URL"], "secret", 100)
                    with (
                        patch.object(tool_module.asyncio, "sleep", new_callable=AsyncMock) as sleep,
                        self.assertLogs(tool_module._logger, level="WARNING"),
                    ):
                        result = await tool_module._execute_tool(fetch, {"url": "https://example.com"})
                self.assertEqual(len(attempts), 3)
                self.assertEqual([call.args[0] for call in sleep.await_args_list], [3, 9])
                self.assertTrue(result[1])
                self.assertEqual(
                    result[0], "The tool service is temporarily unavailable after retries. Try another source."
                )

    async def test_non_retryable_status_codes_are_requested_once(self):
        for status in (400, 401, 402, 403, 404, 413):
            with self.subTest(status=status):
                attempts = []

                def respond(request, attempts=attempts, status=status):
                    attempts.append(request)
                    return httpx.Response(status, json={"error": {"message": "failure"}})

                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                    fetch = tool_module._make_deepseek_fetch(client, TEST_ENV["DS_TOOL_URL"], "secret", 100)
                    with patch.object(tool_module.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                        if status in (400, 401, 402):
                            with (
                                self.assertLogs(tool_module._logger, level="ERROR"),
                                self.assertRaises(httpx.HTTPStatusError),
                            ):
                                await tool_module._execute_tool(fetch, {"url": "https://example.com"})
                        else:
                            await tool_module._execute_tool(fetch, {"url": "https://example.com"})
                self.assertEqual(len(attempts), 1)
                sleep.assert_not_awaited()


class ToolExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_400_is_a_visible_developer_error(self):
        await self._assert_developer_error(400, "Check the request format")

    async def test_401_is_a_visible_credentials_error(self):
        await self._assert_developer_error(401, "Check the tool credentials")

    async def test_402_is_a_visible_balance_error(self):
        await self._assert_developer_error(402, "Top up the tool account")

    async def test_403_observation_explains_the_security_policy(self):
        tool = tool_module._ProviderTool("fetch", "url", AsyncMock(side_effect=_http_error(403)), provider="deepseek")
        result = await tool_module._execute_tool(tool, {"url": "https://example.com"})
        self.assertTrue(result[1])
        self.assertEqual(result[0], "The Fetch request was rejected by the security policy.")

    async def test_404_observation_describes_unusable_page_content(self):
        tool = tool_module._ProviderTool("fetch", "url", AsyncMock(side_effect=_http_error(404)), provider="deepseek")
        result = await tool_module._execute_tool(tool, {"url": "https://example.com"})
        self.assertEqual(
            result[0],
            "Fetch could not retrieve usable page content, for example due to an unavailable URL, "
            "a URL blocked by site policy, an error from the target site, an unsupported content type, "
            "or a page with no extractable Markdown content.",
        )
        self.assertNotIn("HTTPStatusError", result[0])
        self.assertNotIn("/v1/tools/fetch", result[0])
        self.assertNotIn("developer.mozilla.org", result[0])

    async def test_413_observation_explains_request_size(self):
        tool = tool_module._ProviderTool("fetch", "url", AsyncMock(side_effect=_http_error(413)), provider="deepseek")
        result = await tool_module._execute_tool(tool, {"url": "https://example.com"})
        self.assertEqual(result[0], "The request body is too large.")

    async def test_deepseek_descriptions_do_not_mislabel_other_providers(self):
        tool = tool_module._ProviderTool("fetch", "url", AsyncMock(side_effect=_http_error(403)), provider="crawl4ai")
        with self.assertLogs(tool_module._logger, level="ERROR"), self.assertRaises(httpx.HTTPStatusError):
            await tool_module._execute_tool(tool, {"url": "https://example.com"})

    async def test_programming_errors_are_reported_and_propagated(self):
        tool = tool_module._ProviderTool("fetch", "url", AsyncMock(side_effect=KeyError("content")))
        with self.assertLogs(tool_module._logger, level="ERROR") as logs, self.assertRaises(KeyError):
            await tool_module._execute_tool(tool, {"url": "https://example.com"})
        self.assertIn("KeyError", "\n".join(logs.output))

    async def test_transport_timeout_returns_an_actionable_observation(self):
        tool = tool_module._ProviderTool("fetch", "url", AsyncMock(side_effect=httpx.ReadTimeout("internal endpoint")))
        result = await tool_module._execute_tool(tool, {"url": "https://example.com"})
        self.assertEqual(result[0], "The tool service could not be reached after retries. Try another source.")

    async def test_tool_budget_timeout_returns_an_actionable_observation(self):
        tool = tool_module._ProviderTool("fetch", "url", AsyncMock(side_effect=TimeoutError), timeout=0.01)
        result = await tool_module._execute_tool(tool, {"url": "https://example.com"})
        self.assertEqual(result[0], "The tool call timed out after 0.01s. Try another source.")

    async def _assert_developer_error(self, status, expected_action):
        tool = tool_module._ProviderTool(
            "fetch", "url", AsyncMock(side_effect=_http_error(status)), provider="deepseek"
        )
        with self.assertLogs(tool_module._logger, level="ERROR") as logs, self.assertRaises(httpx.HTTPStatusError):
            await tool_module._execute_tool(tool, {"url": "https://example.com"})
        text = "\n".join(logs.output)
        self.assertIn(f"HTTP {status}", text)
        self.assertIn(expected_action, text)
        self.assertIn("server diagnostic", text)
        self.assertNotIn("test-secret", text)


def _http_error(status):
    request = httpx.Request(
        "POST", "https://ds.example/v1/tools/fetch", headers={"Authorization": "Bearer test-secret"}
    )
    response = httpx.Response(status, request=request, json={"error": {"message": "server diagnostic test-secret"}})
    return httpx.HTTPStatusError("failure", request=request, response=response)


def _parse_args(*, flags=()):
    parser = argparse.ArgumentParser()
    with patch.dict(os.environ, TEST_ENV, clear=True):
        add_harness_args(parser, tools=True)
    return parser.parse_args(flags)


if __name__ == "__main__":
    unittest.main()
