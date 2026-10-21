"""Tool provider contracts and CLI integration; no network or real credentials needed."""

import asyncio
import contextlib
import io
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from eval import browsecomp, gaia, webwalkerqa
from eval.core import harness, model, runner

__all__ = []

BENCHMARKS = (browsecomp, gaia, webwalkerqa)
TEST_ENV = {
    "MODEL_NAME": "test-model",
    "DS_TOOL_URL": "https://ds.example/v1/tools/",
    "DS_TOOL_KEY": "ds-secret",
    "SERPER_API_KEY": "serper-secret",
    "CRAWL4AI_URL": "https://crawler.example",
    "CRAWL4AI_TOKEN": "crawler-secret",
}


class ToolCliTests(unittest.TestCase):
    def test_all_entrypoints_and_provider_combinations(self):
        for module, search, fetch in itertools.product(BENCHMARKS, ("serper", "deepseek"), ("crawl4ai", "deepseek")):
            with self.subTest(module=module.__name__, search=search, fetch=fetch):
                flags = ["--search-provider", search, "--fetch-provider", fetch]
                args = _parse_args(module, flags)
                self.assertEqual((args.search_provider, args.fetch_provider), (search, fetch))
        for module in BENCHMARKS:
            args = _parse_args(module)
            self.assertEqual((args.search_provider, args.fetch_provider), ("deepseek", "deepseek"))

    def test_credentials_load_from_env_and_cli_can_override(self):
        flags = ["--search-provider", "deepseek", "--fetch-provider", "deepseek"]
        for module in BENCHMARKS:
            args = _parse_args(module, flags, {key: value for key, value in TEST_ENV.items() if key.startswith("DS_")})
            self.assertEqual(args.ds_tool_key, "ds-secret")
            args = _parse_args(
                module,
                [*flags, "--ds-tool-url", "https://override.example/v1", "--ds-tool-key", "override"],
            )
            self.assertEqual((args.ds_tool_url, args.ds_tool_key), ("https://override.example/v1", "override"))
            args = _parse_args(module, flags, {})
            self.assertEqual((args.ds_tool_url, args.ds_tool_key), ("", ""))

    def test_invalid_providers_and_search_result_counts_are_rejected(self):
        for module in BENCHMARKS:
            for flags in (
                ["--search-provider", "invalid"],
                ["--fetch-provider", "invalid"],
                ["--search-results", "0"],
                ["--search-results", "51"],
            ):
                with self.subTest(module=module.__name__, flags=flags), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exc:
                        _parse_args(module, flags)
                    self.assertEqual(exc.exception.code, 2)

    def test_run_config_records_providers_and_excludes_credentials(self):
        for module in BENCHMARKS:
            args = _parse_args(module, ["--search-provider", "deepseek", "--fetch-provider", "deepseek"])
            # Stop at artifact creation, before any model, judge, or tool request.
            with (
                patch.object(module, "parse_args", return_value=args),
                patch.object(module, "load_tasks", return_value=[]),
                patch.object(runner, "prepare_run_dir", side_effect=RuntimeError("stop")) as prepare,
                self.assertRaisesRegex(RuntimeError, "stop"),
            ):
                module.main()
            with tempfile.TemporaryDirectory() as temp, patch.object(runner, "OUTPUT_DIR", Path(temp)):
                folder = runner.prepare_run_dir(*prepare.call_args.args, **prepare.call_args.kwargs)
                config_text = (folder / "config.json").read_text()
            config = json.loads(config_text)
            self.assertEqual(config["search_provider"], "deepseek")
            self.assertEqual(config["fetch_provider"], "deepseek")
            self.assertEqual([tool["function"]["name"] for tool in config["tools"]], ["search", "fetch"])
            for credential in ("ds-secret", "serper-secret", "crawler-secret"):
                self.assertNotIn(credential, config_text)


class SerperSearchTests(unittest.IsolatedAsyncioTestCase):
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
            tools = harness.make_tools(client, client, args)
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
            tools = harness.make_tools(client, client, args)
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
            fetch = harness.make_tools(client, client, args)[1]
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
            search = harness.make_tools(client, client, args)[0]
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
            search = harness.make_tools(client, client, args)[0]
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
                fetch = harness.make_tools(client, client, args)[1]
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
            fetch = harness.make_tools(client, client, args)[1]
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
                    fetch = harness._make_deepseek_fetch(client, TEST_ENV["DS_TOOL_URL"], "secret", 100)
                    with (
                        patch.object(harness.asyncio, "sleep", new_callable=AsyncMock) as sleep,
                        self.assertLogs(harness._logger, level="WARNING"),
                    ):
                        result = await harness._execute_tool(fetch, {"url": "https://example.com"})
                self.assertEqual(len(attempts), 3)
                self.assertEqual([call.args[0] for call in sleep.await_args_list], [3, 9])
                self.assertEqual(
                    result, "The tool service is temporarily unavailable after retries. Try another source."
                )

    async def test_non_retryable_status_codes_are_requested_once(self):
        for status in (400, 401, 402, 403, 404, 413):
            with self.subTest(status=status):
                attempts = []

                def respond(request, attempts=attempts, status=status):
                    attempts.append(request)
                    return httpx.Response(status, json={"error": {"message": "failure"}})

                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                    fetch = harness._make_deepseek_fetch(client, TEST_ENV["DS_TOOL_URL"], "secret", 100)
                    with patch.object(harness.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                        if status in (400, 401, 402):
                            with (
                                self.assertLogs(harness._logger, level="ERROR"),
                                self.assertRaises(httpx.HTTPStatusError),
                            ):
                                await harness._execute_tool(fetch, {"url": "https://example.com"})
                        else:
                            await harness._execute_tool(fetch, {"url": "https://example.com"})
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
        tool = harness.Tool(harness.FETCH_SCHEMA, "url", AsyncMock(side_effect=_http_error(403)), provider="deepseek")
        result = await harness._execute_tool(tool, {"url": "https://example.com"})
        self.assertEqual(result, "The Fetch request was rejected by the security policy.")

    async def test_404_observation_describes_unusable_page_content(self):
        tool = harness.Tool(harness.FETCH_SCHEMA, "url", AsyncMock(side_effect=_http_error(404)), provider="deepseek")
        result = await harness._execute_tool(tool, {"url": "https://example.com"})
        self.assertEqual(
            result,
            "Fetch could not retrieve usable page content, for example due to an unavailable URL, "
            "a URL blocked by site policy, an error from the target site, an unsupported content type, "
            "or a page with no extractable Markdown content.",
        )
        self.assertNotIn("HTTPStatusError", result)
        self.assertNotIn("/v1/tools/fetch", result)
        self.assertNotIn("developer.mozilla.org", result)

    async def test_413_observation_explains_request_size(self):
        tool = harness.Tool(harness.FETCH_SCHEMA, "url", AsyncMock(side_effect=_http_error(413)), provider="deepseek")
        result = await harness._execute_tool(tool, {"url": "https://example.com"})
        self.assertEqual(result, "The request body is too large.")

    async def test_deepseek_descriptions_do_not_mislabel_other_providers(self):
        tool = harness.Tool(harness.FETCH_SCHEMA, "url", AsyncMock(side_effect=_http_error(403)), provider="crawl4ai")
        with self.assertLogs(harness._logger, level="ERROR"), self.assertRaises(httpx.HTTPStatusError):
            await harness._execute_tool(tool, {"url": "https://example.com"})

    async def test_programming_errors_are_reported_and_propagated(self):
        tool = harness.Tool(harness.FETCH_SCHEMA, "url", AsyncMock(side_effect=KeyError("content")))
        with self.assertLogs(harness._logger, level="ERROR") as logs, self.assertRaises(KeyError):
            await harness._execute_tool(tool, {"url": "https://example.com"})
        self.assertIn("KeyError", "\n".join(logs.output))

    async def test_transport_timeout_returns_an_actionable_observation(self):
        tool = harness.Tool(harness.FETCH_SCHEMA, "url", AsyncMock(side_effect=httpx.ReadTimeout("internal endpoint")))
        result = await harness._execute_tool(tool, {"url": "https://example.com"})
        self.assertEqual(result, "The tool service could not be reached after retries. Try another source.")

    async def test_tool_budget_timeout_returns_an_actionable_observation(self):
        tool = harness.Tool(harness.FETCH_SCHEMA, "url", AsyncMock(side_effect=TimeoutError), timeout=0.01)
        result = await harness._execute_tool(tool, {"url": "https://example.com"})
        self.assertEqual(result, "The tool call timed out after 0.01s. Try another source.")

    async def test_developer_errors_stop_react_without_being_sent_to_the_model(self):
        tool = harness.Tool(harness.FETCH_SCHEMA, "url", AsyncMock(side_effect=_http_error(402)), provider="deepseek")
        ctx = model.Context(MagicMock(), model.ModelParams("model", 0.0, 1.0, 1024, 42), asyncio.Semaphore(1))
        calls = [{"id": "fetch_1", "name": "fetch", "arguments": {"url": "https://example.com"}}]
        with (
            patch.object(
                harness, "call_model", new_callable=AsyncMock, return_value=("", calls, "", None, "tool_calls")
            ) as call,
            self.assertLogs(harness._logger, level="ERROR"),
        ):
            result = await harness.react(
                ctx, "question", tools=[tool], system_prompt="system", max_steps=3, task_timeout=1
            )
        call.assert_awaited_once()
        self.assertIn("HTTPStatusError", result.sys_error or "")
        self.assertIsNone(result.final_content)
        self.assertFalse(any(message["role"] == "tool" for message in result.messages))
        self.assertFalse(any("Top up" in str(message) for message in result.messages))

    async def test_page_failure_is_forwarded_as_a_model_observation_and_react_continues(self):
        tool = harness.Tool(harness.FETCH_SCHEMA, "url", AsyncMock(side_effect=_http_error(404)), provider="deepseek")
        ctx = model.Context(MagicMock(), model.ModelParams("model", 0.0, 1.0, 1024, 42), asyncio.Semaphore(1))
        calls = [{"id": "fetch_1", "name": "fetch", "arguments": {"url": "https://example.com"}}]
        with patch.object(harness, "call_model", new_callable=AsyncMock) as call:
            call.side_effect = [("", calls, "", None, "tool_calls"), ("answer", [], "", None, "stop")]
            result = await harness.react(
                ctx, "question", tools=[tool], system_prompt="system", max_steps=3, task_timeout=1
            )
        self.assertEqual(call.await_count, 2)
        self.assertIsNone(result.sys_error)
        observations = [message["content"] for message in result.messages if message["role"] == "tool"]
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].startswith("Fetch could not retrieve usable page content"))
        self.assertEqual(call.await_args_list[1].args[3][3]["content"], observations[0])

    async def _assert_developer_error(self, status, expected_action):
        tool = harness.Tool(
            harness.FETCH_SCHEMA, "url", AsyncMock(side_effect=_http_error(status)), provider="deepseek"
        )
        with self.assertLogs(harness._logger, level="ERROR") as logs, self.assertRaises(httpx.HTTPStatusError):
            await harness._execute_tool(tool, {"url": "https://example.com"})
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


def _parse_args(module=browsecomp, flags=(), env=None):
    with (
        patch.dict(os.environ, TEST_ENV if env is None else env, clear=True),
        patch.object(module, "load_dotenv"),
        patch("sys.argv", [module.__name__, *flags]),
    ):
        return module.parse_args()


if __name__ == "__main__":
    unittest.main()
