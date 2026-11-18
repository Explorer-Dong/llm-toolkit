"""Environment state, workspace confinement, and real Docker/browser lifecycles."""

import asyncio
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from eval.harness.env import BrowserEnv, DockerEnv, Env, NoEnv, WorkspaceEnv
from eval.harness.tool import env_server, open_toolbox

__all__ = []


class EnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_env_has_no_tools(self):
        async with NoEnv() as env, open_toolbox(env_server(env)) as tools:
            self.assertEqual(env.caps, frozenset())
            self.assertEqual(tools.schemas, [])
            with self.assertRaises(NotImplementedError):
                await env.exec("echo host")

    async def test_failed_setup_and_cancelled_scope_both_teardown(self):
        class Fails(Env):
            closed = False

            async def setup(self):
                raise RuntimeError("setup")

            async def teardown(self):
                self.closed = True

        env = Fails()
        with self.assertRaisesRegex(RuntimeError, "setup"):
            async with env:
                self.fail("setup failed")
        self.assertTrue(env.closed)
        active = asyncio.Event()
        workspace = WorkspaceEnv()

        async def work():
            async with workspace:
                active.set()
                await asyncio.Event().wait()

        future = asyncio.create_task(work())
        await active.wait()
        path = workspace.workdir
        future.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await future
        assert path is not None
        self.assertFalse(path.exists())


class WorkspaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_template_is_isolated_tools_round_trip_and_snapshot_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template"
            template.mkdir()
            (template / "source.txt").write_text("original")
            async with WorkspaceEnv(template) as first, WorkspaceEnv(template) as second:
                first_path = first.workdir
                async with open_toolbox(env_server(first)) as tools:
                    self.assertEqual(
                        {t["function"]["name"] for t in tools.schemas}, {"read_file", "write_file", "list_dir"}
                    )
                    self.assertFalse(
                        (await tools.call("write_file", {"path": "nested/out", "content": "changed"})).is_error
                    )
                    self.assertEqual((await tools.call("read_file", {"path": "nested/out"})).content, "changed")
                    self.assertIn("nested/", (await tools.call("list_dir", {})).content)
                self.assertEqual(await second.list_dir(), ["source.txt"])
                self.assertEqual((template / "source.txt").read_text(), "original")
                await first.snapshot(root / "evidence")
            assert first_path is not None
            self.assertFalse(first_path.exists())
            self.assertEqual((root / "evidence/nested/out").read_text(), "changed")

    async def test_parent_absolute_and_symlink_escapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "secret"
            outside.write_text("secret")
            async with WorkspaceEnv() as env:
                assert env.workdir is not None
                (env.workdir / "escape").symlink_to(outside)
                for path in ("../secret", str(outside), "escape"):
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        await env.read_file(path)
                    with self.assertRaises(ValueError):
                        await env.write_file(path, "overwritten")
                async with open_toolbox(env_server(env)) as tools:
                    result = await tools.call("read_file", {"path": "escape"})
                    self.assertTrue(result.is_error)
                    self.assertNotIn("secret", result.content)
                with self.assertRaises(ValueError):
                    await env.snapshot(Path(tmp) / "snapshot")
                self.assertEqual(outside.read_text(), "secret")

    async def test_operations_require_an_open_workspace(self):
        env = WorkspaceEnv()
        with self.assertRaises(RuntimeError):
            await env.read_file("file")
        async with env:
            with self.assertRaises(NotImplementedError):
                await env.exec("echo host")


@unittest.skipUnless(shutil.which("docker"), "Docker CLI is required")
class DockerTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_container_files_exec_snapshot_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            async with DockerEnv("python:3.13-slim-bookworm") as env:
                container = env.container
                await env.write_file("nested/quote' $value.txt", "unicode 中文\n$literal")
                self.assertEqual(await env.read_file("nested/quote' $value.txt"), "unicode 中文\n$literal")
                self.assertEqual(await env.list_dir("nested"), ["quote' $value.txt"])
                async with open_toolbox(env_server(env)) as tools:
                    self.assertIn("shell", {t["function"]["name"] for t in tools.schemas})
                    result = await tools.call("shell", {"cmd": "printf ok; printf err >&2; exit 7"})
                    self.assertTrue(result.is_error)
                    self.assertIn('"exit_code": 7', result.content)
                await env.snapshot(Path(tmp) / "snapshot")
            self.assertIsNone(env.container)
            process = await asyncio.create_subprocess_exec(
                "docker", "inspect", container, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
            self.assertNotEqual(await process.wait(), 0)
            self.assertEqual((Path(tmp) / "snapshot/nested/quote' $value.txt").read_text(), "unicode 中文\n$literal")

    async def test_failed_setup_removes_partial_container(self):
        env = DockerEnv("harness-invalid-image-that-does-not-exist:missing")
        with self.assertRaises(RuntimeError):
            async with env:
                self.fail("Missing image")
        self.assertIsNone(env.container)


@unittest.skipUnless(importlib.util.find_spec("playwright"), "Install eval[browser]")
class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_browser_mcp_actions_snapshot_and_cleanup(self):
        candidates = sorted((Path.home() / ".cache/ms-playwright").glob("chromium-*/chrome-linux*/chrome"))
        executable = str(candidates[-1]) if candidates else None
        env = BrowserEnv(executable_path=executable)
        with tempfile.TemporaryDirectory() as tmp:
            async with env, open_toolbox(env_server(env)) as tools:
                result = await tools.call(
                    "browser_goto",
                    {
                        "url": (
                            "data:text/html,<title>Harness</title><input id='name'>"
                            "<button onclick=\"document.querySelector('p').textContent="
                            "document.querySelector('input').value\">Go</button><p>ready</p>"
                        )
                    },
                )
                self.assertFalse(result.is_error)
                self.assertFalse((await tools.call("browser_fill", {"selector": "#name", "value": "Vincent"})).is_error)
                self.assertFalse((await tools.call("browser_click", {"selector": "button"})).is_error)
                self.assertIn("Vincent", (await tools.call("browser_text", {})).content)
                await env.snapshot(Path(tmp) / "browser")
            self.assertIsNone(env.page)
            self.assertTrue((Path(tmp) / "browser/page.png").is_file())
            self.assertIn("Vincent", (Path(tmp) / "browser/page.html").read_text())

    async def test_navigation_failure_is_a_recoverable_mcp_response(self):
        candidates = sorted((Path.home() / ".cache/ms-playwright").glob("chromium-*/chrome-linux*/chrome"))
        async with (
            BrowserEnv(executable_path=str(candidates[-1]) if candidates else None) as env,
            open_toolbox(env_server(env)) as tools,
        ):
            invalid = await tools.call("browser_goto", {"url": "http://127.0.0.1:1/"})
            self.assertTrue(invalid.is_error)
            self.assertIsNone(invalid.fatal)


if __name__ == "__main__":
    unittest.main()
