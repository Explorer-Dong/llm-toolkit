"""Sample state and environment primitives; the caller owns their lifetime."""

import asyncio
import json
import posixpath
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

__all__ = ["BrowserEnv", "DockerEnv", "Env", "ExecResult", "NoEnv", "WorkspaceEnv"]


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


class Env:
    caps: frozenset[str] = frozenset()

    async def setup(self) -> None:
        return None

    async def teardown(self) -> None:
        return None

    async def __aenter__(self):
        try:
            await self.setup()
        except BaseException:
            await self.teardown()
            raise
        return self

    async def __aexit__(self, *args) -> None:
        await self.teardown()

    async def exec(self, cmd: str, timeout: float = 60) -> ExecResult:
        raise NotImplementedError("This environment has no exec capability")

    async def read_file(self, path: str) -> str:
        raise NotImplementedError("This environment has no files capability")

    async def write_file(self, path: str, content: str) -> None:
        raise NotImplementedError("This environment has no files capability")

    async def list_dir(self, path: str = ".") -> list[str]:
        raise NotImplementedError("This environment has no files capability")

    async def snapshot(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=False)


class NoEnv(Env):
    pass


class WorkspaceEnv(Env):
    caps = frozenset({"files"})

    def __init__(self, template: Path | None = None) -> None:
        self.template = template
        self.workdir: Path | None = None

    async def setup(self) -> None:
        if self.workdir is not None:
            raise RuntimeError("Workspace is already open")
        self.workdir = Path(tempfile.mkdtemp(prefix="harness-workspace-"))
        if self.template is not None:
            shutil.copytree(self.template, self.workdir, dirs_exist_ok=True, symlinks=True)

    async def teardown(self) -> None:
        if self.workdir is not None:
            shutil.rmtree(self.workdir)
            self.workdir = None

    async def read_file(self, path: str) -> str:
        return self._resolve_path(path).read_text(encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def list_dir(self, path: str = ".") -> list[str]:
        return sorted(entry.name + ("/" if entry.is_dir() else "") for entry in self._resolve_path(path).iterdir())

    async def snapshot(self, dest: Path) -> None:
        if self.workdir is None:
            raise RuntimeError("Workspace is not open")
        for source in self.workdir.rglob("*"):
            if source.is_symlink():
                self._resolve_path(str(source.relative_to(self.workdir)))
        shutil.copytree(self.workdir, dest)

    def _resolve_path(self, path: str) -> Path:
        if self.workdir is None:
            raise RuntimeError("Workspace is not open")
        target = (self.workdir / path).resolve()
        if not target.is_relative_to(self.workdir.resolve()):
            raise ValueError("File path must stay inside the workspace")
        return target


class DockerEnv(Env):
    caps = frozenset({"exec", "files"})

    def __init__(self, image: str, *, workdir: str = "/workspace", template: Path | None = None) -> None:
        if not PurePosixPath(workdir).is_absolute():
            raise ValueError("Docker workdir must be absolute")
        self.image = image
        self.workdir = posixpath.normpath(workdir)
        self.template = template
        self.container: str | None = None

    async def setup(self) -> None:
        if self.container is not None:
            raise RuntimeError("Container is already open")
        self.container = f"harness-{uuid.uuid4().hex}"
        result = await _run_docker(
            "run",
            "-d",
            "--pull=never",
            "--init",
            "--network",
            "none",
            "--name",
            self.container,
            "--workdir",
            self.workdir,
            self.image,
            "sleep",
            "infinity",
        )
        if result.exit_code:
            raise RuntimeError(f"Docker setup failed: {result.stderr.strip()}")
        if self.template is not None:
            result = await _run_docker("cp", f"{self.template.resolve()}/.", f"{self.container}:{self.workdir}")
            if result.exit_code:
                raise RuntimeError(f"Docker template copy failed: {result.stderr.strip()}")

    async def teardown(self) -> None:
        if self.container is not None:
            result = await _run_docker("rm", "-f", self.container)
            self.container = None
            if result.exit_code and "No such container" not in result.stderr:
                raise RuntimeError(f"Docker teardown failed: {result.stderr.strip()}")

    async def exec(self, cmd: str, timeout: float = 60) -> ExecResult:
        return await _run_docker(
            "exec", "-w", self.workdir, self._require_container(), "/bin/sh", "-lc", cmd, timeout=timeout
        )

    async def read_file(self, path: str) -> str:
        result = await _run_docker("exec", self._require_container(), "cat", "--", self._resolve_path(path))
        if result.exit_code:
            raise OSError(result.stderr.strip())
        return result.stdout

    async def write_file(self, path: str, content: str) -> None:
        target = self._resolve_path(path)
        parent = str(PurePosixPath(target).parent)
        result = await _run_docker("exec", self._require_container(), "mkdir", "-p", "--", parent)
        if result.exit_code:
            raise OSError(result.stderr.strip())
        result = await _run_docker(
            "exec",
            "-i",
            self._require_container(),
            "/bin/sh",
            "-c",
            'cat > "$1"',
            "write-file",
            target,
            stdin=content.encode(),
        )
        if result.exit_code:
            raise OSError(result.stderr.strip())

    async def list_dir(self, path: str = ".") -> list[str]:
        result = await _run_docker("exec", self._require_container(), "ls", "-1A", "--", self._resolve_path(path))
        if result.exit_code:
            raise OSError(result.stderr.strip())
        return result.stdout.splitlines()

    async def snapshot(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=False)
        result = await _run_docker("cp", f"{self._require_container()}:{self.workdir}/.", str(dest))
        if result.exit_code:
            raise OSError(result.stderr.strip())

    def _resolve_path(self, path: str) -> str:
        target = posixpath.normpath(posixpath.join(self.workdir, path))
        if not PurePosixPath(target).is_relative_to(self.workdir):
            raise ValueError("File path must stay inside the container workspace")
        return target

    def _require_container(self) -> str:
        if self.container is None:
            raise RuntimeError("Container is not open")
        return self.container


class BrowserEnv(Env):
    caps = frozenset({"browser"})

    def __init__(self, *, url: str | None = None, headless: bool = True, executable_path: str | None = None) -> None:
        self.url = url
        self.headless = headless
        self.executable_path = executable_path
        self.page: Any = None
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None

    async def setup(self) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless, executable_path=self.executable_path
        )
        self._context = await self._browser.new_context()
        self.page = await self._context.new_page()
        if self.url is not None:
            await self.page.goto(self.url)

    async def teardown(self) -> None:
        try:
            if self._context is not None:
                await self._context.close()
        finally:
            try:
                if self._browser is not None:
                    await self._browser.close()
            finally:
                if self._playwright is not None:
                    await self._playwright.stop()
                self.page = self._context = self._browser = self._playwright = None

    async def snapshot(self, dest: Path) -> None:
        if self.page is None:
            raise RuntimeError("Browser is not open")
        dest.mkdir(parents=True, exist_ok=False)
        await self.page.screenshot(path=str(dest / "page.png"), full_page=True)
        (dest / "page.html").write_text(await self.page.content(), encoding="utf-8")
        (dest / "state.json").write_text(json.dumps({"url": self.page.url, "title": await self.page.title()}))


async def _run_docker(*args: str, stdin: bytes | None = None, timeout: float = 60) -> ExecResult:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    return ExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode or 0)
