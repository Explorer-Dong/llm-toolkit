# Harness

A Python package for ReAct agents, MCP runtimes, and trajectory recording. Requires Python 3.12+.

## Install

Add the local package to your project:

```bash
uv add /path/to/llm-toolkit/harness
```

Configure an OpenAI-compatible endpoint:

```bash
export MODEL_NAME="your-model"
export BASE_URL="https://your-endpoint/v1"
export API_KEY="your-api-key"
```

## Basic usage

```python
import asyncio
import os
from pathlib import Path

from harness import ModelParams, Task, make_context, open_toolbox, run_episode


async def main():
    params = ModelParams(
        model=os.environ["MODEL_NAME"],
        temperature=0,
        top_p=1,
        max_tokens=4096,
        seed=42,
    )
    ctx = make_context(os.environ["BASE_URL"], os.environ["API_KEY"], params)
    task = Task(
        id="example",
        system="Answer carefully.",
        prompt="What is 6 * 7?",
        max_turns=1,
        timeout=120,
    )
    try:
        async with open_toolbox() as runtime:
            trace = await run_episode(task, ctx, runtime, out=Path("trajectory.json"))
        print(trace.final_answer)
    finally:
        await ctx.client.close()


asyncio.run(main())
```

`run_episode()` returns a `Trajectory`; `out` also saves it as JSON.

| Parameter | Meaning |
| --- | --- |
| `Task.max_turns` | Maximum model calls; `1` for QA, larger values for multi-turn ReAct |
| `Task.timeout` | Episode timeout in seconds; `None` or `0` disables it |
| `ModelParams.enable_thinking` | Thinking mode; defaults to `True` |
| `ModelParams.reasoning_effort` | Reasoning effort; defaults to `"medium"` |
| `make_context(..., concurrency=N)` | Maximum concurrent model requests sharing the context |

## Runtime examples

Replace the `open_toolbox()` block inside `main()` with either example below. Keep the client cleanup.

### Search and fetch

Set `DS_TOOL_URL` and `DS_TOOL_KEY`. Both common operations use DeepSeek by default.

```python
from harness import common_server

task = Task(
    id="search",
    system="Use search and fetch to find the answer. Include the source URL.",
    prompt="When was Python 3.12.0 released?",
    max_turns=8,
    timeout=120,
)
async with open_toolbox(common_server()) as runtime:
    trace = await run_episode(task, ctx, runtime, out=Path("search.json"))
print(trace.final_answer)
```

### File workspace

The caller opens and closes the runtime, and collects files before closing it.

```python
from harness import WorkspaceEnv, env_server

task = Task(
    id="files",
    system="Use the runtime to create the requested file.",
    prompt="Write answer.txt containing 42, then confirm completion.",
    max_turns=4,
    timeout=120,
)
async with WorkspaceEnv() as env:
    async with open_toolbox(env_server(env)) as runtime:
        trace = await run_episode(task, ctx, runtime, out=Path("workspace.json"))
    print(await env.read_file("answer.txt"))
```

Other runtimes use the same `open_toolbox(...)` interface:

| Runtime | API |
| --- | --- |
| Docker shell and files | `DockerEnv(image, workdir="/workspace")` with `env_server(env)` |
| Browser | `BrowserEnv()` with `env_server(env)`; install the `browser` extra and Chromium |
| External MCP server | `StdioSpec(cmd, args=[...])` or `HttpSpec(url)` |
| Shared MCP connections | `runtime.scoped(...)` adds connections for one episode |

## Read a trajectory

```python
from pathlib import Path

from harness import load_trajectory

trace = load_trajectory(Path("trajectory.json"))
print(trace.final_answer)
print(trace.usage, trace.latency_sec)
print(trace.stop_reason, trace.error)
```

Trajectories include messages, reasoning, runtime calls and results, token usage, timing, and execution errors.
