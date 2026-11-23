"""WebShop Abstain rollouts through the shared harness."""

import argparse
import asyncio
import copy
import gc
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.types import CallToolResult, TextContent, Tool

from ... import ROOT
from ...agent import Task, Trajectory, react
from ...harness import add_harness_args
from ...harness.env import Env
from ...harness.tool import env_server, open_toolbox
from ...harness.utils import save_trajectory
from ...model import AssistantTurn, add_model_args, build_model_params, make_client
from .._common.resume import finalize_rollout_dir, init_rollout_dir, parse_rollout_args
from .._common.storage import cleanup_cancelled_task, trajectory_path
from .._common.tasks import run_tasks

__all__ = ["WebShopCatalog", "WebShopEnv", "parse_args", "load_tasks", "rollout"]
DATA = Path(__file__).parent / "data"
DATA_FILE = "webshop_abstain.json"
_logger = logging.getLogger(__name__)
_DEPENDENCY_WARNINGS = (
    (r"Importing 'parser\.split_arg_string' is deprecated", DeprecationWarning, r"spacy\.cli\._util"),
    (r"Using slow pure-python SequenceMatcher", UserWarning, r"thefuzz\.fuzz"),
    (
        r"Call to deprecated method findAll\.",
        DeprecationWarning,
        r"eval\.benchmarks\.webshop_abstain\.data\.text_environment\.envs\.web_agent_text_env",
    ),
    (r"The 'text' argument to find\(\)-type methods is deprecated", DeprecationWarning, r"bs4\._deprecation"),
)

ABSTAIN_FIRST_TURN_PROMPT = """
You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions,
you have to respond based on the state and instruction.
You can use `search` if search is available.
You can use `click` to click one of the buttons in clickables, including product links, options, and Buy Now.
You can also choose to abstain: directly explain in a final text response why the instruction cannot be resolved with the available information and further exploration is unlikely to help. Do not call a tool when abstaining.

If the instruction is ambiguous, logically inconsistent, or requires information unavailable in the environment, you should explain the issue and abstain.

If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions when that call executes.
Remember that your keywords in search should be carefully designed.
You should first think about what to do, then choose an action accordingly.
You may call one or more tools in a response; calls are executed in order against the updated page state. When abstaining or when the environment reports that the episode has ended, provide a final text response without tool calls.
You can use whatever tool is available and do not need permission from the user.
Your final response must be one JSON object without tool calls:
{"prediction":"buy" or "abstain","reason":"A concise explanation of the purchase outcome or why you abstain."}
Use buy when you complete or attempt to complete the requested purchase. Use abstain when you deliberately
decline to purchase or request necessary clarification. Describe the actual outcome truthfully.
""".strip()


ABSTAIN_MULTI_TURN_PROMPT = ABSTAIN_FIRST_TURN_PROMPT


FIRST_TURN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search products in WebShop by entering free-form keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Search keywords to submit in the WebShop search bar.",
                    }
                },
                "required": ["keywords"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Interact with a clickable value on the current WebShop page: open a product or page, select an option, or execute Buy Now to purchase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "One of the available clickables on the current page.",
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    },
]


VALID_FIRST_TURN_TOOLS = {
    "search": "search",
    "click": "click",
}


GROUPS = {
    "T1": ("subjective", "主观要求"),
    "T2": ("underspecified_intent", "意图不明确"),
    "T3": ("false_premises", "错误前提"),
    "T4": ("missing_target", "目标商品缺失"),
}


class WebShopCatalog:
    """Share immutable product libraries; each episode owns its sessions."""

    def __init__(self, catalogs: Path, java_home: str | None = None) -> None:
        self.catalogs = catalogs.resolve()
        self.java_home = java_home
        self.servers: dict[str, Any] = {}
        self.condition = asyncio.Condition()
        self.active_sessions = 0
        self.goal_indices = json.loads((DATA / "goal_indices.json").read_text())
        indices = list(range(len(self.goal_indices)))
        random.Random(233).shuffle(indices)
        self.reset_indices = {pre: post for post, pre in enumerate(indices)}

    async def get_server(self, variant: str):
        async with self.condition:
            await self.condition.wait_for(lambda: variant in self.servers or self.active_sessions == 0)
            if variant not in self.servers:
                self.close()
                self.servers[variant] = await asyncio.to_thread(self._bootstrap, variant)
                gc.collect()
                gc.freeze()
            self.active_sessions += 1
            return self.servers[variant]

    async def release_server(self) -> None:
        async with self.condition:
            self.active_sessions -= 1
            self.condition.notify_all()

    def reset_index(self, task: Task) -> int:
        record = task.raw
        if record["env_variant"] == "pruned_missing_target_251":
            return 0
        base = record.get("base_goal_index")
        return self.reset_indices[int(base) if base is not None else self.goal_indices[str(record["source_id"])]]

    def close(self) -> None:
        if not self.servers:
            return
        for server in self.servers.values():
            server.user_sessions.clear()
        self.servers.clear()
        gc.unfreeze()
        gc.collect()

    def _bootstrap(self, variant: str):
        java = self.java_home or os.getenv("JAVA_HOME")
        if not java:
            executable = shutil.which("java")
            if executable:
                java = str(Path(executable).resolve().parents[1])
        if not java or not (Path(java) / "bin/java").is_file():
            raise ValueError("WebShop requires Java 21; set JAVA_HOME or --java-home")
        os.environ["JAVA_HOME"] = java
        # Pyserini constructs an unused dense-encoder client during its Lucene import.
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        with _filter_dependency_startup_output():
            # Gym installs a general deprecation filter during import.
            import gym  # noqa: F401

            for message, category, module in _DEPENDENCY_WARNINGS:
                warnings.filterwarnings("ignore", message=message, category=category, module=module)
            from .data.text_environment import utils
            from .data.text_environment.engine import engine
            from .data.text_environment.envs import web_agent_text_env
        from jnius import autoclass

        lucene_logger = autoclass("java.util.logging.Logger").getLogger(
            "org.apache.lucene.store.MemorySegmentIndexInputProvider"
        )
        lucene_logger.setLevel(autoclass("java.util.logging.Level").WARNING)

        products = self.products_path(variant)
        attributes = self.catalogs / "product_attributes.json"
        instructions = self.catalogs / "human_instructions.json"
        catalog_directory = products.parent
        for module in (utils, engine):
            module.DEFAULT_FILE_PATH = str(products)
            module.DEFAULT_ATTR_PATH = str(attributes)
            module.HUMAN_ATTR_PATH = str(instructions)
        for path in (products, attributes, instructions, catalog_directory / "search_index"):
            if not path.exists():
                raise FileNotFoundError(f"Missing WebShop catalog file: {path}; see data/README.md")

        def init_search_engine(num_products=None):
            indexes = {
                100: "search_index_100",
                1000: "search_index_1k",
                100000: "search_index_100k",
                None: "search_index",
            }
            return engine.LuceneSearcher(str(catalog_directory / indexes[num_products]))

        engine.init_search_engine = init_search_engine  # ty: ignore[invalid-assignment]
        web_agent_text_env.init_search_engine = init_search_engine  # ty: ignore[invalid-assignment]
        bootstrap = web_agent_text_env.WebAgentTextEnv(
            observation_mode="text",
            file_path=str(products),
            human_goals=True,
            session_prefix="bootstrap-",
        )
        server = bootstrap.server
        server.user_sessions.pop(bootstrap.session, None)
        bootstrap.close()
        return server

    def products_path(self, variant: str) -> Path:
        if variant == "full":
            return self.catalogs / "full/products.json"
        if variant == "pruned_missing_target_251":
            return self.catalogs / "missing_target/products.json"
        raise ValueError(f"Unsupported WebShop variant: {variant}")


class WebShopEnv(Env):
    caps = frozenset({"tools"})
    # Invalid actions use the original WebShop observations and host receipts.
    validate_tool_arguments = False

    def __init__(self, task: Task, catalog: WebShopCatalog) -> None:
        self.task = task
        self.catalog = catalog
        self.environment: Any = None
        self.terminated_by: str | None = None
        self.reward = 0.0
        self.turn_index = self.max_turns = self.remaining_calls = 0
        self.call_name = ""
        self.borrowed_catalog = False

    @property
    def terminal(self) -> bool:
        return self.terminated_by is not None

    async def setup(self) -> None:
        record = self.task.raw
        server = copy.copy(await self.catalog.get_server(record["env_variant"]))
        self.borrowed_catalog = True
        server.user_sessions = {}
        from .data.text_environment.envs.web_agent_text_env import WebAgentTextEnv

        self.environment = WebAgentTextEnv(
            observation_mode="text",
            file_path=str(self.catalog.products_path(record["env_variant"])),
            server=server,
            human_goals=True,
            session_prefix=f"instruction-set-{self.task.id}-",
        )
        server.user_sessions.pop(self.environment.session, None)
        self.environment.reset(self.catalog.reset_index(self.task))
        state = server.user_sessions[self.environment.session]
        state["goal"] = dict(state["goal"])
        state["goal"]["instruction_text"] = record["instruction"]
        self.environment.browser.get(
            f"{self.environment.base_url}/{self.environment.session}", session_id=self.environment.session
        )
        self.environment.instruction_text = record["instruction"]
        self.environment.prev_obs = [self.environment.observation]
        self.environment.prev_actions = []

    async def teardown(self) -> None:
        try:
            if self.environment is not None:
                self.environment.server.user_sessions.pop(self.environment.session, None)
                self.environment.close()
                self.environment = None
        finally:
            if self.borrowed_catalog:
                self.borrowed_catalog = False
                await self.catalog.release_server()

    async def prepare_task(self, task: Task) -> Task:
        self.max_turns = task.max_turns
        message = _build_initial_message(self.environment.observation, self.environment.get_available_actions())
        return replace(task, prompt=message["content"])

    def observe_turn(self, turn: AssistantTurn, index: int) -> None:
        self.turn_index = index
        self.remaining_calls = len(turn.tool_calls)

    def prepare_tool_call(self, call: dict) -> dict:
        self.call_name = call["function"]["name"]
        classification = _classify_tool_call([call], self.environment.get_available_actions())
        if classification["invalid"]:
            return {**call, "function": {"name": "search", "arguments": json.dumps({"_invalid": classification})}}
        return call

    async def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                inputSchema=tool["function"]["parameters"],
            )
            for tool in FIRST_TURN_TOOLS
        ]

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        self.remaining_calls -= 1
        reward, done, executed = 0, False, False
        if self.terminal:
            label = f"skipped_tool_call[{self.call_name or name}]: episode already ended"
        else:
            classification = arguments.get("_invalid") or _classify_tool_call(
                [{"function": {"name": name, "arguments": arguments}}],
                self.environment.get_available_actions(),
            )
            action = _normalize_action(classification)
            if action is not None:
                _, reward, done, _ = self.environment.step(action)
                executed = True
                self.reward = reward
            if done and action is not None:
                self.terminated_by = "buy_now" if action.lower() == "click[buy now]" else "env_done"
            label = (
                action
                if executed and action is not None
                else f"invalid_tool_call: {classification.get('invalid_reason') or classification.get('invalid_arguments_reason')}"
            )
        if self.turn_index == self.max_turns and self.remaining_calls == 0 and not self.terminal:
            self.terminated_by = "round_limit"
        content = _build_tool_output(
            action=label,
            observation=self.environment.observation,
            available_actions=self.environment.get_available_actions(),
        )
        content += (
            f"\n\nExecuted: {executed}. Reward: {reward if executed else 0}. Done: {done if executed else False}."
        )
        if self.terminal:
            content += f"\n\nThe episode has ended ({self.terminated_by}). Provide your final response based on the observed outcome. Do not call any more tools."
        return CallToolResult(content=[TextContent(type="text", text=content)])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="WebShop Abstain")
    add_model_args(parser, max_tokens=4096, timeout=600)
    add_harness_args(parser, max_turns=30, concurrency=4)
    parser.set_defaults(
        benchmark="webshop_abstain",
        top_p=None,
        seed=None,
        stream=None,
        include_reasoning=True,
        max_retries=0,
    )
    parser.add_argument("--request-attempts", type=int, default=3, help="attempts per model request")
    parser.add_argument("--retry-delay", type=float, default=1, help="seconds between failed model requests")
    parser.add_argument("--dataset-file", default=str(DATA / DATA_FILE))
    parser.add_argument("--dataset-indices", type=int, nargs="+")
    parser.add_argument("--catalogs-dir", type=Path, default=DATA / "catalogs")
    parser.add_argument("--java-home")
    args = parse_rollout_args(parser, argv)
    if args.request_attempts < 1 or args.retry_delay < 0:
        parser.error("request-attempts must be positive and retry-delay must be nonnegative")
    if args.model.rsplit("/", 1)[-1].lower() == "qwen3.8-27b" and args.reasoning_effort not in {
        "low",
        "medium",
        "xhigh",
    }:
        parser.error("Qwen3.8-27B reasoning effort must be low, medium or xhigh")
    return args


def load_tasks(args: argparse.Namespace) -> list[Task]:
    records = json.loads(Path(args.dataset_file).read_text(encoding="utf-8"))
    seen = set()
    for record in records:
        index = record["dataset_index"]
        if type(index) is not int or index in seen:
            raise ValueError(f"Invalid or duplicate dataset index: {index}")
        seen.add(index)
        if type(record.get("should_abstain_expected")) is not bool:
            raise ValueError("should_abstain_expected must be a boolean")
        if record["env_variant"] not in {"full", "pruned_missing_target_251"}:
            raise ValueError(f"Unsupported WebShop variant: {record['env_variant']}")
        if record["should_abstain_expected"]:
            _abstention_type(record)
    selected = getattr(args, "dataset_indices", None)
    if selected is not None:
        if len(set(selected)) != len(selected) or set(selected) - seen:
            raise ValueError("Dataset indices must be unique and present in the dataset")
        records = [record for record in records if record["dataset_index"] in selected]
    records = records[: args.limit]
    if not records:
        raise ValueError("No WebShop Abstain tasks selected")
    max_turns = getattr(args, "max_turns", 30)
    timeout = getattr(args, "task_timeout", 3600.0)
    tasks: list[Task] = []
    for record in records:
        meta = {
            key: record.get(key) for key in ("dataset_index", "source_id", "variant_type", "category", "env_variant")
        }
        tasks.append(
            Task(
                str(record["dataset_index"]),
                ABSTAIN_MULTI_TURN_PROMPT,
                record["instruction"],
                max_turns,
                timeout,
                meta=meta,
                gold="correct_abstain" if record["should_abstain_expected"] else "correct_buy",
                raw=record,
                final_prompt=None,
                strict_response=True,
            )
        )
    return tasks


async def rollout(args: argparse.Namespace, tasks: list[Task]) -> list[Trajectory]:
    started = time.perf_counter()
    run_dir, config, saved_config, completed, pending = init_rollout_dir(args, tasks)
    new_traces: list[Trajectory] = []
    catalog = WebShopCatalog(args.catalogs_dir, args.java_home)
    try:
        if pending:
            model_params = {
                **build_model_params(args),
                "request_attempts": getattr(args, "request_attempts", 1),
                "retry_delay": getattr(args, "retry_delay", 1.0),
            }
            async with make_client(
                args.base_url, args.api_key, timeout=args.timeout, max_retries=getattr(args, "max_retries", 3)
            ) as client:

                async def run_task(task: Task) -> Trajectory:
                    task_started = time.perf_counter()
                    trajectory_file = trajectory_path(run_dir, task.id)
                    trace = Trajectory(
                        task.id,
                        messages=[{"role": "system", "content": task.system}, {"role": "user", "content": task.prompt}],
                        meta=dict(task.meta),
                    )
                    phase = "setup"
                    try:
                        env = WebShopEnv(task, catalog)
                        async with env:
                            task = await env.prepare_task(task)
                            phase = "tools"
                            async with open_toolbox(env_server(env)) as tools:
                                phase = "episode"
                                trace = await react(task, client, tools, model_params, env=env)
                                phase = "tools_cleanup"
                            phase = "teardown"
                    except asyncio.CancelledError:
                        cleanup_cancelled_task(run_dir, task.id)
                        raise
                    except Exception as exc:
                        _logger.exception("Runtime %s failed during %s", task.id, phase)
                        trace.episode_status = "environment_error"
                        detail = f"{phase}: {type(exc).__name__}: {exc}"
                        trace.error = f"{trace.error}; {detail}" if trace.error else detail
                    trace.runtime_latency_sec = round(time.perf_counter() - task_started, 6)
                    save_trajectory(trace, trajectory_file)
                    return trace

                new_traces = await run_tasks(pending, run_task, concurrency=args.concurrency, desc=args.benchmark)
    finally:
        try:
            catalog.close()
        finally:
            by_id = dict(completed)
            by_id.update({trace.task_id: trace for trace in new_traces})
            traces = [by_id[task.id] for task in tasks] if len(by_id) == len(tasks) else list(by_id.values())
            finalize_rollout_dir(
                run_dir,
                config,
                saved_config,
                started=started,
                resumed_count=len(completed),
                benchmark=args.benchmark,
                traces=traces,
            )
    return traces


@contextmanager
def _filter_dependency_startup_output():
    from gym_notices.notices import notices

    # JVM startup writes directly to stderr's file descriptor.
    with tempfile.TemporaryFile() as captured:
        sys.stderr.flush()
        original_stderr = os.dup(2)
        try:
            os.dup2(captured.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(original_stderr, 2)
            os.close(original_stderr)
            captured.seek(0)
            startup_output = captured.read().decode("utf-8", errors="replace")
            startup_output = startup_output.replace(notices.get("0.24.0", ""), "")
            for line in startup_output.splitlines(keepends=True):
                if line.strip() and line.strip() != "WARNING: Using incubator modules: jdk.incubator.vector":
                    sys.stderr.write(line)
            sys.stderr.flush()


def _build_initial_message(observation: str, available_actions: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"The initial observation:\n{observation}\n\nAvailable Actions:\n{available_actions}",
    }


def _build_tool_output(
    *,
    action: str,
    observation: str,
    available_actions: dict[str, Any],
) -> str:
    return f"Action: {action}\n\nObservation:\n{observation}\n\nAvailable Actions:\n{available_actions}"


def _validate_search_arguments(arguments: Any, available_actions: dict[str, Any]) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    if set(arguments.keys()) != {"keywords"}:
        return "unexpected_argument_keys"
    keywords = arguments.get("keywords")
    if not available_actions.get("has_search_bar", False):
        return "search_unavailable"
    if not isinstance(keywords, str) or not keywords.strip():
        return "invalid_keywords"
    return None


def _validate_click_arguments(arguments: Any, available_actions: dict[str, Any]) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    if set(arguments.keys()) != {"value"}:
        return "unexpected_argument_keys"
    value = arguments.get("value")
    if not isinstance(value, str):
        return "invalid_value"
    normalized_clickables = {str(item).lower() for item in available_actions.get("clickables", [])}
    if value.lower() not in normalized_clickables:
        return "value_not_available"
    return None


def _classify_tool_call(tool_calls: list[dict[str, Any]], available_actions: dict[str, Any]) -> dict[str, Any]:
    result = {
        "predicted_class": "invalid",
        "predicted_tool_name": None,
        "predicted_arguments": None,
        "invalid": True,
        "invalid_reason": None,
        "invalid_arguments": False,
        "invalid_arguments_reason": None,
    }

    if not tool_calls:
        result["invalid_reason"] = "no_tool_call"
        return result

    tool_call = tool_calls[0]
    function = tool_call.get("function") or {}
    function_name = str(function.get("name") or "").removeprefix("webshop__")
    raw_arguments = function.get("arguments", "{}")
    result["predicted_tool_name"] = function_name

    if function_name not in VALID_FIRST_TURN_TOOLS:
        result["invalid_reason"] = "unknown_tool"
        return result

    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except Exception:
        result["invalid_reason"] = "arguments_parse_error"
        return result

    result["predicted_class"] = VALID_FIRST_TURN_TOOLS[function_name]
    result["predicted_arguments"] = arguments
    result["invalid"] = False
    result["invalid_reason"] = None

    if function_name == "search":
        invalid_argument_reason = _validate_search_arguments(arguments, available_actions)
    else:
        invalid_argument_reason = _validate_click_arguments(arguments, available_actions)

    if invalid_argument_reason is not None:
        result["invalid"] = True
        result["invalid_arguments"] = True
        result["invalid_arguments_reason"] = invalid_argument_reason

    return result


def _normalize_action(classification: dict[str, Any]) -> str | None:
    if classification.get("invalid"):
        return None

    predicted_class = classification.get("predicted_class")
    arguments = classification.get("predicted_arguments")
    if predicted_class == "search":
        if isinstance(arguments, dict):
            keywords = arguments.get("keywords")
            if isinstance(keywords, str):
                return f"search[{keywords}]"
        return None
    if predicted_class == "click":
        if isinstance(arguments, dict):
            value = arguments.get("value")
            if isinstance(value, str):
                return f"click[{value}]"
        return None
    return None


def _abstention_type(record: dict) -> str:
    variant = record["variant_type"]
    category = record.get("category") if variant == "rewrite" else variant
    if category not in {name for name, _ in GROUPS.values()}:
        raise ValueError(f"Unknown abstention type: {category}")
    return category


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(rollout(args, load_tasks(args)))
