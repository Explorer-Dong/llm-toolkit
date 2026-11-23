"""WebShop Abstain rollouts through the shared harness."""

import argparse
import asyncio
import copy
import fcntl
import gc
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import warnings
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from functools import lru_cache
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
CATALOG_CACHE_FILE = "catalog.sqlite3"
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

EXECUTION_PRINCIPLES = """
Follow these execution principles:
1. Keep decisions consistent with your findings: if your reasoning or tool outputs reveal missing prerequisites, contradictions, errors, or unresolved risks, stop before taking consequential actions and report the issue directly. Never proceed with an action and merely note the problem afterward.
2. Never guess or compromise: satisfy every constraint in the request strictly. When requirements are underspecified, conflicting, or only partially matched, abstain and clarify; never assume defaults, drop constraints, or use an approximate substitute.
3. Treat exploration as reversible: using tools to gather information creates no obligation to finish the task. Stop and abstain at any turn once verification fails; proceed when all requirements are confirmed.
""".strip()

SYSTEM_PROMPT = ABSTAIN_MULTI_TURN_PROMPT.replace(
    "\nYour final response must be one JSON object without tool calls:",
    f"\n\n{EXECUTION_PRINCIPLES}\n\nYour final response must be one JSON object without tool calls:",
    1,
)


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


class _CatalogStore(Mapping[str, dict[str, Any]]):
    """Read-only SQLite product store with an in-memory LRU cache."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self._load_entry = lru_cache(maxsize=512)(self._fetch_entry)
        self.prices = _CatalogPrices(self)

    def _fetch_entry(self, asin: str) -> tuple[float, bytes] | None:
        row = self._conn.execute("SELECT price, data FROM products WHERE asin = ?", (asin,)).fetchone()
        return (float(row[0]), bytes(row[1])) if row is not None else None

    def __getitem__(self, asin: str) -> dict[str, Any]:
        entry = self._load_entry(asin)
        if entry is None:
            raise KeyError(asin)
        return json.loads(zlib.decompress(entry[1]))

    def __contains__(self, asin: object) -> bool:
        return isinstance(asin, str) and self._load_entry(asin) is not None

    def __iter__(self) -> Iterator[str]:
        return (row[0] for row in self._conn.execute("SELECT asin FROM products"))

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM products").fetchone()
        return int(row[0]) if row is not None else 0

    def price(self, asin: str) -> float:
        entry = self._load_entry(asin)
        if entry is None:
            raise KeyError(asin)
        return entry[0]

    def fingerprint(self) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
        return bytes(row[0]).decode("utf-8") if row is not None else None

    def load_goals(self) -> list[dict[str, Any]]:
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'goals'").fetchone()
        if row is None:
            raise RuntimeError(f"Missing goals metadata in {self.db_path}")
        return json.loads(zlib.decompress(row[0]))

    def close(self) -> None:
        self._load_entry.cache_clear()
        self._conn.close()


class _CatalogPrices(Mapping[str, float]):
    def __init__(self, store: _CatalogStore) -> None:
        self._store = store

    def __getitem__(self, asin: str) -> float:
        return self._store.price(asin)

    def __contains__(self, asin: object) -> bool:
        return asin in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)


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
            if isinstance(getattr(server, "product_item_dict", None), _CatalogStore):
                server.product_item_dict.close()
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
            from .data.text_environment.engine import engine, goal
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

        store = _open_catalog_store(products, attributes, instructions, engine, goal)

        def init_search_engine(num_products=None):
            indexes = {
                100: "search_index_100",
                1000: "search_index_1k",
                100000: "search_index_100k",
                None: "search_index",
            }
            return engine.LuceneSearcher(str(catalog_directory / indexes[num_products]))

        def load_products(filepath, num_products=None, human_goals=True):
            return [], store, store.prices, {}

        def get_goals(all_products, product_prices, human_goals=True):
            return store.load_goals()

        engine.init_search_engine = init_search_engine  # ty: ignore[invalid-assignment]
        web_agent_text_env.init_search_engine = init_search_engine  # ty: ignore[invalid-assignment]
        web_agent_text_env.load_products = load_products  # ty: ignore[invalid-assignment]
        web_agent_text_env.get_goals = get_goals  # ty: ignore[invalid-assignment]
        try:
            bootstrap = web_agent_text_env.WebAgentTextEnv(
                observation_mode="text",
                file_path=str(products),
                human_goals=True,
                session_prefix="bootstrap-",
            )
        except Exception:
            store.close()
            raise
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
                SYSTEM_PROMPT,
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


def _open_catalog_store(
    products: Path,
    attributes: Path,
    instructions: Path,
    engine: Any,
    goal: Any,
) -> _CatalogStore:
    db_path = products.parent / CATALOG_CACHE_FILE
    fingerprint = _catalog_fingerprint(products, attributes, instructions)
    store = _try_open_catalog_store(db_path, fingerprint)
    if store is not None:
        return store
    lock_path = products.parent / f".{CATALOG_CACHE_FILE}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        store = _try_open_catalog_store(db_path, fingerprint)
        if store is not None:
            return store
        _build_catalog_db(db_path, fingerprint, products, attributes, instructions, engine, goal)
    store = _try_open_catalog_store(db_path, fingerprint)
    if store is None:
        raise RuntimeError(f"Failed to initialize WebShop SQLite catalog cache at {db_path}")
    return store


def _try_open_catalog_store(db_path: Path, fingerprint: str) -> _CatalogStore | None:
    if not db_path.is_file():
        return None
    store: _CatalogStore | None = None
    try:
        store = _CatalogStore(db_path)
        if store.fingerprint() == fingerprint:
            return store
    except sqlite3.Error:
        pass
    if store is not None:
        store.close()
    return None


def _build_catalog_db(
    db_path: Path,
    fingerprint: str,
    products: Path,
    attributes: Path,
    instructions: Path,
    engine: Any,
    goal: Any,
) -> None:
    attributes_map = json.loads(attributes.read_text(encoding="utf-8"))
    human_map = json.loads(instructions.read_text(encoding="utf-8"))
    fd, tmp_name = tempfile.mkstemp(dir=db_path.parent, prefix=".catalog-", suffix=".sqlite3.tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    rng_state = random.getstate()
    random.seed(233)
    try:
        conn = sqlite3.connect(tmp_path)
        try:
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA journal_mode = OFF")
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
            conn.execute("CREATE TABLE products (asin TEXT PRIMARY KEY, price REAL NOT NULL, data BLOB NOT NULL)")
            asins: set[str] = set()
            goal_items: list[dict[str, Any]] = []
            goal_prices: dict[str, float] = {}
            batch: list[tuple[str, float, bytes]] = []
            for raw_item in _iter_json_array(products):
                formatted = _format_product(raw_item, asins, attributes_map, human_map, engine)
                if formatted is None:
                    continue
                asin, price, item = formatted
                if "instructions" in item:
                    goal_items.append(
                        {
                            "asin": asin,
                            "category": item["category"],
                            "query": item["query"],
                            "name": item["name"],
                            "product_category": item["product_category"],
                            "instructions": item["instructions"],
                        }
                    )
                    goal_prices[asin] = price
                payload = zlib.compress(json.dumps(item, ensure_ascii=False).encode("utf-8"), level=1)
                batch.append((asin, price, payload))
                if len(batch) >= 5000:
                    with conn:
                        conn.executemany("INSERT INTO products VALUES (?, ?, ?)", batch)
                    batch.clear()
            if batch:
                with conn:
                    conn.executemany("INSERT INTO products VALUES (?, ?, ?)", batch)
                batch.clear()
            del attributes_map, human_map, asins
            goals = goal.get_goals(goal_items, goal_prices, True)
            goals_blob = zlib.compress(json.dumps(goals, ensure_ascii=False).encode("utf-8"), level=1)
            with conn:
                conn.executemany(
                    "INSERT INTO meta VALUES (?, ?)",
                    [("fingerprint", fingerprint.encode("utf-8")), ("goals", goals_blob)],
                )
        finally:
            conn.close()
        os.replace(tmp_path, db_path)
    finally:
        random.setstate(rng_state)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _catalog_fingerprint(products: Path, attributes: Path, instructions: Path) -> str:
    files = {}
    for label, path in (("products", products), ("attributes", attributes), ("instructions", instructions)):
        stat = path.stat()
        files[label] = [stat.st_size, stat.st_mtime_ns]
    return json.dumps({"version": 1, "files": files}, sort_keys=True)


def _iter_json_array(path: Path, chunk_size: int = 1 << 22) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        pos = 0
        while True:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer = buffer[pos:] + chunk
                pos = 0
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n,[":
                    pos += 1
                if pos >= len(buffer) or buffer[pos] == "]":
                    break
                try:
                    item, pos = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                yield item
            if not chunk:
                break


def _format_product(
    item: dict[str, Any],
    asins: set[str],
    attributes: dict[str, Any],
    human_attributes: dict[str, Any],
    engine: Any,
) -> tuple[str, float, dict[str, Any]] | None:
    engine.clean_product_keys([item])
    asin = item["asin"]
    if asin == "nan" or len(asin) > 10 or asin in asins:
        return None
    asins.add(asin)
    item["Title"] = item["name"]
    item["Description"] = item["full_description"]
    item["Reviews"] = []
    item["Rating"] = "N.A."
    small = item["small_description"]
    item["BulletPoints"] = small if isinstance(small, list) else [small]
    raw_pricing = item.get("pricing")
    if not raw_pricing:
        pricing = [100.0]
        price_tag = "$100.0"
    else:
        pricing = [float(Decimal(re.sub(r"[^\d.]", "", price))) for price in raw_pricing.split("$")[1:]]
        if len(pricing) == 1:
            price_tag = f"${pricing[0]}"
        else:
            price_tag = f"${pricing[0]} to ${pricing[1]}"
            pricing = pricing[:2]
    item["pricing"] = pricing
    item["Price"] = price_tag
    options: dict[str, list[str]] = {}
    option_to_image: dict[str, Any] = {}
    customization_options = item["customization_options"]
    if customization_options:
        for option_name, option_contents in customization_options.items():
            if option_contents is None:
                continue
            option_key = option_name.lower()
            option_values = []
            for option_content in option_contents:
                option_value = option_content["value"].strip().replace("/", " | ").lower()
                option_values.append(option_value)
                option_to_image[option_value] = option_content.get("image", None)
            options[option_key] = option_values
    item["options"] = options
    item["option_to_image"] = option_to_image
    if asin in attributes and "attributes" in attributes[asin]:
        item["Attributes"] = attributes[asin]["attributes"]
    else:
        item["Attributes"] = ["DUMMY_ATTR"]
    if asin in human_attributes:
        item["instructions"] = human_attributes[asin]
    item["MainImage"] = item["images"][0]
    item["query"] = item["query"].lower().strip()
    price = 100.0 if not pricing else (pricing[0] if len(pricing) == 1 else random.uniform(*pricing[:2]))
    return asin, price, item


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
