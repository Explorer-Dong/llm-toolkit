"""Run WebWalkerQA benchmark with the three methods from the paper: WebWalker, ReAct and Reflexion."""

import argparse
import asyncio
import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json5
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from prompts import FORCED_ANSWER, REFLECTION_GENERATE, SYSTEM_REACT, SYSTEM_REFLEXION
from qwen_agent.tools.base import BaseTool
from tqdm import tqdm
from utils import clean_markdown, get_content_between_a_b, process_url
from webwalker import TOOL_DESC, WebWalker, normalize_react_markers

# Endpoint config lives in .env at the bench root so the agent and judge models
# can use different endpoints; real environment variables take precedence.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

AGENT_API_KEY = os.getenv("AGENT_API_KEY")
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL")

# Chromium does NOT inherit http_proxy/https_proxy env vars, so without this
# it connects directly and foreign sites get intermittently reset
# (net::ERR_CONNECTION_CLOSED). Point the browser at the local proxy.
CRAWL_PROXY = os.getenv("CRAWL_PROXY") or os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")

_write_lock = threading.Lock()

TRAJECTORY_MAX_CHARS = 8000  # cap fed into reflection / forced-answer prompts


async def _fetch_async(url: str) -> Tuple[str, str]:
    browser_config = BrowserConfig(proxy=CRAWL_PROXY) if CRAWL_PROXY else BrowserConfig()
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url, config=CrawlerRunConfig(page_timeout=60000))
        return result.html, clean_markdown(result.markdown)


# crawl4ai's global cache holds an asyncio.Semaphore bound to the event loop
# that first used it; creating a fresh loop per fetch (asyncio.run) eventually
# triggers "Semaphore is bound to a different event loop". Route every fetch
# through one persistent loop on a daemon thread instead.
_fetch_loop = asyncio.new_event_loop()
threading.Thread(target=_fetch_loop.run_forever, daemon=True, name="crawl-loop").start()


def fetch_page(url: str, attempts: int = 4) -> Tuple[str, str]:
    """Fetch a page as (html, cleaned markdown); retry on site resets."""
    last_error = None
    for i in range(attempts):
        try:
            future = asyncio.run_coroutine_threadsafe(_fetch_async(url), _fetch_loop)
            html, markdown = future.result(timeout=180)
            if html:
                return html, markdown
            last_error = "empty response"
        except Exception as e:  # noqa: BLE001 - retry on any crawl failure
            last_error = e
        time.sleep(min(2**i, 8))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


class VisitPageTool(BaseTool):
    """Headless visit_page tool: fetch a page, return its markdown content
    plus the clickable buttons (link texts) found on it.

    It keeps its state (root url + button->url map) per instance, so
    concurrent queries cannot interfere with each other.
    """

    name = "visit_page"
    description = (
        "A tool analyzes the content of a webpage and extracts buttons associated with sublinks. "
        "Simply input the button which you want to explore, and the tool will return both the "
        "markdown-formatted content of the corresponding page of button and a list of new clickable "
        "buttons found on the new page."
    )
    parameters = [
        {
            "name": "button",
            "type": "string",
            "description": "the button you want to click",
            "required": True,
        }
    ]

    def __init__(self, root_url: str):
        super().__init__()
        self.root_url = root_url
        self.button_map: Dict[str, str] = {}
        self.actions = 0

    # -- link extraction ----------------------------------------------------
    def _extract_links(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        links = []

        def add(url, text):
            if not url or not text:
                return
            if "javascript" in url or url.endswith((".jpg", ".png", ".gif", ".jpeg", ".pdf")):
                return
            url = process_url(self.root_url, url)
            if url.startswith(self.root_url):
                links.append({"url": url, "text": text})

        for a_tag in soup.find_all("a", href=True):
            add(a_tag["href"], "".join(a_tag.stripped_strings))
        for a_tag in soup.find_all("a", onclick=True):
            match = re.search(r"window\.location\.href='([^']*)'", a_tag["onclick"])
            if match:
                add(match.group(1), "".join(a_tag.stripped_strings))
        for a_tag in soup.find_all("a", attrs={"data-url": True}):
            add(a_tag["data-url"], "".join(a_tag.stripped_strings))
        for a_tag in soup.find_all("a", class_="herf-mask"):
            url = a_tag.get("href")
            text = a_tag.get("title") or "".join(a_tag.stripped_strings)
            add(url, text)
        for button in soup.find_all("button", onclick=True):
            match = re.search(r"window\.location\.href='([^']*)'", button["onclick"])
            if match:
                url = match.group(1)
                text = button.get("title") or button.get("aria-label") or "".join(button.stripped_strings)
                add(url, text)

        unique_links = {f"{item['url']}_{item['text']}": item for item in links}
        for temp in unique_links.values():
            self.button_map[temp["text"]] = temp["url"]
        info = ""
        for i in unique_links.values():
            info += "<button>" + i["text"] + "<button>" + "\n"
        return info

    def observe_root(self) -> str:
        """Fetch the root page and return the initial observation text."""
        html, markdown = fetch_page(self.root_url)
        buttons = self._extract_links(html)
        response = "website information:\n\n" + markdown + "\n\nclickable button:\n\n" + buttons
        response += "\n\nEach button is wrapped in a <button> tag"
        return response

    # -- repair truncated / malformed JSON params ---------------------------
    def _parse_params(self, params: str) -> Optional[dict]:
        params = params.strip()
        if not params.endswith("}"):
            if "}" in params:
                params = "{" + get_content_between_a_b("{", "}", params) + "}"
            elif not params.endswith('"'):
                params += '"}'
            else:
                params += "}"
        params = "{" + get_content_between_a_b("{", "}", params) + "}"
        try:
            parsed = json5.loads(params)
            if "button" not in parsed and "action_input" in parsed:
                # some outputs use {"action": ..., "action_input": <button>}
                parsed["button"] = parsed["action_input"]
            return parsed if "button" in parsed else None
        except Exception:  # noqa: BLE001
            return None

    def call(self, params: str, **kwargs) -> str:
        parsed = self._parse_params(params)
        if parsed is None:
            return "Your input is invalid, plase output the action input correctly!"
        button = parsed["button"].replace("<button>", "")
        if button not in self.button_map:
            return "The button can not be clicked, please retry a new botton!"
        self.actions += 1
        html, markdown = fetch_page(self.button_map[button])
        response_buttons = self._extract_links(html)
        if markdown:
            response = "The web information is:\n\n" + markdown + "\n\n"
        else:
            response = "The information of the current page is not accessible\n\n"
        response += "Clickable buttons are wrapped in <button> tag" + response_buttons
        return response


def build_start_prompt(query: str, root_url: str, observation: str) -> str:
    return f"query:\n{query} \nofficial website:\n{root_url}\nObservation:" + observation + "\n\n"


def detect_tool(text: str, tool_name: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str], str]:
    """Port of agent.WebWalker._detect_tool (with the same format-drift fixes)."""
    text = normalize_react_markers(text)
    special_func_token = "\nAction:"
    special_args_token = "\nAction Input:"
    special_obs_token = "\nObservation:"
    func_name, func_args = None, None
    i = text.rfind(special_func_token)
    j = text.rfind(special_args_token)
    k = text.rfind(special_obs_token)
    if 0 <= i < j:
        if k < j:
            text = text.rstrip() + special_obs_token
        k = text.rfind(special_obs_token)
        func_name = text[i + len(special_func_token) : j].strip()
        func_args = text[j + len(special_args_token) : k].strip()
        text = text[:i]
    elif i >= 0 and j < 0 and tool_name:
        # "Action:" followed directly by a JSON body, no "Action Input:" line
        match = re.search(r"\{.*\}", text[i + len(special_func_token) :], re.S)
        if match:
            func_name = tool_name
            func_args = match.group(0)
            text = text[:i]
    return (func_name is not None), func_name, func_args, text


def tool_desc_for(tool: BaseTool) -> str:
    function = tool.function
    return TOOL_DESC.format(
        name_for_human=function.get("name_for_human", function.get("name")),
        name_for_model=function.get("name_for_model", function.get("name")),
        description_for_model=function["description"],
        parameters=json.dumps(function["parameters"], ensure_ascii=False),
        args_format=function.get("args_format", ""),
    ).rstrip()


def chat_once(client: OpenAI, model: str, prompt: str, system: Optional[str] = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stop=["Observation:", "Observation:\n"],
    )
    return response.choices[0].message.content


def forced_answer(client: OpenAI, model: str, query: str, information: str) -> str:
    information = re.sub(r"Thought:\s*$", "", information[-TRAJECTORY_MAX_CHARS:].rstrip())
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You must answer the question directly. Do not output thoughts, actions or JSON. "
                "Reply with the answer text only.",
            },
            {"role": "user", "content": FORCED_ANSWER.format(query=query, information=information)},
        ],
    )
    return response.choices[0].message.content


def run_react_episode(
    client: OpenAI,
    model: str,
    tool: VisitPageTool,
    query: str,
    start_query_block: str,
    max_rounds: int,
    reflections: Optional[List[str]] = None,
) -> Tuple[Optional[str], str]:
    """One ReAct episode. Returns (final_answer or None, trajectory_prompt)."""
    if reflections:
        system = SYSTEM_REFLEXION.format(
            tool_descs=tool_desc_for(tool),
            tool_names=tool.name,
            reflections="\nReflections on previous failed attempts:\n"
            + "\n".join(f"{i}. {r}" for i, r in enumerate(reflections, 1))
            + "\n",
            query=start_query_block,
        )
    else:
        system = SYSTEM_REACT.format(
            tool_descs=tool_desc_for(tool),
            tool_names=tool.name,
            query=start_query_block,
        )
    prompt = system
    for _ in range(max_rounds):
        output = chat_once(client, model, prompt)
        if "Final Answer:" in normalize_react_markers(output):
            return normalize_react_markers(output).split("Final Answer:", 1)[1].strip(), prompt + output
        has_action, action, action_input, _ = detect_tool("\n" + output, tool_name=tool.name)
        if not has_action:
            prompt += output + "\nThought: "
            continue
        observation = tool.call(action_input)
        prompt += output + f"\nObservation: {observation}\nThought: "
    return None, prompt


def run_react(
    client: OpenAI, model: str, tool: VisitPageTool, query: str, root_url: str, max_rounds: int
) -> Tuple[str, dict]:
    observation = tool.observe_root()
    start_query_block = f"query:\n{query} \nofficial website:\n{root_url}"
    answer, trajectory = run_react_episode(
        client,
        model,
        tool,
        query,
        start_query_block + "\nObservation:" + observation + "\n\n",
        max_rounds,
    )
    stats = {"rounds": tool.actions}
    if answer is None:
        answer = forced_answer(client, model, query, trajectory)
        stats["forced"] = True
    return answer, stats


def run_reflexion(
    client: OpenAI,
    model: str,
    tool: VisitPageTool,
    query: str,
    root_url: str,
    max_rounds: int,
    attempts: int,
) -> Tuple[str, dict]:
    observation = tool.observe_root()
    start_query_block = f"query:\n{query} \nofficial website:\n{root_url}"
    start_prompt = start_query_block + "\nObservation:" + observation + "\n\n"
    reflections: List[str] = []
    stats = {"rounds": 0, "attempts_used": 0, "forced": False}
    trajectory = ""
    for attempt in range(1, attempts + 1):
        stats["attempts_used"] = attempt
        answer, trajectory = run_react_episode(
            client, model, tool, query, start_prompt, max_rounds, reflections or None
        )
        stats["rounds"] = tool.actions
        if answer is not None:
            return answer, stats
        reflection_prompt = REFLECTION_GENERATE.format(query=query, trajectory=trajectory[-TRAJECTORY_MAX_CHARS:])
        reflections.append(chat_once(client, model, reflection_prompt))
    stats["forced"] = True
    return forced_answer(client, model, query, trajectory), stats


def run_webwalker(
    model: str,
    api_key: str,
    model_server: str,
    tool: VisitPageTool,
    query: str,
    root_url: str,
    max_rounds: int,
    max_input_tokens: int = 120000,
) -> Tuple[str, dict]:
    # max_input_tokens is qwen-agent's rough input truncation threshold; it must
    # fit the endpoint's real context window (e.g. 28000 for 32k-context models)
    # or oversized observations/trajectories get rejected with a 400.
    llm_cfg = {
        "model": model,
        "api_key": api_key,
        "model_server": model_server,
        "generate_cfg": {"top_p": 0.8, "max_input_tokens": max_input_tokens, "max_retries": 20},
        "query": query,
        "action_count": max_rounds,
    }
    bot = WebWalker(llm=llm_cfg, function_list=[tool])
    observation = tool.observe_root()
    start_prompt = build_start_prompt(query, root_url, observation)
    messages = [{"role": "user", "content": start_prompt}]
    final = None
    for step in bot.run(messages=messages, lang="en"):
        content = step[0]["content"]
        if "Final Answer" in content:
            final = content.split("Final Answer:", 1)[1].strip()
    stats = {"rounds": tool.actions, "forced": False}
    if final is None:
        # rounds exhausted without the critic declaring sufficiency; still
        # force an answer so the item is comparable (never a blank pred)
        memory = "-".join(bot.momery) or "(no useful information was collected)"
        final = forced_answer(bot.client, model, query, memory)
        stats["forced"] = True
    return final, stats


def answer_one(
    client: OpenAI,
    model: str,
    item: dict,
    method: str,
    max_rounds: int,
    attempts: int,
    max_input_tokens: int = 120000,
) -> dict:
    query, root_url = item["question"], item["root_url"]
    tool = VisitPageTool(root_url)
    started = time.time()
    try:
        if method == "webwalker":
            pred, stats = run_webwalker(
                model,
                AGENT_API_KEY,
                AGENT_BASE_URL,
                tool,
                query,
                root_url,
                max_rounds,
                max_input_tokens,
            )
        elif method == "reflexion":
            pred, stats = run_reflexion(client, model, tool, query, root_url, max_rounds, attempts)
        else:
            pred, stats = run_react(client, model, tool, query, root_url, max_rounds)
        return {
            "question": query,
            "pred": pred,
            "method": method,
            "model": model,
            "elapsed_s": round(time.time() - started, 1),
            **stats,
        }
    except Exception as e:  # noqa: BLE001 - record failures instead of aborting the run
        return {
            "question": query,
            "pred": "",
            "method": method,
            "model": model,
            "elapsed_s": round(time.time() - started, 1),
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="Run WebWalkerQA benchmark with a given method.")
    parser.add_argument("--method", required=True, choices=["webwalker", "react", "reflexion"])
    parser.add_argument("--output_path", required=True, help="jsonl output file (resumable)")
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--split", default="main")
    parser.add_argument("--max_rounds", type=int, default=15, help="max tool calls per episode (paper uses 15)")
    parser.add_argument("--max_input_tokens", type=int, default=102400, help="128K ctw, 100K in")
    parser.add_argument("--attempts", type=int, default=3, help="episodes per question (reflexion only)")
    parser.add_argument("--workers", type=int, default=4, help="parallel questions")
    parser.add_argument("--limit", type=int, default=None, help="only first N questions (smoke test)")
    args = parser.parse_args()

    if not AGENT_API_KEY or not AGENT_BASE_URL:
        raise SystemExit("Please set AGENT_API_KEY and AGENT_BASE_URL (in .env or the environment).")

    ds = load_dataset("callanwu/WebWalkerQA", split=args.split)
    items = [{"question": q, "root_url": r} for q, r in zip(ds["question"], ds["root_url"])]
    if args.limit:
        items = items[: args.limit]

    visited = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, encoding="utf-8") as f:
            visited = {json.loads(line)["question"] for line in f if line.strip()}
    todo = [it for it in items if it["question"] not in visited]
    print(f"{args.method}/{args.model}: {len(todo)} to run, {len(visited)} already done")

    client = OpenAI(api_key=AGENT_API_KEY, base_url=AGENT_BASE_URL)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                answer_one,
                client,
                args.model,
                item,
                args.method,
                args.max_rounds,
                args.attempts,
                args.max_input_tokens,
            ): item
            for item in todo
        }
        with tqdm(total=len(todo), desc=args.method) as pbar:
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                with _write_lock, open(args.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pbar.update(1)


if __name__ == "__main__":
    main()
