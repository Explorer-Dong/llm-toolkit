from __future__ import annotations

import time
from typing import Any

from openai import OpenAI


class LLMClient:
    def __init__(self, model: str, base_url: str, api_key: str, request_timeout: float) -> None:
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        top_k: int | None,
        max_tokens: int,
        seed: int | None = None,
    ) -> tuple[str, float]:
        extra_body: dict[str, Any] = {}
        if top_k is not None:
            extra_body["top_k"] = top_k
        if seed is not None:
            extra_body["seed"] = seed

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        start = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        latency_sec = time.perf_counter() - start

        if not response.choices:
            raise RuntimeError("LLM response contained no choices")

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM response choice contained no message content")

        return content, latency_sec
