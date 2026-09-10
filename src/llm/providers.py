"""LLM providers.

Groq, Mistral and Gemini all expose an OpenAI-compatible /chat/completions, so a
single client covers all three. Only the base URL and the key change.
"""
from __future__ import annotations

import json
import os
import random
import time

import httpx

from src.config import CACHE_PATH
from src.llm.base import LLMResponse, Message, ToolCall
from src.llm.cache import LLMCache, make_key

ENDPOINTS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
    ),
}

# Price per 1M tokens (input, output). Filled in by hand for your own plan;
# on a free tier these are zeros, but the field is needed so the ablation can
# compute cost.
PRICES: dict[str, tuple[float, float]] = {}

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
_CACHE = LLMCache(CACHE_PATH)
_CACHE_READ = os.getenv("LLM_CACHE", "1") != "0"


class OpenAICompatProvider:
    def __init__(self, provider: str, model: str, timeout: float = 45.0):
        if provider not in ENDPOINTS:
            raise ValueError(f"Unknown provider {provider!r}. Available: {list(ENDPOINTS)}")
        base_url, env_key = ENDPOINTS[provider]
        api_key = os.getenv(env_key)
        if not api_key:
            raise RuntimeError(f"{env_key} is not set in .env")
        self.name = provider
        self.model = model
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        params = {"temperature": temperature, "max_tokens": max_tokens, "tools": tools}
        key = make_key(self.name, self.model, messages, **params)

        # Cache reads can be turned off (LLM_CACHE=0). This is needed for exactly
        # two things that are IMPOSSIBLE to measure with the cache on: the spread
        # between runs, and honest latency. Writing continues regardless, so the
        # next normal run is free again.
        if _CACHE_READ and (hit := _CACHE.get(key)) is not None:
            return self._parse(hit, latency_ms=0.0, cached=True)

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        started = time.perf_counter()
        data = self._post_with_retry("/chat/completions", body)
        latency_ms = (time.perf_counter() - started) * 1000

        _CACHE.put(key, self.name, self.model, data)
        return self._parse(data, latency_ms=latency_ms, cached=False)

    def _post_with_retry(self, path: str, body: dict, max_attempts: int = 6) -> dict:
        """Exponential backoff with jitter. On a free tier a 429 is normal rather
        than an error, so retries are mandatory."""
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                resp = self._client.post(path, json=body)
                if resp.status_code in RETRY_STATUS:
                    wait = self._retry_after(resp) or (2**attempt + random.random())
                    print(f"    [{self.name}] {resp.status_code}, attempt "
                          f"{attempt+1}/{max_attempts}, waiting {min(wait,60):.0f}s", flush=True)
                    time.sleep(min(wait, 60))
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                print(f"    [{self.name}] {type(exc).__name__}, attempt "
                      f"{attempt+1}/{max_attempts}", flush=True)
                time.sleep(min(2**attempt + random.random(), 60))
        raise RuntimeError(
            f"{self.name}: request failed after {max_attempts} attempts"
        ) from last_error

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        raw = resp.headers.get("retry-after")
        try:
            return float(raw) if raw else None
        except ValueError:
            return None

    def _parse(self, data: dict, latency_ms: float, cached: bool) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}

        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                # The model returned invalid JSON. This is a normal situation,
                # handled at the agent level rather than by crashing here.
                args = {"__malformed__": raw_args}
            calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""),
                                  arguments=args,
                                  extra={k: v for k, v in tc.items()
                                         if k not in ("id", "type", "function")}))

        return LLMResponse(
            text=msg.get("content") or "",
            model=self.model,
            provider=self.name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            cached=cached,
            tool_calls=calls,
            raw=data,
        )

    def cost_usd(self, r: LLMResponse) -> float:
        inp, out = PRICES.get(f"{self.name}/{self.model}", (0.0, 0.0))
        return (r.prompt_tokens * inp + r.completion_tokens * out) / 1_000_000
