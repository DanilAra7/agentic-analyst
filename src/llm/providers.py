"""Провайдеры LLM.

Groq, Mistral и Gemini отдают OpenAI-совместимый /chat/completions, поэтому
одного клиента хватает на всех троих. Меняется только базовый URL и ключ.
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

# Цена за 1 млн токенов (input, output). Заполняется вручную под свой тариф;
# на free tier это нули, но поле нужно, чтобы ablation считала стоимость.
PRICES: dict[str, tuple[float, float]] = {}

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
_CACHE = LLMCache(CACHE_PATH)
_CACHE_READ = os.getenv("LLM_CACHE", "1") != "0"


class OpenAICompatProvider:
    def __init__(self, provider: str, model: str, timeout: float = 45.0):
        if provider not in ENDPOINTS:
            raise ValueError(f"Неизвестный провайдер {provider!r}. Доступны: {list(ENDPOINTS)}")
        base_url, env_key = ENDPOINTS[provider]
        api_key = os.getenv(env_key)
        if not api_key:
            raise RuntimeError(f"Не задан {env_key} в .env")
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

        # Чтение кеша можно выключить (LLM_CACHE=0). Это нужно ровно для двух
        # вещей, которые с кешем измерить НЕВОЗМОЖНО: разброса между прогонами
        # и честной латентности. Запись при этом продолжается: следующий
        # обычный прогон снова будет бесплатным.
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
        """Экспоненциальный backoff с джиттером. На free tier 429 это норма,
        а не ошибка, поэтому ретраи обязательны."""
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                resp = self._client.post(path, json=body)
                if resp.status_code in RETRY_STATUS:
                    wait = self._retry_after(resp) or (2**attempt + random.random())
                    print(f"    [{self.name}] {resp.status_code}, попытка "
                          f"{attempt+1}/{max_attempts}, пауза {min(wait,60):.0f}s", flush=True)
                    time.sleep(min(wait, 60))
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                print(f"    [{self.name}] {type(exc).__name__}, попытка "
                      f"{attempt+1}/{max_attempts}", flush=True)
                time.sleep(min(2**attempt + random.random(), 60))
        raise RuntimeError(
            f"{self.name}: не удалось выполнить запрос за {max_attempts} попыток"
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
                # Модель вернула невалидный JSON. Это штатная ситуация,
                # обрабатывается на уровне агента, а не падением здесь.
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
