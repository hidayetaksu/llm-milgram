"""Async OpenRouter chat client with retries, cost accounting, and catalog validation.

Mirrors the reference paper's collection discipline: failures are retried with
exponential backoff and never enter the data; every successful response carries
provider, latency, token usage (incl. cached tokens), and cost.
"""

from __future__ import annotations

import asyncio
import os
import random
import time

import httpx

BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 520, 522, 524}


class OpenRouterError(Exception):
    pass


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, timeout_s: float = 120, max_retries: int = 6):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set (env var or .env). "
                "Use --mock for a dry run without an API key."
            )
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/llm-milgram",
                "X-Title": "llm-milgram obedience census",
            },
        )
        self.total_cost = 0.0
        self.total_requests = 0
        self.retry_log: list[dict] = []

    async def close(self):
        await self._client.aclose()

    async def chat(self, model: str, messages: list[dict], temperature: float,
                   max_tokens: int, disable_reasoning: bool = True,
                   tools: list[dict] | None = None,
                   reasoning_budget: int = 0, **_) -> dict:
        """One chat completion. Returns a normalized record; raises after max retries.

        reasoning_budget: thinking-token budget for this request. 0 (the default
        census arm) is sent as OpenRouter's provider-agnostic disable form --
        a literal max_tokens=0 is rejected by several providers, so "zero
        thinking tokens" is realized as reasoning disabled. A positive budget is
        sent as reasoning.max_tokens; providers with a higher documented minimum
        may clamp it, which is why per-response reasoning_tokens are logged.
        """
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if tools:
            body["tools"] = tools
        if reasoning_budget > 0:
            body["reasoning"] = {"max_tokens": reasoning_budget}
        elif disable_reasoning:
            body["reasoning"] = {"enabled": False}

        last_err = None
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = await self._client.post("/chat/completions", json=body)
                latency_ms = int((time.monotonic() - t0) * 1000)
                if resp.status_code == 400 and disable_reasoning and attempt == 0:
                    # Some endpoints reject the reasoning flag outright; retry once
                    # without it and record the fallback (reference-paper style logging).
                    self.retry_log.append({"model": model, "reason": "reasoning_flag_rejected"})
                    body.pop("reasoning", None)
                    continue
                if resp.status_code in RETRYABLE_STATUS:
                    raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    err = data["error"]
                    if (err.get("code") or 0) in RETRYABLE_STATUS:
                        raise OpenRouterError(str(err)[:200])
                    raise httpx.HTTPStatusError(str(err)[:500], request=resp.request, response=resp)
                choice = data["choices"][0]
                usage = data.get("usage", {}) or {}
                cost = float(usage.get("cost") or 0.0)
                self.total_cost += cost
                self.total_requests += 1
                tool_calls = choice["message"].get("tool_calls") or []
                assistant_message = None
                if tool_calls:
                    assistant_message = {"role": "assistant",
                                         "content": choice["message"].get("content"),
                                         "tool_calls": tool_calls}
                return {
                    "completion": choice["message"].get("content") or "",
                    "tool_calls": tool_calls,
                    "assistant_message": assistant_message,
                    "finish_reason": choice.get("finish_reason"),
                    "reasoning_present": bool(choice["message"].get("reasoning")),
                    "provider": data.get("provider"),
                    "model_reported": data.get("model"),
                    "gen_id": data.get("id"),
                    "latency_ms": latency_ms,
                    "usage": usage,
                    "reasoning_tokens": ((usage.get("completion_tokens_details") or {})
                                         .get("reasoning_tokens")),
                    "cost": cost,
                }
            except (httpx.TimeoutException, httpx.TransportError, OpenRouterError) as e:
                last_err = e
                self.retry_log.append({"model": model, "attempt": attempt, "reason": str(e)[:200]})
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(min(60.0, (2 ** attempt) + random.random()))
        raise OpenRouterError(f"{model}: giving up after {self.max_retries + 1} attempts: {last_err}")

    async def catalog(self) -> dict:
        """Fetch the model catalog: {model_id: {input_price_per_1m, output_price_per_1m, ...}}."""
        resp = await self._client.get("/models")
        resp.raise_for_status()
        out = {}
        for m in resp.json().get("data", []):
            pricing = m.get("pricing", {}) or {}
            out[m["id"]] = {
                "input_per_1m": float(pricing.get("prompt") or 0) * 1e6,
                "output_per_1m": float(pricing.get("completion") or 0) * 1e6,
                "context": m.get("context_length"),
                "supported_parameters": m.get("supported_parameters", []),
            }
        return out
