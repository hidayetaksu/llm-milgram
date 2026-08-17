"""Unit tests for the OpenRouter client (httpx.MockTransport, no network)."""

import asyncio
import json

import httpx
import pytest

import src.openrouter as orm
from src.openrouter import OpenRouterClient, OpenRouterError


def make_client(handler, monkeypatch, max_retries=2):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(orm.asyncio, "sleep", lambda s: real_sleep(0))
    c = OpenRouterClient(timeout_s=5, max_retries=max_retries)
    c._client = httpx.AsyncClient(base_url=orm.BASE_URL,
                                  transport=httpx.MockTransport(handler),
                                  headers=c._client.headers)
    return c


def ok_payload(content="hi\nACTION: SHOCK", tool_calls=None, reasoning=None):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning:
        msg["reasoning"] = reasoning
    return {"id": "gen-1", "model": "m-reported", "provider": "P",
            "choices": [{"message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001,
                      "completion_tokens_details": {"reasoning_tokens": 7}}}


def run(coro):
    return asyncio.run(coro)


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        OpenRouterClient()


def test_chat_success_with_tools_and_reasoning_flag(monkeypatch):
    seen = {}
    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=ok_payload(
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "administer_shock", "arguments": "{}"}}],
            reasoning="thinking..."))
    c = make_client(handler, monkeypatch)
    r = run(c.chat("m", [{"role": "user", "content": "u"}], 1.0, 100,
                   tools=[{"type": "function"}]))
    assert seen["tools"] and seen["reasoning"] == {"enabled": False}
    assert r["completion"].startswith("hi")
    assert r["tool_calls"][0]["id"] == "c1"
    assert r["assistant_message"]["tool_calls"]
    assert r["reasoning_present"] is True
    assert r["provider"] == "P" and r["cost"] == 0.001
    assert r["reasoning_tokens"] == 7
    assert c.total_cost == 0.001 and c.total_requests == 1
    run(c.close())


def test_reasoning_budget_modes(monkeypatch):
    seen = {}
    def handler(req):
        seen.clear(); seen.update(json.loads(req.content))
        return httpx.Response(200, json=ok_payload())
    c = make_client(handler, monkeypatch)
    run(c.chat("m", [{"role": "user", "content": "u"}], 1.0, 100,
               reasoning_budget=1024))
    assert seen["reasoning"] == {"max_tokens": 1024}     # positive budget
    run(c.chat("m", [{"role": "user", "content": "u"}], 1.0, 100,
               reasoning_budget=0))
    assert seen["reasoning"] == {"enabled": False}      # zero budget -> disabled
    run(c.close())


def test_reasoning_flag_rejected_fallback(monkeypatch):
    calls = []
    def handler(req):
        body = json.loads(req.content)
        calls.append("reasoning" in body)
        if "reasoning" in body:
            return httpx.Response(400, json={"error": "no reasoning param"})
        return httpx.Response(200, json=ok_payload())
    c = make_client(handler, monkeypatch)
    r = run(c.chat("m", [{"role": "user", "content": "u"}], 1.0, 100))
    assert calls == [True, False]
    assert r["assistant_message"] is None and r["tool_calls"] == []
    assert c.retry_log[0]["reason"] == "reasoning_flag_rejected"
    run(c.close())


def test_retryable_status_then_success(monkeypatch):
    n = {"i": 0}
    def handler(req):
        n["i"] += 1
        return (httpx.Response(429, text="slow down") if n["i"] == 1
                else httpx.Response(200, json=ok_payload()))
    c = make_client(handler, monkeypatch)
    r = run(c.chat("m", [{"role": "user", "content": "u"}], 1.0, 100,
                   disable_reasoning=False))
    assert r["gen_id"] == "gen-1"
    assert n["i"] == 2
    run(c.close())


def test_exhausted_retries_raises(monkeypatch):
    handler = lambda req: httpx.Response(503, text="down")
    c = make_client(handler, monkeypatch, max_retries=1)
    with pytest.raises(OpenRouterError, match="giving up"):
        run(c.chat("m", [{"role": "user", "content": "u"}], 1.0, 100,
                   disable_reasoning=False))
    run(c.close())


def test_non_retryable_http_error_propagates(monkeypatch):
    handler = lambda req: httpx.Response(401, text="bad key")
    c = make_client(handler, monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        run(c.chat("m", [{"role": "user", "content": "u"}], 1.0, 100,
                   disable_reasoning=False))
    run(c.close())


def test_error_dict_retryable_and_fatal(monkeypatch):
    n = {"i": 0}
    def handler(req):
        n["i"] += 1
        if n["i"] == 1:
            return httpx.Response(200, json={"error": {"code": 429, "message": "rate"}})
        return httpx.Response(200, json=ok_payload())
    c = make_client(handler, monkeypatch)
    r = run(c.chat("m", [{"role": "user", "content": "u"}], 1.0, 100,
                   disable_reasoning=False))
    assert r["model_reported"] == "m-reported"

    fatal = lambda req: httpx.Response(200, json={"error": {"code": 402, "message": "billing"}})
    c2 = make_client(fatal, monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        run(c2.chat("m", [{"role": "user", "content": "u"}], 1.0, 100,
                    disable_reasoning=False))
    run(c.close()); run(c2.close())


def test_timeout_then_success(monkeypatch):
    n = {"i": 0}
    def handler(req):
        n["i"] += 1
        if n["i"] == 1:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json=ok_payload())
    c = make_client(handler, monkeypatch)
    r = run(c.chat("m", [{"role": "user", "content": "u"}], 1.0, 100,
                   disable_reasoning=False))
    assert r["finish_reason"] == "stop"
    run(c.close())


def test_catalog(monkeypatch):
    def handler(req):
        return httpx.Response(200, json={"data": [
            {"id": "a/b", "pricing": {"prompt": "0.000001", "completion": "0.000002"},
             "context_length": 8192, "supported_parameters": ["tools"]},
            {"id": "c/d", "pricing": {}},
        ]})
    c = make_client(handler, monkeypatch)
    cat = run(c.catalog())
    assert cat["a/b"]["input_per_1m"] == 1.0
    assert cat["a/b"]["output_per_1m"] == 2.0
    assert "tools" in cat["a/b"]["supported_parameters"]
    assert cat["c/d"]["input_per_1m"] == 0.0
    run(c.close())
