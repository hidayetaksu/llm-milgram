"""Unit tests for the Opik tracing layer (fake opik module, no network)."""

import sys
import types

import pytest

from src.opik_tracing import OpikTracer


class FakeSpanRecorder:
    def __init__(self):
        self.traces = []
        self.flushed = 0


class FakeTrace:
    def __init__(self, recorder, **kw):
        self.kw = kw
        self.spans = []
        recorder.traces.append(self)

    def span(self, **kw):
        self.spans.append(kw)


def install_fake_opik(monkeypatch, recorder, fail_init=False, fail_trace=False):
    mod = types.ModuleType("opik")

    class Opik:
        def __init__(self, **kw):
            if fail_init:
                raise RuntimeError("no account")
            self.kw = kw

        def trace(self, **kw):
            if fail_trace:
                raise RuntimeError("api down")
            return FakeTrace(recorder, **kw)

        def flush(self):
            recorder.flushed += 1

    mod.Opik = Opik
    monkeypatch.setitem(sys.modules, "opik", mod)
    return mod


SPEC = {"model": "m/x", "condition": "baseline", "language": "en",
        "temperature": 1.0, "rep": 0}
RESP = {"completion": "ok\nACTION: SHOCK", "model_reported": "m/x", "provider": "P",
        "finish_reason": "stop", "reasoning_present": False, "gen_id": "g1",
        "latency_ms": 120, "cost": 0.001,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                  "cost": 0.001}}


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    t = OpikTracer({"workspace": "w", "project_name": "p"})
    assert not t.enabled
    assert t.start_session(SPEC, "k", "sys", False) is None
    t.log_turn(None, 0, 15, "u", RESP, "SHOCK", 0)   # no-ops
    t.end_session(None, {}, 0.0)
    t.flush()


def test_disabled_flag(monkeypatch):
    monkeypatch.setenv("OPIK_API_KEY", "k")
    assert not OpikTracer({}, disabled=True).enabled


def test_init_failure_disables(monkeypatch):
    monkeypatch.setenv("OPIK_API_KEY", "k")
    install_fake_opik(monkeypatch, FakeSpanRecorder(), fail_init=True)
    assert not OpikTracer({"workspace": "w", "project_name": "p"}).enabled


def test_happy_path_buffered_emission(monkeypatch):
    monkeypatch.setenv("OPIK_API_KEY", "k")
    rec = FakeSpanRecorder()
    install_fake_opik(monkeypatch, rec)
    t = OpikTracer({"workspace": "w", "project_name": "p"})
    assert t.enabled
    buf = t.start_session(SPEC, "key1", "system prompt", resumed=True)
    t.log_turn(buf, 0, 15, "u1", RESP, "SHOCK", 0)
    t.log_turn(buf, 1, 30, "u2", RESP, "STOP", 2)
    t.end_session(buf, {"outcome": "defiant", "max_voltage": 15}, 0.002)
    t.flush()
    assert rec.flushed == 1
    assert len(rec.traces) == 1
    tr = rec.traces[0]
    assert tr.kw["output"]["outcome"] == "defiant"
    assert "resumed" in tr.kw["tags"] and "defiant" in tr.kw["tags"]
    assert len(tr.spans) == 2
    assert tr.spans[0]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5,
                                    "total_tokens": 15}
    assert "(prod 2)" in tr.spans[1]["name"]
    # buffered log_turn after disable is a no-op
    t.enabled = False
    t.log_turn(buf, 2, 45, "u3", RESP, None, 0)
    assert len(buf["turns"]) == 2


def test_end_session_with_error_and_no_turns(monkeypatch):
    monkeypatch.setenv("OPIK_API_KEY", "k")
    rec = FakeSpanRecorder()
    install_fake_opik(monkeypatch, rec)
    t = OpikTracer({})
    buf = t.start_session(SPEC, "key2", "sys", resumed=False)
    t.end_session(buf, None, 0.0, error="boom")
    tr = rec.traces[0]
    assert tr.kw["output"]["outcome"] == "attrition_or_abort"
    assert tr.kw["output"]["error"] == "boom"
    assert tr.spans == []
    t.end_session(None, {}, 0.0)  # None buffer no-op


def test_consecutive_errors_self_disable(monkeypatch):
    monkeypatch.setenv("OPIK_API_KEY", "k")
    rec = FakeSpanRecorder()
    install_fake_opik(monkeypatch, rec, fail_trace=True)
    t = OpikTracer({})
    for i in range(5):
        buf = t.start_session(SPEC, f"k{i}", "sys", False)
        t.end_session(buf, {"outcome": "obedient", "max_voltage": 450}, 0.0)
    assert not t.enabled  # 5 consecutive emission failures
    t.flush()  # no-op once disabled
    assert rec.flushed == 0
