"""Unit tests for the collection runner (no network; fake chat functions)."""

import argparse
import asyncio
import json

import pytest

import src.openrouter as orm
import src.runner as rn
from src.trial_engine import TrialEngine, load_prompts


def run(coro):
    return asyncio.run(coro)


def spec_for(model="mock/x", condition="baseline", rep=0, temp=1.0):
    return {"model": model, "condition": condition, "language": "en",
            "temperature": temp, "rep": rep}


def scripted_chat(actions):
    """Chat fn yielding canned SHOCK/STOP/None replies in sequence."""
    it = iter(actions)
    async def chat(model, messages, temperature, max_tokens, **kw):
        a = next(it)
        text = {"SHOCK": "ok\nACTION: SHOCK", "STOP": "no\nACTION: STOP",
                None: "mumble"}[a]
        return {"completion": text, "tool_calls": [], "assistant_message": None,
                "finish_reason": "stop", "reasoning_present": False,
                "provider": "fake", "model_reported": model, "gen_id": "g",
                "latency_ms": 5, "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "cost": 0.001}
    return chat


def exp_cfg_of(root):
    return json.loads((root / "config" / "experiment.json").read_text())


# ---------- helpers ----------

def test_slug_key_utcnow():
    assert rn.slug("a/b:c") == "a_b_c"
    assert rn.session_key("m", "c", "en", 0.0, 3) == "m|c|en|T0|r3"
    assert rn.session_key("m", "c", "en", 1.0, 3) == "m|c|en|T1|r3"
    assert rn.session_key("m", "c", "en", 1.0, 3, 1024) == "m|c|en|T1|Rb1024|r3"
    assert rn.session_key("m", "c", "en", 1.0, 3, 0) == "m|c|en|T1|r3"
    assert rn.arm_label(0) == "none" and rn.arm_label(1024) == "b1024"
    assert "T" in rn.utcnow()


def test_build_thinking_specs():
    cfg = {"thinking_arm": {"enabled": True, "conditions": ["baseline"],
                            "budget_tokens": 1024, "reps_per_cell": 3,
                            "reps_per_cell_frontier": 1, "max_tokens_per_turn": 4000},
           "languages": ["en"], "temperature": 1.0}
    catalog = {"m/think": {"supported_parameters": ["reasoning"]},
               "m/basic": {"supported_parameters": []}}
    specs = rn.build_thinking_specs(["m/think", "m/basic", "m/frontier"], cfg,
                                    dict(catalog, **{"m/frontier": catalog["m/think"]}),
                                    frontier_ids={"m/frontier"})
    by_model = {}
    for s in specs:
        by_model.setdefault(s["model"], []).append(s)
    assert len(by_model["m/think"]) == 3 and len(by_model["m/frontier"]) == 1
    assert "m/basic" not in by_model
    assert all(s["reasoning_budget"] == 1024 and s["max_tokens"] == 4000 for s in specs)
    cfg["thinking_arm"]["enabled"] = False
    assert rn.build_thinking_specs(["m/think"], cfg, catalog, set()) == []


def test_scan_log_restart_semantics():
    recs = [
        {"type": "session_start", "key": "k1"},
        {"type": "turn", "key": "k1", "turn_idx": 0},
        {"type": "session_start", "key": "k1"},          # restart resets turns
        {"type": "turn", "key": "k1", "turn_idx": 0},
        {"type": "session_start", "key": "k2"},
        {"type": "turn", "key": "k2", "turn_idx": 0},
        {"type": "session_end", "key": "k2"},
        {"type": "session_start", "key": "k3"},
        {"type": "session_error", "key": "k3"},
    ]
    completed, partial, errored = rn.scan_log(recs)
    assert completed == {"k2"}                 # k3 errored: retriable, not complete
    assert errored == {"k3"}
    assert len(partial["k1"]) == 1             # restart marker reset its turns
    assert partial["k3"] == []                 # errored before any turn -> clean restart
    assert "k2" not in partial


def test_tool_action_and_messages(prompts_cfg):
    e = TrialEngine(prompts_cfg, "tool_actuation")
    e.start()
    # no calls -> plain assistant message, no action
    a, msgs = rn.tool_action_and_messages(
        {"completion": "hm", "tool_calls": [], "assistant_message": None}, e)
    assert a is None and msgs == [{"role": "assistant", "content": "hm"}]
    # first known call wins; extra ignored; unknown tool answered
    calls = [{"id": "1", "function": {"name": "administer_shock"}},
             {"id": "2", "function": {"name": "refuse_and_stop"}},
             {"id": "3", "function": {"name": "self_destruct"}}]
    resp = {"completion": "", "tool_calls": calls,
            "assistant_message": {"role": "assistant", "content": None, "tool_calls": calls}}
    a, msgs = rn.tool_action_and_messages(resp, e)
    assert a == "SHOCK"
    assert msgs[0]["tool_calls"] == calls
    assert "Shock delivered: 15 volts" in msgs[1]["content"]
    assert "Ignored" in msgs[2]["content"]
    assert "Unknown tool" in msgs[3]["content"]


# ---------- run_session ----------

def test_run_session_defiant_with_tracer(workspace, prompts_cfg, dummy_tracer):
    log = rn.SessionLog()
    cfg = exp_cfg_of(workspace)
    chat = scripted_chat(["SHOCK"] * 3 + ["STOP"] * 5)
    s = run(rn.run_session(spec_for(), chat, log, prompts_cfg, cfg, None,
                           {"cap": None, "spent": lambda: 0}, tracer=dummy_tracer))
    assert s["outcome"] == "defiant" and s["max_voltage"] == 45
    assert dummy_tracer.started[0][1] is False
    assert dummy_tracer.ended[0][1] == "defiant"
    recs = log.read_all("mock/x")
    assert recs[0]["type"] == "session_start"
    assert "You are ORDIS" in recs[0]["system_prompt"]   # verbatim, self-contained
    assert recs[-1]["type"] == "session_end"


def test_run_session_resume_and_mismatch(workspace, prompts_cfg, dummy_tracer):
    log = rn.SessionLog()
    cfg = exp_cfg_of(workspace)
    # partial session: 2 SHOCK turns logged, then interrupted
    run(rn.run_session(spec_for(rep=1), scripted_chat(["SHOCK", "SHOCK"] + [RuntimeError]),
                       log, prompts_cfg, cfg, None,
                       {"cap": None, "spent": lambda: 0}))  # errors at turn 3
    recs = log.read_all("mock/x")
    _, partial, _ = rn.scan_log(recs)
    key = rn.session_key("mock/x", "baseline", "en", 1.0, 1)
    # session_error marked it complete; simulate an interrupt instead by
    # stripping the error record
    recs = [r for r in recs if r["type"] != "session_error"]
    p = log.path("mock/x")
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    _, partial, _ = rn.scan_log(log.read_all("mock/x"))
    assert len(partial[key]) == 2
    # resume: replay 2 turns, then defiance
    s = run(rn.run_session(spec_for(rep=1), scripted_chat(["STOP"] * 5), log,
                           prompts_cfg, cfg, partial[key],
                           {"cap": None, "spent": lambda: 0}, tracer=dummy_tracer))
    assert s["outcome"] == "defiant" and s["max_voltage"] == 30
    assert dummy_tracer.started[-1][1] is True  # resumed
    # only ONE session_start for the resumed run
    starts = [r for r in log.read_all("mock/x")
              if r["type"] == "session_start" and r["key"] == key]
    assert len(starts) == 1

    # mismatch: corrupt the partial's first user_msg -> clean restart
    bad = [dict(t, user_msg="DIFFERENT") for t in partial[key]]
    s = run(rn.run_session(spec_for(rep=2), scripted_chat(["STOP"] * 5), log,
                           prompts_cfg, cfg, bad, {"cap": None, "spent": lambda: 0}))
    assert s["outcome"] == "defiant" and s["max_voltage"] == 0


def test_run_session_resume_replays_to_terminal(workspace, prompts_cfg):
    """Partial log already contains the terminal turn -> summary written on resume."""
    log = rn.SessionLog()
    cfg = exp_cfg_of(workspace)
    run(rn.run_session(spec_for(rep=3), scripted_chat(["SHOCK"] + ["STOP"] * 5 + [RuntimeError]),
                       log, prompts_cfg, cfg, None, {"cap": None, "spent": lambda: 0}))
    key = rn.session_key("mock/x", "baseline", "en", 1.0, 3)
    recs = [r for r in log.read_all("mock/x")
            if not (r["type"] in ("session_end", "session_error") and r["key"] == key)]
    log.path("mock/x").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    _, partial, _ = rn.scan_log(log.read_all("mock/x"))
    s = run(rn.run_session(spec_for(rep=3), scripted_chat([]), log, prompts_cfg,
                           cfg, partial[key], {"cap": None, "spent": lambda: 0}))
    assert s["outcome"] == "defiant"


def test_run_session_chat_error_and_budget(workspace, prompts_cfg, dummy_tracer):
    log = rn.SessionLog()
    cfg = exp_cfg_of(workspace)

    async def broken(**kw):
        raise RuntimeError("api dead")
    s = run(rn.run_session(spec_for(rep=4), broken, log, prompts_cfg, cfg, None,
                           {"cap": None, "spent": lambda: 0}, tracer=dummy_tracer))
    assert s is None
    assert dummy_tracer.ended[-1][2] == "api dead"

    s = run(rn.run_session(spec_for(rep=5), scripted_chat(["SHOCK"]), log,
                           prompts_cfg, cfg, None,
                           {"cap": 1.0, "spent": lambda: 99.0}, tracer=dummy_tracer))
    assert s is None
    assert dummy_tracer.ended[-1][2] == "budget_cap_abort"


def test_run_session_tool_condition(workspace, prompts_cfg):
    log = rn.SessionLog()
    cfg = exp_cfg_of(workspace)
    calls = [{"id": "c1", "type": "function",
              "function": {"name": "refuse_and_stop", "arguments": "{}"}}]

    async def tool_chat(model, messages, temperature, max_tokens, tools=None, **kw):
        assert tools, "tool defs must be passed in tool_actuation"
        return {"completion": "I refuse.\nACTION: SHOCK",  # dissociating text
                "tool_calls": calls,
                "assistant_message": {"role": "assistant", "content": "I refuse.",
                                      "tool_calls": calls},
                "finish_reason": "stop", "reasoning_present": False,
                "provider": "fake", "model_reported": model, "gen_id": "g",
                "latency_ms": 5, "usage": {}, "cost": 0.0}
    s = run(rn.run_session(spec_for(condition="tool_actuation"), tool_chat, log,
                           prompts_cfg, cfg, None, {"cap": None, "spent": lambda: 0}))
    assert s["outcome"] == "defiant" and s["max_voltage"] == 0
    t = [r for r in log.read_all("mock/x") if r["type"] == "turn"][0]
    assert t["tool_call_name"] == "refuse_and_stop"
    assert t["text_action_hint"] == "SHOCK"       # dissociation captured
    assert t["tool_results"][0]["role"] == "tool"


def test_run_session_tool_condition_resume(workspace, prompts_cfg):
    """Interrupted tool_actuation session resumes by replaying tool-call turns."""
    log = rn.SessionLog()
    cfg = exp_cfg_of(workspace)

    def tool_reply(name, n):
        calls = [{"id": f"c{n}", "type": "function",
                  "function": {"name": name, "arguments": "{}"}}]
        return {"completion": "", "tool_calls": calls,
                "assistant_message": {"role": "assistant", "content": None,
                                      "tool_calls": calls},
                "finish_reason": "stop", "reasoning_present": False,
                "provider": "fake", "model_reported": "mock/x", "gen_id": "g",
                "latency_ms": 1, "usage": {}, "cost": 0.0}

    seq = iter([tool_reply("administer_shock", 1), tool_reply("administer_shock", 2),
                RuntimeError])
    async def chat1(**kw):
        r = next(seq)
        if r is RuntimeError:
            raise RuntimeError("cut")
        return r
    run(rn.run_session(spec_for(condition="tool_actuation", rep=9), chat1, log,
                       prompts_cfg, cfg, None, {"cap": None, "spent": lambda: 0}))
    key = rn.session_key("mock/x", "tool_actuation", "en", 1.0, 9)
    recs = [r for r in log.read_all("mock/x")
            if not (r["type"] == "session_error" and r["key"] == key)]
    log.path("mock/x").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    _, partial, _ = rn.scan_log(log.read_all("mock/x"))
    assert len(partial[key]) == 2

    seq2 = iter([tool_reply("refuse_and_stop", n) for n in range(3, 9)])
    async def chat2(**kw):
        return next(seq2)
    s = run(rn.run_session(spec_for(condition="tool_actuation", rep=9), chat2, log,
                           prompts_cfg, cfg, partial[key],
                           {"cap": None, "spent": lambda: 0}))
    assert s["outcome"] == "defiant" and s["max_voltage"] == 30


# ---------- collect / build_specs / estimate_cost ----------

def test_collect_skips_complete_and_survives_crash(workspace, prompts_cfg, dummy_tracer, capsys):
    cfg = exp_cfg_of(workspace)
    specs = [spec_for(rep=0), spec_for(rep=1),
             spec_for(condition="nonexistent_condition", rep=0)]  # crashes in engine

    async def chat(**kw):
        return await scripted_chat(["STOP"] * 5)(**kw)  # fresh iterator each call

    calls = {"n": 0}
    async def stop_chat(model, messages, temperature, max_tokens, **kw):
        calls["n"] += 1
        return {"completion": "no\nACTION: STOP", "tool_calls": [],
                "assistant_message": None, "finish_reason": "stop",
                "reasoning_present": False, "provider": "fake",
                "model_reported": model, "gen_id": "g", "latency_ms": 1,
                "usage": {}, "cost": 0.0}
    run(rn.collect(specs, stop_chat, prompts_cfg, cfg, None, lambda: 0.0,
                   tracer=dummy_tracer))
    out = capsys.readouterr().out
    assert "[error]" in out                    # crashed spec reported, run continued
    assert dummy_tracer.flushed == 1
    snap = rn.RAW_DIR / f"_config_snapshot_v{prompts_cfg['version']}.json"
    assert snap.exists()                       # stimulus snapshot beside raw logs
    assert json.loads(snap.read_text())["prompts"]["version"] == prompts_cfg["version"]
    # second run: everything valid is complete
    run(rn.collect(specs[:2], stop_chat, prompts_cfg, cfg, None, lambda: 0.0))
    assert "2 already complete" in capsys.readouterr().out


def test_collect_retries_errored_session_and_resumes(workspace, prompts_cfg, capsys):
    """A session killed by an API failure is retried (and resumed) on the next run."""
    cfg = exp_cfg_of(workspace)
    spec = spec_for(rep=0)

    async def dying_chat(model, messages, temperature, max_tokens, **kw):
        if len([m for m in messages if m["role"] == "assistant"]) >= 2:
            raise RuntimeError("429 rate limited")
        return {"completion": "ok\nACTION: SHOCK", "tool_calls": [],
                "assistant_message": None, "finish_reason": "stop",
                "reasoning_present": False, "provider": "fake",
                "model_reported": model, "gen_id": "g", "latency_ms": 1,
                "usage": {}, "cost": 0.0}
    run(rn.collect([spec], dying_chat, prompts_cfg, cfg, None, lambda: 0.0))
    log = rn.SessionLog()
    completed, partial, errored = rn.scan_log(log.read_all("mock/x"))
    key = rn.session_key("mock/x", "baseline", "en", 1.0, 0)
    assert key in errored and key not in completed
    assert len(partial[key]) == 2               # two good turns survived

    # the fix is applied (API healthy again) -> re-running resumes and finishes
    async def healthy_chat(model, messages, temperature, max_tokens, **kw):
        return {"completion": "no\nACTION: STOP", "tool_calls": [],
                "assistant_message": None, "finish_reason": "stop",
                "reasoning_present": False, "provider": "fake",
                "model_reported": model, "gen_id": "g", "latency_ms": 1,
                "usage": {}, "cost": 0.0}
    run(rn.collect([spec], healthy_chat, prompts_cfg, cfg, None, lambda: 0.0))
    out = capsys.readouterr().out
    assert "1 retrying after an earlier failure" in out
    completed, _, errored = rn.scan_log(log.read_all("mock/x"))
    assert key in completed and key not in errored
    ends = [r for r in log.read_all("mock/x") if r["type"] == "session_end"]
    assert ends[0]["summary"]["max_voltage"] == 30   # resumed from the 2 logged shocks

    # --skip-errored leaves a failed session alone
    run(rn.collect([spec_for(rep=1)], dying_chat, prompts_cfg, cfg, None, lambda: 0.0))
    capsys.readouterr()
    run(rn.collect([spec_for(rep=1)], healthy_chat, prompts_cfg, cfg, None, lambda: 0.0,
                   skip_errored=True))
    assert "1 errored sessions skipped" in capsys.readouterr().out


def test_collect_circuit_breaker_account_fatal(workspace, prompts_cfg, capsys):
    """Out-of-credits trips the breaker on the FIRST failure; queued sessions
    behind the semaphore issue no further API calls."""
    cfg = dict(exp_cfg_of(workspace), abort_after_consecutive_failures=99,
               concurrency_global=2, concurrency_per_model=1)
    specs = [spec_for(rep=i) for i in range(40)]

    calls = {"n": 0}
    async def broke_chat(**kw):
        calls["n"] += 1
        raise RuntimeError("Client error '402 Payment Required' for url ...")
    run(rn.collect(specs, broke_chat, prompts_cfg, cfg, None, lambda: 0.0))
    out = capsys.readouterr().out
    assert "ABORTING" in out and "account-level API failure" in out
    assert "no collected data is lost" in out
    assert calls["n"] <= 3          # first failure trips it; only in-flight peers follow
    # every session remains pending for the next run
    run(rn.collect(specs, broke_chat, prompts_cfg, cfg, None, lambda: 0.0))
    assert "already complete; 40 to run" in capsys.readouterr().out


def test_collect_breaker_stops_queued_sessions(workspace, prompts_cfg):
    """Sessions already waiting on the semaphore when the breaker trips must not
    issue any API call (the census bug: 4.6k sessions called a dead endpoint)."""
    cfg = dict(exp_cfg_of(workspace), abort_after_consecutive_failures=99,
               concurrency_global=1, concurrency_per_model=1)
    specs = [spec_for(rep=i) for i in range(5)]

    calls = {"n": 0}
    async def slow_fatal_chat(**kw):
        calls["n"] += 1
        await asyncio.sleep(0.05)          # lets every peer queue on the semaphore
        raise RuntimeError("Client error '402 Payment Required' for url ...")
    run(rn.collect(specs, slow_fatal_chat, prompts_cfg, cfg, None, lambda: 0.0))
    assert calls["n"] == 1                 # only the first session ever called the API


def test_collect_circuit_breaker_consecutive(workspace, prompts_cfg, capsys):
    """Non-fatal failures trip the breaker only after the consecutive limit."""
    cfg = dict(exp_cfg_of(workspace), abort_after_consecutive_failures=3,
               concurrency_global=1, concurrency_per_model=1)
    specs = [spec_for(rep=i) for i in range(30)]

    calls = {"n": 0}
    async def flaky_chat(**kw):
        calls["n"] += 1
        raise RuntimeError("connection reset")
    run(rn.collect(specs, flaky_chat, prompts_cfg, cfg, None, lambda: 0.0))
    out = capsys.readouterr().out
    assert "consecutive session failures" in out
    assert 3 <= calls["n"] < len(specs)


def test_build_specs_frontier_and_t0():
    specs = rn.build_specs(["a", "b"], ["baseline"], ["en"], reps=4, reps_t0=2,
                           temperature=1.0, frontier_ids={"b"}, reps_frontier=1)
    a = [s for s in specs if s["model"] == "a"]
    b = [s for s in specs if s["model"] == "b"]
    assert len([s for s in a if s["temperature"] == 1.0]) == 4
    assert len([s for s in b if s["temperature"] == 1.0]) == 1
    assert len([s for s in a if s["temperature"] == 0.0]) == 2


def test_estimate_cost():
    models = [{"id": "cheap"}, {"id": "pricey"}, {"id": "missing"}]
    catalog = {"cheap": {"input_per_1m": 1.0, "output_per_1m": 2.0},
               "pricey": {"input_per_1m": 10.0, "output_per_1m": 20.0}}
    cfg = {"frontier_price_threshold_per_1m_input": 5.0, "reps_per_cell": 2,
           "reps_per_cell_frontier": 1, "reps_t0_per_cell": 0}
    est = rn.estimate_cost(models, catalog, cfg, n_conditions=1)
    assert est == pytest.approx(2 * (55000 * 1 + 1600 * 2) / 1e6
                                + 1 * (55000 * 10 + 1600 * 20) / 1e6)


# ---------- main_async / main ----------

class FakeClient:
    """Stands in for OpenRouterClient in main_async tests."""
    catalog_data = {}

    def __init__(self, **kw):
        self.total_cost = 0.0
        self.total_requests = 0
        self.retry_log = []

    async def catalog(self):
        return dict(self.catalog_data)

    async def chat(self, model, messages, temperature, max_tokens, **kw):
        self.total_requests += 1
        return {"completion": "no\nACTION: STOP", "tool_calls": [],
                "assistant_message": None, "finish_reason": "stop",
                "reasoning_present": False, "provider": "fake",
                "model_reported": model, "gen_id": "g", "latency_ms": 1,
                "usage": {}, "cost": 0.0}

    async def close(self):
        pass


def make_args(**kw):
    base = dict(mock=False, pilot=False, full=False, validate=False,
                models=None, conditions=None, reps=None, budget=None,
                no_opik=True, skip_errored=False)
    base.update(kw)
    return argparse.Namespace(**base)


def small_catalog(with_tools=True):
    sp = ["tools"] if with_tools else []
    return {"m/one": {"input_per_1m": 1.0, "output_per_1m": 2.0, "context": 8192,
                      "supported_parameters": sp},
            "m/expensive": {"input_per_1m": 9.0, "output_per_1m": 20.0,
                            "context": 8192, "supported_parameters": ["tools", "reasoning"]}}


@pytest.fixture
def small_config(workspace):
    (workspace / "config" / "models.json").write_text(json.dumps({
        "version": "t", "models": [
            {"id": "m/one", "family": "f1"},
            {"id": "m/expensive", "family": "f2"},
            {"id": "m/gone", "family": "f3"}]}))
    exp = exp_cfg_of(workspace)
    exp["reps_per_cell"] = 1
    exp["reps_per_cell_frontier"] = 1
    exp["reps_t0_per_cell"] = 0
    exp["pilot"] = {"models": ["m/one", "m/gone"], "conditions": ["baseline"],
                    "reps_per_cell": 1, "reps_t0_per_cell": 0}
    (workspace / "config" / "experiment.json").write_text(json.dumps(exp))
    return workspace


def test_main_async_mock_branch(workspace, monkeypatch, capsys):
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    run(rn.main_async(make_args(mock=True, conditions="baseline", reps=1)))
    assert "to run" in capsys.readouterr().out
    assert list((workspace / "data" / "raw").glob("mock_*.jsonl"))


def test_main_async_validate(small_config, monkeypatch, capsys):
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    FakeClient.catalog_data = small_catalog()
    monkeypatch.setattr(orm, "OpenRouterClient", FakeClient)
    run(rn.main_async(make_args(validate=True)))
    out = capsys.readouterr().out
    assert "MISSING FROM CATALOG" in out and "m/gone" in out
    assert "Upper-bound cost projection" in out


def test_main_async_full_with_tool_filter(small_config, monkeypatch, capsys):
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    FakeClient.catalog_data = small_catalog(with_tools=False)
    FakeClient.catalog_data["m/expensive"]["supported_parameters"] = []
    monkeypatch.setattr(orm, "OpenRouterClient", FakeClient)
    run(rn.main_async(make_args(full=True)))
    out = capsys.readouterr().out
    assert "WARNING: not in catalog" in out
    assert "tool_actuation: skipped" in out
    assert (small_config / "data" / "retry_log.json").exists()


def test_main_async_pilot_and_adhoc(small_config, monkeypatch, capsys):
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    FakeClient.catalog_data = small_catalog()
    monkeypatch.setattr(orm, "OpenRouterClient", FakeClient)
    run(rn.main_async(make_args(pilot=True)))
    assert "Total cost this run" in capsys.readouterr().out
    run(rn.main_async(make_args(models="m/one", conditions="baseline", reps=1)))
    assert "already complete" in capsys.readouterr().out


def test_main_cli_env_and_interrupt(workspace, monkeypatch, capsys):
    (workspace / ".env").write_text("# comment\nMALFORMED\nTESTVAR=42\n")
    monkeypatch.delenv("TESTVAR", raising=False)
    seen = {}

    async def fake_main(args):
        seen["args"] = args
    monkeypatch.setattr(rn, "main_async", fake_main)
    monkeypatch.setattr("sys.argv", ["runner", "--mock"])
    rn.main()
    assert seen["args"].mock is True
    import os
    assert os.environ["TESTVAR"] == "42"

    async def interrupted(args):
        raise KeyboardInterrupt
    monkeypatch.setattr(rn, "main_async", interrupted)
    rn.main()
    assert "Interrupted" in capsys.readouterr().err
