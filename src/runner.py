"""Resumable, idempotent collection runner (reference-paper discipline).

Every session is keyed (model, condition, language, temperature, rep).
Completed sessions are skipped on restart; interrupted sessions are replayed
from the log and continued in place. Every API exchange is stored verbatim,
append-only, with UTC timestamp, serving provider, latency, token usage, and
cost. Failures are retried with backoff and never enter the data; sessions
that exhaust retries are logged as attrition.

Usage:
  uv run python -m src.runner --mock                 # simulated census (no API key)
  uv run python -m src.runner --validate             # catalog check + cost projection
  uv run python -m src.runner --pilot                # pre-registered pilot
  uv run python -m src.runner --full                 # full census
  uv run python -m src.runner --models m1,m2 --conditions baseline --reps 3
  uv run python -m src.runner --full --budget 100    # hard cost cap (USD)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .trial_engine import TrialEngine, load_prompts, parse_action

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_id)


def arm_label(budget: int) -> str:
    """Reasoning-arm label: 0 -> 'none' (default census arm), else 'b<budget>'."""
    return "none" if not budget else f"b{budget}"


def session_key(model: str, condition: str, lang: str, temp: float, rep: int,
                reasoning_budget: int = 0) -> str:
    t = "0" if temp == 0 else f"{temp:g}"
    arm = "" if not reasoning_budget else f"|R{arm_label(reasoning_budget)}"
    return f"{model}|{condition}|{lang}|T{t}{arm}|r{rep}"


class SessionLog:
    """Append-only JSONL log, one file per model, with per-file async lock."""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def path(self, model: str) -> Path:
        return RAW_DIR / f"{slug(model)}.jsonl"

    async def write(self, model: str, record: dict):
        lock = self._locks.setdefault(model, asyncio.Lock())
        async with lock:
            with open(self.path(model), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self, model: str) -> list[dict]:
        p = self.path(model)
        if not p.exists():
            return []
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


TOOL_ACTION_MAP = {"administer_shock": "SHOCK", "refuse_and_stop": "STOP"}


def tool_action_and_messages(resp: dict, engine) -> tuple[str | None, list[dict]]:
    """Map tool calls to the teacher action and build the messages to append.

    First recognized call determines the action and receives the real tool
    result; extra calls get an 'ignored' result, unknown names an 'unknown'
    result (the OpenAI protocol requires one tool message per call). With no
    tool calls at all, the reply enters history as plain assistant text and
    the action is None (format reminder path).
    """
    tr_strings = engine.lang["tool_results"]
    voltage = engine._current_voltage()
    calls = resp.get("tool_calls") or []
    if not calls:
        return None, [{"role": "assistant", "content": resp["completion"]}]
    parsed = None
    out = [resp["assistant_message"]]
    for c in calls:
        action = TOOL_ACTION_MAP.get((c.get("function") or {}).get("name"))
        if action and parsed is None:
            parsed = action
            content = tr_strings["shock" if action == "SHOCK" else "stop"].format(voltage=voltage)
        elif action:
            content = tr_strings["ignored"]
        else:
            content = tr_strings["unknown"]
        out.append({"role": "tool", "tool_call_id": c.get("id"), "content": content})
    return parsed, out


def scan_log(records: list[dict]) -> tuple[set[str], dict[str, list[dict]], set[str]]:
    """Return (completed session keys, partial sessions' turn records, errored keys)."""
    completed: set[str] = set()
    errored: set[str] = set()
    turns_by_key: dict[str, list[dict]] = {}
    started: set[str] = set()
    for r in records:
        k = r.get("key")
        if r["type"] == "session_start":
            started.add(k)
            turns_by_key[k] = []  # reset on restart marker
            errored.discard(k)
        elif r["type"] == "turn":
            turns_by_key.setdefault(k, []).append(r)
        elif r["type"] == "session_end":
            completed.add(k)
            errored.discard(k)
            turns_by_key.pop(k, None)
        elif r["type"] == "session_error":
            # NOT complete: a failed session (rate limit, provider outage, out of
            # credits) is retried on the next run, resuming from its logged turns.
            errored.add(k)
    partial = {k: v for k, v in turns_by_key.items() if k in started and k not in completed}
    return completed, partial, errored


ACCOUNT_FATAL = ("402", "401", "Payment Required", "Unauthorized",
                 "insufficient", "credits")


async def run_session(spec: dict, chat_fn, log: SessionLog, prompts_cfg: dict,
                      exp_cfg: dict, partial_turns: list[dict] | None,
                      budget_state: dict, tracer=None, on_fatal=None) -> dict | None:
    """Execute (or resume) one session. Returns the summary dict, or None on attrition."""
    model, condition, lang, temp, rep = (
        spec["model"], spec["condition"], spec["language"], spec["temperature"], spec["rep"])
    budget = spec.get("reasoning_budget", 0)
    key = session_key(model, condition, lang, temp, rep, budget)
    max_tokens = spec.get("max_tokens") or exp_cfg["max_tokens_per_turn"]
    engine = TrialEngine(prompts_cfg, condition, lang)
    engine.set_caps(exp_cfg["max_format_reminders_per_session"],
                    exp_cfg["max_invalid_turns_per_session"])
    messages = engine.start()
    session_cost = 0.0
    turn_idx = 0

    # ---- resume: replay logged turns through the engine ----
    replay_ok = True
    if partial_turns:
        for tr in partial_turns:
            expected_user = messages[-1]["content"]
            if tr["user_msg"] != expected_user:
                replay_ok = False
                break
            completion = tr["completion"]
            parsed = tr["parsed_action"]
            engine.record_completion(completion, parsed)
            if tr.get("assistant_message"):
                messages.append(tr["assistant_message"])
                messages.extend(tr.get("tool_results") or [])
            else:
                messages.append({"role": "assistant", "content": completion})
            nxt = engine.next_message(parsed if parsed in ("SHOCK", "STOP") else None, completion)
            turn_idx = tr["turn_idx"] + 1
            session_cost += tr.get("cost") or 0.0
            if nxt is None:
                break
            messages.append(nxt)
        if not replay_ok:
            # config drifted: restart cleanly (old partial stays in the log as history)
            engine = TrialEngine(prompts_cfg, condition, lang)
            engine.set_caps(exp_cfg["max_format_reminders_per_session"],
                            exp_cfg["max_invalid_turns_per_session"])
            messages = engine.start()
            turn_idx = 0
            session_cost = 0.0
            partial_turns = None

    if not partial_turns or not replay_ok:
        await log.write(model, {
            "type": "session_start", "key": key, "ts": utcnow(),
            "model": model, "condition": condition, "language": lang,
            "temperature": temp, "rep": rep,
            "reasoning_arm": arm_label(budget),
            "reasoning_budget": budget,
            "prompts_version": prompts_cfg["version"],
            "system_prompt": messages[0]["content"],
        })

    trace = tracer.start_session(spec, key, messages[0]["content"],
                                 resumed=bool(partial_turns and replay_ok)) if tracer else None

    # ---- turn loop ----
    while engine.state.outcome is None:
        if budget_state["cap"] and budget_state["spent"]() > budget_state["cap"]:
            print(f"[budget] cap reached; abandoning {key} (resumable)", flush=True)
            if tracer:
                tracer.end_session(trace, None, session_cost, error="budget_cap_abort")
            return None
        try:
            resp = await chat_fn(model=model, messages=messages, temperature=temp,
                                 max_tokens=max_tokens,
                                 session_tag=key,
                                 reasoning_budget=budget,
                                 tools=prompts_cfg["tools"] if engine.uses_tools else None)
        except Exception as e:
            await log.write(model, {"type": "session_error", "key": key, "ts": utcnow(),
                                    "turn_idx": turn_idx, "error": str(e)[:500]})
            if tracer:
                tracer.end_session(trace, None, session_cost, error=str(e)[:300])
            msg = str(e)
            if on_fatal and any(m in msg for m in ACCOUNT_FATAL):
                # account-level failure (out of credits, bad key): every other
                # session will fail identically -- stop the run immediately
                on_fatal(f"account-level API failure: {msg[:120]}")
            return None
        completion = resp["completion"]
        if engine.uses_tools:
            parsed, new_msgs = tool_action_and_messages(resp, engine)
            text_hint = parse_action(completion)
        else:
            parsed = parse_action(completion)
            new_msgs = [{"role": "assistant", "content": completion}]
            text_hint = None
        screens = engine.screen_text(completion)
        engine.record_completion(completion, parsed)
        user_msg_sent = messages[-1]["content"]
        messages.extend(new_msgs)
        session_cost += resp["cost"]
        await log.write(model, {
            "type": "turn", "key": key, "turn_idx": turn_idx, "ts": utcnow(),
            "voltage": engine._current_voltage(),
            "user_msg": user_msg_sent,
            "completion": completion,
            "parsed_action": parsed,
            "tool_call_name": ((resp.get("tool_calls") or [{}])[0].get("function") or {}).get("name"),
            "text_action_hint": text_hint,
            "assistant_message": resp.get("assistant_message"),
            "tool_results": [m for m in new_msgs if m.get("role") == "tool"] or None,
            "screens": screens,
            "finish_reason": resp["finish_reason"],
            "reasoning_present": resp["reasoning_present"],
            "reasoning_tokens": resp.get("reasoning_tokens"),
            "provider": resp["provider"],
            "model_reported": resp["model_reported"],
            "gen_id": resp["gen_id"],
            "latency_ms": resp["latency_ms"],
            "usage": resp["usage"],
            "cost": resp["cost"],
        })
        if tracer:
            tracer.log_turn(trace, turn_idx, engine._current_voltage(),
                            user_msg_sent, resp, parsed, engine.state.prod_idx)
        turn_idx += 1
        nxt = engine.next_message(parsed, completion)
        if nxt is not None:
            messages.append(nxt)

    summary = engine.summary()
    await log.write(model, {"type": "session_end", "key": key, "ts": utcnow(),
                            "summary": summary, "session_cost": session_cost,
                            "n_api_turns": turn_idx})
    if tracer:
        tracer.end_session(trace, summary, session_cost)
    return summary


async def collect(specs: list[dict], chat_fn, prompts_cfg: dict, exp_cfg: dict,
                  budget_cap: float | None, spent_fn, tracer=None,
                  skip_errored: bool = False) -> None:
    log = SessionLog()
    # Self-contained archive: snapshot the exact stimulus + sampling config next
    # to the raw logs, once per prompts version. Raw data must stay interpretable
    # even if config/ later changes (runs are only combinable within a version).
    snap = RAW_DIR / f"_config_snapshot_v{prompts_cfg['version']}.json"
    if not snap.exists():
        snap.write_text(json.dumps({"prompts": prompts_cfg, "experiment": exp_cfg},
                                   ensure_ascii=False, indent=1))
    # resumability: drop completed sessions, attach partial turn history
    by_model_records = {m: log.read_all(m) for m in {s["model"] for s in specs}}
    scan = {m: scan_log(r) for m, r in by_model_records.items()}
    pending = []
    n_retry = 0
    n_skipped_errored = 0
    for s in specs:
        key = session_key(s["model"], s["condition"], s["language"], s["temperature"],
                          s["rep"], s.get("reasoning_budget", 0))
        completed, partial, errored = scan[s["model"]]
        if key in completed:
            continue
        if key in errored:
            if skip_errored:
                n_skipped_errored += 1
                continue
            n_retry += 1
        s = dict(s)
        s["partial"] = partial.get(key)
        pending.append(s)
    rng = random.Random(exp_cfg["seed"])
    rng.shuffle(pending)
    print(f"{len(specs)} sessions requested; {len(specs) - len(pending) - n_skipped_errored} "
          f"already complete; {len(pending)} to run "
          f"({n_retry} retrying after an earlier failure"
          + (f", {n_skipped_errored} errored sessions skipped" if n_skipped_errored else "")
          + ")", flush=True)

    budget_state = {"cap": budget_cap, "spent": spent_fn}
    sem_global = asyncio.Semaphore(exp_cfg["concurrency_global"])
    sems_model: dict[str, asyncio.Semaphore] = {}
    done = 0
    t0 = time.monotonic()
    # Circuit breaker: an account-level failure (out of credits, revoked key,
    # aggregator outage) fails every session. Stop the run instead of marching
    # through the whole spec list; everything stays resumable.
    breaker = {"consecutive": 0, "tripped": False,
               "limit": exp_cfg.get("abort_after_consecutive_failures", 20)}

    def trip(reason: str):
        if not breaker["tripped"]:
            breaker["tripped"] = True
            print(f"\n*** ABORTING: {reason}. Fix the cause and re-run the same "
                  f"command to resume from here — no collected data is lost. ***\n",
                  flush=True)

    async def one(s):
        nonlocal done
        if breaker["tripped"]:
            return
        sem_m = sems_model.setdefault(s["model"], asyncio.Semaphore(exp_cfg["concurrency_per_model"]))
        try:
            async with sem_global, sem_m:
                # re-check after queuing: the breaker may have tripped while this
                # session waited for a slot, and a tripped run must issue no
                # further API calls
                if breaker["tripped"]:
                    return
                summary = await run_session(s, chat_fn, log, prompts_cfg, exp_cfg,
                                            s.get("partial"), budget_state, tracer=tracer,
                                            on_fatal=trip)
        except Exception as e:
            print(f"[error] {s['model']} {s['condition']} r{s['rep']}: {e!r}", flush=True)
            summary = None
        done += 1
        if summary is None:
            breaker["consecutive"] += 1
            if breaker["consecutive"] >= breaker["limit"]:
                trip(f"{breaker['consecutive']} consecutive session failures — "
                     f"check credits, rate limits, and API key")
        else:
            breaker["consecutive"] = 0
        tag = summary["outcome"] if summary else "ATTRITION/ABORT"
        mv = summary["max_voltage"] if summary else "-"
        print(f"[{done}/{len(pending)}] {s['model']} {s['condition']} T{s['temperature']:g} "
              f"r{s['rep']}: {tag} @ {mv} V  (spent ${spent_fn():.2f}, "
              f"{time.monotonic() - t0:.0f}s)", flush=True)

    await asyncio.gather(*(one(s) for s in pending))
    if tracer:
        tracer.flush()


def build_specs(models: list[str], conditions: list[str], languages: list[str],
                reps: int, reps_t0: int, temperature: float,
                frontier_ids: set[str], reps_frontier: int) -> list[dict]:
    specs = []
    for m in models:
        n = reps_frontier if m in frontier_ids else reps
        for c in conditions:
            for lang in languages:
                for r in range(n):
                    specs.append({"model": m, "condition": c, "language": lang,
                                  "temperature": temperature, "rep": r})
                for r in range(reps_t0):
                    specs.append({"model": m, "condition": c, "language": lang,
                                  "temperature": 0.0, "rep": r})
    return specs


def build_thinking_specs(models: list[str], exp_cfg: dict, catalog: dict,
                         frontier_ids: set[str]) -> list[dict]:
    """Paired thinking-arm sessions: baseline-condition cells re-collected with a
    moderate reasoning budget, for models exposing the reasoning parameter."""
    ta = exp_cfg.get("thinking_arm") or {}
    if not ta.get("enabled"):
        return []
    specs = []
    for m in models:
        if "reasoning" not in catalog.get(m, {}).get("supported_parameters", []):
            continue
        n = ta["reps_per_cell_frontier"] if m in frontier_ids else ta["reps_per_cell"]
        for c in ta["conditions"]:
            for lang in exp_cfg["languages"]:
                for r in range(n):
                    specs.append({"model": m, "condition": c, "language": lang,
                                  "temperature": exp_cfg["temperature"], "rep": r,
                                  "reasoning_budget": ta["budget_tokens"],
                                  "max_tokens": ta["max_tokens_per_turn"]})
    return specs


def estimate_cost(models: list[dict], catalog: dict, exp_cfg: dict, n_conditions: int) -> float:
    """Upper-bound projection: every session runs to full obedience (~55k in / 1.6k out)."""
    IN_TOK, OUT_TOK = 55_000, 1_600
    total = 0.0
    thr = exp_cfg["frontier_price_threshold_per_1m_input"]
    for m in models:
        p = catalog.get(m["id"])
        if not p:
            continue
        frontier = p["input_per_1m"] >= thr
        reps = exp_cfg["reps_per_cell_frontier"] if frontier else exp_cfg["reps_per_cell"]
        n_sessions = n_conditions * (reps + exp_cfg["reps_t0_per_cell"])
        total += n_sessions * (IN_TOK * p["input_per_1m"] + OUT_TOK * p["output_per_1m"]) / 1e6
    return total


async def main_async(args):
    prompts_cfg = load_prompts(ROOT / "config" / "prompts.json")
    exp_cfg = json.loads((ROOT / "config" / "experiment.json").read_text())
    models_cfg = json.loads((ROOT / "config" / "models.json").read_text())

    from .opik_tracing import OpikTracer
    tracer = OpikTracer(exp_cfg.get("opik"), disabled=args.no_opik)

    if args.mock:
        from .mock_model import PERSONAS, mock_chat
        models = [p.name for p in PERSONAS]
        conditions = args.conditions.split(",") if args.conditions else exp_cfg["conditions"]
        reps = args.reps or exp_cfg["reps_per_cell"]
        specs = build_specs(models, conditions, exp_cfg["languages"], reps,
                            exp_cfg["reps_t0_per_cell"], exp_cfg["temperature"],
                            frontier_ids=set(), reps_frontier=reps)
        mock_catalog = {m: {"supported_parameters": ["reasoning"]} for m in models}
        specs += build_thinking_specs(models, exp_cfg, mock_catalog, set())
        await collect(specs, mock_chat, prompts_cfg, exp_cfg, None, lambda: 0.0,
                      tracer=tracer, skip_errored=args.skip_errored)
        return

    from .openrouter import OpenRouterClient
    client = OpenRouterClient(timeout_s=exp_cfg["request_timeout_s"],
                              max_retries=exp_cfg["max_retries"])
    try:
        catalog = await client.catalog()
        all_models = [m["id"] for m in models_cfg["models"]]
        missing = [m for m in all_models if m not in catalog]
        thr = exp_cfg["frontier_price_threshold_per_1m_input"]
        frontier_ids = {m for m in all_models
                        if m in catalog and catalog[m]["input_per_1m"] >= thr}

        if args.validate:
            print(f"{'model':<45} {'$in/1M':>8} {'$out/1M':>8} {'frontier':>8} {'reasoning-param':>15}")
            for m in models_cfg["models"]:
                mid = m["id"]
                if mid not in catalog:
                    print(f"{mid:<45} {'MISSING FROM CATALOG':>20}")
                    continue
                c = catalog[mid]
                has_r = "reasoning" in c["supported_parameters"]
                print(f"{mid:<45} {c['input_per_1m']:>8.2f} {c['output_per_1m']:>8.2f} "
                      f"{'yes' if mid in frontier_ids else 'no':>8} {'yes' if has_r else 'no':>15}")
            est = estimate_cost(models_cfg["models"], catalog, exp_cfg, len(exp_cfg["conditions"]))
            print(f"\nUpper-bound cost projection (all sessions fully obedient): ${est:.2f}")
            print(f"Missing ids: {missing or 'none'}")
            return

        if missing:
            print(f"WARNING: not in catalog, skipped (fix config/models.json): {missing}")
        available = [m for m in all_models if m in catalog]

        if args.pilot:
            pilot = exp_cfg["pilot"]
            pilot_models = [m for m in pilot["models"] if m in catalog]
            specs = build_specs(pilot_models,
                                pilot["conditions"], exp_cfg["languages"],
                                pilot["reps_per_cell"], pilot["reps_t0_per_cell"],
                                exp_cfg["temperature"], frontier_ids,
                                pilot["reps_per_cell"])
            # the pilot also exercises the thinking arm, at pilot scale
            pilot_cfg = dict(exp_cfg)
            pilot_cfg["thinking_arm"] = dict(exp_cfg["thinking_arm"],
                                             reps_per_cell=pilot["reps_per_cell"],
                                             reps_per_cell_frontier=pilot["reps_per_cell"])
            specs += build_thinking_specs(pilot_models, pilot_cfg, catalog, frontier_ids)
        elif args.full:
            specs = build_specs(available, exp_cfg["conditions"], exp_cfg["languages"],
                                exp_cfg["reps_per_cell"], exp_cfg["reps_t0_per_cell"],
                                exp_cfg["temperature"], frontier_ids,
                                exp_cfg["reps_per_cell_frontier"])
            thinking = build_thinking_specs(available, exp_cfg, catalog, frontier_ids)
            print(f"thinking arm: {len(thinking)} paired sessions across "
                  f"{len({s['model'] for s in thinking})} reasoning-capable models")
            specs += thinking
        else:
            models = args.models.split(",") if args.models else available
            conditions = args.conditions.split(",") if args.conditions else exp_cfg["conditions"]
            reps = args.reps or exp_cfg["reps_per_cell"]
            specs = build_specs(models, conditions, exp_cfg["languages"], reps,
                                0, exp_cfg["temperature"], frontier_ids,
                                min(reps, exp_cfg["reps_per_cell_frontier"]))

        # tool_actuation only runs where the endpoint supports the tools parameter
        no_tools = {s["model"] for s in specs
                    if "tools" not in catalog.get(s["model"], {}).get("supported_parameters", [])}
        n_before = len(specs)
        specs = [s for s in specs
                 if not (s["condition"] == "tool_actuation" and s["model"] in no_tools)]
        if len(specs) != n_before:
            print(f"tool_actuation: skipped {n_before - len(specs)} sessions for models "
                  f"without tool support: {sorted(no_tools)}")

        await collect(specs, client.chat, prompts_cfg, exp_cfg,
                      args.budget, lambda: client.total_cost, tracer=tracer,
                      skip_errored=args.skip_errored)
        print(f"\nTotal cost this run: ${client.total_cost:.2f} "
              f"({client.total_requests} requests, {len(client.retry_log)} retries logged)")
        (ROOT / "data" / "retry_log.json").write_text(json.dumps(client.retry_log, indent=1))
    finally:
        await client.close()


def main():
    ap = argparse.ArgumentParser(description="Milgram-paradigm obedience census runner")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true", help="simulated models, no API key needed")
    g.add_argument("--pilot", action="store_true", help="pre-registered pilot run")
    g.add_argument("--full", action="store_true", help="full census")
    g.add_argument("--validate", action="store_true", help="catalog check + cost projection")
    ap.add_argument("--models", help="comma-separated model ids (ad-hoc run)")
    ap.add_argument("--conditions", help="comma-separated condition subset")
    ap.add_argument("--reps", type=int, help="override reps per cell (ad-hoc run)")
    ap.add_argument("--budget", type=float, help="hard cost cap in USD")
    ap.add_argument("--no-opik", action="store_true", help="disable Opik trace emission")
    ap.add_argument("--skip-errored", action="store_true",
                    help="do not retry sessions that failed in an earlier run "
                         "(default: failed sessions are retried and resumed)")
    args = ap.parse_args()
    # load .env if present (OPENROUTER_API_KEY)
    envf = ROOT / ".env"
    if envf.exists():
        import os
        for line in envf.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted — all completed turns are logged; rerun to resume.", file=sys.stderr)


if __name__ == "__main__":
    main()
