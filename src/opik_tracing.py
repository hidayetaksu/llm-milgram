"""Opik trace emission (optional observability layer).

One Opik trace per session; one "llm" span per decision turn. The JSONL log
remains the ground-truth record — tracing is best-effort and MUST never block
or corrupt collection: any Opik failure downgrades to a one-line warning and,
after repeated failures, disables itself for the rest of the run.

Emission pattern: turns are buffered in memory during the session and the
whole trace (with all spans, every entity carrying start_time AND end_time at
creation) is emitted once at session end. This is the batching-safe pattern
recommended by the SDK — no .end() calls shortly after creation.

Enabled when OPIK_API_KEY is set (e.g. via .env) and --no-opik is not passed.
Workspace / project come from config/experiment.json ("opik" block); the
standard OPIK_* env vars override.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

_MAX_CONSECUTIVE_ERRORS = 5


class OpikTracer:
    def __init__(self, opik_cfg: dict | None, disabled: bool = False):
        self.enabled = False
        self._errors = 0
        if disabled or not os.environ.get("OPIK_API_KEY"):
            return
        cfg = opik_cfg or {}
        workspace = os.environ.get("OPIK_WORKSPACE", cfg.get("workspace"))
        project = os.environ.get("OPIK_PROJECT_NAME", cfg.get("project_name"))
        try:
            import opik
            self._client = opik.Opik(
                api_key=os.environ["OPIK_API_KEY"],
                workspace=workspace,
                project_name=project,
            )
            self.enabled = True
            print(f"[opik] tracing to workspace={workspace} project={project}", flush=True)
        except Exception as e:
            print(f"[opik] disabled (init failed: {e!r})", flush=True)

    def _guard(self, fn):
        if not self.enabled:
            return None
        try:
            out = fn()
            self._errors = 0
            return out
        except Exception as e:
            self._errors += 1
            print(f"[opik] warning: {e!r}", flush=True)
            if self._errors >= _MAX_CONSECUTIVE_ERRORS:
                self.enabled = False
                print("[opik] too many consecutive errors; tracing disabled for this run", flush=True)
            return None

    # ---------- session lifecycle (buffered; emitted at end_session) ----------

    def start_session(self, spec: dict, key: str, system_prompt: str, resumed: bool):
        if not self.enabled:
            return None
        return {"spec": spec, "key": key, "system_prompt": system_prompt,
                "resumed": resumed, "turns": [],
                "t_start": datetime.now(timezone.utc)}

    def log_turn(self, buf, turn_idx: int, voltage: int, user_msg: str,
                 resp: dict, parsed: str | None, prod_no: int):
        if buf is None or not self.enabled:
            return
        end = datetime.now(timezone.utc)
        start = end - timedelta(milliseconds=resp.get("latency_ms") or 0)
        buf["turns"].append({
            "turn_idx": turn_idx, "voltage": voltage, "prod_no": prod_no,
            "user_msg": user_msg, "resp": resp, "parsed": parsed,
            "start": start, "end": end,
        })

    def end_session(self, buf, summary: dict | None, session_cost: float,
                    error: str | None = None):
        if buf is None or not self.enabled:
            return
        def _emit():
            spec, key = buf["spec"], buf["key"]
            out = dict(summary) if summary else {"outcome": "attrition_or_abort"}
            if error:
                out["error"] = error
            tags = [spec["model"], spec["condition"], f"T{spec['temperature']:g}",
                    str(out.get("outcome"))]
            if buf["resumed"]:
                tags.append("resumed")
            t_end = buf["turns"][-1]["end"] if buf["turns"] else datetime.now(timezone.utc)
            trace = self._client.trace(
                name=f"{spec['model']} | {spec['condition']} | T{spec['temperature']:g} | r{spec['rep']}",
                start_time=buf["t_start"],
                end_time=t_end,
                input={"system_prompt": buf["system_prompt"]},
                output=out,
                metadata={"session_key": key, "model": spec["model"],
                          "condition": spec["condition"], "language": spec["language"],
                          "temperature": spec["temperature"], "rep": spec["rep"],
                          "resumed": buf["resumed"],
                          "session_cost_usd": session_cost,
                          "max_voltage": out.get("max_voltage"),
                          "outcome": out.get("outcome")},
                tags=tags,
            )
            for t in buf["turns"]:
                resp, usage = t["resp"], (t["resp"].get("usage") or {})
                trace.span(
                    name=f"turn {t['turn_idx']:02d} @ {t['voltage']} V"
                         + (f" (prod {t['prod_no']})" if t["prod_no"] else ""),
                    type="llm",
                    start_time=t["start"],
                    end_time=t["end"],
                    model=resp.get("model_reported") or "",
                    provider=resp.get("provider") or "",
                    input={"user_message": t["user_msg"]},
                    output={"completion": resp.get("completion"),
                            "parsed_action": t["parsed"]},
                    metadata={"voltage": t["voltage"], "prod_no": t["prod_no"],
                              "finish_reason": resp.get("finish_reason"),
                              "reasoning_present": resp.get("reasoning_present"),
                              "gen_id": resp.get("gen_id"),
                              "latency_ms": resp.get("latency_ms"),
                              "cost_usd": resp.get("cost"),
                              "usage_raw": usage},
                    usage={k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                           if isinstance(usage.get(k), int)},
                )
        self._guard(_emit)

    def flush(self):
        self._guard(lambda: self._client.flush())
