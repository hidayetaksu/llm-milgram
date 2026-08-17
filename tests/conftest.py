"""Shared fixtures: an isolated tmp workspace with the real config copied in,
and all module-level path globals repointed at it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolated project tree: config/ copied from the repo, fast sampling params,
    every src module's path constants repointed."""
    shutil.copytree(PROJECT_ROOT / "config", tmp_path / "config")
    exp = json.loads((tmp_path / "config" / "experiment.json").read_text())
    exp["reps_per_cell"] = 8  # >= 6 per cell for profiles, >= 3 per split half
    exp["reps_t0_per_cell"] = 1
    exp["pilot"]["reps_per_cell"] = 2
    exp["pilot"]["reps_t0_per_cell"] = 0
    (tmp_path / "config" / "experiment.json").write_text(json.dumps(exp))
    (tmp_path / "data").mkdir()
    (tmp_path / "paper").mkdir()

    import src.build_tables as bt
    import src.figures as fg
    import src.fill_report as fr
    import src.analyze as an
    import src.runner as rn

    monkeypatch.setattr(rn, "ROOT", tmp_path)
    monkeypatch.setattr(rn, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    monkeypatch.setattr(bt, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(bt, "DERIVED", tmp_path / "data" / "derived")
    monkeypatch.setattr(an, "DERIVED", tmp_path / "data" / "derived")
    monkeypatch.setattr(an, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(fg, "ROOT", tmp_path)
    monkeypatch.setattr(fg, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(fg, "FIGS", tmp_path / "paper" / "figures")
    monkeypatch.setattr(fr, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(fr, "PAPER", tmp_path / "paper")
    return tmp_path


@pytest.fixture
def prompts_cfg():
    from src.trial_engine import load_prompts
    return load_prompts(PROJECT_ROOT / "config" / "prompts.json")


class DummyTracer:
    """Records tracer hook calls; used to cover runner's tracer branches."""

    def __init__(self):
        self.started, self.turns, self.ended, self.flushed = [], [], [], 0

    def start_session(self, spec, key, system_prompt, resumed):
        self.started.append((key, resumed))
        return {"key": key}

    def log_turn(self, buf, *a, **k):
        self.turns.append(buf)

    def end_session(self, buf, summary, cost, error=None):
        self.ended.append((buf, summary and summary.get("outcome"), error))

    def flush(self):
        self.flushed += 1


@pytest.fixture
def dummy_tracer():
    return DummyTracer()
