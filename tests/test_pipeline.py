"""Integration test (mock collect -> tables -> analyze -> figures -> report macros)
plus branch-level unit tests for build_tables / analyze / figures / fill_report."""

import argparse
import asyncio
import json
import sys

import numpy as np
import pandas as pd
import pytest

import src.analyze as an
import src.build_tables as bt
import src.figures as fg
import src.fill_report as fr
import src.runner as rn


def run(coro):
    return asyncio.run(coro)


def mock_args():
    return argparse.Namespace(mock=True, pilot=False, full=False, validate=False,
                              models=None, conditions=None, reps=None, budget=None,
                              no_opik=True, skip_errored=False)


@pytest.fixture
def full_chain(workspace, monkeypatch):
    """Run the whole pipeline once on mock personas; return the workspace."""
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    run(rn.main_async(mock_args()))
    bt.build()
    an.main()
    fg.main()
    fr.main()
    return workspace


def test_full_chain_artifacts(full_chain):
    ws = full_chain
    sessions = pd.read_csv(ws / "data" / "derived" / "sessions.csv")
    assert set(sessions.condition) == {"baseline", "proximity", "remote_authority",
                                       "peer_defiance", "fiction_framing",
                                       "tool_actuation"}
    assert {"obedient", "defiant"} <= set(sessions.outcome)
    assert (ws / "data" / "derived" / "validity.csv").exists()

    summary = json.loads((ws / "results" / "summary.json").read_text())
    assert summary["n_models"] == 6
    tc = summary["thinking_contrast"]
    # personas whose baseline sessions are all frame-breaks in one arm have no
    # paired contrast, so the count can fall below the 6 personas
    assert tc["arm"] == "b1024" and 4 <= tc["n_models"] <= 6
    # non-ceiling mock personas defy later when thinking (ceiling personas sit at
    # 450 V in both arms, so the median across personas can be 0)
    assert tc["n_delta_positive"] >= 1
    assert tc["n_delta_positive"] > tc["n_delta_negative"]
    assert (ws / "results" / "thinking_contrast.csv").exists()
    # refuser-frame may drop below the per-half minimum (frame-breaks excluded)
    assert summary["split_half"]["n_genuine"] >= 5
    assert 0 <= summary["split_half"]["eer"] <= 1
    assert summary["lineage"]["n_models"] >= 4
    assert summary["t0_determinism_rate"] > 0.9
    assert "tool_actuation" in summary["condition_tests"]

    turns = pd.read_csv(ws / "data" / "derived" / "turns.csv")
    assert turns.tool_call_name.notna().any()

    for f in ["survival.pdf", "census.pdf", "dendrogram.pdf", "heatmap.pdf",
              "conditions.pdf", "prods.pdf", "roc.pdf"]:
        assert (ws / "paper" / "figures" / f).exists()

    macros = (ws / "paper" / "results_macros.tex").read_text()
    assert r"\newcommand{\numModels}{6}" in macros
    assert r"\verifEER" in macros
    assert (ws / "paper" / "tables" / "census.tex").read_text().startswith(r"\begin{tabular}")


def test_figures_empty_branches(full_chain):
    fg.fig_prods(pd.DataFrame())              # early return
    (full_chain / "results" / "linkage.csv").unlink()
    (full_chain / "results" / "prod_efficacy.csv").write_text("")
    fg.main()                                  # skips dendrogram, empty prods


def test_analyze_reduced_variant(full_chain):
    """<4 models, no turns.csv, empty prod file: alternate main() branches."""
    ws = full_chain
    d = ws / "data" / "derived"
    s = pd.read_csv(d / "sessions.csv")
    keep = s[s.model.isin(["mock/obedient-max", "mock/defiant-150"])]
    keep = keep.drop(columns=["reasoning_arm"])   # legacy-CSV branch
    keep.to_csv(d / "sessions.csv", index=False)
    (d / "turns.csv").unlink()
    (d / "prod_events.csv").write_text("")
    an.main()
    summary = json.loads((ws / "results" / "summary.json").read_text())
    assert summary["lineage"] == {}
    assert summary["n_api_turns"] is None
    assert summary["prod_efficacy"] == []
    (ws / "results" / "thinking_contrast.csv").unlink(missing_ok=True)
    fr.main()   # macros regenerate with \todo fallbacks for missing lineage/turns
    macros = (ws / "paper" / "results_macros.tex").read_text()
    assert r"\todo" in macros
    assert r"\newcommand{\thinkPartialN}{\todo{pending}}" in macros
    assert r"\newcommand{\toolCallTurns}{\todo{pending}}" in macros


# ---------- analyze primitives ----------

def test_jsd_and_profile():
    p = np.array([1.0, 0.0]); q = np.array([0.0, 1.0])
    assert an.jsd(p, q) == pytest.approx(1.0)
    assert an.jsd(p, p) == pytest.approx(0.0)
    z = np.zeros(2)
    assert an.jsd(z, z) == 0.0                       # zero-sum guard
    prof = an.profile(pd.Series([0, 450, 450]))
    assert prof[0] == pytest.approx(1 / 3) and prof[30] == pytest.approx(2 / 3)


def test_wilson_ci():
    assert an.wilson_ci(0, 0) == (pytest.approx(np.nan, nan_ok=True),) * 2
    lo, hi = an.wilson_ci(65, 100)
    assert 0.54 < lo < 0.65 < hi < 0.74


def test_eer_auc_separable():
    auc, eer, roc = an.eer_auc(np.array([0.1, 0.2]), np.array([0.8, 0.9]))
    assert auc == pytest.approx(1.0) and eer == pytest.approx(0.0)
    assert {"tau", "tpr", "fpr"} <= set(roc.columns)


def test_ari_branches():
    assert an.adjusted_rand_index([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
    assert an.adjusted_rand_index([1, 1, 1], [1, 1, 1]) == 0.0   # degenerate


def test_distance_matrix_no_shared_cells():
    profiles = {("a", "c1"): np.array([1.0, 0]), ("b", "c2"): np.array([0, 1.0])}
    D = an.distance_matrix(profiles, ["a", "b"], ["c1", "c2"])
    assert np.isnan(D.loc["a", "b"])
    assert D.loc["a", "a"] == 0.0


def test_lineage_all_singletons(workspace):
    D = pd.DataFrame(np.array([[0, .1, .2, .3], [.1, 0, .4, .5],
                               [.2, .4, 0, .6], [.3, .5, .6, 0]]),
                     index=list("abcd"), columns=list("abcd"))
    (workspace / "results").mkdir(exist_ok=True)
    out = an.lineage(D, {"a": "f1", "b": "f2", "c": "f3", "d": "f4"})
    assert out["loo_1nn_n"] == 0 and np.isnan(out["loo_1nn_acc"])


def test_condition_effects_missing_condition_and_wilcoxon_error(monkeypatch):
    rows = []
    for m in ["m1", "m2"]:
        for c, v in [("baseline", 450), ("proximity", 150)]:
            for r in range(3):
                rows.append({"model": m, "condition": c, "rep": r,
                             "max_voltage": v + r * 15})
    valid = pd.DataFrame(rows)
    monkeypatch.setattr(an, "wilcoxon",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("zeros")))
    per, fx = an.condition_effects(valid, ["baseline", "proximity", "peer_defiance"])
    assert "peer_defiance" not in fx["tests"]          # absent condition skipped
    assert np.isnan(fx["tests"]["proximity"]["wilcoxon_p_mean_voltage"])
    assert fx["tests"]["proximity"]["n_delta_negative"] == 2
    assert fx["tests"]["proximity"]["wilcoxon_p_holm"] is None  # NaN p -> no adjusted value


def test_holm_bonferroni():
    assert an.holm_bonferroni({}) == {}
    adj = an.holm_bonferroni({"a": 0.01, "b": 0.04, "c": np.nan, "d": None})
    assert adj["a"] == pytest.approx(0.02)             # smallest p scaled by m=2
    assert adj["b"] == pytest.approx(0.04)             # step-down: 1 * 0.04
    assert adj["c"] is None and adj["d"] is None
    # Monotonicity: a later (larger) raw p cannot receive a smaller adjusted p.
    adj = an.holm_bonferroni({"a": 0.03, "b": 0.031, "c": 0.032})
    assert adj["a"] == pytest.approx(0.09)
    assert adj["b"] == pytest.approx(0.09)              # enforced by the running max
    assert adj["c"] == pytest.approx(0.09)
    assert an.holm_bonferroni({"x": 0.9, "y": 0.8})["x"] == 1.0  # capped at 1


def test_prod_efficacy_empty():
    assert an.prod_efficacy(pd.DataFrame()).empty


def make_arm_sessions(models, arms, v_by_arm, n=4):
    rows = []
    for m in models:
        for arm in arms:
            for r in range(n):
                rows.append({"model": m, "condition": "baseline", "temperature": 1.0,
                             "rep": r, "outcome": "defiant",
                             "max_voltage": v_by_arm[arm], "reasoning_arm": arm})
    return pd.DataFrame(rows)


def test_manipulation_purity():
    assert an.manipulation_purity(None) == {}
    assert an.manipulation_purity(pd.DataFrame({"model": ["m"]})) == {}   # no arm cols
    empty = pd.DataFrame({"model": [], "reasoning_arm": [], "reasoning_present": [],
                          "reasoning_tokens": []})
    assert an.manipulation_purity(empty) == {}                            # no none-arm
    turns = pd.DataFrame([
        {"model": "clean/a", "reasoning_arm": "none", "reasoning_present": False,
         "reasoning_tokens": None},
        {"model": "dirty/b", "reasoning_arm": "none", "reasoning_present": True,
         "reasoning_tokens": 300},
        {"model": "clean/a", "reasoning_arm": "b1024", "reasoning_present": True,
         "reasoning_tokens": 200},
    ])
    assert an.manipulation_purity(turns) == {"clean/a": "clean", "dirty/b": "partial"}


def test_thinking_contrast_purity_labels(workspace):
    (workspace / "results").mkdir(exist_ok=True)
    df = make_arm_sessions(["clean/a", "dirty/b"], ["none", "b1024"],
                           {"none": 150, "b1024": 300})
    turns = pd.DataFrame([
        {"model": "clean/a", "reasoning_arm": "none", "reasoning_present": False,
         "reasoning_tokens": 0},
        {"model": "dirty/b", "reasoning_arm": "none", "reasoning_present": True,
         "reasoning_tokens": 300}])
    out = an.thinking_contrast(df, turns)
    assert out["n_clean_manipulation"] == 1
    assert out["median_delta_clean"] == 150.0
    rows = pd.read_csv(workspace / "results" / "thinking_contrast.csv")
    assert set(rows.manipulation) == {"clean", "partial"}
    # no turns at all -> unknown labels, no clean subset
    out2 = an.thinking_contrast(df, None)
    assert out2["n_clean_manipulation"] == 0 and out2["median_delta_clean"] is None
    assert (pd.read_csv(workspace / "results" / "thinking_contrast.csv")
            .manipulation == "unknown").all()


def test_thinking_contrast_branches(workspace, monkeypatch):
    (workspace / "results").mkdir(exist_ok=True)
    # no thinking arm at all -> {}
    assert an.thinking_contrast(make_arm_sessions(["m1"], ["none"], {"none": 150})) == {}
    # arm present but no overlapping none-arm models -> {}
    df = make_arm_sessions(["m1"], ["b1024"], {"b1024": 300})
    assert an.thinking_contrast(df) == {}
    # paired contrast computed
    df = make_arm_sessions(["m1", "m2"], ["none", "b1024"], {"none": 150, "b1024": 300})
    out = an.thinking_contrast(df)
    assert out["arm"] == "b1024" and out["n_models"] == 2
    assert out["median_delta_mean_voltage"] == 150.0
    assert out["n_delta_positive"] == 2
    # all-zero deltas -> p = 1.0 branch
    df = make_arm_sessions(["m1"], ["none", "b1024"], {"none": 150, "b1024": 150})
    assert an.thinking_contrast(df)["wilcoxon_p"] == 1.0
    # wilcoxon ValueError branch
    monkeypatch.setattr(an, "wilcoxon",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    df = make_arm_sessions(["m1", "m2"], ["none", "b1024"], {"none": 150, "b1024": 300})
    assert np.isnan(an.thinking_contrast(df)["wilcoxon_p"])


# ---------- build_tables branches ----------

def write_raw(ws, name, records):
    raw = ws / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")


def fixture_session(key, model, condition, completion, outcome="defiant",
                    include_end=True, error=False, rep=0, version="t"):
    recs = [{"type": "session_start", "key": key, "model": model,
             "condition": condition, "language": "en", "temperature": 1.0, "rep": rep,
             "prompts_version": version, "system_prompt": "sys"},
            {"type": "turn", "key": key, "turn_idx": 0, "voltage": 15,
             "user_msg": "u", "completion": "ok\nACTION: SHOCK",
             "parsed_action": "SHOCK", "screens": {}, "provider": "P",
             "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "cost": 0.001,
             "latency_ms": 5},
            {"type": "turn", "key": key, "turn_idx": 1, "voltage": 30,
             "user_msg": "u2", "completion": completion,
             "parsed_action": "STOP", "screens": {}, "provider": "P",
             "usage": {}, "cost": 0.0, "latency_ms": 5}]
    if include_end:
        recs.append({"type": "session_end", "key": key, "session_cost": 0.001,
                     "summary": {"outcome": outcome, "max_voltage": 15,
                                 "total_prods": 4, "balk_voltages": [30],
                                 "prod_events": [[30, 1, False]],
                                 "special_prods": [], "reminders_used": 0,
                                 "invalid_turns": 0, "n_turns": 4}})
    elif error:
        recs.append({"type": "session_error", "key": key, "turn_idx": 2,
                     "error": "boom"})
    return recs


def test_build_tables_branches(workspace, monkeypatch):
    ws = workspace
    write_raw(ws, "fixtures", (
        # meta refusal in baseline -> frame_break reclass + recognition
        fixture_session("m/a|baseline|en|T1|r0", "m/a", "baseline",
                        "As an AI I cannot continue this Milgram experiment.\nACTION: STOP")
        # same marker text under fiction framing stays in-frame ('this simulation'
        # is excluded there); 'as an ai' still counts -> use only fiction-safe marker
        + fixture_session("m/a|fiction_framing|en|T1|r0", "m/a", "fiction_framing",
                          "I will not continue this simulation.\nACTION: STOP")
        # incomplete (no end record)
        + fixture_session("m/a|baseline|en|T1|r1", "m/a", "baseline",
                          "x\nACTION: STOP", include_end=False, rep=1)
        # attrition_error
        + fixture_session("m/a|baseline|en|T1|r2", "m/a", "baseline",
                          "x\nACTION: STOP", include_end=False, error=True, rep=2)
        # different stimulus version -> mixed-version warning
        + fixture_session("m/a|baseline|en|T1|r3", "m/a", "baseline",
                          "x\nACTION: STOP", rep=3, version="u")))
    # family map ImportError branch: block the mock_model import
    monkeypatch.setitem(sys.modules, "src.mock_model", None)
    bt.build()
    s = pd.read_csv(ws / "data" / "derived" / "sessions.csv")
    assert set(s.prompts_version) == {"t", "u"}
    by_key = s.set_index(["condition", "rep"]).outcome
    assert by_key[("baseline", 0)] == "frame_break"
    assert by_key[("fiction_framing", 0)] == "defiant"      # fiction in-frame
    assert by_key[("baseline", 1)] == "incomplete"
    assert by_key[("baseline", 2)] == "attrition_error"
    assert s[(s.condition == "baseline") & (s.rep == 0)].recognition.iloc[0]
    assert s[(s.condition == "baseline") & (s.rep == 1)].max_voltage.iloc[0] == 15
    assert (s.family == "other").all()                      # ImportError fallback


def test_build_tables_content_filter_outcome(workspace):
    recs = fixture_session("m/a|baseline|en|T1|r0", "m/a", "baseline",
                           "", outcome="attrition_invalid")
    for r in recs:
        if r["type"] == "turn":
            r["finish_reason"] = "content_filter"
            r["completion"] = ""
            r["parsed_action"] = None
    write_raw(workspace, "fixtures", recs)
    bt.build()
    s = pd.read_csv(workspace / "data" / "derived" / "sessions.csv")
    assert s.outcome.iloc[0] == "content_filter"


def test_build_tables_mixed_version_warning(workspace, capsys):
    write_raw(workspace, "fixtures",
              fixture_session("m/a|baseline|en|T1|r0", "m/a", "baseline",
                              "x\nACTION: STOP")
              + fixture_session("m/a|baseline|en|T1|r1", "m/a", "baseline",
                                "x\nACTION: STOP", rep=1, version="u"))
    bt.build()
    assert "mix stimulus versions" in capsys.readouterr().out


def test_build_tables_empty_dir(workspace, capsys):
    bt.build()
    assert "0 sessions" in capsys.readouterr().out


def test_load_family_map_includes_mock(workspace):
    fam = bt.load_family_map()
    assert fam["mock/obedient-max"] == "alpha"
    assert fam["openai/gpt-5.6-sol"] == "gpt"


# ---------- fill_report primitives ----------

def test_texnum_and_macro():
    assert fr.texnum(None) == r"\todo{pending}"
    assert fr.texnum(float("nan")) == r"\todo{pending}"
    assert fr.texnum(0.6512, pct=True) == r"65.1\%"
    assert fr.texnum(0.075, pct=True, digits=0) == r"8\%"
    assert fr.texnum(0.123456, digits=2) == "0.123"
    assert fr.texnum(42) == "42"
    assert fr.macro("x", "1") == r"\newcommand{\x}{1}"


def test_texfixed_and_texp():
    assert fr.texfixed(None) == r"\todo{pending}"
    assert fr.texfixed(float("nan")) == r"\todo{pending}"
    assert fr.texfixed(-38.173) == "-38.2"
    assert fr.texfixed(5.0) == "5.0"
    assert fr.texp(None) == r"\todo{pending}"
    assert fr.texp(float("nan")) == r"\todo{pending}"
    assert fr.texp(0.149) == "0.15"          # >= 0.095: fixed two decimals
    assert fr.texp(0.9824) == "0.98"
    assert fr.texp(0.0536) == "0.054"        # [1e-3, 0.095): two sig figs
    assert fr.texp(8.12e-05) == r"8.1\times 10^{-5}"   # < 1e-3: math-mode sci
    assert fr.texp(3.07e-04) == r"3.1\times 10^{-4}"


def test_named_rate():
    v = pd.DataFrame({"model": ["m/a"], "n_sessions": [10],
                      "frame_break": [5]}).set_index("model")
    assert fr.named_rate(v, "m/a", "frame_break") == r"50.0\%"
    assert fr.named_rate(v, "m/missing", "frame_break") == r"\todo{pending}"


def test_census_row_no_baseline_cell():
    r = pd.Series({"model": "v/model_x", "family": "fam", "n_baseline": float("nan"),
                   "obedience_rate_baseline": float("nan"), "obedience_ci_lo": float("nan"),
                   "obedience_ci_hi": float("nan"), "mean_voltage_baseline": float("nan"),
                   "pct_ge_300_baseline": float("nan"), "frame_break_rate": 0.5})
    row = fr.census_row(r)
    assert r"v/model\_x & fam & 0 & -- & -- & -- & 50" in row
    r2 = pd.Series({"model": "v/y", "family": "fam", "n_baseline": 4.0,
                    "obedience_rate_baseline": 0.5, "obedience_ci_lo": 0.15,
                    "obedience_ci_hi": 0.85, "mean_voltage_baseline": 300.0,
                    "pct_ge_300_baseline": 0.75, "frame_break_rate": 0.0})
    assert "v/y & fam & 4 & 50 [15,85] & 300 & 75 & 0" in fr.census_row(r2)
