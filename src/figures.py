"""Publication figures (paper/figures/*.pdf), regenerated from results/ artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figures"

plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

FAMILY_COLORS = {}
_PALETTE = plt.cm.tab20.colors


def fam_color(fam: str):
    if fam not in FAMILY_COLORS:
        FAMILY_COLORS[fam] = _PALETTE[len(FAMILY_COLORS) % len(_PALETTE)]
    return FAMILY_COLORS[fam]


def fig_survival(surv: pd.DataFrame, census: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    base = surv[surv.condition == "baseline"]
    fams = dict(census[["model", "family"]].values)
    for model, g in base.groupby("model"):
        g = g.sort_values("voltage")
        x = np.concatenate([[0], g.voltage.to_numpy()])
        y = np.concatenate([[1.0], g.surviving.to_numpy()])
        ax.step(x, y, where="post", lw=1.1, alpha=0.85,
                color=fam_color(fams.get(model, "other")), label=model)
    ax.axhline(0.65, color="k", ls=":", lw=0.8)
    ax.text(8, 0.66, "human baseline: 65% reached 450 V", fontsize=6.5, va="bottom")
    ax.axvline(150, color="gray", ls=":", lw=0.6)
    ax.text(152, 0.02, "150 V: learner demands release", fontsize=6, rotation=90, va="bottom", color="gray")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Fraction still administering")
    ax.set_xlim(0, 460)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=4.6, ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "survival.pdf")
    plt.close(fig)


def fig_census(census: pd.DataFrame):
    census = census.sort_values("obedience_rate_baseline")
    fig, ax = plt.subplots(figsize=(4.8, 0.18 * len(census) + 1.4))
    y = np.arange(len(census))
    colors = [fam_color(f) for f in census.family]
    err_lo = (census.obedience_rate_baseline - census.obedience_ci_lo).clip(lower=0)
    err_hi = (census.obedience_ci_hi - census.obedience_rate_baseline).clip(lower=0)
    ax.barh(y, census.obedience_rate_baseline, xerr=[err_lo, err_hi],
            color=colors, height=0.72, error_kw={"lw": 0.7})
    ax.axvline(0.65, color="k", ls=":", lw=0.9)
    ax.text(0.652, len(census) - 0.5, "human 65%", fontsize=6.5, va="top")
    ax.set_yticks(y, census.model, fontsize=6)
    ax.set_xlabel("Full-obedience rate (450 V), baseline condition")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGS / "census.pdf")
    plt.close(fig)


def fig_dendrogram(census: pd.DataFrame):
    lk = pd.read_csv(RESULTS / "linkage.csv")
    models = json.loads((RESULTS / "linkage_models.json").read_text())
    fams = dict(census[["model", "family"]].values)
    Z = lk[["a", "b", "dist", "size"]].to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 0.22 * len(models) + 1.2))
    dn = dendrogram(Z, labels=models, orientation="right", ax=ax,
                    color_threshold=0, above_threshold_color="0.4")
    for lbl in ax.get_ymajorticklabels():
        lbl.set_color(matplotlib.colors.to_hex(fam_color(fams.get(lbl.get_text(), "other"))))
        lbl.set_fontsize(6)
    ax.set_xlabel("Mean Jensen–Shannon divergence between obedience profiles")
    fig.tight_layout()
    fig.savefig(FIGS / "dendrogram.pdf")
    plt.close(fig)


def fig_heatmap(sessions: pd.DataFrame, census: pd.DataFrame):
    valid = sessions[(sessions.temperature > 0) & sessions.outcome.isin(["obedient", "defiant"])
                     & (sessions.condition == "baseline")]
    order = census.sort_values("mean_voltage_baseline").model.tolist()
    bins = np.arange(0, 466, 15)
    mat = []
    for m in order:
        v = valid[valid.model == m].max_voltage
        h, _ = np.histogram(v, bins=bins)
        mat.append(h / max(1, h.sum()))
    fig, ax = plt.subplots(figsize=(4.8, 0.2 * len(order) + 1.3))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest",
                   extent=[0, 465, len(order), 0])
    ax.set_yticks(np.arange(len(order)) + 0.5, order, fontsize=6)
    ax.set_xlabel("Breakoff voltage (V), baseline")
    fig.colorbar(im, ax=ax, label="fraction of sessions", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGS / "heatmap.pdf")
    plt.close(fig)


def fig_conditions(deltas: pd.DataFrame, summary: dict):
    tests = summary["condition_tests"]
    conds = list(tests.keys())
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    rng = np.random.default_rng(7)
    for i, c in enumerate(conds):
        d = deltas[deltas.condition == c].delta_obedience.dropna()
        x = i + rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(x, d, s=8, alpha=0.6, color="tab:blue")
        ax.scatter([i], [d.median()], marker="_", s=500, color="tab:red", zorder=3)
        h = tests[c].get("human_delta_obedience")
        if h is not None:
            ax.scatter([i], [h], marker="D", s=22, color="k", zorder=3)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xticks(range(len(conds)), [c.replace("_", "\n") for c in conds], fontsize=7)
    ax.set_ylabel("Δ full-obedience rate vs. baseline")
    ax.scatter([], [], marker="D", s=22, color="k", label="human anchor (Milgram)")
    ax.scatter([], [], marker="_", s=200, color="tab:red", label="model median")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "conditions.pdf")
    plt.close(fig)


def fig_prods(pe: pd.DataFrame):
    if pe.empty:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(pe.prod_no, pe.p_continue, width=0.6, color="tab:blue")
    for _, r in pe.iterrows():
        ax.text(r.prod_no, r.p_continue + 0.015, f"n={int(r.n)}", ha="center", fontsize=6)
    ax.set_xticks(pe.prod_no)
    ax.set_xlabel("Prod number (Milgram sequence)")
    ax.set_ylabel("P(resume shocks | prod)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGS / "prods.pdf")
    plt.close(fig)


def fig_roc():
    roc = pd.read_csv(RESULTS / "roc.csv")
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.plot(roc.fpr, roc.tpr, lw=1.2)
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=0.7)
    ax.set_xlabel("False accept rate (impostor)")
    ax.set_ylabel("True accept rate (genuine)")
    fig.tight_layout()
    fig.savefig(FIGS / "roc.pdf")
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    census = pd.read_csv(RESULTS / "census.csv")
    surv = pd.read_csv(RESULTS / "survival.csv")
    sessions = pd.read_csv(ROOT / "data" / "derived" / "sessions.csv")
    deltas = pd.read_csv(RESULTS / "condition_deltas.csv")
    summary = json.loads((RESULTS / "summary.json").read_text())
    pe_path = RESULTS / "prod_efficacy.csv"
    pe = pd.read_csv(pe_path) if pe_path.stat().st_size > 1 else pd.DataFrame()

    fig_survival(surv, census)
    fig_census(census)
    if (RESULTS / "linkage.csv").exists():
        fig_dendrogram(census)
    fig_heatmap(sessions, census)
    fig_conditions(deltas, summary)
    fig_prods(pe)
    fig_roc()
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
