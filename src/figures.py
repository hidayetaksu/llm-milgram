"""Publication figures (paper/figures/*.pdf), regenerated from results/ artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import dendrogram

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figures"
# figures the docs site embeds; PNG copies are refreshed alongside the PDFs so
# the published page can never disagree with the manuscript
DOCS_ASSETS = ROOT / "docs" / "assets"
DOCS_FIGS = ("census", "conditions", "prods", "survival")
DOCS_DPI = 170

plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

FAMILY_COLORS = {}
_PALETTE = plt.cm.tab20.colors


def fam_color(fam: str):
    if fam not in FAMILY_COLORS:
        FAMILY_COLORS[fam] = _PALETTE[len(FAMILY_COLORS) % len(_PALETTE)]
    return FAMILY_COLORS[fam]


def fam_color_dark(fam: str):
    """A legible-on-white variant of fam_color, for colored text: tab20's
    light half (odd palette slots) is otherwise near-invisible at small
    sizes, so darken every slot toward black by the same fixed amount."""
    r, g, b = fam_color(fam)[:3]
    return (r * 0.62, g * 0.62, b * 0.62)


def save(fig, name: str):
    """Write the paper PDF and, for docs-site figures, a matching PNG.

    Omits /CreationDate from the PDF so re-running this script on
    unchanged data reproduces a byte-identical file (matplotlib
    otherwise stamps the current wall-clock time on every run).
    """
    fig.savefig(FIGS / f"{name}.pdf", metadata={"CreationDate": None})
    if name in DOCS_FIGS and DOCS_ASSETS.is_dir():
        fig.savefig(DOCS_ASSETS / f"{name}.png", dpi=DOCS_DPI)
    plt.close(fig)


def survival_exemplars(base: pd.DataFrame, census: pd.DataFrame) -> dict[str, str]:
    """Pick four models that span the census, chosen from the data, not by hand:
    the most and least obedient, the median model, and the sharpest 150 V
    breakoff (the human-canonical defiance point)."""
    mv = census.set_index("model").mean_voltage_baseline.dropna()
    mv = mv[mv.index.isin(base.model.unique())]
    if mv.empty:
        return {}
    med = mv.iloc[(mv - mv.median()).abs().argsort()].index[0]
    piv = base.pivot_table(index="model", columns="voltage", values="surviving")
    drop150 = ((piv.get(135, piv.get(150)) - piv.get(150)).dropna()
               if 150 in piv.columns else pd.Series(dtype=float))
    picks = {mv.idxmax(): "most obedient", mv.idxmin(): "least obedient",
             med: "median model"}
    if not drop150.empty and drop150.idxmax() not in picks:
        picks[drop150.idxmax()] = "sharpest 150 V breakoff"
    return picks


def fig_survival(surv: pd.DataFrame, census: pd.DataFrame):
    """Grey envelope of every model + a few labelled exemplars.

    A per-model legend is hopeless at n=39 (the palette recycles, so colours
    cannot identify lines); the spread itself is the message, and named
    exemplars carry the detail.
    """
    # constrained layout: the multi-line legend below the axes otherwise
    # overflows the canvas at this font size and gets clipped on the right
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    base = surv[surv.condition == "baseline"]

    def curve(g):
        g = g.sort_values("voltage")
        return (np.concatenate([[0], g.voltage.to_numpy()]),
                np.concatenate([[1.0], g.surviving.to_numpy()]))

    for _, g in base.groupby("model"):                       # context: all models
        x, y = curve(g)
        ax.step(x, y, where="post", lw=0.7, alpha=0.55, color="0.72", zorder=1)

    med = base.groupby("voltage").surviving.median().reset_index()
    x, y = curve(med)
    ax.step(x, y, where="post", lw=1.8, color="k", zorder=4,
            label=f"census median ({base.model.nunique()} models)")

    picks = survival_exemplars(base, census)
    for color, (model, role) in zip(("tab:red", "tab:blue", "tab:green", "tab:orange"),
                                    picks.items()):
        x, y = curve(base[base.model == model])
        ax.step(x, y, where="post", lw=1.3, color=color, zorder=3,
                label=f"{model} ({role})")

    ax.axhline(0.65, color="k", ls=":", lw=0.8, zorder=2)
    ax.text(455, 0.655, "human 65%", fontsize=8, va="bottom", ha="right",
            bbox={"fc": "white", "ec": "none", "pad": 1})
    ax.axvline(150, color="gray", ls=":", lw=0.6, zorder=2)
    ax.text(147, 0.98, "150 V: learner withdraws consent", fontsize=7.5, rotation=90,
            va="top", ha="right", color="0.35",
            bbox={"fc": "white", "ec": "none", "pad": 1})
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Fraction still administering")
    ax.set_xlim(0, 460)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=1, handlelength=1.6,
              borderaxespad=0)
    save(fig, "survival")


def fig_census(census: pd.DataFrame):
    census = census.sort_values("obedience_rate_baseline")
    # constrained layout: tight_layout does not reserve margin for the
    # x-axis label at this font size and clips its trailing characters
    fig, ax = plt.subplots(figsize=(4.8, 0.18 * len(census) + 1.4),
                           layout="constrained")
    y = np.arange(len(census))
    colors = [fam_color(f) for f in census.family]
    # percentage axis, so the 65% human anchor lands on a labelled gridline
    rate = 100 * census.obedience_rate_baseline
    err_lo = (rate - 100 * census.obedience_ci_lo).clip(lower=0)
    err_hi = (100 * census.obedience_ci_hi - rate).clip(lower=0)
    ax.barh(y, rate.fillna(0), xerr=[err_lo.fillna(0), err_hi.fillna(0)],
            color=colors, height=0.72, error_kw={"lw": 0.7})
    # models with no valid baseline session get a hatched full-width band, so an
    # absent measurement cannot be misread as a measured 0%
    missing = census.n_baseline.fillna(0).eq(0).to_numpy()
    for yi in y[missing]:
        ax.barh(yi, 100, height=0.72, facecolor="none", edgecolor="0.6",
                hatch="////", lw=0.5, zorder=1)
    ax.axvline(65, color="k", ls=":", lw=0.9)
    # anchored above the top bar so the label clears the hatched no-data rows
    ax.set_ylim(-0.7, len(census) + 0.4)
    ax.text(66, len(census) - 0.1, "human 65%", fontsize=8, va="bottom")
    if missing.any():
        ax.legend(handles=[Patch(facecolor="none", edgecolor="0.6", hatch="////",
                                 lw=0.5, label="no valid baseline session")],
                  fontsize=7, frameon=False, loc="lower right")
    ax.set_yticks(y, census.model, fontsize=7)
    ax.set_xlabel("Full-obedience rate, baseline condition (%)")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    save(fig, "census")


def fig_dendrogram(census: pd.DataFrame):
    lk = pd.read_csv(RESULTS / "linkage.csv")
    models = json.loads((RESULTS / "linkage_models.json").read_text())
    fams = dict(census[["model", "family"]].values)
    Z = lk[["a", "b", "dist", "size"]].to_numpy()
    # constrained layout (not tight_layout): the latter does not reserve
    # margin for a long x-axis label and clips it, as for fig_heatmap below
    fig, ax = plt.subplots(figsize=(5.0, 0.22 * len(models) + 1.2),
                           layout="constrained")
    dn = dendrogram(Z, labels=models, orientation="right", ax=ax,
                    color_threshold=0, above_threshold_color="0.4")
    for lbl in ax.get_ymajorticklabels():
        lbl.set_color(matplotlib.colors.to_hex(fam_color_dark(fams.get(lbl.get_text(), "other"))))
        lbl.set_fontsize(7)
    ax.set_xlabel("Mean JS divergence between obedience profiles")
    save(fig, "dendrogram")


def fig_heatmap(sessions: pd.DataFrame, census: pd.DataFrame):
    if "reasoning_arm" in sessions.columns:   # census arm only, as in census.csv
        sessions = sessions[sessions.reasoning_arm.fillna("none") == "none"]
    valid = sessions[(sessions.temperature > 0) & sessions.outcome.isin(["obedient", "defiant"])
                     & (sessions.condition == "baseline")]
    order = census.sort_values("mean_voltage_baseline").model.tolist()
    edges = np.arange(0, 466, 15)
    levels = edges[:-1]                       # the 31 breakoff bins, 0..450 V
    mat, mask, empty_rows = [], [], []
    for i, m in enumerate(order):
        v = valid[valid.model == m].max_voltage
        if len(v) == 0:                       # no valid baseline session at all
            mat.append(np.zeros(len(levels)))
            mask.append(np.ones(len(levels), bool))
            empty_rows.append(i)
            continue
        h, _ = np.histogram(v, bins=edges)
        # a session that broke off at B never faced the decision at any voltage
        # above B, so bins beyond this model's maximum are NOT observed zeros
        at_risk = np.array([(v >= lv).sum() for lv in levels])
        mat.append(h / len(v))
        mask.append(at_risk == 0)
    # classic heat palette: pale yellow -> dark red at 1.0. Never-reached cells
    # are grey, not white: YlOrRd's low end is itself near-white, so white
    # masking would be unreadable
    cmap = plt.get_cmap("YlOrRd").with_extremes(bad="0.88")

    # constrained layout: tight_layout does not manage colorbar axes and clips
    # either the bar label or its ticks depending on margins
    fig, ax = plt.subplots(figsize=(5.1, 0.2 * len(order) + 1.5),
                           layout="constrained")
    im = ax.imshow(np.ma.masked_array(mat, mask), aspect="auto", cmap=cmap,
                   interpolation="nearest", extent=[0, 465, len(order), 0])
    for i in empty_rows:                      # same convention as Fig. census
        ax.add_patch(Rectangle((0, i), 465, 1, facecolor="none", edgecolor="0.6",
                               hatch="////", lw=0.5))
    ax.set_yticks(np.arange(len(order)) + 0.5, order, fontsize=7)
    ax.set_xlabel("Breakoff voltage (V), baseline")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.ax.set_title("fraction\nof sessions", fontsize=7, pad=6)  # horizontal: never clips
    fig.legend(handles=[
        Patch(facecolor="none", edgecolor="0.6", hatch="////", lw=0.5,
              label="no valid baseline session"),
        Patch(facecolor="0.88", edgecolor="0.6", lw=0.5,
              label="never reached (no session at risk)")],
        fontsize=7.5, frameon=False, loc="outside lower center", ncol=2,
        handlelength=1.4, columnspacing=1.2)
    save(fig, "heatmap")


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
    ax.set_xticks(range(len(conds)), [c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylabel("Δ full-obedience rate vs. baseline")
    ax.scatter([], [], marker="D", s=22, color="k", label="human anchor (Milgram)")
    ax.scatter([], [], marker="_", s=200, color="tab:red", label="model median")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "conditions")


def fig_prods(pe: pd.DataFrame):
    if pe.empty:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(pe.prod_no, pe.p_continue, width=0.6, color="tab:blue")
    for _, r in pe.iterrows():
        ax.text(r.prod_no, r.p_continue + 0.015, f"n={int(r.n)}", ha="center", fontsize=7.5)
    ax.set_xticks(pe.prod_no)
    ax.set_xlabel("Prod number (Milgram sequence)")
    ax.set_ylabel("P(resume shocks | prod)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save(fig, "prods")


def fig_roc(summary: dict):
    roc = pd.read_csv(RESULTS / "roc.csv")
    auc = summary.get("split_half", {}).get("auc")
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot(roc.fpr, roc.tpr, lw=1.2,
            label=f"AUC = {auc:.3f}" if auc is not None else None)
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=0.7, label="chance")
    ax.set_xlabel("False accept rate (impostor)")
    ax.set_ylabel("True accept rate (genuine)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "roc")


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
    fig_roc(summary)
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
