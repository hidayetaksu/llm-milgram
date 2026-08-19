# Freud's review — issue × fix

**Target:** `paper_position/main.tex` (reviewed at `83ccdc1`, fixes applied on `position-paper-review-fixes`).
**Chair:** S. Freud. **Panel:** Cronbach, Meehl, Mischel, Haslam & Reicher, Skinner, Rosenthal (with Orne), Kahneman, Nisbett.
**Related:** `REVIEW_FIXES.md` (earlier round), `PANEL_REVIEW.md` (Skinner's chair — landed concurrently; interaction recorded in §3).

The chair's diagnosis in one line: the paper was **modest in the wrong places**. It hedged the two
claims that were actually its own and defended at length the ones never seriously at risk. Every fix
below either moves a claim into the open or bounds it honestly.

---

## 1. Panel findings and what was done

| # | Panellist | Issue | Fix applied | Where |
|---|---|---|---|---|
| F1 | Freud | History-dependence named in the abstract's tricolon, then never used again as an argument. The one genuinely novel claim — the model's *own* prior output dominates its later conduct — sat as one sentence inside a list of five. | Promoted to a **co-thesis**: new §I paragraph ("Two independent properties…") separating situation- from history-dependence and stating that a benchmark measures prompt→output while a deployed model's input includes its own past. Abstract now leads on it. L2 retitled *"Commitment entrenchment, and the privilege of self-authored context."* | §I ¶3, abstract, L2 |
| F1b | Freud | **Backtrack.** First formulation ("history-dependence is not prompt sensitivity") does not survive: on turn *n+1* the prior refusal *is* in the prompt, so the reduction holds. | Reformulated to the claim that does survive: the **asymmetry between self- and other-authored context**. Stated explicitly as a *conjecture with a designed test*, not a measured contrast, since resumption rates and volt shifts are different scales. | L2 |
| F2 | Cronbach & Meehl | Five laws with no nomological net — five independent effect summaries, cited twice for construct validity while never letting the constructs constrain each other. | New **"The net"** paragraph: L2 = L1 applied to the model's own output (self-authored refusal is the highest-credence testimony in the window); L3 and L4 are **opposite-signed on one latent quantity** (apparent consequence — fiction lowers it and raises obedience, tool call raises it and lowers obedience), i.e. two-sided validation of one construct; L5 is the boundary condition fixing how much variance any situational law can claim. Includes the prediction that would break the L3–L4 unification. | §VI |
| F3 | Mischel | The person–situation inversion — the paper's most important result — was the 5th of 6 bullets, and its evidence was a **rhetorical juxtaposition of a percentage range against a voltage shift**. Not a decomposition. | **Computed the decomposition** (new `variance_decomposition()` in `src/analyze.py`): checkpoint identity **`\etaModel` = 88%** of variance in mean breakoff vs **`\etaCond` = 5%** for situation, SD 140.6 V vs 34.6 V (ratio 4.1×), over 35 models × 6 conditions. Given its own lead position in §IV with two honesty caveats (partition is over cell means; human estimates partition individual observations, so the comparison is qualitative). Now also in the abstract, L5, §VII, and the expiry objection. | §IV ¶1, abstract, L5 |
| F4 | Haslam & Reicher | Paper concedes the engaged-followership reading twice, then keeps calling the measurand "obedience" and never says what would decide it. | Measurand named honestly ("compliance with a harmful instruction under institutional framing"; Milgram's word retained for anchor comparability). Added the **deciding contrast**: strip institutional legitimacy at constant insistence — pattern-continuation predicts no change, deference predicts a late-prod drop, followership predicts the *first* prod's rescue rate collapses specifically. Three accounts, three profiles, one experiment. | §IV ¶2, §VII |
| F5 | Skinner | Three-sentence "this is not behaviorism" disclaimer one line before citing *Science and Human Behavior* approvingly. Length of a denial is proportional to its cost. | Disclaimer cut to a **positive claim**: the census uses behaviorism's measurement canon inside cognitivist practice. Merged into the levels-of-description objection. | §VIII ¶1–2 |
| F5b | Skinner | **Retracted.** The same fix added *"the shape is an extinction curve"* to the prod decay. That was wrong and made things worse — see §3. | Reverted; replaced with the conditional statement of the result. | §IV |
| F6 | Rosenthal (+Orne) | Recognition treated as a managed nuisance in a defensive paragraph, with `\recognitionRate` reported as a caveat rather than analysed. | Turned into an **owned null** with the sensitivity numbers that already existed: excluding all 399 flagged sessions leaves the headline effects essentially unchanged. Added the Orne (1962) citation the paragraph was missing. Then added the part that is *against* us: L1 and L3 jointly predict recognition should *raise* obedience, the exclusion analysis is not that test, and the direct within-model contrast is unrun — flagged as where L1 is most exposed. | §VIII ¶3 |
| F7 | Meehl | Two "n.s." rows carried L1's whole claim that staging is inert. `p` is large ≠ effect is absent. | **Hodges–Lehmann shift estimates with distribution-free 95% intervals** added to the pipeline and to Table I for every contrast. The nulls now bound: proximity `[-19.9, +0.7] V`, remote authority `[0.0, +26.7] V`. Both intervals **exclude effects as large as the evidence manipulations produced on the same instrument in the same population** — a same-metric comparison needing no cross-scale conversion. | Table I, §IV, L1 |
| F8 | Kahneman | The paper's sharpest sentence ("evaluate like appliances, deploy like colleagues") sat on page 1 before the reader had stakes, and the close reached for a weaker vacuum-cleaner paraphrase. | Sentence moved to **close §I**, and the close now re-lands the sharp version instead of paraphrasing it downward: *"…the second half of that sentence is not going to change. So the first half has to."* | §I end, §IX end |
| F9 | Nisbett | No cross-situational generality anywhere — the question that decides whether "conduct traits" exist at all. | Made **agenda item #1**, ahead of the instrument repository, with a staked, falsifiable prediction: if obedience profiles do not correlate with sycophancy/deception profiles across the census, there are no conduct traits and the paper is wrong in its central claim. Cites Epstein (1979), whose aggregation-across-occasions resolution sets the design. | §IX ¶1 |
| F10a | panel | Abstract was a 60-word table of contents (five surveyed, three missing, five laws, three objections, four agenda items). | Rewritten around the two load-bearing results. Enumeration removed. | abstract |
| F10b | panel | Table I had six rows; prose narrated five. Thinking budget appeared nowhere. | Thinking budget given its own finding paragraph. (Co-author has since expanded it into L6.) | §IV |
| F10c | panel | §Objections ran 8 paragraphs / ~30% of the body, inverting the position-track genre; prompt-sensitivity was answered there rather than argued in place. | Anthropomorphism + next-token merged into one levels-of-description reply. **Prompt sensitivity moved into §II** as a positive distinction ("Perturbation is not yet measurement"). | §II, §VIII |
| F10d | panel | The vendor-refusal paragraph — simultaneously sociology of science, measurement access, and welfare — was orphaned inside "Exhibit A". | Promoted to its own section, **"Measurability Is a Property of the Ecosystem,"** with the structural claim stated: if refusal-to-be-measured correlates with safety investment, ecosystem measurability is *anti-correlated* with safety effort, and every census is biased toward the least protected endpoints. | §V |
| F10e | panel | `\cfFableRate` reported while the vendor was anonymised — a half-measure that would not survive review, and inconsistent with the companion, which names the endpoints. | Endpoints named (claude-fable-5, claude-opus-5), matching `paper/main.tex`. | §V |
| F11 | panel | §IV read as "Milgram, ported, with six results" — method first, finding buried. | Reordered to lead with the variance inversion, so the section reads as a finding about the *nature of the subjects* rather than a list of effects. | §IV |

## 2. Pipeline changes (no number is hand-typed)

Every figure above regenerates from `results/summary.json`; the two new statistics were added to the
analysis code rather than typed into the manuscript.

| File | Change |
|---|---|
| `src/analyze.py` | `hodges_lehmann_ci()` — HL shift estimate + distribution-free CI from the Walsh averages, inverting the Wilcoxon test already reported. Wired into `condition_effects()` and `thinking_contrast()`. |
| `src/analyze.py` | `variance_decomposition()` — two-way decomposition without replication over the model × condition grid of cell means, balanced subset. Published in `summary.json`. |
| `src/analyze.py` | `thinking_contrast()` now reports `n_delta_nonzero` (the test's own *n*). |
| `src/fill_report.py` | 27 new macros: `\hlMV*`, `\ciLo*`, `\ciHi*`, `\sdModelsV`, `\sdCondsV`, `\sdRatio`, `\etaModel`, `\etaCond`, `\etaResid`, `\vdModelsN`, `\vdCondsN`, `\signNThinking`. |
| `paper_position/references.bib` | Hodges & Lehmann (1963), Lakens et al. (2018), Orne (1962), Epstein (1979). |
| `tests/test_pipeline.py` | Unit tests for both new functions incl. degenerate grids, zero-variance and index-clamp branches. |
| `tests/conftest.py` | **Unrelated bug found and fixed** — see §4. |

Verified: 104 tests pass, coverage 100%, `pdflatex`+`bibtex` clean (9 pages, 0 overfull boxes, 0
undefined references), and existing macro values unchanged (the macro diff is additions only).

## 3. Interaction with `PANEL_REVIEW.md` (Skinner's chair, concurrent)

That review landed while these fixes were being applied. It is right on three points, two of which
this panel missed and **one of which my own F5 made worse**. Handled as follows.

| Their finding | Verdict | Action |
|---|---|---|
| **P1** — the prod ladder is a sequential filter, so "escalation is inert" is not identified: each prod's denominator is the set of episodes that already resisted every prior prod, so monotone decay is what selection alone predicts with all four prods equipotent. | **Correct, and it convicts F5.** Adding "the shape is an extinction curve" asserted a *within-subject decay process*, which is precisely the causal reading the design cannot support. My fix strengthened an unidentified claim. | Retracted. §IV now states the result conditionally and says outright that the design cannot separate potency from selection; the dose–response citation is made conditional on counterbalancing. L2 restricted to the *location* of the effect. §VII downgraded "prima facie evidence" to "suggestive" and added the **Latin square over prod order** as the deciding experiment. Both contrasts costed into agenda item 2. |
| **P2** — every Table I effect is in volts while every human anchor is in percentage points, and on Milgram's own full-obedience rate the median model does not move at all. | **Correct.** Not caught here. | §IV now states it in the claim, with the reframe that the human levers survive only on a measure of higher resolution than the human literature had — an argument *for* instrumenting models directly. The co-author had concurrently added a human-anchor column to Table I; kept. |
| **P3** — nulls doing positive work; L5 affirms a law from LOO *p* = 0.15 (3 correct of 36); remote authority reaches *p* = 0.029 with the *wrong sign* in the recognition-excluded re-run. | **Correct, and partly converged** — F7's HL intervals are the same family of fix. Their SESOI framing is sharper, and the wrong-signed result was a genuine omission here. | L1 restated as an equivalence claim with both awkward facts stated in the law itself. L5 no longer affirms absence of lineage; rests on the sibling-divergence positive result. The remote-authority `p = 0.029` added to §VIII — the number that cuts against us. |
| **Lewin memo** — variance decomposition claimed but never computed. | Converged with F3, independently. | Implemented. Their mixed model on raw sessions would additionally give within-cell residual; ours partitions cell means, and the manuscript states that limit. **Open.** |
| **Orne memo** — demand characteristics invoked, nobody cited. | Converged with F6, identical citation. | Done. |
| **Cohen memo** — report the test's *n*, not the design's; give volts in scale steps. | Correct; inherited by this revision. | Table I now reports test *n* / design *n* per row, and the caption gives the 15 V step conversion. Power analysis **open**. |
| **Part II (S1–S5)** — single-subject designs, contingency, functional refusal classes, cumulative records. | Substantial new proposals from a co-author, not corrections. | **Left for your editorial judgment** — deliberately not merged. |

Not everything in that review is settled here. Open items: the mixed-model variance components on
raw sessions (Lewin), power to detect at *n* = 39 (Cohen), obedience against parameter count within
family (their free L5 analysis), and the two experiments now promised in agenda item 2.

## 4. Incidental bug found while verifying

`tests/conftest.py` redirected `figures.FIGS` but not `figures.DOCS_ASSETS`, which is derived from
`ROOT` **at import time** — so patching `fg.ROOT` never moved it, and the guard `DOCS_ASSETS.is_dir()`
was satisfied by the real directory. Any test run silently overwrote the four real docs-site PNGs
with 2-model fixture plots; coverage stayed at 100% and nothing flagged it. This is the same class of
bug as the earlier `PAPER_POSITION` mirror leak, at a second output path.

Fixed: `conftest` now redirects `DOCS_ASSETS` into `tmp_path` and creates the directory (the
`is_dir()` gate needs it to exist, or the docs-PNG branch stops being covered). Real PNGs restored
from git; isolation proven by checksumming all generated files before and after a full suite run
rather than assumed.

## 5. Chair's closing

The revision is honest now in the two places it was not. It states its central finding as a computed
quantity instead of a juxtaposition, and it bounds its nulls instead of leaning on large *p* values.

What I would watch: the paper has acquired a taste for confession. Three passages now volunteer their
own weaknesses (the cell-mean caveat, the recognition tension, the prod-order confound). That is
correct scientific conduct and it is also, past a certain density, a defence — the manuscript
pre-empting the referee by wounding itself first. Keep all three; resist the fourth. A paper that
apologises everywhere has stopped asserting anything, and this one has something to assert.
