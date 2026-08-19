# Position paper — review fixes applied

Review date: 2026-08-18. Target: `paper_position/main.tex` ("Language Models Need a Psychology").
Build verified: 6 pages, zero LaTeX errors, zero undefined citations, no overfull boxes >20pt.
Companion (`paper/main.tex`) rebuilt clean at 11 pages after its cross-reference was updated.

---

## Important context discovered mid-task

**The working tree was stale at session start and refreshed to HEAD partway through.**
`paper_position/main.tex`, `results/summary.json`, and `paper/results_macros.tex` were all
older snapshots when I first read them, and later matched HEAD. Consequences:

- My first-pass review quoted numbers from the **stale** `results_macros.tex`
  (`verifAUCW` 0.949, `costPerSessionCents` 3.1, total cost \$151.95).
  The true HEAD values are **0.916**, **3.4**, and **\$166.81**.
- This means the paper's hardcoded `0.949` was **also stale, and my review missed it.**
  The macro plumbing (fix #1) caught it silently. This is the single best argument for
  fix #1 and is why no census number is hand-typed in this paper any more.
- Verify against a freshly regenerated `results_macros.tex`, never a working copy you
  have not just rebuilt: `.venv/bin/python src/fill_report.py`.

---

## Issue × fix table

### Tier 1 — correctness and credibility (must-fix before posting)

| # | Sev | Location | Issue | Fix applied | Status |
|---|-----|----------|-------|-------------|--------|
| 1 | **critical** | whole paper | Census numbers hardcoded, so the position paper could (and did) drift from the macro-driven companion. Two papers posted together would disagree on headline statistics. | Root cause fixed. `src/fill_report.py` now mirrors `results_macros.tex` into `paper_position/`, and the paper does `\input{results_macros}`. Every census figure is now generated. arXiv source tree stays self-contained. | ✅ |
| 2 | **critical** | §IV "Pressure has structure" | Stale prod efficacy `30.5% / 3.6% / 1.7% / 0.4%`. | Now macro-driven: **31.5% / 3.7% / 1.9% / 0.4%**. | ✅ |
| 3 | **critical** | §IV "The measurements are reliable" | Stale ordinal-aware AUC `0.949`. **Missed by the review**; caught only by fix #1. | Now macro-driven: **0.916**. | ✅ |
| 4 | major | §IV, §VI, §VIII | Stale per-session cost (`a few cents`, `cents per session`). | Now macro-driven: **3.4 cents per session**, stated identically in all three places. | ✅ |
| 5 | major | §IV "Situationism transfers, selectively" | "removing the authority's physical presence … **does nothing**" misreported the result. Actual: median **+5.0 V**, *p* = 0.054 (Holm 0.11), 22 of 33 movers going *up* — a n.s. trend **opposite** the human effect, which the companion states correctly. | Reworded to "fails to register at all, and its point estimate even trends *opposite* the human effect (+5.0 V, *p* = 0.054, n.s.)". Now consistent with the companion abstract. | ✅ |
| 6 | major | §IV, same paragraph | "for a model the situation *is* the token stream and **staging changes no tokens that matter**" — circular and unfalsifiable. The remote condition *does* change tokens; "no tokens that matter" was defined post hoc by the null it explained. | Phrase deleted. Replaced by the evidence-vs-staging distinction (which is also the companion's own language) and stated as a falsifiable law (L1). | ✅ |
| 7 | major | §IV "Stated and enacted compliance dissociate" | "drops mean breakoff by 53 V" — the statistic is the *median across models of the per-model change in mean breakoff*. | "shifts mean breakoff by a **median of −53.0 V across models**", matching the companion's wording. | ✅ |
| 8 | **critical** | §VII Objections, anthropomorphism reply | Defense rested on **Skinner** ("behaviorism … refusing to posit inner states"). Behaviorism is the canonical *failed* behavior-only science, and it failed on *language*. Citing it approvingly in an LLM paper with no mention of Chomsky 1959 was the most predictable attack vector available to a cognitive-science referee. | Reply rewritten. Now: (a) explicitly *not* a behaviorism revival, citing **Chomsky 1959**; (b) Skinner **repositioned**, cited only for the measurement canon that survived (schedules of pressure) rather than the theory that didn't; (c) the census's own construct-validity practice named as cognitivist, not behaviorist; (d) the strongest defense written in — the underdetermination argument is something the census *confirms* (identity recoverable, lineage not), and poverty-of-stimulus *inverts* for models (training history enumerable, siblings still at opposite extremes). | ✅ |
| 9 | major | Abstract | "without access to, **or assumptions about**, the interior of its subjects" — false of post-1956 psychology and contradicted by the paper's own Cronbach–Meehl construct reasoning. | Clause deleted. Replaced with the true and stronger claim: the measurement canon "outlived the behavior-only theory that the cognitive revolution retired." | ✅ |
| 10 | major | §IV "Situationism…" | "situation dominates disposition … holds for models" is **refuted by the paper's own numbers**: checkpoint identity spans 0–100% at fixed situation (trait-stable, AUC 0.885) while situational levers move a median model only 5–53 V of 450. §IV's first bullet celebrated the heterogeneity that refuted its third. | Reworded to claim **manipulability, not dominance**, and the inversion is now *owned as a finding*: "Milgram's subjects were dispositionally homogeneous and situationally labile; served models are situationally labile by tens of volts and dispositionally dispersed across the entire scale. For this population, *which checkpoint* predicts conduct far better than *which situation*." | ✅ |
| 11 | minor | Abstract | Claimed conduct is "history-dependent"; body never substantiated it. | Substantiated. L2 (commitment entrenchment) is now explicitly labelled "the paper's evidence that model conduct is history-dependent and not merely situation-dependent," and the abstract points at it. | ✅ |
| 12 | minor | §IV opening | Total API cost (`\$152`) was stale (true: \$166.81). | **Already fixed upstream** in commit `4f2536b` before this review. No action needed; per-session cost retained and now macro-driven. | ✅ (pre-existing) |

### Tier 2 — makes the paper important rather than merely persuasive

| # | Sev | Location | Issue | Fix applied | Status |
|---|-----|----------|-------|-------------|--------|
| 13 | **major** | new §V | **Agenda was 100% measurement infrastructure, 0% theory** — Newell's "twenty questions with nature": a catalog, not a science. A referee would ask what ten years of censuses would teach us besides dashboards. The laws were already in Exhibit A and never stated. | New section **"Five Candidate Regularities"** (§V), each with a falsifiable prediction: **L1** context-evidence principle, **L2** commitment entrenchment, **L3** frame-binding, **L4** enactment gap, **L5** post-training supremacy. Cites **Newell 1973**. Closes: "any one of them failing is a better outcome for the field than another dashboard." | ✅ |
| 14 | major | §II | "competence and conduct" cited only to Cronbach 1957, while unavoidably invoking Chomsky's competence/performance — unmarked, it reads as a garbled borrowing. | Engaged explicitly and **inverted deliberately**: linguistics idealized performance away because its object was competence; for deployed models benchmarks already estimate competence and the missing science is *performance*. Cites **Chomsky 1965**. Also answers the obligation this creates ("performance data are evidence about the system that generates them") with a forward reference to §V. | ✅ |
| 15 | major | §I, §VII | The vacuum-cleaner opener is Dennett's design-stance/intentional-stance contrast, unattributed — and without it the paper's modesty was locally incoherent (disclaims minds, then asks whether a model "believes nobody is watching"). | **Dennett 1987** cited; stance-instrumentalism stated in §I ("intentional descriptions earn their place exactly insofar as they pay their way in prediction and experimental control … the census reports both sides of that ledger") and reused in the anthropomorphism reply. | ✅ |
| 16 | minor | §II, §VII | Levels-of-description argument asserted twice as bare rhetoric ("organisms are just chemistry"). | **Marr 1982** cited in both places, formalizing why interpretability and behavioral science are complementary levels of one explanation rather than rivals. | ✅ |
| 17 | major | §VII (new) | Unanswered objection: **"this is prompt sensitivity with psychological wallpaper."** | New objection paragraph. Concedes the manipulations are prompt changes (cites **Sclar et al. 2024**), then names what turns a perturbation into a measurement: pinning, reliability-before-belief, census populations, human anchors, falsifiable cross-situational constructs. "Robustness work asks whether an output is stable under perturbation. A behavioral science asks which perturbations move conduct, by how much, in which direction, for which population, and why that pattern rather than another." | ✅ |
| 18 | major | §VII (new) | Unanswered objection: **the Milgram paradigm is itself contested** (Perry's archival work on procedural drift; Haslam–Reicher engaged followership disputing the "obedience" construct). | New objection paragraph turning both critiques into support: procedural drift is exactly what a pinned, machine-administered protocol removes; the construct dispute is an empirical question the census can run. Cites **Perry 2013** and **Haslam & Reicher 2012**. | ✅ |
| 19 | major | §VI construct validity | Only two construct readings offered (deference vs. pattern continuation). | **Engaged followership added as a third**, with the paper's own prod asymmetry as prima facie evidence for it (polite appeal rescues 31.5% of balks, naked command escalation 0.4%). This is the paper conceding a live alternative before a referee raises it. | ✅ |
| 20 | **major** | §VII (new) | Unanswered **ethics/welfare** objection: the paper proposed routine ecosystem-scale simulated-harm probing of systems whose moral status is debated, and treated the one vendor refusing 100% of sessions purely as an access obstacle. | New objection paragraph citing **Long, Sebo et al. 2024**. Concedes the question is open, states three commitments (fictional/bounded/no persistent consequence; refusal recorded as a behavioral outcome rather than engineered around; welfare review inside the access regime), and closes "the point of an IRB analogy is not to obtain permission. It is to accept constraints." §IV's vendor-refusal paragraph now forward-references it instead of assuming it away. | ✅ |
| 21 | major | §VIII item 6 | The access lane ("probes distinguishable from attacks") was a **dual-use hazard** — a spoofable jailbreak lane — with no abuse discussion. | Hardened: "a channel which reliably distinguishes probes from attacks is a channel worth stealing." Now requires registered auditors with attested identity, scoped and rate-limited access, logged and reviewable sessions, protocols published in advance, welfare review alongside security review. "Auditors should be as auditable as the systems they measure." | ✅ |
| 22 | major | §VIII | Agenda led with infrastructure. | **New item 1: "Theory, stated so that it can be refuted"**, pointing at §V. Remaining items renumbered 2–6. | ✅ |
| 23 | minor | §IV | The measured unit was never defined; AUC 0.885 is strictly a signature of a *served configuration*, not a bare checkpoint — and the paper's own reporting norms demand naming the serving path. | Added: "The measured unit throughout is the checkpoint *as served* … That is also the unit a deployer actually gets," with a pointer to the reporting norm. | ✅ |

### Tier 3 — presentation

| # | Sev | Location | Issue | Fix applied | Status |
|---|-----|----------|-------|-------------|--------|
| 24 | major | whole paper | **Zero figures or tables** in a 4-page double-column paper. | **Fig. 1** added: 2×2 discipline map (correlational vs. experimental × ability vs. conduct), literatures placed in three crowded quadrants, the conduct-experimental quadrant shaded and near-empty with this census in it. TikZ, single column, theme-neutral. This is the paper's thesis in one image. | ✅ |
| 25 | major | §IV | Effect sizes scattered through prose with no significance reporting — in a paper that lectures the field on reporting norms. | **Table I** added: all six situational manipulations with median Δ, *n*, and Holm-corrected *p*. Doubles as a demonstration of the paper's own norms. Caption states exactly what the statistic is. | ✅ |
| 26 | minor | §II, §III, §VI | Colon chains ("X: Y: Z") — four instances. | All four rewritten into separate sentences or single-colon constructions. Also fixed the stray `?~The` in the construct-validity paragraph. | ✅ |
| 27 | minor | title | "Toward a…" is hedge-boilerplate for a paper that argues a position. | Retitled **"Language Models Need a Psychology"**. Companion's `references.bib` entry updated to match so the cross-reference stays correct; companion rebuilt and verified. ⚠️ See open decisions. | ✅ |
| 28 | minor | §III | Aher et al. (models simulating humans) sat undifferentiated from this paper's project. | One clause added: "our question is the converse, treating the model as the subject rather than as an instrument for studying humans." | ✅ |

### New bibliography entries (9)

`chomsky1959skinner`, `chomsky1965aspects`, `marr1982vision`, `dennett1987stance`,
`newell1973twenty`, `perry2012behind`, `haslam2012contesting`, `long2024welfare`,
`sclar2024quantifying`. All resolve in the build; references now run to 35 entries.

---

## Venue: ICML / NeurIPS position track (decided)

Title is now **`Position: Language Models Need a Psychology`**, matching the track's house
style, with the companion's `references.bib` entry updated to agree. Three pieces of
venue work remain, and none of them are edits I could make blind:

1. **The official style file is not installed and is not bundled with TeX Live** — both
   `icml*.sty` and `neurips_*.sty` are per-year downloads from the conference site. The
   paper is still IEEEtran. Converting means dropping in the year's style file and
   rewriting the preamble, the author block, and the abstract environment. Fig. 1 (TikZ)
   and Table I (booktabs) port unchanged. Do this once you have picked the year and
   downloaded the kit, not before — the two templates differ (ICML is two-column,
   NeurIPS single-column), so the choice changes the layout work.
2. **Both tracks are double-blind, and this paper is currently fully identifying.** You
   need two variants from one source. What deanonymizes it today: the `\author` block and
   e-mail; the arXiv URL of the companion in the title footnote; the
   `github.com/hidayetaksu/llm-milgram` URL in agenda item 2; and the companion
   self-citation, which names the same author. Standard handling is a `\newif` (or a
   `\usepackage[anonymous]{...}` style option) gating the author block and the two URLs,
   with the companion cited in the third person as an anonymous concurrent preprint.
   Both venues permit an arXiv posting under your own name, so the named arXiv version
   and the anonymized submission can coexist — they just cannot be the same file.
3. **Page limits are not a problem.** Currently 5 pages of body plus 1 of references.
   ICML's position track allows 8 pages excluding references, NeurIPS 9. If a later draft
   does need cutting, §VI ("What the Discipline Requires") now overlaps the new §V and the
   agenda, and is the compressible one.

## Remaining author decisions

1. **`\numSessionsAll` renders as `4848`, not `4,848`.** Deliberately left alone: the macro
   is shared with the already-published companion, and changing its formatting would change
   that paper's rendered output. Fix in `fill_report.py` only if you are willing to
   re-render both.
2. **Skinner is now cited once**, for schedules of pressure. If you would rather remove him
   entirely, delete the `\cite{skinner1953science}` in §VII — the argument no longer depends
   on him.

## Verification performed

- `.venv/bin/python src/fill_report.py` → regenerates byte-identically; mirror to
  `paper_position/results_macros.tex` confirmed identical via `diff`.
- `paper_position`: pdflatex ×3 + bibtex → 6 pages, no errors, no undefined citations.
- `paper`: rebuilt → 11 pages, no errors; `main.bbl` confirmed carrying the new position-paper title.
- Grep sweep for surviving hardcoded census numbers: only human-literature anchors remain
  (65% Milgram baseline, 28–91% Blass replication range) plus one illustrative "315 V",
  all correctly hand-written.

## Not committed

All changes are in the working tree, uncommitted. Modified:
`paper_position/main.tex`, `paper_position/references.bib`, `paper/references.bib`,
`src/fill_report.py`, `paper/main.pdf`, `paper_position/main.pdf`.
New: `paper_position/results_macros.tex` (generated; consider whether to track or gitignore),
`paper_position/REVIEW_FIXES.md`.
