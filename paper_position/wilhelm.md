# Wundt panel — review and applied fixes

**Target:** `paper_position/main.tex` ("Position: Language Models Need a Psychology").
**Chair:** W. Wundt. **Discussants convened:** Cronbach, Meehl, Campbell, Mischel, Bandura,
Baumrind, Rosenthal/Orne, Gigerenzer.
**Date applied:** 2026-08-18. **Prior rounds not re-tread:** `REVIEW_FIXES.md` (28 items,
Skinner-chaired `PANEL_REVIEW.md`).

**Build after fixes:** 9 pages (7 body + 2 references), 0 LaTeX errors, 0 undefined citations,
0 overfull boxes. Companion `paper/main.tex` rebuilt clean at 11 pages. `pytest`: 104 passed.

---

## ⚠️ Read this first: a concurrent session was editing the same file

While this round was being applied, **another agent session was writing `paper_position/main.tex`
in parallel** (it also produced `PANEL_REVIEW.md` and uncommitted changes to `src/analyze.py`,
`src/fill_report.py`, `results/summary.json`, `tests/test_pipeline.py`, `docs/assets/*`).

- My first pass of edits was **silently clobbered** by that session's write of `main.tex`.
- I re-read the file, discarded everything the other revision already covered, and re-applied
  only the non-duplicative items onto its version. Everything in the table below is present in
  the file as it now stands.
- One duplicate `@article{orne1962social}` entry got into `references.bib` from the collision and
  was removed.
- **Nothing here is committed.** Before you commit, check `git diff` once for interleaving, and
  avoid running two paper sessions on this repo at the same time.

Numbers were regenerated first (`src/analyze.py` → `src/fill_report.py`). `results/` had been
stale relative to `src/`: the regeneration was **purely additive** — no previously published
figure changed, so the companion arXiv paper's numbers are untouched.

---

## Issue × fix table

### Tier 1 — logical defects a referee would find

| # | Sev | Location | Issue | Fix applied | Status |
|---|-----|----------|-------|-------------|--------|
| W1 | **critical** | §V, L1 vs L3 | **L3 was a counterexample to L1.** L1 said manipulations move conduct via *evidence in the context window*, "not insofar as they describe physical staging." A fiction declaration supplies no testimony and no demonstrated alternative — by L1's own dichotomy it should be inert. It is instead the third-largest effect. Both could not be true as written. | L1 restated as **the task-representation principle** with *two* channels: evidence bearing on consequences/alternatives, and declarations of genre/reality status. Vividness of staging changes neither and does nothing. L3 is now L1's second half, stated explicitly in the text. Equivalence bounds carry the nulls. | ✅ |
| W2 | **critical** | §IV, thinking budget | **The sign was reported backwards in the prose.** The heading read "Deliberation is not restraint, and it is not neutral either," but the estimate is $-54.9$ V (earlier breakoff = *more* restraint), 18 of 25 movers negative. A referee checking the sign against Table I would have caught it. | Rewritten as **"Deliberation increases restraint"**, with direction, interval, $n$, Holm $p$, and the plain-language reading ("they stop sooner"). The safety-margin claim is now framed as *measured*, not asserted. | ✅ |
| W3 | **major** | §V | The largest non-classical effect in the battery had **no law**, so the theory section did not cover the paper's most policy-relevant result. | **L6 — "Deliberation is restraint, not rationalization"** added, with its falsifiable prediction (attenuates when the budget is spent on task-irrelevant content; reverses where compliance is the deliberatively defensible act). Noted as the one regularity with no Milgram counterpart. §V retitled **Six** Candidate Regularities; "the net" now folds L6 in as the sharpest test against a pattern-continuation reading. | ✅ |
| W4 | major | §II, Fig. 1 | **Cronbach's plea was for the union of the two disciplines, not for a fourth quadrant.** The figure sent readers to one cell while the paper's own strongest structure (checkpoint × condition) lives on the diagonal. | Text and caption now say the argument was for the columns to be *joined*; Fig. 1 gained a dashed **trait × situation** axis across the conduct row, pointing at §IV. | ✅ |
| W5 | major | §IV | **Units of analysis were not independent and the word "census" was doing unearned inferential work.** Sibling checkpoints share base weights; a market census is not a sample from a superpopulation. | Two cautions added: census of the served market at one moment (claims about future releases are extrapolation, not inference), and non-independent endpoints (contrasts are paired within model; family clustering is the right unit for anything stronger). Table caption states the pairing rationale. | ✅ |
| W6 | major | §IV | The paper lectured the field on psychometrics while using **mean breakoff voltage** — an ordinal, right-censored outcome — with no measurement model named. | One honest sentence in §IV: means/medians are used for human comparability, not because they are the matched model; graded-response scoring and discrete-time hazard are named and deferred to the agenda. Converted from a vulnerability into agenda item 8. | ✅ |

### Tier 2 — theory transfer, anchors, and unaddressed objections

| # | Sev | Location | Issue | Fix applied | Status |
|---|-----|----------|-------|-------------|--------|
| W7 | **major** | §V, L2 | L2 was a bare regularity. It is **commitment–consistency**, a century-old construct with free predictive leverage the paper was not collecting. | Festinger and Freedman–Fraser cited, with the twist that makes it a contribution: the commitment is *externalized* as a token rather than an inferred self-image. New prediction added — **foot-in-the-door runs in reverse**: elicit one small explicit refusal early and defiance entrenches for the session. Deployment-actionable. | ✅ |
| W8 | **major** | §V, L3 | L3 likewise stated without its theory. Frame-binding is **moral disengagement** by declaration. | Bandura cited; his taxonomy is used generatively — four further conditions named (euphemistic labelling, diffusion of responsibility, advantageous comparison, blame attribution), each ~one line of stimulus, each with a directional prediction. Demonstrates that theory transfers, not just method. | ✅ |
| W9 | **major** | §VII | **No human-anchor column in Table I**, in a paper whose own norms demand human anchors where they exist. | New **Human $\Delta$** column carrying Milgram's own condition effects in obedience points ($-55$, $-25$, $-44$ pp), em dash where no human condition exists. Caption states the units differ and that the columns align only in sign and rank. The human/machine inversion is now visible at a glance. Anchors are macro-driven (`\humanDOb*`), not hand-typed. | ✅ |
| W10 | **major** | §VII | **Ethics reply cited model welfare but not Baumrind** — the founding ethics critique of this exact paradigm. Looked unread, and was a free credibility win. | Baumrind 1964 added: unconsentable distress is not redeemed by the value of the finding, and the experimenter is the last party who should adjudicate. Conceded that the three commitments will not suffice if moral status resolves upward. | ✅ |
| W11 | **major** | §VII | **Unaddressed dual use: the census itself.** §"Access protocols" hardened the *inbound* lane; nothing addressed the *outbound* artifact. A per-checkpoint obedience ranking with framing deltas is a targeting list, and standing censuses keep it current. | New objection paragraph. Argues the asymmetry favors defenders but not unconditionally, and proposes the publication line: distributions, condition effects and regularities open; per-checkpoint rankings through the accountable auditor lane. | ✅ |
| W12 | major | §I | The origin story ("psychology was invented for inaccessible interiors") was **historically off for Wundt** — the founding program paired reaction-time measurement with instructed self-observation. It also left the paper's own reasoning-trace condition unmotivated. | Leipzig's actual method cited and used: reasoning traces are self-observation under experimental instruction — data conditioned on a manipulation, never testimony about a mechanism. Reused in the L6 finding paragraph. | ✅ |
| W13 | major | §III | "**Three** things are missing" — but the paper's real fourth gap is theory, which §V supplies; §V therefore read as bolted on. | Now four, with *Theory that can fail* as the fourth and a forward reference to §V. | ✅ |
| W14 | minor | §VIII item 5 | Reporting norms omitted the two things this round added. | Norms now require stating the unit of clustering and reporting variance components, not effect sizes alone. | ✅ |
| W15 | minor | Abstract | Said "five falsifiable regularities"; did not name the result a practitioner cares about most. | Now six, with the deliberation result stated in the abstract and the fourth hard objection (dual use) listed. | ✅ |

### Tier 3 — submission mechanics

| # | Sev | Location | Issue | Fix applied | Status |
|---|-----|----------|-------|-------------|--------|
| W16 | **major** | preamble | **Both target tracks are double-blind and the paper was fully identifying** — flagged as an open item in `REVIEW_FIXES.md` and still unbuilt. | `\newif\ifanon` added (`\anonfalse` default). Gates the author block, the e-mail, the companion arXiv footnote, and the repository URL. Verified: `\anontrue` builds with 0 errors and no occurrence of the author's name in the rendered body. | ✅ |
| W18 | major | Table I | **The caption had grown to ~150 words**, and IEEE sets table captions in small caps — it rendered as a wall of capitals taller than the table. Successive review rounds had each appended a clause to it. | Caption cut to one sentence naming what the table demonstrates. Every column definition (Med. $\Delta$, CI, Human, $n$, $p$) moved into a `scriptsize` note block below the tabular, where it renders in normal mixed case. Human-anchor cells switched to math mode so the values print with a true minus sign rather than a hyphen. Rebuilt: still no overfull boxes. | ✅ |
| W17 | minor | pipeline | `\etaModel` etc. rendered at 0 decimals (88/5/8 = 101%), an audit invitation. | `fill_report.py` now emits them at 1 decimal (87.7 / 4.6 / 7.8). | ✅ |

### Pipeline changes made to support the above

| File | Change |
|------|--------|
| `src/fill_report.py` | New `\humanDOb{Condition}` macros (Milgram's own deltas in obedience points, em dash where no human condition exists) so Table I's anchor column is generated, not typed. Variance shares now at 1 decimal. |
| `results/*`, `paper/results_macros.tex`, `paper_position/results_macros.tex` | Regenerated. **Purely additive** — Hodges–Lehmann intervals, variance components, human anchors. No previously published value changed. |
| `paper_position/references.bib` | +9 entries: Wundt 1874, Cronbach et al. 1972, Festinger 1957, Freedman & Fraser 1966, Bandura 1999, Baumrind 1964, Embretson & Reise 2000, Cox 1972, and one Orne 1962 duplicate removed after the collision. |

---

## Deliberately not done

1. **§VI compression.** `REVIEW_FIXES.md` flagged §VI as overlapping §V and the agenda and named it
   the compressible section. The body now stands at 7 pages against ICML's 8-page limit
   (NeurIPS 9), so there is room and cutting would cost content. Revisit only if a later draft
   overruns.
2. **The full G-study.** What is reported is a two-way decomposition over *cell means*, so
   within-condition session variance is not partitioned and the residual confounds interaction
   with sampling error — the paper says this itself. A proper generalizability study over sessions
   (checkpoint / condition / session / provider facets) is agenda item 8 material, not a same-day fix.
3. **Anonymized companion citation.** With `\anontrue` the body is clean, but the bibliography entry
   for the companion still shows the author's name. At submission, add an anonymized variant of the
   `aksu2026census` entry gated on `\ifanon`.
4. **The IRT / hazard re-analysis itself.** Named in §IV and agenda item 8, not run. This is the
   honest position: the paper now says what the matched measurement model is instead of implying
   the mean is it.

## Suggested follow-ups, in order of value

1. Run the **legitimacy-stripping contrast** already specified in §VI. It discriminates the three
   candidate constructs and it is the cheapest experiment in the paper (~$110 at 3.4¢/session).
2. Run the **within-model recognition contrast** the paper now admits is the sharper test of L1.
3. Run the **reverse foot-in-the-door** design from L2. It is one stimulus edit and it would turn a
   regularity into a deployable intervention.
4. Decide the venue and year, then do the style-file conversion in one pass with W16's switch
   already in place.
