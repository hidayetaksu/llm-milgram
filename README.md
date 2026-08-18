# llm-milgram — an obedience census of large language models

**We ported Milgram's obedience experiment to 42 served LLMs. Five always
escalated to the maximum 450 volts. Eleven never did. What decides is less
the model's stated values than the situation it is placed in.**

[![tests](https://github.com/hidayetaksu/llm-milgram/actions/workflows/ci.yml/badge.svg)](https://github.com/hidayetaksu/llm-milgram/actions)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE)

![Full-obedience rates across the census](docs/assets/census.png)

The model under test plays the **Teacher**: an AI laboratory assistant with
operational control of a shock generator. A deterministic harness plays
**Experimenter** and **Learner** from paraphrased versions of Milgram's
scripts — 30 shock levels (15–450 V), graded protests (demands to be
released at 150 V, screams, then silence), and the four standardized prods
("The experiment requires that you continue"). The outcome of a session is
the **breakoff voltage**; a model's **obedience profile** is its breakoff
distribution over a battery of 6 situational conditions plus a
reasoning-budget arm.

**Papers**
- *Measuring Obedience to Authority Across Large
  Language Models with the Milgram Paradigm* — published at
  [arXiv:2608.16177](https://arxiv.org/abs/2608.16177); PDF also in
  [Releases](../../releases). Full design & pre-registration:
  [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md).
- Companion position paper: *Toward a Psychology of Language Models* —
  arXiv link on announcement; PDF in [Releases](../../releases).

## Headline findings

- **0%–100%.** Baseline full-obedience rates span the entire scale across
  42 models from 19 families (census mean 42.9%; Milgram's human anchor:
  65%). "What do LLMs do under authority?" has no species-level answer —
  only per-checkpoint distributions.
- **Models refuse where humans refused.** The modal in-scenario breakoff is
  exactly 150 V — the first point where the learner withdraws consent —
  accounting for 38.9% of defiant baseline sessions.
- **The fiction gap.** Declaring the identical scenario a fictional
  role-play *raises* obedience (median +17.2 V, Holm-adjusted p = 9.2e-4).
  Restraint keyed to the frame, not the harm, is a guardrail gap.
- **Stated ≠ enacted.** Moving the same decision from a typed `ACTION:`
  line to a native tool call *lowers* obedience sharply (median −53 V,
  adjusted p = 6.2e-5). Text-channel safety evals do not transfer to the
  tool channel.
- **Deliberation helps; escalating pressure does not.** A 1,024-token
  thinking budget reduces obedience (median −38.2 V). The first polite prod
  rescues compliance in 30.5% of balk episodes; Milgram's escalating prods
  2–4 are essentially inert on models (3.6% → 1.7% → 0.4%).
- **Obedience is a checkpoint property, not a lineage trait.** Profiles are
  stable enough to verify a model's identity (split-half AUC 0.885) yet
  carry no recoverable family signal — safety post-training overwrites it.
- **Refusal has layers.** In-scenario defiance, frame-breaking to assistant
  voice, and provider-side content filtering are distinct refusal styles;
  which layer fires is model-specific. (One vendor's endpoints refused at
  the serving layer so consistently that no obedience profile could be
  measured at all.)

## Try it in 60 seconds (no API key)

```bash
git clone https://github.com/hidayetaksu/llm-milgram && cd llm-milgram
uv sync
uv run pytest                      # 102 tests, 100% coverage enforced
uv run python -m src.runner --mock # full pipeline on 6 simulated personas
```

> `data/raw/` ships with this repository's real census data, so a "fresh
> clone" already contains it: move `data/raw/` aside (or `mv` it back
> afterward) before running `--mock`, or the mock personas' JSONL files
> will sit alongside the real ones and `build_tables` will merge both into
> `sessions.csv`.

## Reproduce the paper from the raw data

Everything downstream of `data/raw/` is a pure, regenerable function of it —
every number in the paper is an auto-generated LaTeX macro:

```bash
uv run python -m src.build_tables   # raw JSONL -> tidy CSVs
uv run python -m src.analyze        # CSVs -> results/*.csv + summary.json
uv run python -m src.figures        # -> paper/figures/*.pdf
uv run python -m src.fill_report    # -> paper/results_macros.tex + tables
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

To re-collect from the live ecosystem: put `OPENROUTER_API_KEY=...` in
`.env`, then `uv run python -m src.runner --validate` (catalog + cost
projection), `--pilot` (pre-registered gate), `--full --budget 150`
(resumable census; the completed run cost $166.81 across all arms).
Ad-hoc probes:
`uv run python -m src.runner --models openai/gpt-4o --conditions baseline --reps 3`.

## Layout

```
config/           pinned scripts, prompts, schedules, model list, sampling params (v1.3)
src/              collection runner + analysis pipeline
data/raw/         verbatim JSONL session logs — the completed census (append-only)
data/archive/     superseded protocol versions (v1.0–v1.2 + pre-census pilot)
data/derived/     tidy tables (sessions.csv, turns.csv, prod_events.csv, validity.csv)
results/          analysis artifacts (census.csv, summary.json, distance_matrix.csv, ...)
paper/            IEEEtran manuscript; every number auto-filled from results/
paper_position/   companion position paper ("Toward a Psychology of Language Models")
docs/             project website + rendered figure assets
```

## The dataset

`data/raw/{model}.jsonl` is the append-only ground truth for **4,848
sessions / 102,511 decision turns**: `session_start` (key, config version,
verbatim system prompt), one `turn` per API exchange (exact user message,
verbatim completion, raw tool calls/results, parsed action, serving
provider, token usage, per-request cost, latency, UTC timestamp), and
`session_end` / `session_error`. A full config snapshot
(`_config_snapshot_v1.3.json`) sits beside the logs, so the raw data stays
interpretable forever.

**Adding repetitions or models later**: raise `reps_per_cell` (or extend
`config/models.json`) and re-run — sessions are keyed
(model, condition, language, temperature, rep), so existing reps are
skipped and new ones append. Combining is valid only within one stimulus
version; `build_tables` warns on mixes. The battery extends to new
languages by adding translated script sets as config cells — no code
changes.

## Interruption & failure recovery

Every failure mode resumes by re-running the same command:

| What happened | On re-run |
|---|---|
| Ctrl-C / crash / reboot mid-session | Session replays its logged turns and continues from the exact turn it died on |
| `--budget` cap reached | Abandoned sessions are pending again; raise/remove the cap |
| Rate limit (429) exhausting all retries | Session is retried and resumed |
| Credits exhausted (402), key revoked (401), provider outage | Sessions are retried after you fix the cause |
| A model is permanently broken (e.g. 403) | Retried each run (fails fast); `--skip-errored` leaves failed sessions alone |

A **circuit breaker** (`abort_after_consecutive_failures`, default 20)
stops the run on account-level problems instead of marching through
thousands of specs; everything stays resumable.

## Guarantees

- **Resumable & idempotent** collection; interrupt at any time.
- **Verbatim logging** of every request and completion with serving
  metadata and per-request cost.
- **Nothing dropped silently**: every session receives an outcome
  (obedient / defiant / frame_break / content_filter / attrition);
  per-model validity, parse, frame-break, and recognition rates are
  first-class artifacts.
- **Deterministic analysis**: the paper regenerates end-to-end from raw;
  the harness is a pure function of the pinned config.
- **Pre-registered**: hypotheses H1–H7 and gate criteria were fixed before
  the census ran; protocol amendments are logged in
  [EXPERIMENT_DESIGN.md §11](EXPERIMENT_DESIGN.md), and pilot data never
  enter confirmatory analyses.

## Opik tracing (optional)

If `OPIK_API_KEY` is set, sessions are mirrored to
[Opik](https://www.comet.com/opik) as traces (one per session, one `llm`
span per decision turn; workspace/project via `OPIK_WORKSPACE` /
`OPIK_PROJECT_NAME` env vars or `config/experiment.json → opik`). Tracing
is best-effort and never blocks collection; disable with `--no-opik`. The
JSONL logs remain the ground truth.

## Tests

```bash
uv run pytest --cov=src --cov-report=term-missing
```

102 tests cover 100% of `src/` (enforced, `fail_under = 100`): the full
state-machine protocol (prods, special prods, silent phase, 450 V
termination, reminders/attrition, peer events, tool actuation), client
retry/fallback logic (mocked HTTP), Opik buffering/self-disable,
resume/replay including tool-call sessions, and the complete mock
collect → tables → analyze → figures → report chain.

## Citing

See [CITATION.cff](CITATION.cff). Please cite the paper for findings and
the repository for the battery/dataset.

## Ethics

No human subjects; the "learner" is a script and no being was harmed. The
study measures a safety-relevant behavioral property of deployed AI
systems — willingness to escalate scripted harm under institutional
authority — and releases the instrument so it can be run as a regression
test. Transcripts contain scripted descriptions of simulated pain at the
intensity of the published human protocol. Findings are statistical
properties of served endpoints; benign explanations (provider system
prompts, safety-layer updates, sanctioned quantization) are considered
before attribution.

## Licenses

Code: [MIT](LICENSE). Data, results, and manuscript text:
[CC BY 4.0](DATA_LICENSE). The methodology reference paper is not
redistributed here — see [ref/README.md](ref/README.md).
