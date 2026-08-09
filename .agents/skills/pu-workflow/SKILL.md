---
name: pu-workflow
description: >-
  End-to-end PU learning workflow for the pu-learning-toolbox project:
  profile data and diagnose SCAR/SAR assumptions, recommend PU
  algorithms and estimate the class prior, train and evaluate with the
  CLI, and interpret results with actionable advice. Use when the user
  asks to analyze positive-unlabeled data, run a PU analysis, check
  labeling bias or SCAR assumptions, choose a PU method, or estimate a
  class prior.
license: MIT
compatibility: Python >=3.10, uv, pu-toolbox (>= 1.0)
metadata:
  author: shuidisjtu
  version: 1.0.0
---

# pu-workflow

End-to-end PU learning analysis for the
[pu-learning-toolbox](https://github.com/shuidisjtu/pu-learning-toolbox)
project. The user describes a PU analysis task in natural language; you
execute the four steps below, stopping at the checkpoints, and report
results with actionable advice. Follow the steps in order; never skip
data profiling or the checkpoint reviews.

## When to use

Trigger on: "analyze this PU data", "run PU analysis", "which PU method
should I use", "is SCAR plausible here", "estimate the class prior",
"check labeling propensity assumptions". Do NOT use for plain binary
classification with full labels.

## Prerequisites

- Working directory: repository root of pu-learning-toolbox (all paths
  below are relative to it).
- Environment: `uv` installed; dependencies present (`uv sync` if
  missing). Deep methods need optional torch extras.
- Input data contract: feature matrix as CSV (rows = samples, first row
  must be a non-numeric header) or .npy 4-D NCHW images; PU labels as a
  single-column CSV of {1, 0} with a header row. Headerless or
  multi-column files are rejected with a clear error - report it to the
  user, do not work around it.
- The profile/recommend scripts operate on table data (2-D CSV). For
  4-D NCHW image data, go directly to Step 3 and pass
  `--architecture cnn` (plus `--backbone`/`--device` as needed).

## Workflow overview

| Step | What | Executes | Output |
|---|---|---|---|
| 1. Profile & diagnose | Data quality + SCAR/SAR assumption evidence | `scripts/pu_workflow/profile.py` | `profile.json` |
| 2. Recommend & prior | Method ranking + class-prior estimation | `scripts/pu_workflow/recommend.py` | `recommendation.json` |
| 3. Train & evaluate | Full PU pipeline under stratified CV | `pu-toolbox run` (CLI) | `report.json` + `report.md` |
| 4. Interpret & advise | Human-readable conclusions + next actions | you (see below) | summary to the user |

All scripts exit 0 on success and 1 on user/input errors (clear stderr
message, no traceback). Outputs are strict JSON (no NaN/Infinity).

## Step 1 — Profile & diagnose

Run:

```bash
uv run python scripts/pu_workflow/profile.py \
  --data <X.csv> --labels <y_pu.csv> \
  [--true-labels <y_true.csv>] \
  --out-dir <work_dir>
```

Read `<work_dir>/profile.json`:

- `selection_diagnostic.status`: `plausible` / `at_risk` /
  `inconclusive` — SCAR/SAR plausibility evidence.
- `issues[]`: each has `code`, `severity` (`info`/`warning`/`error`),
  `message`, `action`. The `action` field tells the user what to do.

**Checkpoint 1 (mandatory):** if any issue has `severity` in
{`warning`, `error`} or the diagnostic is not `plausible`, STOP. Explain
the findings in plain language, give the suggested `action`, and wait
for the user's decision before continuing.

For deeper context, see `docs/user/howto/data_profiling.md` and
`docs/user/concepts/scar_sar.md` (in-repo, load on demand).

## Step 2 — Recommend & prior

The recommender needs a class prior only for methods that require one
(`requires_class_prior`). Decide how to obtain it, per **Checkpoint 2**:

**Checkpoint 2 (mandatory):** if the user has not supplied a class
prior, STOP and offer exactly two options:

1. Auto-estimate (`recpe`, `pen_l1`, `km1`, or `km2`; pass `--prior-estimator`).
2. Supply a domain value (`--class-prior 0.3`).

Wait for the user's choice. If the user explicitly declines a prior,
run with `--prior-estimator none`; if the user gives a value, keep it
verbatim.

Run with the chosen prior (or `--prior-estimator none` if none is
needed):

```bash
uv run python scripts/pu_workflow/recommend.py \
  --profile <work_dir>/profile.json \
  [--data <X.csv> --labels <y_pu.csv>] \
  [--class-prior <value> | --prior-estimator recpe|pen_l1|km1|km2] \
  [--top-k 5] [--has-gpu] \
  --out-dir <work_dir>
```

Read `<work_dir>/recommendation.json`: `candidates[]` ordered by
`rank`, each with `score`, `reasons[]`, `warnings[]`; `global_warnings[]`
explain profile-level caveats.

Report the top 3 candidates with their reasons to the user. If the
candidate list is empty, explain the filter reasons
(`filters_applied`) and suggest how to relax them.

See `docs/user/concepts/method_selection.md` for the scoring
dimensions.

## Step 3 — Train & evaluate

Use the existing CLI (stable, structured output):

```bash
uv run pu-toolbox run \
  --data <X.csv> --labels <y_pu.csv> \
  [--true-labels <y_true.csv>] [--class-prior <value>] \
  [--classifier <name>|auto] [--prior-estimator <name>|none] \
  [--cv 5] [--metrics <csv>] [--seed 42] [--save-model] \
  --out-dir <work_dir>/run
```

Prefer `--classifier auto` (the recommender picks a method and records
the choice in `report.json` under `provenance.classifier`); pick an
explicit name only when the user asks for a specific method. Let the
user's prior decision flow through: pass `--class-prior <value>` or
`--prior-estimator` exactly as chosen in Step 2.

Read `<work_dir>/run/report.json`: `cv_metrics` (a dict keyed by metric
name; each entry has `available`, `mean`, `std`, `n_computed`,
`reason`), `prior`, `provenance`, `issues[]` (`has_errors` /
`has_warnings` summarized in `report.md`).

**Checkpoint 3 (mandatory):** if `report.json` carries any error or
warning entries (or the run failed with exit 1), STOP and walk the
user through the messages before continuing to interpretation.

See `docs/user/howto/pipeline.md` and `docs/user/howto/cli.md` for
parameter details.

## Step 4 — Interpret & advise

Translate the results into plain language. Follow the Chinese
interpretation guide in
`.claude/skills/pu-workflow/references/interpret.zh-CN.md` (repo-root
relative) for field-by-field meaning and advice templates; write your
final summary in the language the user speaks. Cover:

1. **Assumption verdict**: what the SCAR/SAR diagnostic said and what
   it implies for the chosen method.
2. **Prior estimate**: value, estimator, and sensitivity implications
   (run `scripts/pu_workflow/sensitivity.py` if the user wants an
   assumption sweep:
   `uv run python scripts/pu_workflow/sensitivity.py --data <X.csv> --labels <y_pu.csv> --out-dir <work_dir>`).
3. **Chosen method**: name, why it won, and its caveats (`warnings[]`).
4. **Metric highlights**: the `available` metrics with `mean ± std`.
5. **Next actions**: at most 3 concrete suggestions (fix a data issue,
   supply a better prior, compare a specific alternative method).

Do not invent numbers: cite only values present in the JSON outputs.

## Failure handling

| Symptom | Action |
|---|---|
| Script exit 1 | Read the stderr `error:` message; fix the input and re-run, or ask the user how to proceed |
| Diagnostic `at_risk` / `inconclusive` | Explain what it means and stop at Checkpoint 1, waiting for the user's decision |
| Empty candidate list | Explain `filters_applied`; suggest relaxing filters (e.g. supply a prior, densify sparse data) |
| `pu-toolbox run` exit 1 | Show the `error:` message; three options for the user: fix data and re-run / pick another method / get the full traceback |
| Unknown classifier/prior name | Run `uv run pu-toolbox list-methods` / `uv run pu-toolbox list-priors` for the accepted set |

## References

- `docs/user/quickstart.md`, `docs/user/concepts/pu_problem.md`,
  `docs/user/concepts/scar_sar.md`, `docs/user/concepts/method_selection.md`
- `docs/user/howto/data_profiling.md`, `docs/user/howto/pipeline.md`,
  `docs/user/howto/cli.md`, `docs/user/howto/sensitivity_analysis.md`
- `docs/user/reference/api.md`
- `.claude/skills/pu-workflow/references/interpret.zh-CN.md` (repo-root
  relative; this file only exists on the .claude side)
