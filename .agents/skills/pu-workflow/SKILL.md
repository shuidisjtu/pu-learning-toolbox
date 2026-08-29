---
name: pu-workflow
description: >-
  End-to-end PU learning workflow for the pu-learning-toolbox project:
  profile data and diagnose SCAR/SAR assumptions, recommend PU
  algorithms and estimate the class prior, train and evaluate with the
  CLI, and interpret results with actionable advice. Optional
  extensions cover distribution-shift migration, deployment
  monitoring, and benchmark audits. Use when the user asks to analyze
  positive-unlabeled data, run a PU analysis, check labeling bias or
  SCAR assumptions, choose a PU method, estimate a class prior, audit
  distribution shift, monitor a deployed PU model, or audit benchmark
  results.
license: MIT
compatibility: Python >=3.10, uv, pu-toolbox (>= 1.10.0)
metadata:
  author: shuidisjtu
  version: 1.10.0
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
  single-column CSV of {1, 0} with a header row. Headerless,
  multi-column, or non-finite (NaN/Inf) inputs are rejected with a
  clear error - report it to the user, do not work around it.
- The profile/recommend scripts operate on table data (2-D CSV). For
  4-D NCHW image data, go directly to Step 3 and pass
  `--architecture cnn` (plus `--backbone`/`--device` as needed).

## Workflow overview

| Step | What | Executes | Output |
|---|---|---|---|
| 1. Profile & diagnose | Data quality + SCAR/SAR assumption evidence | `pu-toolbox profile` | `profile.json` |
| 2. Recommend & prior | Method ranking + class-prior estimation | `pu-toolbox recommend` | `recommendation.json` |
| 3. Train & evaluate | Full PU pipeline under stratified CV | `pu-toolbox run` (CLI) | `report.json` + `report.md` |
| 4. Interpret & advise | Human-readable conclusions + next actions | you (see below) | summary to the user |

All commands exit 0 on success and 1 on user/input errors (clear stderr
message, no traceback). Outputs are strict JSON (no NaN/Infinity).

## Step 1 — Profile & diagnose

Run:

```bash
uv run pu-toolbox profile \
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
uv run pu-toolbox recommend \
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
   (run `pu-toolbox sensitivity` if the user wants an assumption sweep:
   `uv run pu-toolbox sensitivity --data <X.csv> --labels <y_pu.csv> --out-dir <work_dir>`).
3. **Chosen method**: name, why it won, and its caveats (`warnings[]`).
4. **Metric highlights**: the `available` metrics with `mean ± std`.
5. **Next actions**: at most 3 concrete suggestions (fix a data issue,
   supply a better prior, compare a specific alternative method).

Do not invent numbers: cite only values present in the JSON outputs.

## Optional extensions

The four steps above cover the standard single-domain analysis. Engage
the scenarios below only when the user's request mentions them, and
keep the same discipline: run the commands, read the JSON outputs, and
stop at the checkpoints.

### A. Distribution-shift migration

Trigger: the user mentions a target domain, covariate shift,
cross-domain transfer, or source-to-target deployment.

1. Audit source-to-target drift and export bounded covariate weights:

```bash
uv run pu-toolbox shift-audit \
  --source-data <X_source.csv> --source-labels <y_source.csv> \
  --target-data <X_target.csv> [--target-labels <y_target.csv>] \
  --out-dir <work_dir>/shift
```

Read `<work_dir>/shift/` outputs for drift evidence and the exported
source weights.

**Checkpoint A (mandatory):** if the audit reports significant drift,
STOP, explain the evidence and what covariate weighting changes, and
wait for the user's decision before running a weighted model.

2. Compare unweighted vs covariate-weighted PU models on the target
   domain:

```bash
uv run pu-toolbox shift-run \
  --source-data <X_source.csv> --source-labels <y_source.csv> \
  --target-data <X_target.csv> --target-labels <y_target.csv> \
  [--target-true-labels <y_target_true.csv>] \
  [--target-class-prior <value>] [--classifier <name>] \
  --out-dir <work_dir>/shift-run
```

Report which arm wins and by how much; cite only numbers present in
the JSON outputs.

See `docs/user/howto/distribution_shift.md` for details.

### B. Deployment monitoring

Trigger: the user asks to monitor a deployed PU model, check
production drift window by window, or set up a human review queue.

1. Append one target window to a persistent shift history:

```bash
uv run pu-toolbox shift-monitor \
  --reference-data <X_ref.csv> --reference-labels <y_ref.csv> \
  --window-data <X_win.csv> [--window-labels <y_win.csv>] \
  --window-id <id> [--history <previous.json>] \
  --out-dir <monitor_dir>
```

Outputs include `shift_history.json`, `window_shift.json` (plus `.md`)
and `window_summary.json`.

**Checkpoint B (mandatory):** if the monitor emits an alert (AUC or
label-rate jump beyond its threshold), STOP, explain what degraded and
by how much, and wait for the user's decision.

2. Build a selective-prediction review queue from a saved model:

```bash
uv run pu-toolbox review \
  --model <model.pkl> --data <X.csv> \
  [--labels <y_pu.csv>] [--true-labels <y_true.csv>] \
  [--query-strategy uncertainty|shift_weighted|diverse_uncertainty] \
  [--query-budget 20] [--min-confidence 0.5] \
  --out-dir <review_dir>
```

`<model.pkl>` comes from Step 3's `--save-model`.

See `docs/user/howto/distribution_shift.md` and
`docs/user/howto/cli.md` for details.

### C. Benchmark audit

Trigger: the user provides a benchmark result directory and asks to
validate it.

```bash
uv run pu-toolbox audit-benchmark \
  --result-dir <dir> [--output <audit.json>]
```

Read the audit report and walk the user through every issue with its
severity before any further use of the results. See
`docs/user/howto/cli.md` for parameter details.

## Failure handling

| Symptom | Action |
|---|---|
| Command exit 1 | Read the stderr `error:` message; fix the input and re-run, or ask the user how to proceed |
| Diagnostic `at_risk` / `inconclusive` | Explain what it means and stop at Checkpoint 1, waiting for the user's decision |
| Empty candidate list | Explain `filters_applied`; suggest relaxing filters (e.g. supply a prior, densify sparse data) |
| `pu-toolbox run` exit 1 | Show the `error:` message; three options for the user: fix data and re-run / pick another method / get the full traceback |
| Unknown classifier/prior name | Run `uv run pu-toolbox list-methods` / `uv run pu-toolbox list-priors` for the accepted set |
| Extension command missing (`shift-*`, `review`, `audit-benchmark`) | The installed package predates the command; suggest `uv pip install -U pu-toolbox` |
| `review` fails for missing `--model` file | Run Step 3 with `--save-model` to produce `model.pkl` first |

## References

- `docs/user/quickstart.md`, `docs/user/concepts/pu_problem.md`,
  `docs/user/concepts/scar_sar.md`, `docs/user/concepts/method_selection.md`
- `docs/user/howto/data_profiling.md`, `docs/user/howto/pipeline.md`,
  `docs/user/howto/cli.md`, `docs/user/howto/sensitivity_analysis.md`,
  `docs/user/howto/distribution_shift.md`, `docs/user/howto/model_tuning.md`,
  `docs/user/howto/self_pu.md`, `docs/user/howto/sar_simulation.md`,
  `docs/user/howto/diagnostic_reports.md`, `docs/user/howto/ui.md`,
  `docs/user/howto/using_skill.md`
- `docs/user/reference/api.md`
- `.claude/skills/pu-workflow/references/interpret.zh-CN.md` (repo-root
  relative; this file only exists on the .claude side)
