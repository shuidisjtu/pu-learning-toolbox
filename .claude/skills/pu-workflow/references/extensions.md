# Drift, monitoring, review and benchmark audits

Read this reference for a requested extension. Use headered CSV tables
and single-column PU labels, with an output directory for each stage.
Check the chosen subcommand's `--help` in the active environment.

## Source-to-target drift

Start with a marginal distribution audit:

```bash
uv run pu-toolbox shift-audit --source-data X_source.csv --source-labels y_source.csv --target-data X_target.csv --target-labels y_target.csv --cv 3 --out-dir work/shift
```

`--target-labels` is optional for the audit. Read `shift_report.json`:
`domain_auc`, `severity`, `adaptation_ready`, `weight_summary`, `issues`
and `provenance`. `source_importance_weights.csv` records source-row
identity and the estimated bounded covariate weights. Report the
source/target separability evidence and overlap diagnostics.

For an authorized paired model comparison, use target PU labels:

```bash
uv run pu-toolbox shift-run --source-data X_source.csv --source-labels y_source.csv --target-data X_target.csv --target-labels y_target.csv --classifier elkan_noto --cv 3 --out-dir work/shift-run
```

Supply source `--class-prior`, `--target-class-prior`, and source/target
truth files when available. Select a classifier with the required
sample-weight support. Record the adaptation assumptions, permitted
budget and any explicit overlap-policy choice before executing that
choice. The default overlap policy determines whether the weighted arm
runs; read the resulting recommendation and executed arms.

Read `shift_comparison.json`: `baseline`, `weighted`, `metric_deltas`,
`primary_metric`, `recommendation` and `provenance`. For each available
comparison, report its evaluation population, metric basis, raw delta
and `improvement` (positive means the weighted arm performed better).
The directory also contains the shift report, source weights and
`target_predictions.csv`.

Completion: the requested audit or paired comparison exists and its
recorded evidence, executed arms and recommendation have been explained.

## Window monitoring

```bash
uv run pu-toolbox shift-monitor --reference-data X_ref.csv --reference-labels y_ref.csv --window-data X_window.csv --window-labels y_window.csv --window-id week-01 --cv 3 --out-dir work/monitor/week-01
```

For the next window, pass `--history` pointing to the previous
`shift_history.json` and use a new window ID/output directory. Record
the AUC-jump and label-rate-jump thresholds if customized.

Read `window_summary.json` for `alert_level`, `alert_codes` and window
statistics. Read `window_shift.json` for the current drift evidence and
`shift_history.json` for the sequence. Describe what changed, the
threshold crossed, and the concrete data/model action relevant to the
requested monitoring task.

Completion: the window and history artifacts exist and the alert or
stable-window result has been explained with its observed values.

## Human-review queue

Use a trusted `model.pkl` produced by the analysis route's
`--save-model`. Match its feature preprocessing at inference time.

```bash
uv run pu-toolbox review --model work/run/model.pkl --data X.csv --labels y_pu.csv --query-strategy diverse_uncertainty --query-budget 20 --min-confidence 0.5 --out-dir work/review
```

Optional inputs include `--true-labels` and a headered single-column
`--importance-weights` file. The `shift_weighted` strategy uses those
importance weights; the shift audit exports weights with a source-row
column, so select its weight column when preparing this input.

Read `uncertainty.json`, `uncertainty.md` and `uncertainty_rows.csv`.
Report coverage, selected query count, selection strategy and source-row
identities. Explain how the user can label the exported review rows.

Completion: selective predictions and the requested review queue have
been produced and their row identities and budget have been checked.

## Benchmark artifact audit

```bash
uv run pu-toolbox audit-benchmark --result-dir benchmark_results --output work/benchmark-audit.json
```

Read the saved report's `passed`, trial/method/seed counts, `errors`,
`warnings` and provenance checks. A failed audit writes the requested
JSON before the command reports failure. Resolve issues within the
authorized task and rerun after a concrete correction. Describe the
checks passed and any actionable artifact repairs using the report.

Completion: an audit artifact exists and its completeness, provenance,
metrics and split findings have been reported.
