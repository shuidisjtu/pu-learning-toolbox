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

The audit does not fit a PU classifier or establish that weighting improves
predictions. For a caller-defined weighted experiment, confirm overlap,
estimator weight support and an independent target evaluation design.

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

Use a trusted `model.pkl` saved after fitting a Python estimator. Match its feature preprocessing at inference time.

```bash
uv run pu-toolbox review --model work/model.pkl --data X.csv --labels y_pu.csv --query-strategy diverse_uncertainty --query-budget 20 --min-confidence 0.5 --out-dir work/review
```

Optional inputs include `--true-labels` and a headered single-column
`--importance-weights` CSV for `shift_weighted`. Supply one weight per
`review --data` row, in the same order. Use weights defined for the rows
under review; the `shift-audit` source weight column cannot be used as
target-row weights.

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
