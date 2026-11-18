# Auditing and Handling PU Distribution Shift

> Prerequisites: source-domain `(X_source, y_source_pu)` from the historical/training environment
> and target-domain `X_target` from the current deployment environment. Weighted adaptation
> also requires target PU labels `y_target_pu` aligned to those target rows.

## Understand the interpretation boundary

The audit tests whether source and target are distinguishable, then estimates a bounded relative
transform of the marginal density ratio `p_target(x) / p_source(x)`. This is a **covariate-shift
weighting baseline**: its adaptation interpretation requires
`p_source(y|x) = p_target(y|x)`.

- High out-of-fold (OOF) domain AUC indicates observable differences; it does not identify
  covariate, prior, or concept shift by itself.
- Low OOF domain AUC means this auditor found no strong difference; it does not prove equality.
- Low effective sample size (ESS), boundary-concentrated weights, or very low relative mass
  before normalization indicates inadequate source coverage; avoid automatic adaptation.
- Source CV metrics are not target performance. Target truth supports separately labeled oracle metrics.

## CLI: audit first

Source and target CSVs need nonnumeric headers and the same feature-column count:

```bash
pu-toolbox shift-audit \
  --source-data source/X.csv \
  --source-labels source/y_pu.csv \
  --target-data target/X.csv \
  --target-labels target/y_pu.csv \
  --out-dir shift_results/
```

Omit `--target-labels` if target labels are unavailable. The audit still runs, but
`adaptation_ready` is `false`.

The output directory contains:

| File | Contents |
|---|---|
| `shift_report.json` | Strict JSON: domain AUC, severity, ESS, weight quantiles, issues, and implementation boundaries |
| `shift_report.md` | Readable interpretation and suggested next steps |
| `source_importance_weights.csv` | `source_row` and source weights normalized to mean one |

Common options: `--alpha 0.1` controls the bounded relative-ratio transform, `--cv 5` sets domain
OOF folds, and `--probability-clip 1e-6` prevents divergent domain-probability odds.

## Python: audit only

```python
from pu_toolbox.diagnostics import analyze_pu_shift

shift = analyze_pu_shift(
    X_source,
    y_source_pu,
    X_target,
    y_target_pu=y_target_pu,  # Optional; required to enable adaptation.
    alpha=0.1,
    cv=5,
    random_state=42,
)

print(shift.domain_auc, shift.severity)
print(shift.weight_summary["effective_sample_fraction"])
shift.save("shift_report.json")
shift.save("source_importance_weights.csv")
```

The weights CSV preserves one-to-one source-row alignment. JSON stores weight summaries and
artifact names rather than the full weight vector, keeping reports bounded as sample counts grow.

## Python: explicitly enable covariate adaptation

```python
from pu_toolbox.workflows import ShiftAwarePUPipeline

workflow = ShiftAwarePUPipeline(
    classifier="elkan_noto",  # Default; the method must actually use sample_weight.
    evaluation_cv=5,
    shift_cv=5,
    random_state=42,
)
report = workflow.fit_evaluate(
    X_source,
    y_source_pu,
    X_target,
    y_target_pu=y_target_pu,
    y_true_target=y_true_target,        # Optional, for oracle evaluation only.
    target_class_prior=0.30,            # Optional; enables prior-dependent target metrics.
)

print(report.shift.to_markdown())
print(report.target_metrics["pu_auc_roc"].mean)
```

The workflow slices source weights by CV training fold and passes them to the final refit.
Methods declaring `sample_weight_support="ignored"`, such as `upu` and `pnu`, fail immediately
instead of silently ignoring adaptation weights. See the
[API reference](../reference/api/classifiers.md#sample_weight-support-levels) for weight-support contracts.

For an audit plus an unweighted source baseline:

```python
report = workflow.fit_evaluate(
    X_source,
    y_source_pu,
    X_target,
    adapt=False,
)
```

Failed coverage checks raise `PipelineError` by default. Controlled research can set
`allow_unstable=True`; warnings remain in the report. This is not recommended for production.

## CLI: compare paired arms before deciding

```bash
pu-toolbox shift-run \
  --source-data source/X.csv \
  --source-labels source/y_pu.csv \
  --target-data target/X.csv \
  --target-labels target/y_pu.csv \
  --target-class-prior 0.30 \
  --classifier elkan_noto \
  --evaluation-cv 5 --shift-cv 5 \
  --out-dir shift_comparison/
```

The command compares unweighted and weighted arms on the same target set.
`--evaluation-cv` controls source-model evaluation; `--shift-cv` controls domain-model folds
used to estimate the importance weights. It writes
`shift_comparison.json/.md`, `shift_report.json`, `source_importance_weights.csv`, and
`target_predictions.csv`. With `--target-true-labels`, the default decision uses oracle AUC;
otherwise it can use target-prior-dependent PU risk. With neither, it still provides comparison
values but concludes `audit_only`. Unstable coverage without `--allow-unstable` skips the weighted
arm and concludes `collect_target_data`.

Oracle-guided comparison requires a designated validation/audit set. Keep final test data
independent; do not use its truth to choose the adaptation arm, threshold, or prior.

## Monitor successive windows

Use aligned reference arrays `X_reference, y_reference_pu` and the next window's
`X_august, y_august_pu`:

```python
from pu_toolbox.diagnostics import PUShiftMonitor

monitor = PUShiftMonitor(X_reference, y_reference_pu, cv=5)
window, audit = monitor.update(
    X_august,
    y_window_pu=y_august_pu,
    window_id="2026-08",
    timestamp="2026-08-01T00:00:00+08:00",
)
print(window.alert_level, window.alert_codes)
monitor.save_history("shift_history.json")
```

Alerts cover high domain shift, adaptation blocks, and abrupt neighboring-window changes in AUC
or labeled fraction. History stores window summaries without features or labels. A new monitor
with the same configuration can call `load_history(...)` and continue appending.

## Separate prior changes from labeling changes

```python
from pu_toolbox.diagnostics import analyze_domain_assumptions

assumptions = analyze_domain_assumptions(
    X_source,
    y_source_pu,
    X_target,
    y_target_pu,
    source_class_prior=0.25,   # If omitted, estimate each domain separately with pen_l1.
    target_class_prior=0.35,
)
print(assumptions.conclusion)
print(assumptions.differences)
```

The report uses `P(S=1)=π·c̄` to separate class prior from mean positive-labeling propensity,
and provides a local 3×3 prior-sensitivity table. `c̄` is an aggregate: equal values across
domains do not establish SCAR or exclude feature-dependent SAR.

## Abstention and active review

Here `fitted_model` is a trained model and `target_relevance_weight` is a finite, nonnegative,
nonzero vector aligned to **target** rows. The source-weight vector from the shift audit cannot
be substituted for target-row weights.

```python
from pu_toolbox.diagnostics import analyze_pu_uncertainty

review = analyze_pu_uncertainty(
    fitted_model,
    X_target,
    y_pu=y_target_pu,
    min_confidence=0.60,
    query_budget=20,
    query_strategy="shift_weighted",
    importance_weight=target_relevance_weight,
)
review.save("uncertainty_rows.csv")
```

`selective_prediction=-1` delegates to human/downstream review; it is not a negative class.
The three query strategies are uncertainty alone, shift weight times uncertainty, and cluster
diversity among uncertain candidates. Labeled positives are excluded from the query pool when
PU labels are supplied. The probability margin is an uncertainty heuristic, not a
coverage-calibrated confidence interval.

## Choosing an action

| Finding | Suggested action |
|---|---|
| Low AUC, high ESS | Continue monitoring and target validation; do not claim absence of shift |
| Moderate/high AUC, high ESS, adequate coverage | Check covariate-shift assumptions, then compare unweighted and weighted models |
| Low ESS or very low relative mass | Collect more target/similar-source data before relying on extreme weights |
| Large change in labeled-positive fraction | Audit prior and propensity separately; labeled fraction is not π |
| Suspected change in `p(y|x)` | Use target PU data for dedicated adaptation/retraining; marginal weighting has no joint-shift guarantee |

## Research joint-shift methods

The AISTATS 2025 joint-shift method estimates `p_target(x,y) / p_source(x,y)` and alternates
shared-feature, classifier-head, and class-conditional weight-model training. The research
package provides:

- `JointShiftPUClassifier`: a sklearn soft class-conditional approximation without Torch.
- `DynamicJointShiftPUClassifier`: a Torch clean-room implementation of the paper's
  Equations (19)/(20), loss correction, shared features, and Algorithm 1 gradient isolation.

The paper states that its source is proprietary and does not release an implementation.
The latter path therefore claims formula-level clean-room implementation, not official-source
equivalence or numerical reproduction. It is outside the registry/`auto`. The
[development checklist](../../dev/distribution_shift_aware_pu_checklist.md) is optional Chinese
implementation background.

## Next steps

- [Exact objects, parameters, and report fields](../reference/api/tools.md#distribution-shift-api).
- [Prior/propensity sensitivity](sensitivity_analysis.md).
- [Design and theoretical boundaries](../../dev/distribution_shift_aware_pu.md)
  (optional Chinese background).
