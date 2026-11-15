# Class-Prior and Labeling-Propensity Sensitivity

> Prerequisites: complete the [quickstart](../quickstart.md) and [pipeline guide](pipeline.md).
> See [the PU problem](../concepts/pu_problem.md) for $`q=\pi\bar c`$ and identifiability.

`analyze_pu_sensitivity` holds model outputs fixed while scanning OS population priors and mean
labeling propensities. It computes PU metrics and checks compatibility with the observed labeled
fraction. This analysis does not refit the model.

## 1. Basic usage

Prepare a fitted classifier and aligned OS validation arrays `X_valid, y_valid`, where `y_valid`
contains PU labels. `fitted_classifier` below must expose `predict` and `decision_function`.

```python
from pu_toolbox.diagnostics import analyze_pu_sensitivity

analysis = analyze_pu_sensitivity(
    y_valid,
    fitted_classifier.predict(X_valid),
    scores=fitted_classifier.decision_function(X_valid),
    class_priors=[0.20, 0.25, 0.30, 0.35, 0.40],
    label_propensities=[0.20, 0.30, 0.40, 0.50, 0.80],
)

print(analysis.to_frame(axis="class_prior"))
print(analysis.metric_ranges)
```

`y_pu` must normalize to `{1, 0}` and contain both labeled-positive and unlabeled groups.
`y_pred` must use `{1, 0}`; single-class predictions are allowed. Supply `class_priors`,
`label_propensities`, or both. See the [API reference](../reference/api.md) for valid domains
and `SensitivityPoint` fields.

- Without `scores`, risk uses `+1/-1` scores mapped from `y_pred`. Estimated precision always
  uses `y_pred`, so explicit scores affect risk only.
- `metric_ranges` summarizes only consistent points with available metrics. An implied
  boundary $`\pi=1`$ can be algebraically consistent, but the PU metrics require `(0,1)`,
  so its metrics are `None`. The explicit `class_priors` grid excludes 1.

## 2. Persistence and experiment records

This interface uses the complete OS input pool to estimate the population risk term;
`class_priors` denotes population priors. Its zero-one risk is
`2*pi*mean_P(score<=0) + mean_all(score>0) - pi`.

```python
analysis.save("artifacts/sensitivity.json")
analysis.save("artifacts/sensitivity.md")
analysis.save("artifacts/sensitivity.csv")
```

JSON is strict, with no `NaN` or `Infinity` literals. CSV supports plotting; JSON preserves
the feasible region, metric ranges, and provenance; Markdown suits experiment reports.

## 3. Interpretation

Fixed-output curves describe metric sensitivity to assumptions, supporting post-training reports,
prior-setting comparisons, and configuration debugging. For training robustness studies, keep
splits/seeds fixed, refit at each prior setting, and report multi-seed means, standard deviations,
and failure rates.

Complete runnable example:

```bash
uv run python examples/minimal/09_sensitivity_analysis.py
```

## Next steps

- Choose another task from the [user documentation](../README.md).
- [Exact API contracts](../reference/api.md) (currently Chinese).
