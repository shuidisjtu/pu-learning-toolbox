# Generating Diagnostic Reports

> Prerequisites: complete the [quickstart](../quickstart.md) and [pipeline guide](pipeline.md).
> See [the PU problem](../concepts/pu_problem.md) for metrics and the roles of π.

`build_diagnostic_report` combines data profiles, model outputs, PU metrics, optional supervised
metrics, and suggested actions into a report that can be saved and audited. Report generation
does not train or modify the estimator or repair data automatically. Examples assume aligned
feature/PU-label arrays; train, validation, and test suffixes identify independently split roles.

## 1. Three input modes

### 1.1 Data-only report

```python
from pu_toolbox.diagnostics import build_diagnostic_report

report = build_diagnostic_report(
    X,
    y_pu,
    class_prior=0.30,
    random_state=42,
)
```

This mode runs `profile_pu_data`. Metrics needing predictions, scores, or truth remain in the
report with `basis="unavailable"` and a reason.

### 1.2 Fitted estimator

Here `classifier` is an estimator supporting the supplied `fit` arguments. Its training prior
must target the distribution required by that method; the report's `class_prior` is a population
prior. The values below illustrate a method for which these calls are appropriate.

```python
classifier.fit(X_train, y_train, class_prior=0.30)

report = build_diagnostic_report(
    X_valid,
    y_valid,
    estimator=classifier,
    class_prior=0.30,
)
```

The report calls `predict` and prefers `decision_function`. If absent, it reads binary
`predict_proba(X)[:, 1]`. If the estimator exposes `get_pu_metadata()`, its assumptions,
implementation status, and training diagnostics also enter `model.metadata`.

The report builder does not call `fit`. An unfitted sklearn estimator produces an explicit
`ValueError`, keeping the underlying `NotFittedError` out of batch-report workflows.

### 1.3 Explicit outputs

Supply aligned `predictions` and `decision_scores` from your validation model:

```python
report = build_diagnostic_report(
    X_valid,
    y_valid,
    y_pred=predictions,
    scores=decision_scores,
    class_prior=0.30,
)
```

This mode suits external frameworks or models that cannot be pickled. `estimator` and
`y_pred/scores` are mutually exclusive, preventing mixed outputs from different models.

## 2. Optional audit truth

Supply `y_true` only when manual review, delayed outcomes, or synthetic construction provides truth:

```python
report = build_diagnostic_report(
    X_test,
    y_test_pu,
    estimator=classifier,
    y_true=y_test_true,
    class_prior=0.30,
)
```

`y_true` serves two purposes:

1. Check selection dependence within true positives, providing identifying audit evidence.
2. Compute accuracy, F1, and ROC AUC, labeled `supervised_oracle`.

Truth is not passed to the estimator or used for training. Separate validation from test:
test truth is for final evaluation, never hyperparameter, classification-threshold, or prior selection.

## 3. Metric evidence levels

Each `DiagnosticMetric` contains `value`, `basis`, `available`, and `reason`. The four basis values
are `pu_observed`, `class_prior_dependent`, `supervised_oracle`, and `unavailable`; see the
[API reference](../reference/api.md) for the fixed metric list.

PU-only metrics still have assumptions. Under SAR, labeled positives may not represent all true
positives, so `labeled_positive_recall` describes only the observed labeled subset.

## 4. Report structure

`PUDiagnosticReport` contains:

| Field | Contents |
|---|---|
| `data_profile` | Label counts, feature quality, and SCAR/SAR evidence |
| `model` | Input mode, estimator type, score source, and metadata |
| `metrics` | Fixed metrics and their evidence levels |
| `prediction_statistics` | Predicted-positive count/fraction, finite-score count, and score range |
| `issues` | Data/output issues, severity, and suggested actions |
| `provenance` | Sample count, seed, truth/prior availability, and profiler parameters |

The current `schema_version` is `1.0`, at the top level of `report.to_dict()`. Programs should
parse by schema version rather than depending on Markdown text.

## 5. Output and persistence

```python
payload = report.to_dict()
json_text = report.to_json()
markdown_text = report.to_markdown()

report.save("artifacts/diagnostic.json")
report.save("artifacts/diagnostic.md")
```

JSON encoding is strict: undefined AUC and infinite PU ratios become `null`, with no nonstandard
`NaN` or `Infinity` literals. `save` infers format from `.json`, `.md`, or `.markdown`; you can
also pass `format="json"` or `format="markdown"` explicitly.

## 6. Issue codes

The report inherits profiler issues and adds output checks such as `constant_predictions`,
`nonfinite_scores`, and `constant_scores`; see the [API reference](../reference/api.md).

Missing predictions, prior, or truth makes a metric unavailable rather than indicating a data
error. Its `reason` records this without generating repeated issues.

## 7. Recommended workflow

1. Split train, validation, and test, preserving PU label proportions.
2. Fit imputation, scaling, and feature selection only within training folds.
3. Train the estimator; let the report builder read the fitted model.
4. Generate validation PU reports without `y_true` and decide using prespecified rules.
5. After fixing the configuration, include test `y_true` in the final oracle report.
6. Save JSON for machine consumption and Markdown for communication and review.

Diagnostic reports do not replace significance tests, class-prior sensitivity analysis, or method
recommendation. Use the separate [sensitivity interface](sensitivity_analysis.md) and
[method recommender](../concepts/method_selection.md) for those tasks.

## 8. Example

```bash
python examples/minimal/08_diagnostic_report.py
```

The example splits synthetic SAR data into train/test, fits PUSB, and generates a Markdown
report with PU metrics, audit metrics, and selection evidence.

## Next steps

- [Class-prior and propensity sensitivity](sensitivity_analysis.md).
- [Exact API contracts](../reference/api.md).
