# Data Profiling and Assumption Evidence

> Prerequisite: complete the [quickstart](../quickstart.md).
> See [SCAR/SAR](../concepts/scar_sar.md) for identifiability limits and evidence types.

`profile_pu_data` checks PU label counts, feature quality, prior consistency, and labeling-mechanism
evidence before training. Its structured `PUDataProfile` can be read in a terminal or serialized
into experiment records. The examples below assume aligned arrays `X, y_pu`; audit examples
add corresponding binary truth `y_true`.

**Interpretation boundary:** observing only $`X`$ and $`S`$ generally does not identify SCAR versus
SAR. `observed_mixture` evidence is screening; `audited_positives` (with `y_true`) can directly
test feature-dependent selection within true positives. `plausible` does not prove SCAR. See
[SCAR/SAR](../concepts/scar_sar.md) for the derivation and evidence table.

## 1. Basic usage

```python
from pu_toolbox.preprocessing import profile_pu_data

report = profile_pu_data(
    X,
    y_pu,
    class_prior=0.30,
    random_state=42,
)

print(report.format_text())
payload = report.to_dict()
```

Public PU labels are normalized to `{1, 0}`. `class_prior` is a supplied or independently
estimated population $`P(Y=1)`$; the profiler does not estimate it from hidden truth.

Enable audit mode when manual review, delayed outcomes, or a synthetic experiment provides truth:

```python
audited_report = profile_pu_data(
    X,
    y_pu,
    y_true=y_true,
    class_prior=0.30,
    cv=5,
    scar_auc_threshold=0.65,
)
```

Every sample with `y_pu == 1` must also be positive in `y_true`; otherwise the interface rejects
the input. This exposes a conflict with the trusted-positive assumption early.

## 2. Returned object

See the [API reference](../reference/api.md) for `PUDataProfile` fields,
`selection_diagnostic["status"]` values, and issue codes such as `no_labeled_positives`.

- `report.has_errors` indicates data errors that prevent reliable training;
  `report.has_warnings` identifies issues to review before training.
- `report.format_text()` produces readable text; `report.to_dict()` produces ordinary
  containers suitable for JSON output.
- Errors are not repaired automatically. Silent imputation, column removal, or label rewriting
  changes the data flow and can introduce training/validation leakage.

## 3. Class-prior consistency

For default OS input, supplying `class_prior=pi` makes the report compute:

```math
\widehat c = \frac{n_{\mathrm{labeled}}/n}{\pi}.
```

This is the implied labeling frequency from the observed labeled fraction and population prior.
If $`\widehat c>1`$, the report records `inconsistent_class_prior`. This alone does not establish
that the population prior is wrong: finite-sample proportions can differ. Explain the discrepancy
in the experiment record.

Use `input_sampling="ts"` for independent P and population U. That design records the extra-P
table fraction and does not interpret it as an OS labeling frequency. See
[prior semantics](../concepts/class_prior_semantics.md).

Obtain priors from independent knowledge, training-side estimation, or nested validation.
Filling them in from test truth leaks information.

## 4. Thresholds and reproducibility

Default configuration:

```python
profile_pu_data(
    X,
    y_pu,
    min_labeled_positives=30,
    max_unlabeled_to_positive=100.0,
    low_variance_threshold=1e-12,
    scar_auc_threshold=0.65,
    cv=5,
    random_state=42,
)
```

The selection diagnostic reduces its CV folds to the size of the smaller label group. If that
group has fewer than two samples, its status is `inconclusive`. Preserve complete parameters,
software versions, and seeds, rather than only the final boolean result.

## 5. Sparse input and leakage

The interface supports scipy sparse matrices, including implicit zeros in variance calculations.
It also detects explicitly stored `NaN` and `inf` values.

Read-only quality checks can precede splitting. Repairs that learn statistics, such as imputation,
scaling, dimensionality reduction, and low-variance filtering, must be fitted inside each training
fold. Audit `y_true` supports assumption diagnostics; do not use it as classifier features or a
shortcut to hyperparameter selection.

## 6. Complete example

Run:

```bash
python examples/minimal/07_data_profiling.py
```

The example generates SCAR and linear SAR data side by side, showing why observational evidence
is non-identifying and how audit mode provides more direct evidence about selection.

## Next steps

- [End-to-end training and evaluation](pipeline.md).
- [Exact API contracts](../reference/api.md) (currently Chinese).
