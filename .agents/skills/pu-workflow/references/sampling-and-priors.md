# Sampling, labeling, and prior targets

Read this reference when constructing PU observations, determining OS/TS input
semantics, estimating a prior, or preparing a method's training prior. For execution
and report persistence, use [analysis.md](analysis.md).

## Identify the distributions

| Setting or target | Meaning | Execution consequence |
|---|---|---|
| OS input | Label some positives in one population sample; U contains the remaining rows | Population estimates use the complete input; native-U prior reflects removal of labeled positives |
| Independent TS input | Obtain extra P from the positive conditional and U independently from the target population | U represents the population; P/U concatenation has its own class proportion |
| Finite-pool design | Select or reuse roles from a named source pool | Record removal, replacement, role overlap, and source identities |
| Population prior | Target-population `P(Y=1)` | Pipeline call-level `class_prior` and report `prior.value` |
| Native-U prior | Positive fraction of input U | Ordinary mixture-prior estimators' declared target |
| Training-table prior | Positive fraction of the P/U concatenation | Consumers with `class_prior_target="training_table"` |
| Risk-pool prior | Positive fraction of the pool entering a loss | Prepare the value required by the selected method and risk view |

Record input sampling, labeling mechanism, selection policy, and training risk view
as separate settings. Inspect the estimator's `class_prior_target` and supported
risk-view parameters when calling it directly. Pipeline applies its default-view
consumer conversions fold by fold and for the final refit.

## Construct same-pool OS observations

Start with features and complete binary truth, then split by original source into
the required training, validation, and test roles. Apply the labeling policy within
each permitted role. Keep row identities aligned with features and both labels.

This minimal example demonstrates fixed-count labeling of an existing clean pool:

```python
import numpy as np

from pu_toolbox.preprocessing import make_fixed_count_labels, make_scar_labels

y_true = np.array([1] * 40 + [0] * 60)
y_fixed = make_fixed_count_labels(y_true, n_labeled=20, random_state=7)
y_scar = make_scar_labels(y_true, c=0.5, random_state=7)
assert y_fixed.sum() == y_scar.sum() == 20
assert y_true[y_fixed == 0].mean() == 0.25
```

`make_fixed_count_labels` selects a specified number of positives uniformly without
replacement. `make_scar_labels` uses a positive-class labeling fraction and a fixed
quota. `make_pu_labels(mechanism="fixed_count", n_labeled=...)` is the dispatcher.
The older `case_control` labeling aliases retain this same-pool construction and
emit a migration warning toward `fixed_count`.

For a complete OS teaching dataset, `make_scar_dataset(n=..., c=..., ...)` returns
`X, y_pu, y_true` with `n` samples per true class. `make_gaussian_pu_data` provides
the teaching case with all generated positives labeled and U drawn from negatives.
Choose the generator by the desired data distribution.

## Generate independent TS data

```python
from pu_toolbox.preprocessing import make_gaussian_case_control_data

X_ts, y_pu_ts, y_true_ts, metadata = make_gaussian_case_control_data(
    n_p=30, n_u=180, class_prior=0.4, n_features=3, random_state=42,
)
assert metadata["input_sampling"] == "ts"
assert metadata["population_class_prior"] == 0.4
assert metadata["unlabeled_class_prior"] == y_true_ts[y_pu_ts == 0].mean()
assert len(set(metadata["sample_ids"])) == len(y_pu_ts)
```

The output is ordered P then U. Preserve `sampling_policy`, `sample_ids`, source
distribution, seed, target `population_class_prior`, and empirical
`unlabeled_class_prior`/`training_class_prior`. Use a batch namespace when combining
multiple generator outputs. Analyze this data through `input_sampling="ts"`.

## Simulate SCAR/SAR labeling from clean features

`make_sar_labels` uses Bernoulli draws from calibrated positive-class propensities.
Choose `mechanism="scar"`, `"linear"`, or `"nonlinear"` explicitly.

```python
from pu_toolbox.preprocessing import make_sar_labels

# Use the generated population U as a clean OS source pool for this example.
X_os = X_ts[y_pu_ts == 0]
y_true_os = y_true_ts[y_pu_ts == 0]
y_pu_os, propensity = make_sar_labels(
    X_os, y_true_os, mechanism="linear", label_frequency=0.4,
    strength=1.5, ensure_labeled=False, random_state=7,
    return_propensity=True,
)
```

The example uses arrays from the TS generator above; a real complete-label dataset
can supply `X_os, y_true_os` directly. Record the target mean propensity, actual
positive labeling fraction, selected count, mechanism strength, seed, and
`ensure_labeled`. With `ensure_labeled=True`, a zero-selection draw labels the
highest-propensity positive. For pure Bernoulli experiments use `False` and check
the realized positive count against the training/CV budget.

Named LBE-A/LBE-B fixed-quota policies are available through experiment generators.
Record their auxiliary-score fitting scope and weighting configuration; use
[research-experiments.md](research-experiments.md) for the four-role workflow.

## Convert and estimate priors

For one-sided OS labeling, let `q` be the labeled fraction of the complete input:

```text
pi_population = q + (1 - q) * pi_U
pi_U = (pi_population - q) / (1 - q)
```

For independent TS, `pi_U = pi_population`. With extra P fraction `r`, the
concatenated table has `pi_table = r + (1 - r) * pi_population`.

Use the shared conversion functions:

```python
from pu_toolbox.prior import (
    native_unlabeled_to_population,
    population_to_native_unlabeled,
)

population_prior = native_unlabeled_to_population(0.25, 0.2, input_sampling="os")
u_prior = population_to_native_unlabeled(0.4, 0.2, input_sampling="os")
assert abs(population_prior - 0.4) < 1e-12
assert abs(u_prior - 0.25) < 1e-12
```

The conversions accept valid probabilities and require a nonempty native U. OS
population priors satisfy `pi_population >= q`. Individual fitting APIs enforce
their own supported probability domain.

Ordinary KM1/KM2, penL1, and ReCPE prior estimators declare
`estimate_target="native_unlabeled"`. Read their declared `estimation_pool` and
metadata. Interpret estimates under the method's representative-positive and
identifiability assumptions. ReCPE records a regrouping approximation; independent
BBE has its own TS holdout contract.

For a standalone recommendation using a KM2 estimate with an explicit target:

```python
from pu_toolbox.advisor import recommend_from_profile
from pu_toolbox.preprocessing import profile_pu_data
from pu_toolbox.prior import KernelMeanPriorEstimator

estimator = KernelMeanPriorEstimator(variant="km2")
estimator.fit(X_ts, y_pu_ts)
raw_prior = float(estimator.estimate())
assert estimator.estimate_target == "native_unlabeled"
resolved_prior = native_unlabeled_to_population(
    raw_prior, float(y_pu_ts.mean()), input_sampling="ts",
)
profile = profile_pu_data(
    X_ts, y_pu_ts, input_sampling="ts", class_prior=resolved_prior,
)
recommendation = recommend_from_profile(
    profile, class_prior=resolved_prior, class_prior_source="estimated",
)
```

Select an estimator matching the user's requested method and budget. For integrated
training, Pipeline performs the target resolution and retains its full provenance.

## Read the training chain

Pipeline resolves an explicit call-level population prior, then a classifier's
constructor prior by its declared target, then required automatic estimation.
Read these `report.prior` fields together:

- `input_value`, `input_target`, `estimator`, `estimation_pool`, and
  `estimation_details`: original input or estimator result and its meaning.
- `value`, `target`, `input_sampling`, and `conversion`: resolved population prior.
- `training_consumption`: target and value actually passed to each training fold
  and final refit.
- `source`, `auto_selected`, and `degraded`: execution provenance.

Synthetic truth supports construction checks and authorized evaluation. Training
priors come from user-specified knowledge, allowed PU estimation, or registered
study settings. Completion: the data construction and prior records identify the
source distribution, observed roles, selection policy, target conversion, and
actual training consumption needed for the task.
