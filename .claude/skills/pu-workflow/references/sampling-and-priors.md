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
| Population prior | Target-population `P(Y=1)` | Population metrics and the PA selection criterion |
| Native-U prior | Positive fraction of input U | Ordinary mixture-prior estimators' declared target |
| Training-table prior | Positive fraction of the P/U concatenation | Consumers declaring a `training_table` target |
| Risk-pool prior | Positive fraction of the pool entering a loss | Prepare the value required by the selected method and risk view |

Record input sampling, labeling mechanism, selection policy, and training risk view
as separate settings. Inspect the estimator's `prior_consumers` and supported
risk-view parameters. Compute conversions from the relevant training partition.

## Construct same-pool OS observations

Pass one already-loaded source partition and explicit positive/negative classes to
`construct_pu_dataset`. Supply original sample IDs when available. Otherwise use a
stable source and partition namespace for generated position IDs. Keep experiment
roles under caller control; the constructor never reads, downloads or splits data.

```python
import numpy as np
from pu_toolbox.preprocessing import PULabelingConfig, SourcePartition, construct_pu_dataset

truth = np.array([1] * 40 + [0] * 60)
features = np.arange(100, dtype=float)[:, None]
result = construct_pu_dataset(
    SourcePartition(features, truth, "example-v1", "train"),
    PULabelingConfig("scar", "fixed_count", random_state=7, n_labeled=20),
    positive_classes=(1,),
    negative_classes=(0,),
)
observed = result.observations
assert observed.labels.sum() == 20
assert result.manifest["audit"]["empirical_native_unlabeled_prior"] == 0.25
```

Use `result.observations` for fitting. Keep `result.audit` separate and save the manifest.

## Generate independent TS data

```python
from pu_toolbox.preprocessing import make_gaussian_case_control_data

X_ts, y_pu_ts, y_true_ts, metadata = make_gaussian_case_control_data(
    n_p=30, n_u=180, population_prior=0.4, n_features=3, random_state=42,
)
assert metadata["input_sampling"] == "ts"
assert metadata["design_population_prior"] == 0.4
assert metadata["empirical_unlabeled_prior"] == y_true_ts[y_pu_ts == 0].mean()
assert len(set(metadata["sample_ids"])) == len(y_pu_ts)
```

The output is ordered P then U. Preserve `sampling_policy`, `sample_ids`, source
distribution, seed, target `design_population_prior`, and empirical
`empirical_unlabeled_prior`/`empirical_training_table_prior`. Use a batch namespace when combining
multiple generator outputs. Analyze this data through `input_sampling="ts"`.

## Simulate SCAR/SAR labeling from clean features

Use `PULabelingConfig("scar", "bernoulli", ...)` for equal positive propensities or
`PULabelingConfig("sar", "bernoulli", ..., propensity_rule="linear")` for feature-dependent
propensities (also supports `"nonlinear"`). Select `c` as a target positive-class marking
rate. Read the actual budget and row identities from the result; `ensure_labeled=True`
explicitly tops up a zero-mark draw, without redefining the recorded propensity.

```python
from pu_toolbox.preprocessing import GaussianSourceConfig, PULabelingConfig, simulate_pu_dataset

result = simulate_pu_dataset(
    GaussianSourceConfig(120, 3, population_prior=0.4, separation=2.0, random_state=11),
    PULabelingConfig("sar", "bernoulli", random_state=7, c=0.3, propensity_rule="linear"),
)
```

Use the same labeling config with `construct_pu_dataset` for an existing partition.
Named LBE-A/LBE-B fixed-quota policies remain with the experiment generators; see
[research-experiments.md](research-experiments.md) for their four-role workflow.

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

For a standalone KM2 estimate and TS profile with an explicit target:

```python
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
    X_ts, y_pu_ts, input_sampling="ts", evaluation_prior=resolved_prior,
)
```

Select an estimator matching the requested method and budget. Interpret the
raw estimate using its declared target and estimation pool. Record the raw
value, sampling design, any conversion, and the value and target passed to
training. Recompute partition-specific values if the caller changes the risk
pool or training data.

Synthetic truth supports construction checks and authorized evaluation. Training
priors come from user-specified knowledge, allowed PU estimation, or registered
study settings. Completion: the data construction and prior records identify the
source distribution, observed roles, selection policy, target conversion, and
actual training consumption needed for the task.
