# Constructing Same-Pool OS and Independent TS Data

Prerequisites: install the toolbox and read [sampling designs](../concepts/pu_sampling.md) and
[prior targets](../concepts/class_prior_semantics.md). This tutorial uses small CPU Gaussian datasets.

## 1. Construct OS observations from one source partition

`construct_pu_dataset` marks positives in an already-loaded partition. Declare positive,
negative and (if needed) excluded source classes. Unmapped or overlapping classes are
rejected. Dataset names identify sources; they do not choose the positive class.

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

Use `c=0.5` instead of `n_labeled=20` to request half the true positives. A fixed count
samples without replacement. For Bernoulli SCAR use `PULabelingConfig("scar", "bernoulli", ...)`;
for SAR use `"sar"`, `"bernoulli"` and a feature-based `propensity_rule` (`"linear"` or
`"nonlinear"`). `c` targets the positive-class marking rate, not the whole-table marked
fraction `q`. Supply original `sample_ids` when available; generated position IDs are local
to the declared source and partition, not proof of independence across sources.

`result.observations` contains the features, PU labels and IDs for training. Read
`result.audit` only for evaluation or audit. The separate manifest records requested and
actual counts, seeds, source and audit-only empirical priors. The constructor does not
read a file, download data, or split and rearrange experiment roles.

To simulate a clean Gaussian source first, pass the same labeling config to
`simulate_pu_dataset` with a `GaussianSourceConfig`:

```python
from pu_toolbox.preprocessing import GaussianSourceConfig, PULabelingConfig, simulate_pu_dataset

result = simulate_pu_dataset(
    GaussianSourceConfig(120, 3, population_prior=0.4, separation=2.0, random_state=11),
    PULabelingConfig("sar", "bernoulli", random_state=7, c=0.3, propensity_rule="linear"),
)
assert result.manifest["input_sampling"] == "os"
```

## 2. Obtain independent positive P and population-mixture U samples

`make_gaussian_case_control_data` draws P from the positive class-conditional Gaussian and
independently draws U from the target population. `class_prior` and the positive/negative class
means define that population; each U observation's class is drawn independently.

```python
import numpy as np
from pu_toolbox.preprocessing import make_gaussian_case_control_data

X, y_pu, y_true, metadata = make_gaussian_case_control_data(
    n_p=80, n_u=800, class_prior=0.4, n_features=2, random_state=7,
)
assert X.shape == (880, 2)
assert np.all(y_true[y_pu == 1] == 1)
assert metadata["population_class_prior"] == 0.4
assert metadata["unlabeled_class_prior"] == y_true[y_pu == 0].mean()
assert len(set(metadata["sample_ids"])) == 880
```

Outputs are ordered P then U. `y_pu` identifies the training roles; `y_true` supports synthetic
auditing and evaluation. `metadata` records the target population prior, empirical U and
training-table priors, sample identities, distribution parameters, and sampling policy.

The `positive:` and `population:` prefixes in `sample_ids` identify draws from the two independent
sources. These identities are unique within one output; add a batch namespace when combining
multiple outputs. Integer seeds reproduce all outputs. Passing a `RandomState` continues its
current random stream and records the metadata seed as `None`.

## 3. Run the paired comparison example

```powershell
uv run python examples/minimal/15_pu_sampling.py --seed 7
```

[Example 15](../../../examples/minimal/15_pu_sampling.py) first generates independent TS data,
then uses its population-sampled U as a clean source pool to construct OS observations. The two
settings share target population prior 0.4 and the same population draw, making the change in U
after removing labeled positives visible. Output distinguishes target parameters, finite-sample
empirical proportions, labeled counts, and sampling policies.

Split research data into train / pu_val / clean_val / test by original identity before generating
labels within permitted partitions. See [custom experiments](custom_experiments.md) for a complete
four-role workflow.

## 4. Keep sampling identities distinct

Same-pool fixed counts are OS, not independent TS. Use the independent Gaussian TS
constructor above for that simulation. The all-positive-labeled Gaussian teaching
case has negative-only U and cannot substitute for either marginal U or OS labeling.

See the [API reference](../reference/api/tools.md#data-generation) for parameter
constraints and return fields.
