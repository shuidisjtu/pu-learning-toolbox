# Constructing Same-Pool OS and Independent TS Data

Prerequisites: install the toolbox and read [sampling designs](../concepts/pu_sampling.md) and
[prior targets](../concepts/class_prior_semantics.md). This tutorial uses small CPU Gaussian datasets.

## 1. Label a fixed quota within one clean pool

`make_fixed_count_labels` selects a specified number of true positives uniformly without
replacement and preserves input row order. For example, labeling 20 positives in a complete pool
of 40 positives and 60 negatives leaves 20 positives and 60 negatives in U.

```python
import numpy as np
from pu_toolbox.preprocessing import make_fixed_count_labels

y_true = np.array([1] * 40 + [0] * 60)
y_pu = make_fixed_count_labels(y_true, n_labeled=20, random_state=7)
assert y_pu.sum() == 20
assert y_true[y_pu == 0].mean() == 0.25
```

To specify the quota as a fraction, use `make_scar_labels(y_true, c=0.5, random_state=7)`.
For Bernoulli or feature-dependent labeling, see [SCAR/SAR simulation](sar_simulation.md).
`make_pu_labels(..., mechanism="fixed_count", n_labeled=20)` is the unified dispatcher.

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

## 4. Migrate existing calls

| Existing usage | Replacement |
|---|---|
| `make_case_control_labels(y_true, n_labeled, random_state)` | `make_fixed_count_labels(y_true, n_labeled, random_state)` |
| `make_pu_labels(..., mechanism="case_control")` | Change to `mechanism="fixed_count"`, retaining other arguments |
| Gaussian teaching code intended to represent independent TS | `make_gaussian_case_control_data(..., class_prior=...)`, accepting four return values |

The legacy `case_control` names retain same-pool behavior and seeded random sequences, emit a
`FutureWarning`, and are scheduled for removal in 2.0. The new fixed-count entry accepts all-positive
inputs; legacy aliases preserve their original two-class validation. `make_gaussian_pu_data`
continues to provide the teaching case where all positives are visible and U comes from negatives;
its third return value is the true positive proportion of that concatenated table.

See the [API reference](../reference/api.md#data-generation) for valid parameter
domains and complete return fields.
