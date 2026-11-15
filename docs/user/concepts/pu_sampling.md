# PU Input Sampling and Training Risk Views

## 1. Establish where U comes from

PU labels `1/0` identify labeled-positive and unlabeled roles. The sampling design
determines the distribution of U. Let $`Y`$ be the true class and let
$`\pi_{population}=P(Y=1)`$ be the target population prior.

| Input design | How P (labeled positives) is obtained | Distribution of native U |
|---|---|---|
| OS / SS (one-sample / single-sample) | Label some positives within one population sample | The remaining unlabeled observations, distributed as $`X\mid S=0`$ |
| Strict TS / CC (two-sample / case-control) | Obtain an additional, independent P sample from the positive class-conditional distribution | A mixture sampled from the target population, with prior $`\pi_{population}`$ |
| Finite-pool benchmark construction | Draw or reuse P from an existing pool | Depends on whether P is removed, sampling uses replacement, or the full pool is reused |

In OS, $`S`$ is a labeling event within the population. In TS, the sampling design
determines P/U membership and their concatenation ratio. Either design can be combined
with SCAR or a named SAR selection mechanism; see [SCAR/SAR](scar_sar.md) for mechanisms
and sampling policies. When additional P is selected under SAR, both the positive
class distribution and the selection mechanism determine its distribution. Risk
estimation must account for that positive-selection bias.

For finite-pool designs, record the source pool, original indices, replacement policy,
and reuse across roles. For example, using the complete training pool as U and selecting
P from its positives places the same source observation in both P and U. Chen et al.
(2026), [PU-Bench](https://openreview.net/forum?id=tb8DabMbMq), use this construction
in Appendix C.2. Tracking observation identities makes reuse traceable and allows
training, validation, and test splits to follow original source identities.

## 2. A prior example for OS and TS

Suppose the OS target population prior is 0.4 and the mean positive labeling rate
is 0.5. The population labeled fraction is $`q=0.4\times0.5=0.2`$. After removing
the labeled positives, the positive proportion in native U is:

```math
\pi_U=\frac{0.4-0.2}{1-0.2}=0.25.
```

In strict TS, U is sampled directly from the same target population, so its target
prior remains 0.4. If additional P makes up 20% of the concatenated TS training table,
that table's target positive proportion is $`0.2+0.8\times0.4=0.52`$. Name the
population, U, and training-table targets separately. Empirical proportions also
vary with finite-sample randomness. See [prior semantics](class_prior_semantics.md)
for the derivation.

## 3. Risk decomposition determines the loss pool

When P represents the true positive class distribution, expected classification risk
can be written as:

```math
R(f)=\pi_{population} E_{X\mid Y=1}[\ell(f,+1)-\ell(f,-1)]
     +E_{population}[\ell(f,-1)].
```

Strict TS uses native U to estimate the final term; OS uses the complete $`L\cup U`$.
Under SAR, labeled positives follow the selected positive distribution, so risk
estimation must also address selection bias. Consult each implementation for its
statistical assumptions and risk decomposition.

The current [TrainingView implementation](../../../pu_toolbox/core/training_views.py)
distinguishes three roles:

| Role | `requested_view="os"` | `requested_view="ts"` |
|---|---|---|
| `positive` | Labeled P | Labeled P |
| `native_unlabeled` | Input U with label 0 | The same native U |
| `loss_unlabeled` | Native U | Native U followed by P |

Here, `ts` constructs a **TS-compatible training risk pool from OS input**.
Positive observations enter both the positive loss and the population-mixture loss;
their PU labels retain their original values. This construction applies to train.
Validation and test keep their respective original views. Source indices are
preserved for each role.

```python
import numpy as np
from pu_toolbox.core.training_views import build_training_view

X = np.arange(20).reshape(10, 2)
y_pu = np.array([1, 1] + [0] * 8)
view = build_training_view(X, y_pu, requested_view="ts")
assert len(view.positive_indices) == 2
assert len(view.native_unlabeled_indices) == 8
assert len(view.loss_unlabeled_indices) == 10
```

This example shows observation roles within a training view. Each algorithm trains
with its supported risk view; the observation pools, mini-batch sampling proportions,
and loss weights jointly determine its risk estimate. Prepare inputs and priors
according to the method's API, and record batch construction and weighting. The
[implementation baseline](../../dev/pu_research_semantics.md#4-历史-os_or_ts-的逐实现清单)
provides optional per-method background in Chinese.

## 4. Current generation entry points and limits

- `make_scar_labels` labels a fixed quota of positives within one clean pool: an OS construction.
- `make_fixed_count_labels` uniformly selects a specified number of P observations from
  the same pool and preserves source row order. `make_pu_labels(mechanism="fixed_count")`
  is its dispatch entry point.
- `make_case_control_labels` and `make_pu_labels(mechanism="case_control")` are
  compatibility names for this same-pool policy. They issue `FutureWarning` and are
  scheduled for removal in 2.0.
- `make_gaussian_pu_data` currently labels every generated positive as P; all U observations
  come from the negative class. Its returned `n_p/(n_p+n_u)` is the true positive
  proportion in that concatenated table.
- For OS teaching data with mixed U, use `make_scar_dataset`; see the
  [API reference](../reference/api.md).
- `make_gaussian_case_control_data` independently generates positive P and population-mixture
  U, returning the configured population prior, empirical U/training-table priors, and
  source identities. The [sampling tutorial](../howto/pu_sampling.md) provides runnable
  usage and a migration table.

For real data, first investigate how observations were obtained. In comparative studies,
split train / pu_val / clean_val / test by original source before constructing PU labels
in permitted partitions. See [custom experiments](../howto/custom_experiments.md) for
a four-role example.

## 5. Sources

- Bekker & Davis (2020), §2.3–2.5, equations (5)–(7):
  [Learning from positive and unlabeled data: a survey](https://doi.org/10.1007/s10994-020-05877-5).
- Mielniczuk & Wawrzeńczyk (2024), §1, equation (2), §2–2.1, equations (16)–(19):
  [Single-sample versus case-control sampling scheme](https://arxiv.org/abs/2312.02095v3).
- Wang et al. (2026), §4, equations (7)–(9), Algorithm 1:
  [Accessible, Realistic, and Fair Evaluation](https://openreview.net/forum?id=5R11h5o44C).
- Chen et al. (2026), Appendix C.2:
  [PU-Bench](https://openreview.net/forum?id=tb8DabMbMq).
