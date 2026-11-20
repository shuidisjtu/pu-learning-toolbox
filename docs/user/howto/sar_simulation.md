# Generating SCAR/SAR Data

> Prerequisite: complete the [quickstart](../quickstart.md).
> Concepts: [SCAR/SAR](../concepts/scar_sar.md) defines the mechanisms and three generation presets.

This module constructs controlled PU benchmarks. True labels support simulation of the labeling
mechanism and classification evaluation; propensity supports labeling-mechanism recovery evaluation.
Training interfaces receive the generated PU views.

## 1. Public interfaces

### 1.1 Compute propensity only

```python
import numpy as np
from pu_toolbox.preprocessing import make_sar_propensity

rng = np.random.RandomState(7)
X = rng.normal(size=(100, 8))
y_true = np.arange(100) % 2
propensity = make_sar_propensity(
    X,
    y_true,
    mechanism="linear",
    label_frequency=0.4,
    strength=1.5,
)
```

The result represents `P(S=1|Y,X)`, so true-negative entries are always zero. The function calibrates
an intercept so mean propensity among true positives equals `label_frequency`.

### 1.2 Generate PU labels from complete labels

The following continues with `X` and `y_true` from above. Choose the source partition,
mechanism, policy, and labeling seed explicitly:

```python
from pu_toolbox.preprocessing import PULabelingConfig, SourcePartition, construct_pu_dataset

result = construct_pu_dataset(
    SourcePartition(X, y_true, "example-source", "train"),
    PULabelingConfig("sar", "bernoulli", 42, c=0.4, propensity_rule="nonlinear"),
    positive_classes=(1,), negative_classes=(0,),
)
y_pu = result.observations.labels
propensity = result.audit.propensity
```

`y_pu` uses `1` for observed positives and `0` for U. Bernoulli draws can label no positives:
`ensure_labeled=False` by default. Opt in to `ensure_labeled=True` to top up such a draw;
the audit propensity remains the pre-top-up probability.

### 1.3 Generate synthetic data directly

```python
from pu_toolbox.preprocessing import GaussianSourceConfig, PULabelingConfig, simulate_pu_dataset

result = simulate_pu_dataset(
    GaussianSourceConfig(1000, 8, 0.3, 2.0, 7),
    PULabelingConfig("sar", "bernoulli", 42, c=0.4,
                     propensity_rule="linear", strength=1.5),
)
X, y_pu = result.observations.features, result.observations.labels
y_true, propensity = result.audit.true_labels, result.audit.propensity
manifest = result.manifest
```

Pass `X, y_pu` to the learner. Use `y_true` and `propensity` for evaluation and audit,
and `manifest` for construction provenance. Record clean support and selection by
[data role](custom_experiments.md#2-prepare-four-data-roles). For SCAR choose
`PULabelingConfig("scar", "bernoulli", 42, c=0.4)`; for a fixed quota choose
`"fixed_count"` instead. No mechanism is implicit. See [SCAR/SAR concepts](../concepts/scar_sar.md).

## 2. Labeling-rate semantics

For Bernoulli construction, `c` is the target mean positive-class propensity (called
`label_frequency` by `make_sar_propensity`):

```math
\frac{1}{n_{+}}\sum_{i\,:\,Y_{i}=1} e(x_{i}) = \bar c.
```

Here $`\bar c`$ is the supplied target and $`e(x)`$ is positive-class labeling
propensity. The expected OS labeled fraction satisfies $`E[q]=\pi_{population}\bar c`$; see
[SCAR/SAR concepts](../concepts/scar_sar.md#1-two-labeling-mechanisms) for notation.

The realized fraction comes from Bernoulli sampling and fluctuates around the target. Save:

- Target mean propensity.
- Realized `sum(y_pu)/sum(y_true)`.
- Positive-class propensity standard deviation.
- Data seed and mechanism strength.

Check both realized labeled counts and mechanisms when comparing experiments, so the effects of
label quantity and selection preference remain distinguishable.

### 2.1 Compare Bernoulli and named fixed-quota policies

```powershell
uv run python examples/minimal/17_sar_policies.py --seed 7 --label-frequency 0.4
```

[Example 17](../../../examples/minimal/17_sar_policies.py) applies four policies to the same clean
OS pool. Each reports a label hash, target/realized rate, counts, seed, and score provenance.

| Policy | Mechanism parameters and score source | Sampling policy | Realized count at seed=7 |
|---|---|---|---:|
| linear | Projection standardized within positives, `strength=1.5`, equal feature weights | Independent Bernoulli, `ensure_labeled=False` | 63 |
| nonlinear | Same projection with a centered quadratic term, `strength=1.5` | Independent Bernoulli, `ensure_labeled=False` | 61 |
| LBE-A | LogisticRegression fitted to this clean pool's true labels, `k=10`, 90% weights plus 10% uniform smoothing | Fixed quota, weighted sampling without replacement | 64 |
| LBE-B | Same fit scope, `k=10, shrink_coef=1` | Fixed quota, weighted sampling without replacement | 64 |

The pool contains 160 true positives and the requested rate is 0.4. The Bernoulli expected count
is 64. LBE uses quota `min(n_pos, max(1, round(c*n_pos)))`, with ties-to-even rounding and a
minimum of one available positive. Interpret LBE weights as relative preferences among positives;
see [named mechanism definitions](../concepts/scar_sar.md#6-paper-definitions-and-named-implementations)
for formulas and sources.

In four-role experiments, the Generator constructs train and pu_val views separately and fits
auxiliary scores within each permitted clean partition. Pass `SARLBEAGenerator()` or
`SARLBEBGenerator()` through `ExperimentRunner(generator=...)`, using
`protocols=[ProtocolOA()]` for clean-validation selection. See
[extending a generator](custom_experiments.md#4-extend-the-label-generator) for component replacement.

## 3. SCAR/SAR comparison benchmark

```bash
python -m benchmarks.assigned_methods.run \
  --config benchmarks/assigned_methods/configs/scar_sar_comparison.json \
  --output benchmarks/assigned_methods/results/scar_sar_comparison
```

For each seed, the configuration reuses the same data-generation process and target labeling
rate across SCAR, linear SAR, and nonlinear SAR. In addition to classification metrics it reports:

- `posterior_spearman`.
- `posterior_kendall`.
- `pairwise_ranking_accuracy`.
- `propensity_mse_positive`.
- `propensity_spearman_positive`.

SCAR truth has constant propensity, so propensity rank correlations are mathematically undefined
and are recorded as null values.

## 4. Common mistakes

- **A negative receives `y_pu=1`:** this violates one-sided PU labeling and is a data-generation error.
- **Mixing supervision roles:** record training, PU selection, OA, and final evaluation labels
  separately for train, pu_val, clean_val, and test.
- **Confusing mean propensity with realized labeling rate:** the former is calibrated; the latter
  has sampling variation.
- **Ignoring feature scale:** the simulator standardizes within true positives before constructing
  propensity. State the scale handling for external custom mechanisms too.
- **Reporting only accuracy:** selection-bias methods should also report ranking and propensity
  recovery quality when those quantities are available in the controlled benchmark.

## Next steps

- Check generated data quality and assumption evidence: [data profiling](data_profiling.md).
- Look up exact parameter contracts: [API reference](../reference/api/tools.md#data-generation).
