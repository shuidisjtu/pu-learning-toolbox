# PU Model Selection: PA, OA, and Supervision Roles

## 1. Define which supervision may be read

A research experiment selects hyperparameters, checkpoints, and thresholds before
evaluating the selected model on test data. The current
[ExperimentRunner](../howto/custom_experiments.md) expresses these boundaries with
four roles:

| Role | Available supervision | Purpose |
|---|---|---|
| train | PU labels | Train candidates and fit preprocessing |
| pu_val | PU labels | PA or a custom PU selection criterion |
| clean_val | Complete true labels | OA selection |
| test | Complete true labels | Final evaluation of the selected models |

Isolate roles at the original-source level. PA reads PU labels from `view="pu"`.
The four-role runner currently receives clean inputs, then a Generator constructs
the train/pu_val views. For existing PU data, call estimators directly and design validation explicitly; use `ExperimentRunner` for four-role experiments.

## 2. PA: conditions and formulas

PA (proxy accuracy) scores candidates using PU validation labels and a population
class prior. Following Wang et al., let $`L`$ denote labeled positives,
$`U`$ the native unlabeled set, and
$`\hat Y=\mathbf1\{f(X)\ge\tau\}`$ the predicted class at threshold $`\tau`$.

For PA and population Accuracy to have the same ranking in expectation, the labeled
positives must represent the target population's positive distribution (for example,
under SCAR), the population prior $`\pi_{population}`$ must be correct, and the
pool used to estimate the population distribution must match the validation sampling
design. The two sampling designs give the following formulas.

**OS: label some positives within one population sample.** The full validation
partition $`L\cup U`$ comes from the target population. Compute the second term,
the predicted-negative proportion, on this full partition:

```math
PA_{OS}=2\pi_{population}\widehat P(\hat Y=1\mid S=1)
        +\widehat P_{L\cup U}(\hat Y=0),
```

**TS: obtain additional positives P independently of a population-mixture U.**
The native $`U`$ comes from the target population. Compute the second term on
$`U`$:

```math
PA_{TS}=2\pi_{population}\widehat P_L(\hat Y=1)
        +\widehat P_U(\hat Y=0).
```

In both formulas, the first term estimates the weighted positive-recall contribution
and the second estimates the predicted-negative proportion in the target population.
The sampling design determines the population-estimation pool; see
[PU input sampling](pu_sampling.md). Let true-positive recall be $`a`$ and the
predicted-positive proportion among true negatives be $`b`$. At population level:

```math
E[PA]=2\pi a+1-\pi a-(1-\pi)b
     =Accuracy+\pi.
```

For a fixed prediction rule and threshold independent of validation data, expected
PA differs from population Accuracy by the constant $`\pi`$, so their expected
rankings agree. Empirical PA fluctuates on finite validation sets; evaluate
post-selection performance on independent test data. PA proxies Accuracy+π and
can exceed 1. Changing the prior changes the positive-term weight and can change
the selected threshold.

The toolbox's `proxy_accuracy` and `ProtocolPA` use the **OS sampling formula**;
their implementation is in
[experiment/strategies.py](../../../pu_toolbox/experiment/strategies.py).

```python
import numpy as np
from pu_toolbox.experiment.strategies import proxy_accuracy

scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
y_pu = np.array([1, 1, 0, 0, 0])
pa = proxy_accuracy(scores, y_pu, threshold=0.5, class_prior=0.4)
assert abs(pa - 1.2) < 1e-12  # 2 * 0.4 * 1 + 2 / 5
```

The toolbox's official Survey workflow compares PA and OA under SCAR and uses OA
under SAR. Record additional identification assumptions and supervision budgets
for custom SAR criteria.

## 3. OA: conditions and formula

OA (oracle accuracy) uses the complete true labels in clean_val to calculate
validation accuracy. Here, oracle means that selection may read true validation
labels. For $`D_{\mathrm{val}}=\{(x_i,y_i)\}_{i=1}^{n_{val}}`$:

```math
OA(f,\tau)=\frac{1}{n_{val}}\sum_{i=1}^{n_{val}}
\mathbf1\!\left[\mathbf1\{f(x_i)\ge\tau\}=y_i\right].
```

The clean_val labels must be reliable for both classes, represent the target
population, and be isolated from training and final test data at the original-source
level. For a fixed prediction rule and threshold independent of validation data,
expected empirical OA equals target-population Accuracy.

Both OS and TS studies can obtain an independent clean_val from the target population
and use this OA formula. It directly compares predicted and true classes, taking
model scores, a threshold, and true labels as input. SCAR/SAR describes the labeling
mechanism of PU data; OA requires complete validation labels. In TS, the concatenated
additional P and U training table has its own prior, while clean_val is sampled from
the target population.

The toolbox's `ProtocolOA` selects on clean_val using this criterion. Independent
test data evaluates the selected model's generalization.

## 4. Independent PA and OA selection

`ProtocolPA` and `ProtocolOA` independently select hyperparameters, a checkpoint,
and a threshold from the same training candidate trajectories, then compute separate
test metrics. Each protocol retains its own candidate, checkpoint, threshold, and
test results.

Built-in protocols min-max normalize nonconstant scores on their respective
validation roles before scanning a threshold grid. Constant scores retain their
original values, with affine parameters recorded as `None`. Threshold ties retain
the first value encountered in the grid, which is the smaller threshold for the
default ascending grid. See the [API reference](../reference/api/tools.md#experiment-layer-experiment) for the exact
contract and [example 14](../../../examples/minimal/14_diy_experiment.py) for an
executable independent-selection workflow.

If training also requires independent clean support, record its source, size, and
usage stage. Interpret `PA+clean-support` separately from PA with purely PU
supervision. Record clean_val labels for OA, training support labels, and final
test labels as separate supervision budgets and stages.

## 5. Make experiments comparable

- Reuse underlying splits, representations, and labeling configurations across methods
  for the same seed. State label-reuse rules when scanning labeling rates.
- Fit preprocessing statistics only on train. Record the source and fit scope of
  pretrained models.
- Record candidate counts, epoch budgets, checkpoints, and threshold grids. Count
  shared training cost once.
- Give PN oracle and PU comparisons the same splits, representations, and budgets;
  train the oracle with complete training labels and select it through OA.
- Report prior provenance, sampling design, supervision budgets, and multi-seed
  statistics. Evaluate on test only after priors, candidates, and thresholds are fixed.

See [model tuning](../howto/model_tuning.md) for selection boundaries and the
[Survey protocol](../../research/pu_survey/pu_survey_protocol.md) for study-specific settings.

## 6. Sources

Wang et al. (2026), §3, Equations (3)/(5)/(6), Proposition 1, and the PAUC extension
in §3.3:
[Accessible, Realistic, and Fair Evaluation](https://openreview.net/forum?id=5R11h5o44C).

The fixed-reference comparison for the OS implementation of PA is optional Chinese
background in the
[implementation baseline](../../dev/pu_research_semantics.md#15-pa-在-os-采样设计下的参考实现差异).
