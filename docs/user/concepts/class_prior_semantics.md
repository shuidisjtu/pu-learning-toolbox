# Class-Prior Targets, Sources, and Conversions

## 1. Specify the target distribution for every prior

The same number can denote positive proportions in different distributions. Before
using a class prior, identify its distribution and the prior target required by the
training, model-selection, or evaluation interface.

| Name | Definition | Typical consumer |
|---|---|---|
| Population prior $`\pi_{population}`$ | $`P(Y=1)`$ in the target population | Population risk, OS PA |
| Native-U prior $`\pi_U`$ | Positive proportion within the input U pool | Native-U mixture estimation and its corresponding U risk |
| Risk-pool prior | Positive proportion in the pool actually entering a loss term | CVIR's `unlabeled_positive_prior` |
| Training-table prior $`\pi_{train}`$ | Positive proportion in the concatenated P/U training table | Training-table moment constraints and prior-quantile thresholds on training scores |

Research records should include the value, target, source (known / estimated /
sensitivity setting), estimation pool, input sampling design, and conversion basis.

## 2. Probability-mass identities within one OS population

Assume $`S=1`$ occurs only in the positive class. Let
$`e(x)=P(S=1\mid Y=1,X=x)`$, $`\bar c=E[e(X)\mid Y=1]`$, and
$`q=P(S=1)`$ be the population labeled fraction. Then:

```math
q=\pi_{population}\bar c,\qquad
\pi_U=\frac{\pi_{population}-q}{1-q},\qquad
\pi_{population}=q+(1-q)\pi_U.
```

These mass decompositions hold under one-sided labeling for both SCAR and SAR.
Under SCAR, $`\bar c=c`$. Estimating $`\pi_U`$ additionally requires the
representativeness and identifiability conditions of the chosen estimator.

For example, $`\pi_{population}=0.4,c=0.5`$ gives $`q=0.2,\pi_U=0.25`$.
If an estimate is a U prior of 0.25 and its target really belongs to the same OS
population:

```python
q = 0.2
pi_u = 0.25
from pu_toolbox.prior import native_unlabeled_to_population

pi_population = native_unlabeled_to_population(pi_u, q, input_sampling="os")
assert abs(pi_population - 0.4) < 1e-12
```

For empirical counts, the positive proportion in a complete OS partition equals
$`n_L/n+(n_U/n)\widehat\pi_U`$. In finite samples, this can differ from the
configured parameter of the generating population. True-label counts here serve
synthetic construction and auditing. Unknown-prior experiments determine training
priors through independent estimation or preregistered settings.

The valid domains are $`0\le q<1`$, $`q\le\pi_{population}\le1`$, and
$`0\le\pi_U\le1`$. At $`q=0`$, the two targets have equal values; at
$`q=1`$, U is empty and its prior is undefined. Report the input, target, and
valid domain for invalid values so their source can be checked. Individual algorithms
validate ranges such as `(0, 1)` according to their own contracts.

## 3. TS and the concatenation ratio

In strict TS, U comes from the target population, so $`\pi_U=\pi_{population}`$.
Let additional P occupy a fraction $`r=n_P/(n_P+n_U)`$ of the training table. Then:

```math
\pi_{train}=r+(1-r)\pi_U.
```

For $`r=0.2,\pi_U=0.4`$, the training-table prior is 0.52. Here, $`r`$ is
the P fraction in the concatenated TS table, whereas $`q`$ is the labeled fraction
in an OS population. Interpret finite-pool reuse designs using their actual counts
and distributions; see [sampling concepts](pu_sampling.md).

## 4. What the toolbox prior estimators estimate

| Entry point | Current estimation pool and output target | Conditions or limits |
|---|---|---|
| `KernelMeanPriorEstimator`, KM1/KM2 | Native U is mixture F; P is component H; returns mixture proportion κ | Interpret as the U positive proportion only when positive representativeness and the corresponding identifiability conditions hold |
| `ClassPriorEstimator` (penL1) | Compares basis-function means for P and native U; returns mixture target θ | Interpretable as the U mixture proportion under the relevant density/identifiability conditions; consumers resolve population conversion |
| `ReCPEEstimator` | Copies high-scoring U into P, retaining complete U as the mixture; defaults to KM2 | Approximates the native-U positive prior through a regrouped component weight; metadata records `regrouping_approximation` |
| `BBEMixtureEstimator` | Held-out U from independent TS; outputs `alpha_U` | Requires explicit `input_sampling="ts"`; requires explicit estimation |

The [semantic baseline](../../dev/pu_research_semantics.md) is optional Chinese
background on implementation locations and focused acceptance work. Public `.estimate()`
methods return scalars; the `estimate_target` and `estimation_pool` class attributes
declare the target and pool. The three ordinary estimators above declare
`native_unlabeled`. Third-party `BasePriorEstimator` subclasses default to
`population`; override the class attributes to declare a native-U target.

Equation (7) of the ReCPE paper gives the auxiliary weight
$`\pi'=\pi_U+(1-\pi_U)P_N(A)`$. `copy_fraction` controls the number of
copied observations, while $`P_N(A)`$ denotes the negative class-conditional mass
in the selected region. Reports preserve the auxiliary target and its approximation status.

## 5. From estimation to reporting and training

For direct estimator calls, choose the prior for the actual risk pool and compute conversions using the training partition. In independent TS, native U and population targets coincide. For OS native-U risk, convert the population prior using the training labeled fraction. For a complete OS marginal risk pool, use the population prior. See [migration](../howto/pipeline.md) and [example 05](../../../examples/minimal/05_recpe_pipeline.py).

Contracts also differ by method: CVIR uses `unlabeled_positive_prior` for the selected
risk pool; RobustPU uses `risk_prior` for warm-up U risk;
VPU and GradPU accept it as a compatibility field. Interpret each parameter by where
the method actually consumes it.

Known-prior, estimated-prior, and prior-sensitivity studies answer different questions;
present their results separately. Select priors, estimation methods, and hyperparameters
using permitted training/validation information. Evaluate test only after selection is complete.

## 6. Sources

- Bekker & Davis (2020), §2.3–2.5, equations (5)–(7):
  [PU survey](https://doi.org/10.1007/s10994-020-05877-5), the relationship between
  labeling rate and population prior.
- Mielniczuk & Wawrzeńczyk (2024), §1, equation (2), §2:
  [Single-sample versus case-control](https://arxiv.org/abs/2312.02095v3), U
  distributions under OS/TS.
- Wang et al. (2026), §4, equations (7)–(9):
  [Accessible, Realistic, and Fair Evaluation](https://openreview.net/forum?id=5R11h5o44C),
  internal label shift and risk adaptation.
- Yao et al. (2022), Algorithm 1, equation (7):
  [Rethinking Class-Prior Estimation](https://arxiv.org/html/2002.03673v2), the
  regrouped component and auxiliary weight.

Distinguish mathematical mass identities from the statistical guarantees of individual
estimators. Paper sources for the estimation methods are in the optional
[prior estimation method card (Chinese)](../../research/method_cards/class_prior_estimation.md).
