# SCAR/SAR mechanisms and identifiability

## 1. Two labeling mechanisms

Let $`Y`$ be the true class and $`S`$ indicate an observed positive label. Two common mechanisms are:

```math
\text{SCAR}: P(S=1\mid Y=1,X)=c,
\qquad
\text{SAR}: P(S=1\mid Y=1,X)=c(X).
```

- **SCAR** (Selected Completely At Random): the positive-class labeling probability is a
  constant $`c`$, independent of features.
- **SAR** (Selected At Random): propensity $`c(x)`$ varies with features, producing
  instance-dependent selection bias.

## 2. Data generation mechanisms

The `mechanism` parameter of `make_sar_propensity` / `make_sar_dataset` supports:

| Mechanism | Definition | Use |
|---|---|---|
| `scar` | $`P(S=1\mid Y=1,X)=c`$, setting `label_frequency` directly | SCAR control |
| `linear` | $`P(S=1\mid Y=1,X)=\sigma(\gamma\,w^\top\tilde{x}+b)`$: sigmoid of a standardized feature projection; $`\gamma`$ is `strength` and $`b`$ a calibrated intercept | Monotone instance-dependent bias |
| `nonlinear` | Adds a centered quadratic term: $`\sigma(\gamma\,(w^\top\tilde{x}+\tfrac12\,q)+b)`$, where $`q`$ is the squared projection centered over true positives | Nonlinear selection bias |

`feature_weights` controls the projection direction. By default, all features receive equal
weights, normalized to unit L2 norm. With `strength=0`, both linear and nonlinear mechanisms
reduce to constant propensity, providing a useful implementation boundary check. See the
[SAR simulation guide](../howto/sar_simulation.md) for generation examples.

**The default is SAR:** omitting `mechanism` in `make_sar_dataset` resolves to `"linear"` and
emits a `UserWarning`. Class-prior estimators and many PU classifiers assume SCAR. Before
using generated data to evaluate such a method, choose the mechanism explicitly; use
`mechanism="scar"` for a SCAR setting.

## 3. Observational screening and positive-class audits

Observational mode compares labeled positives with the unlabeled mixture, providing a screening
signal for distribution differences. Audit mode compares labeled and unlabeled samples **within
known true positives**, testing whether selection depends on positive-class features.

The toolbox separates these evidence types:

| `evidence` | Data used | `is_identifying` | Interpretation |
|---|---|---:|---|
| `observed_mixture` | All labeled positives versus the unlabeled mixture | `False` | Distribution-difference screening between labeled positives and U |
| `audited_positives` | Labeled versus unlabeled samples within known true positives | `True` | Directly checks whether $`S`$ depends on positive-class features |
| `not_evaluated` | Non-finite features prevent evaluation; insufficient sample size instead gives `inconclusive` without changing the evidence type | `False` | Address the reported data-quality issues first |

`plausible` means the current test did not detect strong feature dependence. Detection power
depends on sample size, model capacity, and observed features.

## 4. Consequences for method choice

- Methods assuming SCAR, such as Elkan-Noto, uPU, and nnPU, can be biased when labeling actually
  depends on features.
- SAR methods need their own additional structure. For example, LBE jointly models class and
  labeling propensity, while DGPU depends on conditional generation components. The current
  `pusb` is an observed-P-versus-U logistic baseline; `pusb_kernel` is a separate kernel method.
  See [method selection](method_selection.md) for implementation identities and conditions.
- When profiling reports `sar_signal`, investigate how labels were obtained, choose comparisons
  according to method assumptions, and perform sensitivity analysis.

See the [data profiling guide](../howto/data_profiling.md) for how these
signals are used, and [method selection](method_selection.md) for mismatch risks.

## 5. Generation probabilities, sampling weights, and realized counts

A mechanism describes selection preferences; a sampling policy determines how they become labels.
Record both separately.

| Entry point | Meaning of its values | Policy and counts |
|---|---|---|
| `make_sar_propensity` | Individual Bernoulli labeling probabilities for true positives; zero for negatives | Positive-class mean calibrated to `label_frequency` |
| `make_sar_labels` | Uses those probabilities | Independent Bernoulli draws; the realized count is random; fallback labeling is enabled by default |
| `make_scar_labels` / `SCARGenerator` | c is the requested fraction of true positives to label | Uniform sampling without replacement; count is `min(n_pos, max(1, round(c*n_pos)))`; no positives gives all U |
| `SARLBEAGenerator` / `SARLBEBGenerator` | Relative sampling weights among true positives | Fixed quota, weighted sampling without replacement; weights express relative selection preference |

With `make_sar_labels(ensure_labeled=True)`, if Bernoulli sampling selects no positive, the
highest-propensity true positive is labeled. A tie chooses the first in input order. Returned
propensities describe the Bernoulli phase; record the fallback separately through `ensure_labeled`.
Use `False` for a pure Bernoulli study. Fixed-count rounding and the minimum of one label can also
make the realized fraction differ from the requested rate.

`make_scar_labels` and `SCARGenerator` produce identical labels for identical input labels,
labeling rates, and seeds. Full selection still advances their random streams identically,
which matters when chaining multiple data-construction calls.

The current LBE generators fit an auxiliary LogisticRegression using true labels in the clean
partition passed to that call, then select P only among its true positives. Train and `pu_val`
are generated separately; auxiliary scores are not fitted across roles. Truth used for simulation
is excluded from the PU views passed to the learner or PA.

## 6. Paper definitions and named implementations

Let $`\hat p(x)`$ be the auxiliary positive-class probability. The table distinguishes the R4
paper's Appendix C.3 from inspected implementations so shared names do not conceal formula changes.
Here R4 denotes Chen et al. (2026), listed below.

| Mechanism | R4 paper | Toolbox implementation or reference-code difference |
|---|---|---|
| LBE-A | $`\hat p^{10}`$ weights, fixed-count sampling without replacement | Default `k=10`; after normalization, mix in 10% uniform weight, as in the reference code |
| LBE-B | $`(1-\hat p)^{10}`$ weights, fixed-count sampling without replacement | Uses `clip(1.5+shrink_coef-p, 0)^k`, with defaults `shrink_coef=1, k=10`: the reference code's `(2.5-p)^10` |
| PUSB selection preset | Rank by $`\hat p^{20}`$ and take top-N | The reference uses acceptance/rejection followed by truncation and may return fewer than requested; the toolbox does not yet provide this named generation preset |

R4 Appendix C.2 uses `floor(c*n_pos)` for its quota. The toolbox uses the rounding/minimum-one
policy described above. Record the paper formula, reference implementation, and toolbox formula
with their versions. The toolbox's `pusb` **classifier** is a separate entry from the PUSB
**data selection mechanism** in this table.

A simulation record should include mechanism formula/parameters, auxiliary-score fitting scope,
requested/realized counts and rates, sampling policy, seed, rounding/fallback rules, and source
version. LBE metadata records counts, parameters, auxiliary-truth provenance, and
`sampling_policy`. The general Bernoulli entry point can return propensity. See the
[SAR simulation guide](../howto/sar_simulation.md) and [API reference](../reference/api/tools.md#data-generation)
for operational contracts.

## 7. Sources

- Bekker & Davis (2020), Definitions 1/2:
  [PU survey](https://doi.org/10.1007/s10994-020-05877-5), SCAR/SAR definitions and assumptions.
- Chen et al. (2026), Appendix C.2/C.3:
  [PU-Bench](https://openreview.net/forum?id=tb8DabMbMq), fixed-count mechanisms and paper formulas.
- [Pinned reference code](https://github.com/XiXiphus/PU-Bench/blob/2d95a19eefd72e66ff30128ec2f65e1d1d4cc077/data/data_utils.py):
  LBE-A/B smoothing and weights at lines 443–472; PUSB acceptance/rejection at lines 86–122.
- Toolbox implementations: [propensity/Bernoulli](../../../pu_toolbox/preprocessing/selection_bias.py),
  [fixed-count labeling](../../../pu_toolbox/preprocessing/pu_labeling.py), and
  [experiment generators](../../../pu_toolbox/experiment/strategies.py).
