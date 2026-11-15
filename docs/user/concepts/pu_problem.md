# The PU learning problem

## 1. Problem definition

Positive-unlabeled (PU) learning has two observed supervision roles: **labeled positives** and
**unlabeled samples**. The unlabeled set contains both true positives and true negatives. This
partial supervision distinguishes PU learning from fully supervised classification.

The toolbox accepts existing PU arrays `X, y_pu`, constructs PU labels from fully labeled data
using `make_scar_labels`, `make_fixed_count_labels`, or `make_sar_labels`, and provides synthetic
data generators. For real PU data, record the actual sampling and labeling process. For controlled
experiments, record the source data, sampling policy, and labeling mechanism. Operational details
are in the [sampling guide](../howto/pu_sampling.md) and
[SCAR/SAR simulation guide](../howto/sar_simulation.md).

The standard PU setting assumes one-sided labeling: a labeled sample is truly positive.

```math
S=1 \Longrightarrow Y=1,
\qquad
P(S=1\mid Y=0,X)=0.
```

Here $`S`$ indicates whether a sample receives a positive label and $`Y`$ is its true class.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $`x`$ | Sample features |
| $`y \in \{0,1\}`$ | True class, not fully observed during PU training |
| $`s \in \{0,1\}`$ | Whether the sample is labeled positive |
| $`\pi_{population} = P(Y=1)`$ | Positive prior in the target population; an unsubscripted π below denotes this value |
| $`c = P(s=1 \mid y=1)`$ | Constant positive-class labeling probability under SCAR |
| $`e(x) = P(S=1 \mid Y=1, X=x)`$ | Individual labeling propensity, also written c(x) in older documents |
| $`\bar c = E[e(X)\mid Y=1]`$ | Mean propensity among true positives |
| $`q=P(S=1)`$ | Labeled fraction in the OS population |
| $`\pi_U`$ | Positive proportion in the native unlabeled pool |

Canonical PU labels are $`\{1, 0\}`$: 1 is labeled positive and 0 is unlabeled.
The PNU extension uses $`\{+1, -1, 0\}`$ for labeled positive, labeled negative, and unlabeled.

## 3. The observation identity: q = πc̄

Let $`q = P(S=1)`$ be the observable labeled-positive rate. Then:

```math
q = P(Y=1)P(S=1\mid Y=1) = \pi\bar c,
```

where $`\bar c = c`$ under SCAR and $`\bar c = E[e(X)\mid Y=1]`$ under SAR.
Here q, π, and the labeling event refer to the same OS population. The extra-P fraction r in a
TS concatenation is a different quantity; see [prior semantics](class_prior_semantics.md).

- Given $`\pi`$, the implied mean propensity is $`\bar c = q/\pi`$.
- Given $`\bar c`$, the implied class prior is $`\pi = q/\bar c`$.
- Observed $`q`$ constrains the product; identifying the two factors separately needs additional information.
- $`\pi < q`$ or $`\bar c < q`$ implies a probability greater than 1 and conflicts with the
  observations. Sensitivity analysis marks these settings with `is_consistent=False`.

## 4. Roles of the class prior

Different method families use the prior differently:

- Risk-based methods such as uPU, nnPU, PNU, LLSVM, and Dist-PU, and deep methods such as
  Self-PU, WConPU, and DGPU, use π in a risk or weighted objective. Prior misspecification can
  materially affect results; for example, underestimating π can reduce uPU recall.
- Methods without a mandatory external π include Elkan-Noto, which estimates labeling
  probability c; LBE, which can use optional π for posterior initialization; and InfoMax PU,
  which can estimate π internally. The current `pusb` baseline does not consume a prior.
  Class-prior estimators themselves estimate a prior rather than requiring it as input.

`requires_class_prior=False` means the caller may omit an external prior. It does not imply the
method has no prior-related quantities: LDCE/KLDCE derive a prior from the labeled fraction and
`flip_probability`, and InfoMax can estimate one internally. For the current per-implementation
inventory, see the [semantic baseline](../../dev/pu_research_semantics.md#4-历史-os_or_ts-的逐实现清单)
(optional Chinese implementation background).

The pipeline resolves a call-level population prior first, then a classifier constructor prior
according to its declared target, then estimation when required. See the
[pipeline guide](../howto/pipeline.md) for operations and
[sensitivity analysis](../howto/sensitivity_analysis.md) for misspecification checks.

## 5. Differences from supervised learning

| Dimension | Fully supervised learning | PU learning |
|---|---|---|
| Negative labels | Observed explicitly | Absent; unlabeled samples form a mixture |
| Evaluation | Accuracy and other supervised metrics use true labels | PU risk can depend on π; supervised metrics such as AUC require audit/oracle truth |
| Labeling mechanism | No missing positive labels in the fully labeled setting | SCAR/SAR assumptions affect method choice |

See [SCAR/SAR](scar_sar.md) for definitions and identifiability limits, and the
[data profiling guide](../howto/data_profiling.md) for assumption evidence.

## 6. Describe research settings with independent dimensions

| Dimension | Question | Starting point |
|---|---|---|
| Input sampling | How are P and U obtained? Is U drawn from the population or the remainder after removing P? | [OS/TS and risk views](pu_sampling.md) |
| Labeling mechanism and selection policy | SCAR or SAR? Bernoulli, fixed quota, or ranked selection? | [SCAR/SAR](scar_sar.md) |
| Training risk pool | Which samples enter positive and mixture terms? | [Risk decomposition](pu_sampling.md#3-risk-decomposition-determines-the-loss-pool) |
| Prior target and source | Population, U, or concatenated table? Known or estimated? | [Prior semantics](class_prior_semantics.md) |
| Validation supervision and criterion | Which labels do PA/OA use? How are checkpoints and thresholds selected? | [PU model selection](pu_model_selection.md) |
| Representation and training | Two-dimensional features, a trainable CNN, or a frozen encoder? | [Representations](representations.md) |
| Evaluation and comparison | Which population does test represent? Are splits, seeds, and budgets comparable? | [Custom experiments](../howto/custom_experiments.md) |

For example, “OS input + LBE-A weighted fixed-count labeling + TS-compatible risk pool + known
population prior + OA + trainable ResNet-18” specifies separate choices. SCAR/SAR describes the
labeling mechanism, OS/TS describes input sampling, and historical `os_or_ts` describes a training
risk view.

For existing PU data, read sampling → labeling assumptions → priors → pipeline. For synthetic
or comparative studies, read sampling → labeling policy → risk view → model selection →
representations → DIY experiments. Sources and formula conditions are listed on the relevant
concept pages. The [semantic baseline](../../dev/pu_research_semantics.md) is optional Chinese
background on current implementation and follow-up boundaries.
