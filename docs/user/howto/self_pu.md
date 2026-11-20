# Training a Self-PU classifier

> Prerequisite: complete the [quickstart](../quickstart.md).
> Read [the PU problem](../concepts/pu_problem.md) and
> [prior semantics](../concepts/class_prior_semantics.md) for prior targets.
> The [Self-PU method card](../../research/method_cards/Self-PU.md) provides optional
> Chinese research background.

`SelfPUClassifier` is a sklearn-style PyTorch PU classifier with three core components:
a dynamic self-paced trusted set, clean-validation meta reweighting, and dual student/EMA
teacher distillation. The testable clean-room core does not by itself establish full numerical
reproduction on MNIST, CIFAR-10, or ADNI.

## 1. Data protocol

The training labels contain only P/U roles:

- `y_pu == 1`: reliable labeled positives.
- `y_pu == 0`: unlabeled observations, containing both true classes.
- `risk_prior`: the prior required by the selected training risk view. The default direct
  estimator path consumes the positive
  proportion in native U. With independent TS sampling, native U represents the population.
  Under OS, convert a population prior to native U before a direct default-view fit.
  Do not derive training priors from test truth.

Full self-calibration also needs an independent `(X_val, y_val)`. These are true binary labels,
encoded as `{0, 1}` or `{-1, 1}`, with both classes present. They drive meta-gradients and
final teacher selection. Do not substitute test data for this support/validation role.

Direct calls retain the estimator's risk-view contract; `os_or_ts` controls that view,
separately from how input data were sampled.

## 2. Full training

Assume the training, independent clean-validation, and test arrays have already been prepared.
The prior 0.3 below must match the chosen risk view:

```python
from pu_toolbox.estimators.deep import SelfPUClassifier

classifier = SelfPUClassifier(
    risk_prior=0.3,
    warmup_epochs=10,
    self_paced_start=10,
    self_paced_end=50,
    distill_start=50,
    max_epochs=200,
    max_trust_ratio=0.25,
    pace_1=0.20,
    pace_2=0.30,
    require_validation=True,
    random_state=42,
    device="cuda",
)

classifier.fit(
    X_train,
    y_pu_train,
    validation_data=(X_clean_val, y_clean_val),
)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
scores = classifier.decision_function(X_test)
```

Install the torch extra before using this classifier. The example requests CUDA; use
`device="cpu"` when CUDA is unavailable and choose an appropriate execution budget.

The default backbone is a two-layer MLP for tabular or flattened inputs. Direct estimator calls
also accept NCHW image arrays with a custom PyTorch `backbone` returning shape `(n_samples,)`
or `(n_samples, 1)`. This custom-backbone route does not declare native CNN support or expose
the shared `encoder` injection interface.

## 3. Ablation without clean validation

With no `validation_data` and `require_validation=False`, the classifier warns and:

- Keeps dynamic trusted-set updates.
- Disables validation meta reweighting.
- Keeps student/teacher distillation.
- Sets `calibration_mode_="ablation"`.

Teacher selection then depends on the available data. If independent `pu_validation_data`
is supplied, PU-validation nnPU risk selects a teacher/checkpoint. With neither clean nor PU
validation, training-set nnPU risk selects the final teacher.

This path is not full Self-PU. For full-meta experiments, set `require_validation=True`
so missing clean validation fails immediately. Use the direct estimator API or a trainer that
explicitly supplies isolated support data.

## 4. Training stages and state

The two students have distinct final paces. Their effective trusted ratios are
`min(pace_i, max_trust_ratio)`. Each update reranks the complete native-U pool, so previously
trusted observations can leave. Positive/negative directions are selected in pairs. Targets
are model probabilities rather than hard 0/1 labels.

After fitting, inspect:

| Attribute | Content |
|---|---|
| `trusted_indices_` | Per-student global indices, soft labels, and selection directions |
| `trusted_history_` | Per-epoch targets, realized sizes, entries, and exits |
| `reweight_history_` | Meta activation, fallback, CE support fraction, and weight quality |
| `distillation_history_` | Hard-sample fraction and student/teacher MSE |
| `training_history_` | nnPU, trusted CE, calibrated CE, consistency, and total losses |
| `best_teacher_index_` | Selected teacher 1 or 2 |
| `teacher_selection_basis_` | Clean-validation accuracy, PU-validation nnPU risk, or training nnPU risk in ablation |

`reweight_gamma` limits the fraction of samples receiving CE weights. If all meta influences
in a column are nonpositive, stable uniform fallback is used and recorded as `ce_fallback`
or `pu_fallback`. Non-`None` `sample_weight` raises `NotImplementedError`: weighting only
some losses would change algorithm semantics.

The default `distillation_scope="untrusted_u"` preserves the established restricted scope.
The explicit `"paper_population"` option covers P plus untrusted U for student consistency
and P plus all U for teacher consistency. Record this setting when comparing results;
training-view calibration does not change those identity-based sets.

## 5. Checkpoints and reproduction boundaries

```python
checkpoint = classifier.get_training_checkpoint()
```

The checkpoint includes both students, both teachers, optimizer/scheduler states, trusted sets,
training histories, the class prior, and final teacher selection. It is an independent copy
that can be passed to `torch.save`. The interface does not promise automatic restoration
across different backbones; use the same constructor configuration and architecture when restoring.

Core estimator coverage does not establish completion of these separate reproduction tasks:

- Official MNIST/CIFAR-10/ADNI splits and preprocessing.
- Paper-specific 6-layer, 13-layer, and multi-branch backbones.
- Five-seed, full 200-epoch studies, ablations, and paper-table comparisons.
- Numerical comparisons with the official commit and historical dependency environment.

Run the small executable example:

```bash
python examples/minimal/10_self_pu.py
```

This script is a CPU training smoke example. It converts the declared population prior 0.3
using only the training partition's labeled fraction with
`population_to_native_unlabeled(..., input_sampling="os")` before fitting the default risk view.
Its clean validation labels provide extra supervision for meta/teacher selection; the script
does not report independent-test performance or establish numerical reproduction.

## Next steps

- For an independent support artifact, inspect the explicit TS full-meta/checkpoint route in
  the [method card](../../research/method_cards/Self-PU.md) (Chinese). It records extra label
  budgets and `PA+clean-support`/OA; interpret that supervision separately from purely PU PA,
  and check the applicable versioned Survey recipe before attributing results.
- For the generic workflow and its available input roles, see [pipeline](pipeline.md).
- Use [data profiling](data_profiling.md) for independent diagnostics.
