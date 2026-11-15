# API reference

> **Scope:** this is the authoritative public API contract reference. Each entry contains its
> signature, parameter table, return structure, and minimal example. For concepts and complete
> walkthroughs, follow links to [concepts](../concepts/), task-oriented [guides](../howto/),
> and [examples/minimal](../../../examples/minimal/). Looking up a parameter contract does not
> require following those links.
>
> **Version:** **v1.11.0**, matching `pyproject.toml`; updated with package releases. The
> [release process](../../dev/release_process.md) is optional Chinese maintenance background.
> **Coverage:** `scripts/check_api_docs.py` checks that this page mentions every public export
> and registered algorithm name. New public APIs must be added here for the gate to pass.

## Reading guide

| Task | Location |
|---|---|
| Run a first experiment in five minutes | [Quickstart](../quickstart.md) |
| Understand PU settings, SCAR/SAR, and method selection | [Concepts](../concepts/) |
| Follow a complete task workflow | [Guides](../howto/), starting from the [user index](../README.md) |
| **Look up an API signature, parameters, or return contract** | **This page** |
| Run an example script | [examples/minimal/](../../../examples/minimal/) |
| Read algorithm derivations, formulas, and reproduction status | [Method cards](../../research/method_cards/) (optional Chinese research background) |

> Parameter contracts live here; docstrings are the source of truth for behavioral details
> under ADR-0013. Walkthroughs live in guides/examples, and algorithm research lives in method
> cards. Each information category has one authoritative location, connected by links.

## Symbol quick index

| Module | Section |
|---|---|
| Classifiers and prior estimators; the current set is defined by the registry | [Classifiers and estimators](#classifiers-and-estimators) |
| PUPipeline / PipelineReport | [PUPipeline](#pupipeline) |
| Distribution shift: auditing, adaptation, monitoring, and assumption diagnostics | [Distribution-shift API](#distribution-shift-api) |
| Experiment layer for research workflows | [Experiment layer](#experiment-layer-experiment) |
| Tuning and comparison | [PUTuner](#putuner) · [PUModelComparator](#pumodelcomparator) |
| Utilities | [build_encoder](#build_encoder) · [profile_pu_data](#profile_pu_data) · [recommend](#recommend_methods--recommend_from_profile) · [analyze_pu_sensitivity](#analyze_pu_sensitivity) · [build_diagnostic_report](#build_diagnostic_report) |
| Data generation: SCAR/SAR simulation | [Data generation](#data-generation) |
| Errors and exceptions | [Errors and exceptions](#errors-and-exceptions) |

## Classifiers and estimators

### Standalone BBE mixture proportion estimation

`from pu_toolbox.prior import BBEMixtureEstimator, BBEResult, best_bin_estimate`.
This component estimates **α_U**, rather than the OS workflow's population π. It has no
registered name for automatic population-prior estimation.

`BBEMixtureEstimator(*, input_sampling=None, holdout_fraction=0.2, delta=0.1,
gamma=0.01, classifier=None, random_state=None)` requires an explicit
`input_sampling="ts"`; the caller must ensure independent P and marginal-U sampling.
`fit(X, y_pu, *, sample_ids=None)` accepts dense two-dimensional features and one-dimensional
observed roles (1=P, 0=U). It randomly holds out the specified fraction within each role,
rounding upward; both roles must remain nonempty in both training and holdout partitions.
The default is a train-only StandardScaler and balanced LogisticRegression(max_iter=1000).
An injected classifier must be cloneable and provide `fit`/`predict_proba` with `classes_={0,1}`.
Optional `sample_ids` must be unique nonempty strings aligned to rows; isolation across sources,
groups, and external roles requires separate verification. `fit` returns self;
`estimate_unlabeled_prior()` returns float α_U and raises `NotFittedError` before fitting.
A failed refit clears the previous estimate. `train_indices_`, `holdout_indices_`, and
`provenance_` expose the internal split for auditing. Clean U labels are not accepted.

`best_bin_estimate(positive_scores, unlabeled_scores, *, delta=0.1, gamma=0.01)`
returns an immutable `BBEResult`. Both inputs must be nonempty one-dimensional finite scores
in [0,1], with higher scores indicating positives; the scorer must be independent of these
held-out samples. `delta` and `gamma` are strictly in (0,1). Threshold selection uses the paper's
complete empirical upper tails and upper-bound criterion. It returns the raw, unclipped ratio
and chooses the lowest threshold when criteria tie. Result fields are `alpha_u`, `threshold`,
`selected_index`, `thresholds`, `positive_counts`, `unlabeled_counts`, `ratios`,
`selection_upper_bounds`, `n_positive`, `n_unlabeled`, `delta`, and `gamma`. The selection upper
bound is not an α_U confidence interval for arbitrary inputs. Estimates of 0 or 1 are not
automatically moved to an interior value acceptable to CVIR. Invalid finite-value, shape,
probability, or sampling-design inputs raise `ValueError`.

```python
from pu_toolbox.prior import BBEMixtureEstimator

model = BBEMixtureEstimator(input_sampling="ts", random_state=42).fit(X, y_pu)
alpha_u = model.estimate_unlabeled_prior()
```

This BBE component supplies a standalone two-dimensional engineering path. It does not include
the TEDⁿ classifier, the original deep recipe, TS-compatible calibration, automatic assignment
to population π, or formal numerical qualification. The [BBE method card](../../research/method_cards/BBE.md)
provides optional Chinese background on source differences and identifiability assumptions.

### P3MixClassifier: experimental registration, native OS

`from pu_toolbox import P3MixClassifier`. `P3MixClassifier(*, variant="base",
hidden_dim=128, max_epochs=100, batch_size=128, batching_policy="role_balanced",
learning_rate=0.001, beta=1.0,
gamma=0.9, pool_size=100, mixup_alpha=1.0, early_learning_weight=1.0,
teacher_decay=0.99, model=None, encoder=None, random_state=0, device=None)`.
`variant` accepts `base`, `e`, and `c` for P3Mix, P3Mix-E, and P3Mix-C; E and C are mutually
exclusive variants. `fit(X, y_pu, *, class_prior=None, sample_weight=None, epoch_callback=None, os_or_ts="os")`
implements only the paper's OS training view and rejects explicit TS. `class_prior` is validated
but unused in training; external `sample_weight` is unsupported. The default input is
two-dimensional features. NCHW images require an injected trainable encoder. Model and training
defaults unspecified by the paper are engineering choices, not a paper recipe.
Numeric parameters must be finite real numbers and reject booleans/strings: `gamma∈[0.5,1]`,
`teacher_decay∈[0,1)`, `early_learning_weight≥0` (strictly positive for E), and strictly positive
`learning_rate`, `beta`, and `mixup_alpha`. `hidden_dim`, `max_epochs`, `batch_size`, and `pool_size`
must be positive integers, excluding booleans. The [P3Mix method card](../../research/method_cards/P3MIX.md)
provides optional Chinese background on sources, parameter bounds, and literature results.

`batching_policy="role_balanced"` preserves the existing behavior: `batch_size` caps each role,
and the shorter role cycles. Explicit `"merged_shuffle"` merges and shuffles roles according to
Algorithm 1, uses `batch_size` as the total batch size, and visits each primary sample once per
epoch. A missing role in a single-role batch contributes zero loss, an engineering convention
not specified in the paper.

Standalone image execution uses `python scripts/run_candidate_image_experiment.py --method p3mix`
with four disjoint NCHW files, `train/pu_val/clean_val/test.npz`, each containing `X/y/indices`.
The `--parameters` JSON must explicitly supply `variant`, `hidden_dim`, `max_epochs`, `batch_size`,
`batching_policy`, `learning_rate`, `beta`, `gamma`, `pool_size`, `mixup_alpha`,
`early_learning_weight`, `teacher_decay`, and `device`; see `--help` for other required arguments.
The default generates a plan. Training requires `--execute --out-dir <new-directory>`.
This entry uses native OS and a shared ResNet-18 with training-set normalization statistics,
rather than the paper backbone. PA/OA independently select student snapshots, and unselected
weights are reclaimed. The selection prior does not enter training, and E's teacher is not
used for model selection.

The same image CLI supports the CVIR engineering candidate through `--method cvir --view os/ts`.
Explicit JSON fields are
`hidden_dim/warm_start_epochs/max_epochs/batch_size/learning_rate/data_device_policy/input_sampling/device`.
Use `input_sampling="os"` and `data_device_policy="cpu_stream"`. The population prior is converted
to the actual risk-pool prior using observed P counts; the parameter file cannot preset
`unlabeled_positive_prior/prior_source`. Both stages record costs and are available for PA/OA
selection. Batch normalization is rejected early only when training provides one value per
channel overall; single-image batches are allowed when spatial dimensions suffice. Shape probes
use a private CPU/eval copy and do not change training data, original BN state, or RNG state.
Python callers can use `CandidateImageCheckpointTrainer(..., population_prior_source="declared-source")`.
CVIR images cannot be trained directly with `fit_candidate_image_classifier`, which does not bind
a prior. Formal recipe and resource approval remain outstanding.

### TEDnClassifier: standalone experimental registration, native TS

`from pu_toolbox import TEDnClassifier`, also exported from `pu_toolbox.estimators.deep`.

`TEDnClassifier(*, input_sampling, holdout_fraction=0.2, delta=0.1,
gamma=0.01, hidden_dim=64, warm_start_epochs=5, max_epochs=20, batch_size=128,
learning_rate=0.001, convergence_tol=0.0001, convergence_patience=5,
random_state=None, device=None, encoder=None)`.

Construction requires explicit `input_sampling="ts"`; the caller supplies independent P and
marginal U. There is no OS calibration entry. The CLI accepts
`--classifier tedn --classifier-param input_sampling=ts`; the UI labels this method case-control
and requires this parameter explicitly. Declaring it does not validate the sampling design.
`fit(X, y_pu, *, class_prior=None, sample_weight=None, sample_ids=None,
epoch_callback=None)` returns self and supports dense two-dimensional features; injecting a
trainable encoder also supports four-dimensional NCHW. Observed PU roles follow core
normalization (1=P, 0 or -1=U), rather than true negative labels. `holdout_fraction`, `delta`,
and `gamma` are strictly in (0,1). Holdouts are rounded upward by observed role, leaving both
training and holdout nonempty. `hidden_dim`, `max_epochs`, `batch_size`, and
`convergence_patience` are positive integers; `warm_start_epochs` is a nonnegative integer.
`learning_rate` is finite and positive; `convergence_tol` is finite and nonnegative.
Optional `sample_ids` are unique nonempty strings aligned to rows. The encoder is deep-copied
and jointly trained, producing two-dimensional features. No fitted input preprocessor is added;
external preprocessing and isolation across groups/external roles require separate verification.

Before each TEDⁿ update, BBE estimates α_U from internal holdout scores. The round then ranks
only training U, discards `floor(α_U*n_U_train)`, and trains for one epoch using the **unweighted
sum** of mean P BCE and mean retained-U BCE. α_U=0 retains all U; α_U=1 raises `ValueError` and
invalidates the fit, without clipping or forcibly retaining a row. After warm-start, stopping
uses the post-update sum of training error rates (P error + retained-U negative-class error):
stop after `convergence_patience` consecutive absolute changes ≤ `convergence_tol`, or at
`max_epochs`. This tolerance is an engineering choice, not a convergence proof or model-selection
authorization.

`predict` splits logits at 0; larger `decision_function` scores indicate positives. There is no
calibrated `predict_proba`. `estimate_unlabeled_prior()` returns the final round's pre-update
α_U, not a new estimate from the final model. Reading it before fitting, after failure, or in
a warm-start callback before any TEDn round raises `NotFittedError`. Non-`None` `class_prior`
raises `ValueError`; non-`None` `sample_weight` raises `NotImplementedError`. Core input checks
may raise `ValidationError`; other parameter/shape/probability errors raise `ValueError` or
`TypeError`. Exceptions from internal validation, training, or callbacks are not retried blindly
and invalidate the fitted state.

Auditable attributes are `train_positive_indices_`, `train_unlabeled_indices_`,
`holdout_positive_indices_`, `holdout_unlabeled_indices_`, `selected_negative_indices_`,
`history_`, `bbe_history_`, `training_provenance_`, `epoch_provenance_`, `optimizer_steps_`,
`n_iter_` (TEDn rounds), `epochs_completed_`, `converged_`, and `stop_reason_`.
`history_` contains warm_start_loss/tedn_loss/alpha_u/selected_negative_count/training_error.
Each `bbe_history_` round records pre-update cumulative steps, α_U, threshold, both tail counts,
upper bound, holdout counts, and δ/γ. `checkpoint_epoch_count=warm_start_epochs+max_epochs` is
an upper bound only. Every completed epoch calls back with `(zero-based cumulative epoch, self)`;
stages are warm_start/tedn, and inference snapshots do not support resumed training. Complete
input stays on CPU. Combined train P/N forward passes use at most 2*batch_size rows;
evaluation/prediction uses at most batch_size.

```python
from pu_toolbox.estimators.deep import TEDnClassifier

model = TEDnClassifier(input_sampling="ts", random_state=42, device="cpu").fit(X, y_pu)
scores = model.decision_function(X_new)
```

This is a standalone native-TS engineering implementation. It does not replace CVIR or population
prior estimation and is outside the frozen matrix and formal qualification. α participates in
feedback updates on a repeatedly reused holdout. The original recipe, numerical results,
multi-seed GPU runs, and owner qualification remain pending. The [TEDn method card](../../research/method_cards/TEDn.md)
provides optional Chinese research background.

Classifiers follow the shared `fit(X, y)` + `predict(X)` + `decision_function(X)` +
`get_params()`/`set_params()` contract; class-prior estimators implement `fit` + `estimate()`.

Registered classifiers must declare `label_semantics`: for `"pu"`, 0 is unlabeled; for `"pn"`,
0 is a true negative; `"pnu"` accepts three main-input roles, positive/negative/unlabeled.
The experiment runner and PU workflows conservatively treat undeclared third-party estimators
as `"pu"`; supervised oracles must explicitly declare `"pn"`. The runner checks the generated
view. `PUPipeline` requires PU main inputs and checks at construction, run entry, and after
actual cloning/construction. Comparison, tuning, and CLI/UI execution reuse this gate; CLI
sensitivity checks it before direct fitting too. Direct three-role PNU fitting remains available,
but a PNU model cannot bypass the PU workflow semantics gate. Equal numeric label sets do not
establish equal meaning. The [label-semantics design](../../dev/label_semantics_plan.md) is
optional Chinese implementation background.

### Registry index

This table provides quick navigation and aliases. Complete signatures, parameter tables, and
return behavior are documented in the family sections below. This page owns parameter contracts;
method cards provide optional Chinese research background.

| Registered name (aliases) | Class | Family | Main constructor parameters | Method card (Chinese) |
|---|---|---|---|---|
| `class_prior_estimation` (`cpe`, `pen_l1`) | `ClassPriorEstimator` | class-prior | `sigma` / `reg_lambda` / `theta_grid` / `n_centers` | [class_prior_estimation](../../research/method_cards/class_prior_estimation.md) |
| `recpe` (`re_cpe`) | `ReCPEEstimator` | class-prior | `copy_fraction` / `base_estimator` | [ReCPE](../../research/method_cards/ReCPE.md) |
| `elkan_noto` (`en`) | `ElkanNotoClassifier` | classic | `base_estimator` / `calibration_method` / `n_cv_folds` / `eps` | [Elkan_Noto](../../research/method_cards/Elkan_Noto.md) |
| `upu` (`convex_pu`) | `UPUClassifier` | risk | `class_prior` / `loss` / `reg_lambda` | [Convex uPU](../../research/method_cards/Convex_Formulation_for_PU_DATA_Learning.md) |
| `nnpu` (`nn-pu`) | `NonNegativePUClassifier` | risk | `model` / `encoder` / `class_prior` / `loss` / `optimizer` | [nnPU](../../research/method_cards/nnpu.md) |
| `pnu` | `PNUClassifier` | risk | `class_prior` / `eta` / `reg_lambda` | [PNU](../../research/method_cards/PNU.md) |
| `puet` (`pu_extra_trees`) | `PUExtraTreesClassifier` | risk | `class_prior` / `n_estimators` / `max_features` / `max_candidates` | [PUET](../../research/method_cards/PUET.md) |
| `centroid_pu` (`ldce`) | `LDCEClassifier` | risk | `flip_probability` / `reg_strength` / `centroid_radius` | [LDCE](../../research/method_cards/LDCE.md) |
| `kldce` (`kernelized_ldce`) | `KLDCEClassifier` | risk | `flip_probability` / `sigma` / `reg_strength` | [KLDCE](../../research/method_cards/KLDCE.md) |
| `llsvm` | `LLSVMClassifier` | risk | `alpha` / `beta` / `gamma` / `reg_lambda` / `max_epochs` | [LLSVM](../../research/method_cards/LLSVM.md) |
| `dist_pu` (`distpu`) | `DistPUClassifier` | risk | `class_prior` / `hidden_dim` / `epochs` / `learning_rate` / `distribution_loss` | [Dist-PU](../../research/method_cards/Dist-PU.md) |
| `vpu` (`variational_pu`) | `VPUClassifier` | risk | `hidden_dim` / `max_epochs` / `regularization_weight` / `mixup_alpha` | [VPU](../../research/method_cards/VPU.md) |
| `pulda` (`label_distribution_alignment`) | `PULDAClassifier` | risk | `class_prior` / `warmup_epochs` / `pu_epochs` / `margin` / `mixup_weight` | [PULDA](../../research/method_cards/PULDA.md) |
| `cvir` (`conditional_value_ignoring_risk`) | `CVIRClassifier` | risk | `unlabeled_positive_prior` (U proportion) / `hidden_dim` / `max_epochs` / `warm_start_epochs` | [CVIR](../../research/method_cards/CVIR.md) |
| `pusb` (`biased_pu`) | `PUSBClassifier` | bias-aware | `threshold` / `C` / `max_iter` | [PUSB](../../research/method_cards/PUSB.md) |
| `pusb_kernel` (`kernelized_pusb`) | `PUSBKernelClassifier` | bias-aware | `n_basis` / `cv` / `sigma_grid` / `reg_grid` | [PUSB §7.3](../../research/method_cards/PUSB.md) |
| `lbe` | `LBEClassifier` | bias-aware | `max_iter` / `n_em_iter` / `C` | [LBE](../../research/method_cards/LBE.md) |
| `lbe_neural` (`neural_lbe`) | `NeuralLBEClassifier` | bias-aware / torch | `pretrain_steps` / `n_em_iter` / `m_steps` / `encoder` | [LBE §10](../../research/method_cards/LBE.md#10-2026-10-14-独立神经网络变体) |
| `gradpu` (`grad_pu`) | `GradPUClassifier` | deep | `alpha` / `beta_max` / `model` / `max_epochs` | [GradPU](../../research/method_cards/GradPU.md) |
| `robust_pu` (`robust-pu`) | `RobustPUClassifier` | deep | `class_prior` / `pretrain_epochs` / `episodes` / `spl_type` | [Robust-PU](../../research/method_cards/Robust-PU.md) |
| `split_pu` (`split-pu`) | `SplitPUClassifier` | deep | `class_prior` / `teacher_epochs` / `split_epochs` / `student_epochs` | [Split-PU](../../research/method_cards/Split-PU.md) |
| `tedn` (`transform_estimate_discard`) | `TEDnClassifier` | risk / torch | `input_sampling="ts"` (required) / `holdout_fraction` / `warm_start_epochs` / `max_epochs` | [TEDⁿ](../../research/method_cards/TEDn.md) |
| `lagam` (`la_gam`) | `LaGAMClassifier` | deep | `support_data` (required by fit) / `warmup_epochs` / `num_clusters` | [LaGAM](../../research/method_cards/LaGAM.md) |
| `pan` (`predictive_adversarial_pu`) | `PANClassifier` | deep | `encoder` / `model` / `discriminator` / `adversarial_weight` / `max_epochs` | [PAN](../../research/method_cards/PAN.md) |
| `genpu` (`gen_pu`) | `GenPUClassifier` | deep | `class_prior` / `latent_dim` / `max_epochs` / `classifier_epochs` | [GEN-PU](../../research/method_cards/GEN-PU.md) |
| `holistic_pu` (`holisticpu`) | `HolisticPUClassifier` | deep | `encoder` / `warmup_epochs` / `trend_scale` / `max_epochs` | [Holistic-PU](../../research/method_cards/Holistic-PU.md) |
| `rp` (`rank_pruning`) | `RankPruningClassifier` | classic | `base_estimator` / `n_cv_folds` / `frac_pos2neg` / `min_retained` | [RP](../../research/method_cards/RP.md) |
| `pulns` (`negative_selector_pu`) | `PULNSClassifier` | deep | `support_data` (required by fit) / `episodes` / `discount` / `terminal_weight` | [PULNS](../../research/method_cards/PULNS.md) |
| `self_pu` | `SelfPUClassifier` | deep | `class_prior` / `backbone` / `warmup_epochs` / `self_paced_start` | [Self-PU](../../research/method_cards/Self-PU.md) |
| `infomax_pu` | `InfoMaxPUClassifier` | deep | `class_prior` / `representation_*` / `classifier_*` (see the deep-classifier section below) | [InfoMax-PU](../../research/method_cards/InfoMax-PU.md) |
| `weighted_contrastive_pu` (`wconpu`) | `WeightedContrastivePUClassifier` | deep | `class_prior` / `encoder` / `hidden_dim` / `embedding_dim` | [WConPU](../../research/method_cards/WConPU.md) |
| `p3mix` | `P3MixClassifier` | deep | `variant` / `beta` / `gamma` / `pool_size` / `encoder` | [P3MIX](../../research/method_cards/P3MIX.md) |
| `dgpu` | `DGPUClassifier` | deep | `class_prior` / `generator` / `model` / `hidden_dim` | [DGPU](../../research/method_cards/DGPU.md) |
| — (`km1` / `km2` variant) | `KernelMeanPriorEstimator` | class-prior | `variant="km1"/"km2"` and other kernel-mean settings; supported by `PUPipeline.prior_estimator` | [Kernel_Mean](../../research/method_cards/Kernel_Mean_Class_Prior.md) |

> Registered classifier names work in `PUPipeline(classifier="...")` and CLI `--classifier`;
> aliases are case-insensitive. The family column uses friendly abbreviations. CLI and JSON use
> enum values `class_prior_estimation` / `classic_calibration` / `risk_estimation` / `bias_aware` /
> `deep_pu`. Use `classifier_params` for required constructor arguments other than `class_prior`;
> a configured instance can supply Python-object protocol dependencies. The old prior alias `pe`
> still resolves but emits `FutureWarning`; migrate to `class_prior_estimation` or `cpe`.

### sample_weight support levels

Classifiers retain the common `fit(..., sample_weight=None)` signature and declare behavior
through `get_pu_metadata()["sample_weight_support"]`:

| Value | Behavior | Classifiers (examples) |
|---|---|---|
| `supported` | Weights enter the training objective | `elkan_noto`, `nnpu`, `pusb`, `infomax_pu`, `weighted_contrastive_pu`, `dgpu` |
| `ignored` | Accepted for sklearn API compatibility, unused in training | `llsvm`, `upu`, `pnu`, `centroid_pu`, `kldce`, `dist_pu`, `lbe` |
| `not_implemented` | Non-`None` values raise `NotImplementedError` | `pusb_kernel`, `self_pu`, `gradpu`, `puet`, `vpu`, `pulda` |

Check the field before fitting when weights are required. `ignored` does not claim supplied
weights were applied; consult each method's current metadata for methods beyond these examples.

### Class-prior estimators

#### ClassPriorEstimator (class_prior_estimation; aliases cpe / pen_l1)

penL1 (penalized L1 risk) class-prior estimator. With `sigma=None`, it selects a median-distance
scale adaptively.

```python
ClassPriorEstimator(*, sigma=None, reg_lambda=0.01, theta_grid=None, n_centers=200, standardize=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sigma` | `float \| None` | `None` (adaptive median distance) | RBF width; explicit `sigma` retains the historical fixed-scale behavior |
| `reg_lambda` | `float` | `0.01` | Penalty coefficient |
| `theta_grid` | `np.ndarray \| None` | `None` (generated automatically) | Search grid of θ candidates |
| `n_centers` | `int \| None` | `200` | Number of subsampled centers |
| `standardize` | `bool` | `True` | Standardize X before training |

- Implements `fit(X, y_pu)` + `estimate()`; `estimate_target=estimation_pool="native_unlabeled"`.
  The old alias `pe` is deprecated (`FutureWarning`).
- Optional Chinese research background: [class_prior_estimation method card](../../research/method_cards/class_prior_estimation.md).
  Example: [05_recpe_pipeline.py](../../../examples/minimal/05_recpe_pipeline.py), through `PUPipeline`.

#### ReCPEEstimator (recpe; alias re_cpe)

Regrouping Copy Positive Estimator (ReCPE) for class-prior estimation.

```python
ReCPEEstimator(copy_fraction=0.1, base_estimator=None, classifier=None, classifier_max_iter=1000)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `copy_fraction` | `float` | `0.1` | Fraction of unlabeled observations copied into the positive sample |
| `base_estimator` | estimator \| None | `None` | CPE estimator implementing `estimate()`; `None` uses KM2 |
| `classifier` | estimator \| None | `None` | Binary P-versus-U classifier; positive probabilities rank U |
| `classifier_max_iter` | `int` | `1000` | Maximum iterations for the default logistic classifier |

High-scoring U rows are copied into P while the complete U remains the mixture.
`estimate_target=estimation_pool="native_unlabeled"`; the scalar output is a regrouping
approximation to the native-U prior. `get_metadata()` records `copy_policy`,
`estimate_kind="regrouping_approximation"`, and `auxiliary_target="regrouped_component_weight"`.

- Optional Chinese research background: [ReCPE method card](../../research/method_cards/ReCPE.md).
  Example: [05_recpe_pipeline.py](../../../examples/minimal/05_recpe_pipeline.py).

#### KernelMeanPriorEstimator (km1 / km2 variants)

Kernel-mean class-prior estimator implementing KM1 / KM2.

```python
KernelMeanPriorEstimator(*, variant="km1", kernel_width=None, width_selection="relative",
                         kernel_width_scale=0.1, width_factors=(0.1, 0.316227766, 1.0, 3.16227766, 10.0),
                         epsilon=0.04, lambda_upper_bound=8.0, km2_final_slope_weight=0.2,
                         max_qp_iter=2000, qp_tolerance=1e-7, max_samples_per_group=None,
                         standardize=False, random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `variant` | `"km1"` / `"km2"` | `"km1"` | Algorithm variant |
| `kernel_width` | `float \| None` | `None` | Kernel width; `None` selects it through `width_selection` |
| `width_selection` | `"relative"` / `"mmd_grid"` | `"relative"` | Relative heuristic or MMD-grid width selection |
| `kernel_width_scale` | `float` | `0.1` | Scale for the `relative` strategy |
| `width_factors` | `tuple[float, ...]` | Default grid | Width-factor grid for `mmd_grid` |
| `epsilon` | `float` | `0.04` | KM2 relaxation parameter |
| `lambda_upper_bound` | `float` | `8.0` | Upper scaling-constraint bound |
| `km2_final_slope_weight` | `float` | `0.2` | KM2 final-slope weight |
| `max_qp_iter` / `qp_tolerance` | `int` / `float` | `2000` / `1e-7` | QP iteration limit and tolerance |
| `max_samples_per_group` | `int \| None` | `None` | Maximum samples per group; `None` imposes no cap |
| `standardize` | `bool` | `False` | Standardize before training |
| `random_state` | `int \| None` | `None` | Random seed |

- `PUPipeline.prior_estimator="km1"/"km2"` maps to this estimator.
- `estimate_target=estimation_pool="native_unlabeled"`; returns the U mixture proportion.
- Optional Chinese research background: [Kernel_Mean method card](../../research/method_cards/Kernel_Mean_Class_Prior.md).

#### Prior probability-mass conversion

`from pu_toolbox.prior import native_unlabeled_to_population, population_to_native_unlabeled`.
Both functions accept `(class_prior, labeled_fraction, *, input_sampling="os")` and return
`float`. For OS, they use `q+(1-q)*pi_U` and `(pi-q)/(1-q)`, respectively. TS returns the input
prior. Priors lie in `[0,1]`; the P fraction lies in `[0,1)`. An OS population prior must be at
least the P fraction. Non-real, boolean, nonfinite, and out-of-range inputs raise `TypeError`
or `ValueError` as appropriate.

### Classic wrappers

#### ElkanNotoClassifier (elkan_noto; alias en)

Elkan-Noto empirical-estimation wrapper: single-model calibration followed by probability
correction or weighted retraining.

```python
ElkanNotoClassifier(base_estimator=None, calibration_method="sigmoid", n_cv_folds=3,
                    eps=1e-12, mode="weighted_retraining", random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_estimator` | sklearn estimator | `None` (`LogisticRegression()`) | Base binary classifier; must implement `predict_proba` |
| `calibration_method` | `"sigmoid"` / `"isotonic"` | `"sigmoid"` | Calibration for base estimators other than LogisticRegression; LogisticRegression is used directly |
| `n_cv_folds` | `int` | `3` | OOF stratified CV folds for estimating `c`; must be ≥ 2 |
| `eps` | `float` | `1e-12` | Clipping threshold against division by zero in weights |
| `mode` | `"probability_correction"` / `"weighted_retraining"` | `"weighted_retraining"` | Probability correction (`g(x)/c`) or weighted retraining; the toolbox defaults to weighted retraining (ADR-0016) |
| `random_state` | `int \| None` | `None` | K-fold split seed |

- Optional Chinese research background: [Elkan_Noto method card](../../research/method_cards/Elkan_Noto.md).
  Example: [01_elkan_noto.py](../../../examples/minimal/01_elkan_noto.py).

### Risk-estimation classifiers

#### UPUClassifier (upu; alias convex_pu)

Unbiased PU (uPU / convex PU) classifier minimizing convex PU risk.

```python
UPUClassifier(class_prior, *, loss="squared", reg_lambda=1e-3, basis="linear",
              kernel_width=None, n_centers=None, fit_intercept=True,
              max_iter=1000, tol=1e-6, random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_prior` | `float` | Required | Prior for the training risk pool; the default direct view consumes native-U prior. Can be overridden by the `fit` kwarg `class_prior` |
| `loss` | `"double_hinge"` / `"logistic"` / `"squared"` | `"squared"` | Margin loss: C-DH (convex QP, recommended main variant), C-LL (smooth convex, L-BFGS), or squared (closed-form and fastest, but penalizes large correct margins; toolbox default, ADR-0016) |
| `reg_lambda` | `float` | `1e-3` | ℓ₂ regularization on α; strictly positive; intercept *b* is unregularized |
| `basis` | `"linear"` / `"rbf"` | `"linear"` | Basis type; `"rbf"` subsamples *n_centers* centers from U |
| `kernel_width` | `float \| None` | `None` | RBF width; required for `basis="rbf"` |
| `n_centers` | `int \| None` | `None` (`min(200, n_U)`) | RBF center count; ignored for `basis="linear"` |
| `fit_intercept` | `bool` | `True` | Fit intercept *b* |
| `max_iter` / `tol` | `int` / `float` | `1000` / `1e-6` | Optimization iteration limit and convergence tolerance |
| `random_state` | `int \| None` | `None` | Center-subsampling seed |

- Optional Chinese research background: [Convex uPU method card](../../research/method_cards/Convex_Formulation_for_PU_DATA_Learning.md).
  Example: [02_upu.py](../../../examples/minimal/02_upu.py).

#### NonNegativePUClassifier (nnpu; alias nn-pu)

Nonnegative PU (nnPU) classifier using PyTorch networks.

```python
NonNegativePUClassifier(model=None, *, encoder=None, class_prior=None, loss="sigmoid",
                        beta=0.0, gamma=1.0, optimizer=None, batch_size=256,
                        max_epochs=200, patience=20, random_state=None, device=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module \| None` | `None` (`nn.Linear(d, 1)`) | PyTorch model producing raw scores g(x) |
| `encoder` | `torch.nn.Module \| None` | `None` | Feature encoder replacing the default raw-input path; `model` becomes the score head, or a default `nn.Linear(rep_dim, 1)` head is created. `None` retains the existing path |
| `class_prior` | `float \| None` | `None` | Prior in (0,1) for the training risk pool; the default direct view consumes native-U prior. May also be supplied/overridden through `fit` |
| `loss` | `"sigmoid"` | `"sigmoid"` | Surrogate loss; currently only sigmoid is supported |
| `beta` | `float` | `0.0` | Nonnegative threshold; must be ≥ 0 |
| `gamma` | `float` | `1.0` | Correction-branch step discount in [0,1] |
| `optimizer` | `torch.optim.Optimizer \| None` | `None` (`Adam(lr=1e-3)`) | Optimizer |
| `batch_size` | `int` | `256` | P and U batches independently use `min(batch_size, n_P/n_U)` |
| `max_epochs` | `int` | `200` | Maximum training epochs |
| `patience` | `int` | `20` | Early-stopping patience, active only with `fit(validation_data=...)` |
| `random_state` / `device` | `int \| None` / `str \| None` | `None` / `None` | Seed and torch device |

- Optional Chinese research background: [nnPU method card](../../research/method_cards/nnpu.md).
  Example: [03_nnpu.py](../../../examples/minimal/03_nnpu.py).

#### PNUClassifier (pnu)

Positive-negative-unlabeled (PNU) risk classifier with a common pluggable λ loss and regularizer.

```python
PNUClassifier(class_prior, *, eta=0.0, reg_lambda=1e-3, basis="linear",
              kernel_width=None, n_centers=None, fit_intercept=True, random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_prior` | `float` | Required | Positive-class prior θ_P=P(y=1), strictly in (0,1) |
| `eta` | `float` | `0.0` | PNU tradeoff in [-1,1]: `0`=PN (supervised), `+1`=PU, `-1`=NU |
| `reg_lambda` | `float` | `1e-3` | ℓ₂ regularization; intercept *b* is excluded |
| `basis` / `kernel_width` / `n_centers` | See uPU | `"linear"` / `None` / `None` | Basis and RBF parameters follow uPU semantics |
| `fit_intercept` | `bool` | `True` | Implemented with a constant basis column, following pywsl |
| `random_state` | `int \| None` | `None` | Center-subsampling seed |

- Optional Chinese research background: [PNU method card](../../research/method_cards/PNU.md).
  Example: [04_pnu.py](../../../examples/minimal/04_pnu.py).

#### PUExtraTreesClassifier (puet; alias pu_extra_trees)

PU Extra Trees trained by nnPU quadratic-risk descent; currently a two-dimensional tabular/CPU subset.

```python
PUExtraTreesClassifier(class_prior, *, n_estimators=100, max_depth=None,
                       min_samples_leaf=1, max_features="sqrt", max_candidates=1,
                       bootstrap=False, random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_prior` | `float` | Required | Positive prior $\pi\in(0,1)$ in the actual training risk pool; the default direct view consumes native-U prior. Can be overridden by `fit(..., class_prior=...)` |
| `n_estimators` / `max_depth` | `int` / `int \| None` | `100` / `None` | Tree count and maximum depth |
| `min_samples_leaf` | `int` | `1` | Minimum total P/U observations in each child leaf |
| `max_features` | `"sqrt" \| "all" \| int` | `"sqrt"` | Candidate features per node; `sqrt` rounds upward. Positive integers above the input dimension are capped at that dimension without changing the constructor parameter; this is not author-equivalent behavior |
| `max_candidates` | `int` | `1` | Random thresholds per candidate feature |
| `bootstrap` | `bool` | `False` | Resample P/U separately with replacement; the author code defaults to no resampling |
| `random_state` | `int \| None` | `None` | Forest seed |

`decision_function` is a voting margin in $[-1,1]$, not a calibrated probability; zero margin
predicts negative. Only the paper's nnPU/quadratic variant is implemented, not the full benchmark.
Non-`None` `sample_weight` raises an error.

`bootstrap=True` draws n_P rows from P and draws as many rows as the actual U risk pool contains
(OS: n_U; TS-compatible risk view: n_U+n_P). Each tree's root risk masses are P=the supplied
risk-pool prior and U=1; original-U counts are not incorrectly reused for the TS-compatible pool.
`min_samples_leaf` constrains both children. Thresholds span the complete min/max interval;
only gain>1e-12 is accepted.

- Optional Chinese research background: [PUET method card](../../research/method_cards/PUET.md).

For PUET/RP, optionally import
`from pu_toolbox.experiment.classical_candidate import ClassicalCandidateTrainer`
and inject an instance into `ExperimentRunner.config["trainer"]` to record actual per-tree or
OOF/refit cost. `fit(estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None)` returns
a single-model trajectory without epochs. Successful `training_provenance_` enters the candidate
manifest; failed work is retained in the trainer's `last_receipt`. Optional constructor
`receipt_dir` persists each success/failure cost receipt independently; retries do not overwrite
earlier receipts. `val_pu` is unused for training; PA/OA remain independent runner operations.
Only the exact PUET/RP classes are supported, excluding subclasses and instance tree-builder overrides.

The standalone CPU entry `python scripts/run_classical_candidate_experiment.py --help`
generates a locked plan by default. Fitting requires `--execute --out-dir <new-directory>`.
Inputs are four disjoint two-dimensional `X/y/indices` NPZ files following the four-role contract.
`--parameters` must explicitly include all PUET
`n_estimators/max_depth/min_samples_leaf/max_features/max_candidates/bootstrap` fields, or RP
`n_cv_folds/frac_pos2neg/min_retained`. Seed, population-prior source, and `--view` are separate.
PUET supports OS/TS risk views; RP supports only OS. The entry does not fit or certify input-feature
preprocessing. Exported `selected_PA/OA.pkl` wrap the validation affine transform and selected
threshold in the same module's `ClassicalSelectionPredictor(estimator, selection)`.
`decision_function` retains raw scores; `predict` replays the selected rule. The ordinary
non-epoch runner's `selected_models` do not automatically carry that rule; native `predict`
cannot substitute for the predictions used in the report.

#### LDCEClassifier (centroid_pu; alias ldce)

Label Density Centroid Estimation (LDCE) classifier.

```python
LDCEClassifier(flip_probability, *, reg_strength=1.0, centroid_radius=0.1, mom_groups=10,
               covariance_ridge=0.01, learning_rate=0.01, n_inner_iter=50,
               max_iter=10000, tol=1e-6, random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `flip_probability` | `float` | Required | Probability *h* that a true positive is flipped to an observed negative (censoring rate), strictly in (0,1) |
| `reg_strength` | `float` | `1.0` | Linear-weight ℓ₂ regularization λ |
| `centroid_radius` | `float` | `0.1` | Ellipsoid radius *b* for the centroid constraint; a tuned toolbox default (ADR-0016) |
| `mom_groups` | `int` | `10` | Median-of-means centroid groups *g*; `1` reduces to the ordinary mean |
| `covariance_ridge` | `float` | `1e-2` | Diagonal ridge on centroid covariance; strongly interacts with `centroid_radius` (ADR-0016) |
| `learning_rate` | `float` | `0.01` | Initial subgradient-descent step size |
| `n_inner_iter` | `int` | `50` | Inner gradient steps per outer iteration |
| `max_iter` / `tol` | `int` / `float` | `10000` / `1e-6` | Alternating-optimization limit and tolerance |
| `random_state` | `int \| None` | `None` | Seed |

- Optional Chinese research background: [LDCE method card](../../research/method_cards/LDCE.md).

#### KLDCEClassifier (kldce; alias kernelized_ldce)

Kernelized LDCE with alternating optimization of the ACP formulation.

```python
KLDCEClassifier(flip_probability, *, sigma="scale", reg_strength=1.0, centroid_radius=1.0,
                mom_groups=10, covariance_ridge=0.0, max_acs_iter=50, max_inner_iter=2000,
                inner_tol=1e-6, tol=1e-6, random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `flip_probability` | `float` | Required | Censoring rate *h*, strictly in (0,1) |
| `sigma` | `float \| "scale"` | `"scale"` | RBF bandwidth; `"scale"` uses heuristic `σ = 1/sqrt(n_features)` |
| `reg_strength` | `float` | `1.0` | ℓ₂ regularization λ |
| `centroid_radius` | `float` | `1.0` | Ellipsoid radius *b* |
| `mom_groups` | `int` | `10` | Median-of-means centroid group count |
| `covariance_ridge` | `float` | `0.0` | Diagonal ridge on centroid covariance; `0.0` matches the paper, >0 is a numerical-stabilization variant |
| `max_acs_iter` / `max_inner_iter` / `inner_tol` | `int` / `int` / `float` | `50` / `2000` / `1e-6` | ACS outer limit, inner QP pair-update limit, and KKT tolerance |
| `tol` / `random_state` | `float` / `int \| None` | `1e-6` / `None` | Convergence tolerance and seed |

- Optional Chinese research background: [KLDCE method card](../../research/method_cards/KLDCE.md).

Standalone CPU execution: `python scripts/run_classical_candidate_experiment.py --method kldce --help`.
Explicitly set `--prior-policy derived/population_override` and `--kernel-payload-cap-bytes`;
other four-role/prior-source arguments follow PUET/RP. JSON must supply all
`sigma/reg_strength/centroid_radius/mom_groups/covariance_ridge/max_acs_iter/max_inner_iter/inner_tol/tol`
fields. Manually supplied `flip_probability` is rejected; it is bound to training SCAR's
`1-c_realized`. Only OS is supported; no TS substitution is made.
`KLDCECandidateTrainer(prior_policy=..., kernel_payload_cap_bytes=..., receipt_dir=None)` can be
injected into a nonformal runner. Direct standalone fit takes h from estimator parameters;
without generation validation, it does not claim a verified SCAR binding. Infeasible QP box
constraints or named-matrix payloads above the cap are rejected early. ACS history is not a
neural epoch trajectory. Unobserved total rollback cost is null; the final QP iteration sum
does not stand in for total cost.

#### LLSVMClassifier (llsvm)

Label-dependent linear SVM (LLSVM, LPSVM variant).

```python
LLSVMClassifier(*, alpha=2.0, beta=1.0, gamma=10.0, squash_scale=10.0, reg_lambda=1.0,
                learning_rate=5e-6, max_epochs=3000, n_batches=20, fit_intercept=True,
                intercept_scale=10.0, shuffle=True, random_state=None, early_stopping=True,
                patience=100, tol=5e-4, min_epochs=200)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | `2.0` | Positive squared-hinge loss weight |
| `beta` | `float` | `1.0` | Unlabeled hat-loss weight |
| `gamma` | `float` | `10.0` | Label-calibration loss weight |
| `squash_scale` | `float` | `10.0` | Squash scale *A* in A/π·arctan(f) |
| `reg_lambda` | `float` | `1.0` | ℓ₂ regularization strength |
| `learning_rate` | `float` | `5e-6` | Fixed SGD step size |
| `max_epochs` | `int` | `3000` | Maximum training epochs |
| `n_batches` | `int` | `20` | Mini-batches per epoch |
| `fit_intercept` / `intercept_scale` | `bool` / `float` | `True` / `10.0` | Intercept through constant-feature augmentation and that feature's value |
| `shuffle` | `bool` | `True` | Shuffle each epoch |
| `random_state` | `int \| None` | `None` | Initialization and shuffling seed |
| `early_stopping` / `patience` / `tol` / `min_epochs` | `bool`/`int`/`float`/`int` | `True` / `100` / `5e-4` / `200` | Early stopping on a plateau in the full-data objective; enabled by default as a toolbox choice, while the official code uses a fixed 3000 epochs |

- `get_pu_metadata()["n_epochs"]` reports the actual stopping epoch count.
- Optional Chinese research background: [LLSVM method card](../../research/method_cards/LLSVM.md).

#### DistPUClassifier (dist_pu; alias distpu)

Dist-PU defaults to a clean-room single-stage full-batch MLP. Explicitly combining
`training_recipe="author_staged"` and `distribution_loss="author_histogram"` enables a source-audited
two-stage mini-batch adaptation, including an injected trainable encoder for four-dimensional
images. This adaptation does not port the author's datasets, backbone, or preprocessing, and
does not copy selection by test accuracy.

```python
DistPUClassifier(class_prior, *, hidden_dim=64, epochs=100, batch_size=256, learning_rate=1e-3,
                 alignment_weight=1.0, entropy_weight=0.05, mixup_weight=0.1,
                 distribution_loss="expectation", training_recipe="clean_room",
                 warmup_epochs=60, pu_epochs=60,
                 warmup_learning_rate=5e-4, warmup_weight_decay=5e-3,
                 pu_learning_rate=5e-5, pu_weight_decay=1e-3,
                 warmup_entropy_weight=0.002, pu_entropy_weight=0.004,
                 mixup_alpha=6.0, mixup_entropy_weight=0.04, mixup_bce_weight=5.0,
                 random_state=0, device=None, encoder=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_prior` | `float` | Required | Prior for the training risk pool; the default direct view consumes native-U prior |
| `hidden_dim` | `int` | `64` | Hidden dimension |
| `epochs` | `int` | `100` | Training epochs |
| `batch_size` | `int` | `256` | Ignored by `clean_room`; mini-batch size for both `author_staged` phases |
| `learning_rate` | `float` | `1e-3` | Learning rate |
| `alignment_weight` | `float` | `1.0` | Distribution-alignment loss weight |
| `entropy_weight` | `float` | `0.05` | Entropy-loss weight |
| `mixup_weight` | `float` | `0.1` | MixUp-loss weight |
| `distribution_loss` | `str` | `"expectation"` | Default `"expectation"` is clean-room squared-mean alignment; `"author_histogram"` uses the locked author code's one-bin polar/L1 P/U histogram objective and is required for the author-adapted path |
| `training_recipe` | `str` | `"clean_room"` | `"clean_room"` retains the single-stage full-batch path; opt-in `"author_staged"` runs warm-up and PU mini-batch adaptation |
| `warmup_epochs` / `pu_epochs` | `int` | `60` / `60` | Stage epochs for author adaptation; source scheduler cadence is per epoch and per batch, respectively |
| `warmup_learning_rate` / `warmup_weight_decay` | `float` | `5e-4` / `5e-3` | Warm-up Adam settings |
| `pu_learning_rate` / `pu_weight_decay` | `float` | `5e-5` / `1e-3` | Settings after rebuilding Adam for the PU phase |
| `warmup_entropy_weight` / `pu_entropy_weight` | `float` | `0.002` / `0.004` | Fixed warm-up entropy weight; the PU weight grows from zero to its limit using the source cosine formula |
| `mixup_alpha` / `mixup_entropy_weight` / `mixup_bce_weight` | `float` | `6.0` / `0.04` / `5.0` | PU-phase Beta mixup, mixed-output entropy, and dual-target BCE weights |
| `random_state` / `device` | `int \| None` / `str \| None` | `0` / `None` | Seed and torch device |
| `encoder` | `nn.Module \| None` | `None` | Supported only by staged/histogram training; deep-copied and trained end-to-end, with finite two-dimensional output. Required for images; no silent flattening |

- With an encoder, inputs/labels stay on CPU and training/inference are batched. Pseudo-label
  scans use `min(128,batch_size)` rows; single-row tail batches use BN running statistics.
  Ordinary MLP scans and default training retain their behavior. Scoring restores each layer's
  mode, and old pickles receive missing optional parameters.
- `author_staged` is an engineering adaptation matching the high-level training sequence of
  locked `train.py`. The frozen Pilot retains its original full-batch/feature-adapter path.
  The [Dist-PU method card](../../research/method_cards/Dist-PU.md) provides optional Chinese
  research background and full limitations.

#### VPUClassifier (vpu; alias variational_pu)

Variational PU classification without a class prior. Supports dense two-dimensional features and
four-dimensional NCHW with an injected trainable encoder; the network is an experimental toolbox adaptation.

```python
VPUClassifier(*, hidden_dim=64, depth=2, max_epochs=100, batch_size=128,
              learning_rate=3e-4, regularization_weight=0.03, mixup_alpha=0.3,
              random_state=None, device=None, optimizer_schedule="constant",
              iterations_per_epoch=None, encoder=None, data_device_policy="auto")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hidden_dim` / `depth` | `int` / `int` | `64` / `2` | MLP hidden width and depth; `depth=0` gives a linear head |
| `max_epochs` / `batch_size` | `int` / `int` | `100` / `128` | Training epochs and batch size |
| `learning_rate` | `float` | `3e-4` | Adam learning rate |
| `regularization_weight` | `float` | `0.03` | Log-MixUp consistency weight |
| `mixup_alpha` | `float` | `0.3` | Equal shape parameters of the Beta interpolation distribution |
| `optimizer_schedule` | `str` | `"constant"` | Explicit `"source_reset_halving"` halves the learning rate and rebuilds Adam before epochs 20/40/…, clearing momentum; this is not StepLR |
| `iterations_per_epoch` | `int \| None` | `None` | Positive integer fixes updates per epoch; None retains data-derived counts. The author's script uses 30, which is not adopted by default |
| `encoder` | `torch.nn.Module \| None` | `None` | Required for four-dimensional images; deep-copied and trained end-to-end with finite two-dimensional output followed by the MLP head. The supplied template is unchanged |
| `data_device_policy` | `str` | `"auto"` | Without an encoder, auto retains resident behavior; with one, auto forces cpu_stream. Explicit cpu_stream is accepted; encoder + resident is rejected |
| `random_state` / `device` | `int \| None` / `str \| None` | `None` / `None` | Seed and torch device |

`fit(X, y_pu, *, class_prior=None, sample_weight=None, pu_validation_data=None, epoch_callback=None, os_or_ts="ts")`
validates `class_prior` for compatibility only; it never enters the objective. Non-`None`
`sample_weight` raises an error. `pu_validation_data` accepts a PU-view `DatasetPart` or
`(X_val, y_pu_val)` solely to record per-epoch variational risk. Do not pass a true-label view or
use this record as a Survey PA/OA criterion. `decision_function` returns log-normalized scores
with threshold 0. `predict_proba` returns normalized VPU scores clipped to `[0,1]`, **not separately
calibrated probabilities**. `history_` adds `learning_rate` / `optimizer_generation`;
`optimizer_resets_` counts actual Adam rebuilds, while `optimizer_steps_` remains cumulative
across rebuilds. Default training is unchanged. A failed refit invalidates the fitted state.
The [VPU method card](../../research/method_cards/VPU.md) provides optional Chinese research background.

`cpu_stream` keeps full training/validation pools on CPU, sending at most `batch_size` rows to
the device at once; normalization scans, diagnostic validation, and prediction are batched too.
Validation risk combines the **global** log-sum-exp and positive mean in float64 on CPU, rather
than averaging per-batch risks; last-digit floating-point differences are possible. Original
resident reductions remain unchanged. `data_device_policy_` / `training_data_device_` record
the actual policy and data residency. Per-epoch snapshots include encoder, head, normalization,
and inference batch size; optimizer resumption is unsupported.
`PUPipeline(classifier="vpu", architecture="cnn", ...)` can inject a shared encoder. Optional
Chinese research background: [VPU method card](../../research/method_cards/VPU.md).

#### CVIRClassifier (cvir; alias conditional_value_ignoring_risk)

```python
CVIRClassifier(*, unlabeled_positive_prior=None, hidden_dim=64,
               warm_start_epochs=1, max_epochs=20, batch_size=128,
               learning_rate=1e-3, random_state=None, device=None,
               encoder=None, data_device_policy="auto",
               input_sampling="os", prior_source=None)
```

`unlabeled_positive_prior` must explicitly give the risk-pool positive proportion in `(0,1)`.
For default OS and independent native-TS inputs, this is **α_U in original U**.
`fit(class_prior=...)` always rejects the value; it does not automatically convert population π.
By default, each round stably sorts original-U scores, retains
`max(1,floor((1-α_U)*n_U))` temporary negatives, and minimizes the paper's α_U-weighted BCE.
Defaults are one PvU warm-start epoch and a fixed 20 CVIR rounds. This is neither convergence
stopping, BBE, nor TEDⁿ. The published author code differs in balanced CE, rounding, loaders,
SGD, and warm-start timing. The [source review](../../research/method_cards/CVIR.md) provides
optional Chinese research background.

Two-dimensional features can be used directly. Four-dimensional NCHW requires a
`torch.nn.Module` encoder with finite two-dimensional output, deep-copied and jointly trained
before an MLP head. `data_device_policy="auto"` retains resident behavior without an encoder and
uses cpu_stream with one. Explicit cpu_stream is accepted; encoder + resident is rejected.
CPU streaming retains complete data on CPU and sends at most batch_size rows to the device;
ranking/prediction are batched too. Ranking still needs O(n_U) CPU storage; this is not disk
streaming. `data_device_policy_` / `training_data_device_` record actual policy/residency.

`fit(..., sample_weight=None, epoch_callback=None, os_or_ts="os")` rejects non-`None` sample_weight.
Explicit `ts` first replaces the training risk pool with `U∪P` (U followed by P), then globally
ranks/truncates it while retaining the positive P term. Warm-start uses this same pool.
The supplied prior then targets the restored pool's π, rather than original-U α_U.
Validation/test retain their original views. Applying this calibration on top of native TS input
is rejected. `optimizer_steps_` counts both stages' updates; `epoch_provenance_` records
`stage / stage_epoch / round_index / optimizer_steps`, and `checkpoint_epoch_count` is the sum
of both stage lengths. Inference snapshots contain encoder/head and stage context. They support
inference restoration, not optimizer resumption. Fit failures clear the fitted marker. The public
Pipeline can inject a CNN and accept an explicitly supplied prior.

`input_sampling="ts"` only declares that the caller supplied independent native P/marginal U
and requires nonempty `prior_source`. It performs no OS→TS risk calibration and is distinct from
Survey's `os_or_ts="ts"`. Learning rate must be finite, positive, and real, excluding booleans
and strings; callbacks must be callable, and features finite real values. Encoder probes and
forward inputs are isolated so in-place networks cannot rewrite training/scoring arrays.
Public scoring preserves each layer's mode even when forward/eval raises; empty scoring does
not run the network. The extra current-batch copy is not a measured resource bound; formal
memory/GPU qualification remains separate.

#### PULDAClassifier (pulda; alias label_distribution_alignment)

Two-stage PU classifier combining label-distribution alignment, bidirectional margin, and
pseudo-label MixUp. Supports two-dimensional features and four-dimensional images with an injected CNN.

```python
PULDAClassifier(class_prior, *, hidden_dim=64, depth=2, warmup_epochs=60,
                pu_epochs=60, positive_batch_size=16, unlabeled_batch_size=128,
                warmup_learning_rate=1e-4, learning_rate=1e-3,
                warmup_weight_decay=5e-4, weight_decay=1e-4,
                temperature=3.5, unlabeled_ema=0.85, margin_ema=0.5,
                margin=0.6, mixup_weight=4.2, mixup_alpha=11.0, encoder=None,
                train_transform=None, random_state=0, device=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_prior` | `float` | Required | Positive proportion in the training risk pool, strictly in `(0,1)`; the default direct view consumes native-U prior. Can be overridden by `fit(class_prior=...)` |
| `hidden_dim` / `depth` | `int` / `int` | `64` / `2` | Toolbox MLP hidden width and depth |
| `encoder` | `torch.nn.Module \| None` | `None` | Optional feature encoder, required for four-dimensional images; deep-copied and trained end-to-end, followed by the MLP head |
| `train_transform` | `torch.nn.Module \| None` | `None` | Four-dimensional training images only; deep-copied and resamples augmentation for every P/U batch in both stages, before input MixUp and encoder normalization. Unused for prediction or pseudo-label scans |
| `warmup_epochs` / `pu_epochs` | `int` / `int` | `60` / `60` | Distribution-alignment warm-up and pseudo-label MixUp epochs; cannot both be zero |
| `positive_batch_size` / `unlabeled_batch_size` | `int` / `int` | `16` / `128` | P/U observations per batch |
| `warmup_learning_rate` / `learning_rate` | `float` / `float` | `1e-4` / `1e-3` | Initial Adam learning rates for the two stages |
| `warmup_weight_decay` / `weight_decay` | `float` / `float` | `5e-4` / `1e-4` | Stage-specific weight decay |
| `temperature` | `float` | `3.5` | Label-distribution symmetric-softplus temperature; the margin term fixes it at 1, following author code |
| `unlabeled_ema` / `margin_ema` | `float` / `float` | `0.85` / `0.5` | Unlabeled-mean and margin-moment EMA coefficients in `[0,1)` |
| `margin` / `mixup_weight` / `mixup_alpha` | `float` | `0.6` / `4.2` / `11.0` | Bidirectional margin, MixUp BCE weight, and Beta shape parameter |
| `random_state` / `device` | `int \| None` / `str \| None` | `0` / `None` | Seed and torch device |

`fit(..., sample_weight=None, epoch_callback=None)` rejects non-`None` `sample_weight`.
`history_` records stage, three loss components, total loss, and cumulative updates;
`pseudo_labels_` stores training pseudo-labels from the end of stage two. `predict_proba` gives
sigmoid scores, not independently calibrated probabilities. The [PULDA method card](../../research/method_cards/PULDA.md)
provides optional Chinese research background.


### Additional PU methods (experimental integration)

#### `HolisticPUClassifier` (`holistic_pu`)

```python
HolisticPUClassifier(*, hidden_dim=64, warmup_epochs=30, max_epochs=100,
                     batch_size=64, learning_rate=1e-3, trend_scale=2.0,
                     encoder=None, pseudo_pn_initialization="continue",
                     warmup_selection="fixed", lzo_alpha=0.5,
                     lzo_validation_size=None,
                     random_state=0, device=None)
```

`fit(X, y_pu, *, class_prior=None, sample_weight=None, os_or_ts="os", epoch_callback=None)`
performs balanced P/U resampling, records prediction trajectories, partitions them using the
paper's full-time-series signed trend and variance, then trains with pseudo-label supervision.
It supports a 2-D MLP or an injected end-to-end CNN. 4-D NCHW input requires an `encoder`
returning batch×feature_dim. Each fit deep-copies and trains the encoder in both stages;
template weights and BN state are not shared. Data remain on CPU; optimization, trajectory
scans, and prediction run in batches. Scanning and prediction use eval mode and restore modes.
A singleton final batch uses BN running statistics without dropping rows.
`pseudo_pn_initialization="continue"` continues the same model and Adam by default.
`"reinitialize"` creates a new complete network and Adam for the second stage, randomly resetting
CNN/head parameters and BN running statistics. This aligns only the initialization step in
supplementary Algorithm 2, not the complete author recipe. Parameter-owning modules in custom
encoders must expose `reset_parameters`; legacy weight_norm is rejected. Copying pretrained
weights is not treated as reinitialization.
Fixed warmup is not the paper's automatic LZO stopping rule; `warmup_epochs>=2`.
External priors are unused; sample_weight and TS requests are rejected.
`warmup_selection="lzo_positive_loss"` enables an engineering variant: fixed within-class mixup
using train P only, minimizing mean positive CE over the complete budget. Candidates are epochs
2..warmup, ties select the earliest, and defaults are alpha 0.5 and size n_P.
An independent auxiliary random stream does not advance the training stream. Validation uses
eval mode and preserves BN/torch RNG state. The complete budget runs before truncating the trend
prefix; continue restores selected weights/Adam, and both initialization variants restore the
selected random stream. This is not population accuracy, patience-based early exit, a checkpoint
resume API, or a formal PA/OA criterion. Source-unspecified rules are recorded in the spec.
`selected_warmup_epoch_` / `executed_warmup_epochs_` / `discarded_warmup_optimizer_steps_`,
`lzo_losses_` / `lzo_mixup_indices_` / `lzo_mixup_weights_` / `lzo_selection_spec_` support auditing.
`lzo_evaluated_rows_` / `lzo_forward_batches_` count extra validation; cumulative training cost
does not subtract the discarded suffix.
`prediction_trajectory_` / `pseudo_labels_` / `pseudo_label_indices_` / `breakpoint_` record the
partition. `optimizer_steps_` includes both stages and is not reset by reinitialization;
`stage_optimizer_steps_` separates stages. `pseudo_pn_initialization_` /
`pseudo_pn_optimizer_reset_` record the actual path. The callback runs after each epoch with
continuous zero-based numbering; `history_["phase"]` distinguishes warmup / pseudo_pn.
`checkpoint_epoch_count` counts peak snapshot storage across both stages.
Snapshots support inference replay only; formal PA/OA candidate stages require separate
preregistration. The [method card](../../research/method_cards/Holistic-PU.md) provides optional
Chinese research background.

#### `GenPUClassifier` (registered as `genpu`, alias `gen_pu`)

```python
GenPUClassifier(*, class_prior=None, hidden_dim=128, latent_dim=100,
                max_epochs=100, classifier_epochs=100, batch_size=64,
                learning_rate=3e-4, positive_weight=1.0, negative_weight=1.0,
                unlabeled_weight=1.0, encoder=None, generator_output="identity",
                random_state=0, device=None)
```

`fit(X, y_pu, *, class_prior=None, sample_weight=None, os_or_ts="os", epoch_callback=None)`
requires a population prior; conflicting constructor/fit values are rejected. The paper objective
uses two generators and three discriminators, followed by a PN classifier trained on synthetic
streams. Only Du's real-data pool supports the TS U∪P replacement. It supports 2-D MLP input or
4-D NCHW with an explicitly injected CNN: two dense pixel generators, three independent trainable
discriminator encoders, and a fresh independent PN encoder. Real images are not flattened.
`generator_output="identity"` retains the original default; `"tanh"` requires X already in [-1,1],
without automatic clipping or rescaling. G steps freeze D parameters/BN/dropout while preserving
input gradients. Original data remain on CPU.
`history_` / `optimizer_steps_` / `stage_optimizer_steps_` record complete GAN+PN cost and costs
for all six networks. `history_["optimizer_steps"]` is cumulative cost at each PN epoch end.
Callbacks run only after synthetic PN classifier epochs, with zero-based numbering;
`checkpoint_epoch_count=classifier_epochs`. GAN networks are not classifier snapshots.
Snapshots support classifier inference replay. Save the complete six-model state separately as
a trusted pickle; training resume and formal admission are not supported.
Non-None sample_weight raises an error; sigmoid scores do not guarantee independent probability
calibration. The [GEN-PU method card](../../research/method_cards/GEN-PU.md) provides optional
Chinese research background.

#### `PULNSClassifier` (`pulns`)

```python
PULNSClassifier(*, hidden_dim=32, pretrain_epochs=10, episodes=20,
                classifier_epochs=1, batch_size=128, learning_rate=1e-3,
                selector_learning_rate=1e-3, discount=0.9, terminal_weight=1.0,
                encoder=None, random_state=0, device=None)
```

Experimental adaptation of the paper's RL negative selection.
`fit(..., support_data=(X_support,y_clean), train_indices=None, support_indices=None,
os_or_ts="os", epoch_callback=None)` requires independent support with true positives and
negatives. Do not reuse clean_val/test; support supplies training rewards, making this ineligible
for purely PU PA. When both sets of IDs are supplied, disjointness is checked; otherwise
isolation is unverified. TS and non-None sample_weight are rejected. `history_` records reward
probes, actual classifier accuracy, policy loss, and negative counts. Best support accuracy and
the reward baseline are stored separately in `best_support_accuracy_` and `reward_baseline_`.
`epoch_callback(episode, fitted)` runs once after each completed RL episode (zero-based),
recording the current classifier before the final restoration of the support-selected best.
It does not record pretraining/internal classifier epochs or change default selection.
`checkpoint_epoch_count=episodes`, `checkpoint_stages=("episode",)`; weights-only snapshots
contain the classifier, not selector/optimizer, and cannot resume training.
Input is 2-D MLP data or 4-D NCHW with an explicitly injected `encoder`, without silent flattening.
Each fit deep-copies and unfreezes the encoder. CNN pretraining, reward probes, and the actual
classifier all update end to end. Selector state comes from the final hidden_dim hidden layer;
state/reward gradients stop, and policy updates do not backpropagate into the encoder.
CNN states are extracted in batches and cached on CPU, but the sequential policy graph still
grows with U; constant GPU memory for an entire episode is not guaranteed. Singleton BN uses
running statistics; inference preserves BN state and restores modes.
`stage_optimizer_steps_` separates pretrain/reward_probe/policy/classifier;
`optimizer_steps_` and `history_["optimizer_steps"]` count all actual updates. Returning an
earlier best network does not subtract cost.
The ordinary Pipeline/CLI has no independent reward-set input; direct calls require isolated
support. Explicit research routes use `PULNSExtraSupportTrainer` or
`PULNSCleanSupportCheckpointTrainer` and must record their extra supervision and compatible
protocol. CNN eligibility does not authorize reuse of test/selection labels.
The UI retains capability metadata but marks ui_ready=false and excludes manual/comparison
choices. Importing a configuration requiring an unavailable independent-support input fails
without changing existing widget state; it does not enable PA/OA training automatically.

#### `PANClassifier` (`pan`)

```python
PANClassifier(*, hidden_dim=128, max_epochs=100, batch_size=64,
              learning_rate=1e-4, adversarial_weight=1.0, model=None,
              discriminator=None, encoder=None, random_state=0, device=None)
```

Experimental MLP / injected-CNN adaptation of PAN Equation (7): alternating Adam updates for
C/D, predictions from C, and no external prior. `fit` supports `epoch_callback` and
`os_or_ts="os"`; explicit TS and non-None external sample_weight raise errors.
C-only epoch snapshots support inference, not training resume. `predict_proba` returns sigmoid
scores without a calibration guarantee; this is not admitted to the formal Survey matrix.
4-D NCHW requires an explicit `encoder`. Two independent copies are unfrozen and trained end to
end without modifying the template. Here `model` / `discriminator` are score heads (linear by
default), and the encoder must return finite 2-D features. Pipeline can inject a shared initial
backbone using `architecture="cnn"`.

#### `RankPruningClassifier` (`rp`)

```python
RankPruningClassifier(*, base_estimator=None, n_cv_folds=3,
                      frac_pos2neg=None, min_retained=10, random_state=0)
```

Supports only the PU special case without mislabeled positives. Each candidate runs n_cv_folds
OOF fits plus one reweighted fit, without an external prior.
`fit(os_or_ts="os")` rejects explicit TS and non-None external sample_weight. Internal weighting
requires the base_estimator's fit to support sample_weight.
`rho1_`, `rho0_`, `pi0_`, `estimated_prior_`, `prior_diagnostic_`, and OOF/retained indices are
diagnostics, not formal budget registration or independent probability calibration.

### Bias-aware classifiers

#### `PUSBClassifier` (registered as `pusb`, alias `biased_pu`)

A linear observed-P-vs-U logistic baseline for selection-biased PU data. This implementation
does not implement the paper's partially identified PUSB scoring algorithm, and its sigmoid
output is not a calibrated latent class posterior.

```python
PUSBClassifier(*, threshold=0.5, C=1.0, max_iter=1000)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.5` | Fixed threshold on observed-P-vs-U sigmoid output; `predict` uses this constructor value |
| `C` | `float` | `1.0` | Inverse regularization strength |
| `max_iter` | `int` | `1000` | Maximum optimizer iterations |

- Optional Chinese research background: [PUSB method card](../../research/method_cards/PUSB.md).

#### `PUSBKernelClassifier` (registered as `pusb_kernel`, alias `kernelized_pusb`)

RBF PUSB aligned with the released implementation, using deterministic CV.

```python
PUSBKernelClassifier(*, n_basis=300, cv=5, sigma_grid=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
                     reg_grid=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
                     random_state=2018, max_iter=200, tol=1e-5)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_basis` | `int` | `300` | Number of kernel basis functions |
| `cv` | `int` | `5` | Deterministic CV fold count |
| `sigma_grid` | `Sequence[float]` | Released grid | RBF bandwidth candidates |
| `reg_grid` | `Sequence[float]` | Released grid | Regularization candidates |
| `random_state` | `int \| None` | `2018` | Seed used in the released implementation |
| `max_iter` / `tol` | `int` / `float` | `200` / `1e-5` | Iteration limit and tolerance |

- The regularization gradient matches the released implementation (`0.5·reg_lambda·‖coef‖²`).
- Optional Chinese research background: [PUSB method card §7.3](../../research/method_cards/PUSB.md).

#### `LBEClassifier` (registered as `lbe`)

Label Bias Estimation classifier: estimates the class posterior and instance-dependent labeling propensity.

```python
LBEClassifier(*, max_iter=1000, n_em_iter=20, C=1.0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_iter` | `int` | `1000` | Maximum main-optimizer iterations |
| `n_em_iter` | `int` | `20` | Number of EM iterations |
| `C` | `float` | `1.0` | Inverse regularization strength; finite and greater than 0 |

- Optional Chinese research background: [LBE method card](../../research/method_cards/LBE.md).
- This is linear logistic EM, not the authors' dual-MLP/Adam implementation; `fit(class_prior=...)` affects U posterior initialization only.
- After fitting, `n_iter_` / `converged_` record actual EM iterations and posterior stopping. `observed_log_likelihood_history_` records the sum of observed log likelihoods per iteration, not the regularized optimization objective, and is not guaranteed monotonic.
- `predict_proba` returns class probabilities; `predict_label_proba` returns labeling probabilities. All prediction entry points require a successful fit.

#### `NeuralLBEClassifier` (registered as `lbe_neural`, alias `neural_lbe`)

Separate experimental dual-network generalized EM; **does not replace `lbe` or the frozen Pilot**.

```python
NeuralLBEClassifier(model=None, *, propensity_model=None, encoder=None,
                    hidden_dim=100, pretrain_steps=20, n_em_iter=20, m_steps=40,
                    sample_fraction=0.1, batch_size=256, learning_rate=1e-3,
                    weight_decay=1e-3, random_state=0, device=None)
```

| Parameter | Default | Contract |
|---|---|---|
| `model` / `propensity_model` | `None` | Each outputs one raw logit per row; defaults are independent MLP100/tanh networks; deep-copied without changing templates |
| `encoder` | `None` | Explicitly required for 4-D images; deep-copied into independent trainable encoders for both networks, returning 2-D features |
| `hidden_dim` | `100` | Hidden width of default heads; positive integer |
| `pretrain_steps` | `20` | Adam pretraining updates per network; nonnegative integer; 0 explicitly adapts the recipe without pretraining |
| `n_em_iter` / `m_steps` | `20` / `40` | Positive integers; E-step posteriors stop gradients and remain fixed throughout that iteration's M updates |
| `sample_fraction` | `0.1` | `(0,1]`; each M update resamples `max(1,floor(N*fraction))` training rows without replacement |
| `batch_size` | `256` | Positive integer; maximum device batch for forward passes, gradient accumulation, and prediction, not an additional Adam-update axis |
| `learning_rate` / `weight_decay` | `0.001` / `0.001` | Finite real values, positive / nonnegative respectively |
| `random_state` / `device` | `0` / `None` | Isolated NumPy RNG; preserves CPU/selected CUDA RNG without resetting other CUDA generators; CPU/CUDA only, with auto resolved by the shared resolver |

`fit(X, y_pu, *, class_prior=None, sample_weight=None, epoch_callback=None, os_or_ts="os")`
accepts training PU observations only, with at least two labeled P and one U; it does not read
validation, test, or hidden PN truth. A non-None `class_prior` is checked as a real value in
`(0,1)` but does not affect the source recipe. Non-None `sample_weight` and non-OS views are rejected.
Inputs/posteriors remain on CPU. Row-count-weighted gradients accumulate across batches, with
one Adam update per M step. Defaults total 840 optimizer updates; `checkpoint_epoch_count=20`
counts EM iterations, not 840 weight snapshots. `epoch_callback` runs after each EM iteration;
`optimizer_steps_` and stage provenance record complete pretraining/EM cost.

`predict_proba` returns class posteriors, `predict_label_proba` returns posterior×propensity,
and `decision_function` returns class raw logits. `history_` /
`observed_log_likelihood_history_` are training diagnostics without monotonicity or identifiability
guarantees; they are not general PA/OA criteria. Standalone pickle retains both networks;
runner epoch checkpoints support classifier inference only, not propensity/optimizer training
resume. Defaults come from the separately recorded user-package MLP variant; stable logits,
batching, and CNN support are adaptations. The [LBE method card](../../research/method_cards/LBE.md)
provides optional Chinese background on sources, differences, calibration limits, and open reviews.

### Deep classifiers

#### `LaGAMClassifier` (registered as `lagam`, alias `la_gam`)

Latent-group contrastive learning and clean-support meta-label updates. **Not eligible for
purely PU PA**: support supplies extra clean-label supervision. Use an explicitly configured
clean-support trainer for research runs; the ordinary PU-only Pipeline does not supply support.

```python
LaGAMClassifier(*, hidden_dim=128, warmup_epochs=2, max_epochs=20,
                batch_size=64, support_batch_size=32, num_clusters=5,
                learning_rate=1e-3, meta_lr=1e-3, mix_weight=1.0,
                contrastive_weight=1.0, temperature=0.07, noise_std=0.05,
                rho_start=0.95, rho_end=0.8, random_state=None, device=None,
                encoder=None, data_device_policy="auto", meta_update_bias=True)
```

`fit(X, y_pu, *, class_prior=None, sample_weight=None, support_data=(X_support, y_clean), epoch_callback=None)`
requires an independent clean support set containing both classes; missing support fails
immediately. Non-None `sample_weight` raises an error. The [LaGAM method card](../../research/method_cards/LaGAM.md)
provides optional Chinese research background.

Supports 2-D features or 4-D NCHW with an injected trainable encoder; support must have the same
input shape. A deep-copied encoder precedes the original MLP and must return finite 2-D features.
Auto preserves legacy resident storage without an encoder and forces cpu_stream with an encoder.
Explicit cpu_stream is supported; encoder+resident is rejected. CPU-stream inputs, support,
pseudo-labels, and group indices remain on CPU. Training/clustering/inference device batches
contain at most batch_size rows, support batches at most support_batch_size; complete clustering
features still occupy CPU memory.

`meta_update_bias=True` retains virtual weight+bias updates; False aligns only the author's
classifier weight-only scope. New CNN virtual steps extract frozen features in eval mode and
restore each module's mode without changing actual BN/dropout; actual losses train the encoder.
Hard observed meta-targets, stable BCE, rho endpoints, and Gaussian augmentation remain
adaptations; the [source review](../../research/method_cards/LaGAM.md) is optional Chinese background.

`max_epochs` includes warmup; `checkpoint_epoch_count=max_epochs`. `optimizer_steps_`,
`virtual_steps_`, and `clustering_passes_` count actual updates, virtual second-order steps, and
clustering passes. `epoch_provenance_` and snapshot context distinguish warmup/meta. Image
snapshots retain inference batch size, without optimizer resume. Ordinary fit exceptions clear
fitted state. Aliased overlapping arrays are rejected; callers must check role indices across
distinct copies. PU-only UI/recommendation and ordinary trainers do not create independent
support; use the explicit support adapters. The published OS split and registry TS positioning
remain under review; no new TS interface is implied.

#### `SplitPUClassifier` (registered as `split_pu`, alias `split-pu`)

nnPU teacher → prediction-disagreement split into easy/hard U → easy JS distillation and
hard dual-source consistency. Supports dense 2-D inputs or 4-D images with an injected CNN;
the paper's complete augmentation/SimSiam recipe is not reproduced.

```python
SplitPUClassifier(class_prior=None, *, encoder=None, encoder_feature_layer=None,
                  hidden_dim=100, teacher_epochs=10,
                  split_epochs=10, student_epochs=10, rounds=2, batch_size=64,
                  learning_rate=1e-3, agreement_threshold=0.92,
                  js_teacher_weight=0.7, hard_weight=0.3,
                  feature_weight=0.3, similarity_weight=0.1,
                  noise_std=0.05, random_state=None, device=None)
```

`fit(X, y_pu, *, class_prior=None, sample_weight=None, epoch_callback=None, os_or_ts="os")`.
In the TS risk view, class_prior is population π (the positive proportion in marginal U).
The TS risk-pool replacement applies only to the nnPU teacher; splitter, easy/hard partitioning,
and student still use original U. Non-None sample_weight raises an error.
The [Split-PU method card](../../research/method_cards/Split-PU.md) provides optional Chinese background.

`checkpoint_epoch_count=teacher_epochs+split_epochs+rounds*student_epochs` bounds snapshot count;
splitter early stopping reduces actual snapshots. `optimizer_steps_` counts actual updates
across all three stages and student rounds. Reclamation reduces final retained snapshots, not
peak training storage; this upper bound is not a formal candidate budget.
Data pools and frozen teacher targets remain on CPU; training/prediction transfer batches to
the device, and prediction restores model modes. Original U needs at least two rows so both
easy/hard branches are nonempty; empty prediction input returns an empty array.
Each stage independently deep-copies/trains the encoder; 4-D input without an encoder is rejected.
Low-level MSE uses the selected encoder module (first MaxPool by default, otherwise Conv/Linear);
high-level cosine uses final encoded features. This is not official SimSiam.
Singleton groups temporarily use BN running statistics while preserving gradients; larger
groups update normally.

#### `RobustPUClassifier` (registered as `robust_pu`, alias `robust-pu`)

nnPU pretraining, separate self-paced P/U weights, then weighted candidate-negative training.
Supports 2-D features or 4-D images with an injected CNN, returning raw logits.

```python
RobustPUClassifier(class_prior=None, *, model=None, encoder=None, hidden_dim=100,
                   pretrain_epochs=10, episodes=20, inner_epochs=1,
                   batch_size=64, pretrain_lr=1e-3, learning_rate=1e-4,
                   alpha_p=0.1, alpha_n=0.1, max_thresh_p=2.0,
                   max_thresh_n=2.0, grow_steps=10, spl_type="welsch",
                   temper_p=1.0, temper_n=1.0, phi=0.0,
                   random_state=None, device=None)
```

`fit(X, y_pu, *, class_prior=None, sample_weight=None, epoch_callback=None, os_or_ts="os")`.
The TS risk-pool replacement applies only to nnPU pretraining; subsequent pseudo-negatives
still come from original U. TS is explicitly rejected when pretraining is skipped.
Non-None sample_weight raises an error. This is experimental technical integration, outside
the formal Survey matrix. The [Robust-PU method card](../../research/method_cards/Robust-PU.md)
provides optional Chinese research background.

`checkpoint_epoch_count=pretrain_epochs+episodes`, with one snapshot per episode.
All `inner_epochs` contribute to `optimizer_steps_` without adding snapshots. Refit resets
update counts. The encoder is deep-copied and unfrozen; 4-D images require it. A default MLP
or the supplied model's raw-score head follows it. Images, labels, and original-row self-paced
weights remain on CPU; only training batches move to the device. Prediction batches use eval
mode and restore modes.

#### `GradPUClassifier` (registered as `gradpu`, alias `grad_pu`)

PU classifier with input-gradient penalties and hard-positive weighting, supporting 2-D
features or 4-D images with a BatchNorm-free encoder.

```python
GradPUClassifier(model=None, *, hidden_dim=128, alpha=0.1, beta_max=1.0,
                 batch_size=256, max_epochs=200, learning_rate=1e-3,
                 lr_schedule="constant", lr_final_ratio=0.1,
                 weight_decay=5e-4, encoder=None, activation_memory="device",
                 backward_mode="joint", gp_chunk_size=None, random_state=None, device=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module \| None` | `None` | Network returning one raw score per row; defaults to a single-hidden-layer MLP; BatchNorm unsupported |
| `hidden_dim` | `int` | `128` | Default MLP hidden width |
| `encoder` | `torch.nn.Module \| None` | `None` | Required for 4-D images; BatchNorm-free, deep-copied and trained end to end; a supplied model serves as raw-score head |
| `alpha` / `beta_max` | `float` | `0.1` / `1.0` | Input-gradient penalty strength / maximum positive weight |
| `batch_size` / `max_epochs` | `int` | `256` / `200` | Per-group batch size / training epochs |
| `learning_rate` / `weight_decay` | `float` | `1e-3` / `5e-4` | Adam settings |
| `lr_schedule` / `lr_final_ratio` | `str` / `float` | `"constant"` / `0.1` | Explicit `linear` / `cosine` decays by epoch to the initial rate times a ratio in `(0,1]`; default stays constant and a single epoch uses the initial rate. Curves are toolbox options, not paper-specified schedules |
| `activation_memory` | `{"device", "cpu_offload"}` | `"device"` | Opt-in CPU storage of autograd saved tensors, including second-order penalty graphs; preserves full batches/precision/RNG order at the cost of transfers and host RAM, without guaranteeing every batch fits |
| `backward_mode` | `{"joint", "split"}` | `"joint"` | Joint loss backward or separate PU/gradient-penalty backward before one optimizer update |
| `gp_chunk_size` | `int \| None` | `None` | Positive chunk size for gradient-penalty backward; requires split mode. With active penalty, injected modules must be deterministic and sample-independent; known stochastic layers are rejected |
| `random_state` / `device` | `int \| None` / `str \| None` | `None` / `None` | Seed / PyTorch device |

`fit(X, y_pu, *, class_prior=None, sample_weight=None, epoch_callback=None, os_or_ts="os")`
accepts a prior for interface compatibility, but the paper objective does not use it.
Non-None `sample_weight` raises an error. `decision_function` returns $[-1,1]$ scores, not
calibrated probabilities; the prediction threshold is 0. The 2-D and BatchNorm-free injected
image paths remain outside the formal Survey matrix.

`batch_size` caps each of P and original U; the shorter branch cycles by batch.
Updates per epoch are `max(ceil(n_P/B),ceil(n_U/B))`. OS uses at most 2B physical rows;
TS U risk uses at most 2B and the additional P term gives at most 3B cumulative forward rows.
GP interpolation uses at most 2B; the physical training set is not expanded to 3B.
`history_["learning_rate"]` records actual epoch rates. `training_provenance_` records policy,
actual maximum batch sizes, update counts, and TS/SAR boundaries; the runner saves it in
`candidate_runs[].training_provenance`.

- Optional Chinese research background: [GradPU method card](../../research/method_cards/GradPU.md).

#### `SelfPUClassifier` (registered as `self_pu`)

Self-PU classifier with probability reweighting, distillation, and a dynamic trusted set.

```python
SelfPUClassifier(class_prior, *, backbone=None, hidden_dim=128, warmup_epochs=10,
                 self_paced_start=10, self_paced_end=50, distill_start=50, max_epochs=200,
                 max_trust_ratio=0.25, pace_1=0.2, pace_2=0.3, meta_step_size=1e-3,
                 reweight_gamma=1/16, distillation_alpha=10.0, ema_decay=0.99,
                 student_loss_weight=1.0, teacher_loss_weight=1.0,
                 distillation_scope="untrusted_u", batch_size=256,
                 learning_rate=1e-3, weight_decay=0.0, threshold=0.5,
                 require_validation=False, random_state=None, device=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_prior` | `float` | Required | Prior for the selected risk view; the direct default declares `class_prior_target="native_unlabeled"`, requiring OS population-to-U conversion |
| `backbone` | `Any \| None` | `None` (built-in MLP) | Score backbone; default tabular MLP dimensions are inferred by fit; custom backbones are separate from Pipeline encoder injection |
| `hidden_dim` | `int` | `128` | Hidden dimension |
| `warmup_epochs` | `int` | `10` | Warmup epochs |
| `self_paced_start` / `self_paced_end` | `int` | `10` / `50` | Self-paced calibration epoch interval |
| `distill_start` | `int` | `50` | Distillation start epoch |
| `max_epochs` | `int` | `200` | Maximum training epochs |
| `max_trust_ratio` | `float` | `0.25` | Maximum trusted fraction |
| `pace_1` / `pace_2` | `float` | `0.20` / `0.30` | Final student paces, capped by max_trust_ratio |
| `meta_step_size` | `float` | `1e-3` | Meta-learning step size |
| `reweight_gamma` | `float` | `1/16` | Fraction limit for CE reweighting |
| `distillation_alpha` | `float` | `10.0` | Distillation loss weight |
| `ema_decay` | `float` | `0.99` | Teacher EMA decay |
| `student_loss_weight` / `teacher_loss_weight` | `float` | `1.0` / `1.0` | Student/teacher loss weights |
| `distillation_scope` | `{"untrusted_u", "paper_population"}` | `"untrusted_u"` | Consistency-loss scope; the independent clean-support full-meta adapter uses paper-population scope, preserving the historical Pilot default |
| `batch_size` / `learning_rate` / `weight_decay` | `int` / `float` / `float` | `256` / `1e-3` / `0.0` | Training settings |
| `threshold` | `float` | `0.5` | Decision threshold, untuned; see method card |
| `require_validation` | `bool` | `False` | Require clean `validation_data` in fit for full-meta training; missing data fail instead of entering ablation |
| `random_state` / `device` | `int \| None` / `str \| None` | `None` / `None` | Seed and device |

- `fit(..., validation_data=...)` enables meta reweighting and clean-validation teacher selection. Keep this role independent from train/test.
- `fit(..., pu_validation_data=...)` accepts P/U labels only, records both teachers' nnPU risks each epoch, and restores the lowest-PU-validation-risk teacher checkpoint. It does not enable meta reweighting requiring true labels. Without either validation input and with `require_validation=False`, ablation selects using training nnPU risk.
- Real-valued parameters require finite scalars, rejecting bool, strings, arrays, complex values, and NaN/Inf. Budgets/stages require integers; `require_validation` is boolean; `random_state` is None or an integer in `[0,2**32-1]`. NumPy scalars are supported. Invalid settings raise `ValueError` before seed/model construction; failed refit requires a new successful fit before scoring.
- Optional Chinese background: [Self-PU method card](../../research/method_cards/Self-PU.md). Example: [10_self_pu.py](../../../examples/minimal/10_self_pu.py); its illustrative prior needs the OS conversion explained in the [guide](../howto/self_pu.md).

#### `InfoMaxPUClassifier` (registered as `infomax_pu`)

Deep PU classifier: PURL representation learning → prior estimation → nnPU classification.
For fine-grained control of its many constructor parameters, pass an instance to `PUPipeline`.

```python
InfoMaxPUClassifier(*, class_prior=None, representation_dim=20, hidden_dim=60,
                    representation_epochs=200, classifier_epochs=200, learning_rate=1e-3,
                    representation_ratio_steps=4, representation_encoder_steps=1,
                    representation_weight_decay=5e-4, representation_batch_norm=False,
                    representation_activation=False, representation_batch_size=None,
                    representation_gradient_noise=0.0, classifier_hidden_dims=(),
                    classifier_batch_norm=False, classifier_optimizer="adam",
                    classifier_learning_rate=1e-3, classifier_weight_decay=0.0,
                    classifier_batch_size=256, prior_estimator=None,
                    random_state=None, encoder=None, device=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_prior` | `float \| None` | `None` | Class prior; None estimates through prior_estimator |
| `representation_*` | Mixed | See signature | Representation stage: `representation_dim` (dimension) / `representation_epochs` / `representation_ratio_steps` / `representation_encoder_steps` / `representation_weight_decay` / `representation_batch_norm` / `representation_activation` / `representation_batch_size` / `representation_gradient_noise` |
| `classifier_*` | Mixed | See signature | nnPU stage: `classifier_hidden_dims` / `classifier_batch_norm` / `classifier_optimizer` (`"adam"` / `"adagrad"`) / `classifier_learning_rate` / `classifier_weight_decay` / `classifier_batch_size` |
| `prior_estimator` | `BaseEstimator \| None` | `None` | Prior estimator when class_prior=None |
| `encoder` | `torch.nn.Module \| None` | `None` | External encoder, e.g. `build_encoder("cnn", ...)`; None uses the built-in MLP for backward compatibility |
| `device` | `str \| None` | `None` | Torch device; None/"auto" detects automatically |
| `random_state` / `hidden_dim` / `learning_rate` | — | — | Shared seed, hidden dimension, and learning rate |

- An external encoder replaces the internal `nn.Sequential(Linear...)` encoding component; `ratio_head_` follows its features, and fit accepts 4-D NCHW images.
- Optional Chinese research background: [InfoMax-PU method card](../../research/method_cards/InfoMax-PU.md).

#### `WeightedContrastivePUClassifier` (registered as `weighted_contrastive_pu`, alias `wconpu`)

Weighted contrastive PU classifier with contrastive loss and distribution matching.

```python
WeightedContrastivePUClassifier(class_prior, *, encoder=None, hidden_dim=128, embedding_dim=128,
                                queue_size=8192, temperature=0.07, momentum=0.999,
                                pseudo_label_momentum=0.9, contrastive_weight=0.1,
                                distribution_weight=0.1, hard_negative_quantile=0.25,
                                weak_augmentation=None, strong_augmentation=None, batch_size=256,
                                max_epochs=100, learning_rate=1e-2, optimizer_momentum=0.9,
                                scheduler="none", random_state=None, device=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_prior` | `float` | Required | Class prior |
| `encoder` | `torch.nn.Module \| None` | `None` | Feature encoder; images can use `build_encoder("cnn", ...)` |
| `hidden_dim` / `embedding_dim` | `int` | `128` / `128` | Hidden and embedding dimensions |
| `queue_size` / `temperature` / `momentum` | `int` / `float` / `float` | `8192` / `0.07` / `0.999` | Contrastive queue, temperature, and queue momentum |
| `pseudo_label_momentum` | `float` | `0.9` | Pseudo-label momentum |
| `contrastive_weight` / `distribution_weight` | `float` | `0.1` / `0.1` | Contrastive/distribution loss weights |
| `hard_negative_quantile` | `float` | `0.25` | Hard-negative quantile |
| `weak_augmentation` / `strong_augmentation` | callable \| None | `None` | Weak/strong augmentation transforms; None uses defaults |
| `batch_size` / `max_epochs` / `learning_rate` / `optimizer_momentum` | — | `256` / `100` / `1e-2` / `0.9` | Training settings |
| `scheduler` | `"none"` / `"cosine_annealing"` | `"none"` | Learning-rate schedule |
| `random_state` / `device` | — | `None` / `None` | Seed and device |

- Optional Chinese research background: [WConPU method card](../../research/method_cards/WConPU.md).

#### `DGPUClassifier` (registered as `dgpu`)

Deep generative PU classifier combining a GAN discriminator and iterative pseudo-labeling.

```python
DGPUClassifier(class_prior, generator, *, model=None, hidden_dim=128, rounds=3,
               initialization_epochs=100, annotation_epochs=100, generated_samples=5000,
               pseudo_label_fraction=0.1, confidence_threshold=0.95, debias_strength=0.8,
               distribution_momentum=0.999, batch_size=256, learning_rate=1e-4,
               weak_augmentation=None, strong_augmentation=None, random_state=None, device=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_prior` | `float` | Required | Class prior |
| `generator` | Torch model | Required | Data generator, e.g. a GAN generator |
| `model` | `torch.nn.Module \| None` | `None` | Discriminator/classifier model; None uses the built-in model |
| `hidden_dim` | `int` | `128` | Hidden dimension |
| `rounds` | `int` | `3` | Pseudo-labeling rounds |
| `initialization_epochs` / `annotation_epochs` | `int` | `100` / `100` | Initialization and annotation epochs |
| `generated_samples` | `int` | `5000` | Number of generated samples |
| `pseudo_label_fraction` | `float` | `0.1` | Pseudo-label fraction |
| `confidence_threshold` | `float` | `0.95` | Pseudo-label confidence threshold |
| `debias_strength` | `float` | `0.8` | Debiasing strength |
| `distribution_momentum` | `float` | `0.999` | Distribution momentum |
| `batch_size` / `learning_rate` | — | `256` / `1e-4` | Training settings |
| `weak_augmentation` / `strong_augmentation` | callable \| None | `None` | Weak/strong augmentation |
| `random_state` / `device` | — | `None` / `None` | Seed and device |

- Optional Chinese research background: [DGPU method card](../../research/method_cards/DGPU.md).

## PUPipeline

Usage: [pipeline guide](../howto/pipeline.md). Complete manual prior-estimation workflow:
[05_recpe_pipeline.py](../../../examples/minimal/05_recpe_pipeline.py).

```python
pipe = PUPipeline(
    classifier="auto",          # Registry name / "auto" recommendation / classifier instance.
    classifier_params=None,      # Constructor args for explicit names; unavailable for auto/instances.
    prior_estimator="pen_l1",   # "pen_l1"/"recpe"/"km1"/"km2" (the last two map to
                                # KernelMeanPriorEstimator), estimator instance, or None.
    prior_params=None,          # Args for a named prior estimator, e.g. {"variant": "km2"}; not an instance.
    cv=5,                       # PU-stratified folds or custom splitter.
                                # The actual default cv=None resolves to 5-fold PUStratifiedKFold.
    metrics=DEFAULT_METRICS,    # Metric names from the table below.
                                # The actual default metrics=None resolves to DEFAULT_METRICS.
    random_state=42,
    architecture="mlp",         # "mlp" tables / "cnn" 4-D NCHW with an explicit compatible method.
    backbone="cnn13",           # cnn13/cnn13_no_bn/resnet18/resnet50; used for CNN.
    device=None,                # Torch device; None/"auto" selects CUDA when available, otherwise CPU.
    max_epochs=None,            # Epoch cap forwarded to named constructors accepting max_epochs.
    input_sampling="os",       # "os" one source pool / "ts" independent P and population U.
)
report = pipe.fit_evaluate(
    X,
    y_pu,
    y_true=None,
    class_prior=None,
    sample_weight=None,         # Sliced per CV training fold and passed to final refit.
    refit=True,
    progress_callback=None,     # Called at workflow boundaries.
    cancellation_token=None,    # Interrupted by raise_if_cancelled.
)
```

| Constructor parameter | Type | Default | Description |
|---|---|---|---|
| `classifier` | `str \| BasePUClassifier` | `"auto"` | Registry name / automatic recommendation / instance |
| `classifier_params` | `dict \| None` | `None` | Explicit named classifier's constructor arguments; unavailable for auto/instances |
| `prior_estimator` | `str \| instance \| None` | `"pen_l1"` | Prior estimator; see precedence below |
| `prior_params` | `dict \| None` | `None` | Named prior estimator's constructor arguments |
| `cv` | `int \| splitter \| None` | `None` (→5 folds) | PU-stratified CV |
| `metrics` | `Sequence[str] \| None` | `None` (→DEFAULT_METRICS) | Metric names |
| `random_state` | `int \| None` | `42` | Random seed |
| `architecture` | `str` | `"mlp"` | `"mlp"` / `"cnn"` |
| `backbone` | `str` | `"cnn13"` | `cnn13`/`cnn13_no_bn`/`resnet18`/`resnet50` |
| `device` | `str \| None` | `None` (auto) | Torch device |
| `max_epochs` | `int \| None` | `None` | Epoch cap for compatible named deep methods |
| `input_sampling` | `str` | `"os"` | Input sampling design: `"os"` / `"ts"` |

`refit=False` computes CV metrics only, skipping full-data refit and model diagnostics;
`report.final_model` and `report.diagnostic` are then None. This mode primarily supports search.

`sample_weight` must be one-dimensional, finite, nonnegative, and have at least one positive
entry. After resolving the classifier, a non-None value triggers the `SampleWeightSupport`
check: only `supported` proceeds; `ignored` and `not_implemented` raise `PipelineError`.
`provenance["sample_weight"]` records presence, range, mean, and ESS, not the complete vector.

`provenance["architecture"]` is `"native_mlp"` / `"native_cnn"`.
`provenance["backbone"]` stores the CNN backbone name (None for MLP);
`provenance["device"]` stores `{"requested", "resolved"}`.
`provenance["encoder"]` summarizes the injected encoder as `{"backbone", "in_channels"}`
(None without injection in MLP mode).

### Class-prior resolution precedence

| Priority | Source | Report source |
|---|---|---|
| 1 | Explicit `fit_evaluate(class_prior=...)` | `user` |
| 2 | Classifier instance constructor `class_prior=...` | `constructor` |
| 3 | Automatic `prior_estimator` estimation, default `"pen_l1"` | `estimated` |
| 4 | Method does not need a prior | `none` |

Call-level `class_prior` targets the population; constructor values follow the classifier's
`class_prior_target`. Prior estimators declare their scalar through `estimate_target` and
`estimation_pool`; the default target for third-party estimators is `population`.
Native-U estimates are converted to the population according to the input sampling design.
Estimation runs once on the complete supplied PU data and is shared across folds; training
converts it per fold for the classifier's default-view target: `population`, `native_unlabeled`,
or `training_table`. This is not fold-nested prior estimation.
PUSBKernel also accepts `fit(..., risk_class_prior=...)` to specify the native-U risk prior
separately; `class_prior` supplies the concatenated training-table quantile threshold.
A direct call omitting `risk_class_prior` uses the same value at both consumption points.

`PipelineReport` JSON uses `schema_version="1.1"` for target-aware resolution and consumption.
`PriorInfo` retains its original fields and records:

| Field | Meaning |
|---|---|
| `target` / `value` | `population` and resolved population value |
| `input_value` / `input_target` | Original input/estimate and its target |
| `estimation_pool` / `estimation_details` | Estimation pool and diagnostics; ReCPE includes approximation identity |
| `input_sampling` / `conversion` | OS/TS input design and conversion name |
| `training_consumption` | CV/refit stage/fold, P fraction, and parameter target/value records |

Constructors declaring `input_sampling` receive the workflow's design; a supplied instance's
declaration must agree with the workflow.

- `prior_estimator=None` with a required but missing prior raises `PipelineError` for an explicit classifier. Auto can fall back to methods not requiring an external prior.
- Default OS profiling audits a population prior below the labeled-positive fraction (`inconsistent_class_prior`); independent TS records its extra-P fraction separately.
- In auto mode, prior-estimation failure falls back to prior-free recommendations (`report.prior.degraded`) rather than aborting.

### Metrics

`DEFAULT_METRICS = ("pu_zero_one_risk", "pu_recall", "pu_estimated_precision", "pu_auc_roc")`.
Aliases: `pu_risk`, `auc`/`roc_auc`, `recall`, `precision`, `accuracy`, `f1`,
`negative_rate`, `ap`, `bacc`, `brier`, `ece`.

| Metric | Required inputs | basis |
|---|---|---|
| `pu_recall` / `pu_negative_rate` | `y_pu` and predictions only | `pu_observed` |
| `pu_zero_one_risk` / `pu_estimated_precision` | Predictions and class prior | `class_prior_dependent` |
| `pu_auc_roc` / `average_precision` | `y_true` and continuous scores | `supervised_oracle` |
| `pu_accuracy` / `pu_f1` / `balanced_accuracy` | `y_true` and predictions | `supervised_oracle` |
| `brier_score` / `expected_calibration_error` | `y_true` and actual `predict_proba` | `probability_calibration` |

These functions, including `calibration_bucket_stats`, are directly importable from
`pu_toolbox.metrics`. Missing inputs (truth, continuous scores, predict_proba, or prior) make
the corresponding metric unavailable with a recorded reason (`CVMetric.available=False`),
without aborting. `CVMetric.n_computed` reports computed folds and is serialized in `to_json()`.

### Classifier selection semantics

These workflows accept classifiers declaring main-input `label_semantics="pu"` only.
Explicit PN/PNU mismatches fail before training; numeric label values do not determine semantics.
`y_true` supplies evaluation labels without switching the main input to PN.

| `classifier=` | Behavior |
|---|---|
| `"auto"` (default) | Resolve prior → recommend through `recommend_from_profile` → scan ranks for the first automatically instantiable candidate |
| Registered name such as `"nnpu"` / `"upu"` | Resolve case-insensitively with aliases; configure through `classifier_params`; inject managed arguments when accepted by the constructor |
| Instance such as `UPUClassifier(...)` | Clone per fold, convert/set a constructor prior when present, and inject a constructed CNN encoder when supported; retain the instance's other constructor settings |

Methods with required non-`class_prior` arguments, such as `"ldce"` requiring
`flip_probability`, accept those through `classifier_params` in explicit-name mode.
Auto skips these methods and records reasons in `report.provenance["skipped_candidates"]`.
`class_prior`, `random_state`, and `encoder` are managed by the workflow and cannot be overridden
through `classifier_params`.

### Deep methods and architecture selection

Contracts for `architecture` / `backbone` / `device`:

| Parameter | Default | Values | Semantics |
|---|---|---|---|
| `architecture` | `"mlp"` | `"mlp"` / `"cnn"` | CNN requires an explicit deep classifier with an encoder constructor parameter and matching registry capabilities. Candidates are discovered dynamically; nnPU, WConPU, and InfoMax PU are examples, not an exhaustive list. Unsupported methods such as Self-PU, auto, or nondeep classifiers with CNN raise `PipelineError`; 4-D input with MLP also raises `PipelineError` |
| `backbone` | `"cnn13"` | `"cnn13"` / `"cnn13_no_bn"` / `"resnet18"` / `"resnet50"` | Used for `architecture="cnn"`; invalid values raise `ValueError` |
| `device` | `None` (auto) | `None`/`"auto"`/`"cpu"`/`"cuda"`, etc. | Forwarded by signature for constructed deep classifiers; None/auto resolves to CUDA when torch+CUDA are available, otherwise CPU. Supplied instances retain their device setting |

- CNN integration requires an `encoder` constructor parameter accepting a feature extractor and matching native CNN / 4-D / encoder capability declarations. The pipeline injects `build_encoder` output under signature guards; unsupported paths fail during construction.
- Compatible explicit names pass required-argument checks when supplied appropriately. The prior follows explicit input before estimation. In CNN mode, `fit_evaluate` lazily builds `build_encoder("cnn", backbone=..., in_channels=...)` and injects it.
- Input dimensions: 4-D NCHW + explicit compatible deep classifier + CNN is accepted. Prior estimation/profiling use a flattened view, and CV splits indices. 4-D with MLP or a nondeep classifier, and 2-D with CNN, raise `PipelineError`.
- Deep methods with `cv>1` produce a training-cost warning (n_splits+1 fits with refit). Reduce folds when appropriate; minimum cv is 2.
- Auto uses GPU availability and data size as recommendation scoring inputs. Lower CPU/small-data scores do not categorically exclude deep methods. A selected deep method still receives torch seeding and training-cost warnings.

### Error cases

| Scenario | Exception |
|---|---|
| Invalid classifier / prior name | `PipelineError` during construction |
| Required constructor arguments missing, e.g. ldce without flip_probability | `PipelineError` during construction; supply classifier_params or a configured instance |
| No positives / fewer than `MIN_POSITIVE_SAMPLES` | `ValidationError` |
| Fewer positives than CV folds | `ValidationError` suggesting fewer folds |
| Required prior ultimately unavailable | `PipelineError` |
| User `class_prior` outside (0,1) | `ValueError`, as for invalid constructor settings such as cv/metrics/architecture |
| Estimated prior outside (0,1) | Auto degrades to prior-free recommendations; explicit classifier raises `PipelineError` |
| Prior-estimator exception | Auto degrades, recording prior.degraded; explicit classifier raises `PipelineError` |
| Unknown metric / invalid CV | `ValueError` / `TypeError` during construction |
| Invalid sample_weight shape/values | `ValueError` |
| Classifier ignores or does not implement sample_weight | `PipelineError`, without silent weighted-training failure |

### PipelineReport (return object)

`fit_evaluate` returns a `PipelineReport` with:

| Field | Type | Description |
|---|---|---|
| `profile` | `PUDataProfile` | Data profile |
| `recommendation` | `RecommendationResult \| None` | Auto recommendation; None for explicit classifiers |
| `prior` | `PriorInfo` | Prior value/source/estimator/auto_selected and target provenance |
| `cv_metrics` | `dict[str, CVMetric]` | Metric CV means / standard deviations / availability |
| `cv_provenance` | `dict` | CV metadata, including fold count |
| `final_model` | estimator \| None | Full-data refit model; None with refit=False |
| `diagnostic` | `PUDiagnosticReport \| None` | Structured diagnostics; None with refit=False |
| `issues` | `tuple[ProfileIssue, ...]` | Error/warning issues |
| `provenance` | `dict` | Workflow audit: classifier/prior/architecture/backbone/device/encoder/sample_weight, etc. |

`has_errors` / `has_warnings` indicate issue severity. `summary()` returns a compact readable
summary with CV means±standard deviations, issues, and assumption notes.

## Distribution-shift API

Usage and interpretation: [distribution-shift guide](../howto/distribution_shift.md).
Examples: [11_distribution_shift.py](../../../examples/minimal/11_distribution_shift.py),
[12_shift_decision_tools.py](../../../examples/minimal/12_shift_decision_tools.py), and
[13_dynamic_joint_shift.py](../../../examples/minimal/13_dynamic_joint_shift.py).

### `analyze_pu_shift`

```python
report = analyze_pu_shift(
    X_source,
    y_source_pu,
    X_target,
    y_target_pu=None,
    alpha=0.1,
    probability_clip=1e-6,
    cv=5,
    random_state=42,
    moderate_auc=0.60,
    high_auc=0.75,
    min_effective_sample_fraction=0.50,
    max_boundary_fraction=0.05,
    min_relative_mass=0.10,
)
```

Returns `PUShiftReport`:

| Field | Contract |
|---|---|
| `domain_auc` | Stratified OOF domain-classification ROC AUC, direction-symmetrized into `[0.5,1]` |
| `severity` | low / moderate / high, default boundaries 0.60/0.75 |
| `sample_summary` | Domain sizes, flattened feature count, labeled-positive fractions, and target PU availability |
| `weight_summary` | Normalized weight quantiles, ESS, probability-clipping rate, boundary rate, and relative mass before normalization |
| `adaptation_ready` | Target PU supplied and ESS/boundary/relative-mass coverage gates passed |
| `source_importance_weights` | Source-row-aligned marginal relative weights, normalized to mean one |
| `issues` | ProfileIssue tuple with codes, severities, explanations, and actions |

`report.to_dict()` / `to_json()` omit the full weight vector; `save(.csv)` exports it separately.
`save(.json/.md)` saves structured/readable reports. Weight-estimation scope is fixed to
`marginal_covariate`: a bounded relative transform of the marginal density ratio, not joint
`p_target(x,y)/p_source(x,y)` weights.

### `ShiftAwarePUPipeline`

```python
workflow = ShiftAwarePUPipeline(
    pipeline=None,               # Optional configured PUPipeline.
    classifier="elkan_noto",    # Used when pipeline=None.
    alpha=0.1,
    shift_cv=5,
    allow_unstable=False,
    cv=5,                        # Remaining keywords pass to PUPipeline.
    random_state=42,
)
result = workflow.fit_evaluate(
    X_source,
    y_source_pu,
    X_target,
    y_target_pu=y_target_pu,
    y_true_source=None,
    y_true_target=None,
    class_prior=None,
    target_class_prior=None,
    adapt=True,
    refit=True,
)
```

`adapt=True` requires target PU labels and passes the audit's source weights to `PUPipeline`;
coverage-gate failure raises `PipelineError` by default. `adapt=False` runs the audit and an
unweighted source baseline. `ShiftAwarePipelineReport` separately stores `shift`,
`source_pipeline`, and `target_metrics`; target-truth metrics remain `supervised_oracle`.
Its guarantee is fixed to `covariate_shift_only`. Target PU labels provide an adaptation safety
gate and target evaluation input; they do not enter marginal domain-density-ratio estimation.

### `ShiftAwarePUPipeline.compare`

```python
comparison = workflow.compare(
    X_source, y_source_pu, X_target,
    y_target_pu=...,              # Required.
    y_true_source=None, y_true_target=None,
    class_prior=None, target_class_prior=None,
    primary_metric=None,          # Defaults to first available oracle metric.
    min_improvement=0.0,
)
```

Runs unweighted and weighted arms on the same target set and returns `ShiftComparisonReport`.
`metric_deltas[*].improvement` has unified direction: positive always favors weighting (risk
differences are negated). Automatic `recommendation` uses only target-truth oracle metrics or
target-prior-dependent metrics. With only PU-observed metrics it is `audit_only`; failed
coverage without override yields `collect_target_data`. Oracle-guided arm selection requires
a designated validation/audit set; keep final test truth independent of that decision.

### `PUShiftMonitor`

```python
monitor = PUShiftMonitor(
    X_reference,
    y_reference_pu,
    alpha=0.1,
    cv=5,
    auc_jump_threshold=0.10,
    label_rate_jump_threshold=0.05,
)
window, shift = monitor.update(
    X_window, y_window_pu=y_window_pu, window_id="2026-08", timestamp=None
)
monitor.save_history("history.json")
```

`ShiftWindow` stores current values, neighboring-window deltas, `alert_level`, and `alert_codes`.
Window IDs must be unique; `load_history` requires an exact match between saved configuration
and the current monitor.

### `analyze_domain_assumptions`

```python
report = analyze_domain_assumptions(
    X_source, y_source_pu, X_target, y_target_pu,
    source_class_prior=None, target_class_prior=None,   # Estimate each domain independently if absent.
    prior_estimator="pen_l1",
    prior_shift_threshold=0.05, propensity_shift_threshold=0.05,
    sensitivity_radius=0.05, bootstrap_replicates=0,
    confidence=0.95, random_state=42,
)
```

Accepts source/target features and PU labels with optional domain priors. Missing priors are
estimated independently per domain. Returns `DomainAssumptionReport`, with conclusion
`stable`, `class_prior_shift`, `labeling_mechanism_shift`, `both_shift`, or `inconclusive`.
Mean labeling propensity does not identify SCAR/SAR.

With `bootstrap_replicates>=2`, each domain receives row-level nonparametric resampling and
prior re-estimation per replicate. Percentile intervals propagate to priors, labeling rates,
mean propensities, and the three domain differences. `uncertainty` records requested,
successful, and failed replicates; this reflects sampling/estimator variation, not bias from
identifiability assumptions.

### `analyze_pu_uncertainty`

Builds a selective-prediction and active human-review plan for a fitted model.

```python
report = analyze_pu_uncertainty(
    estimator,                        # Fitted model; calls predict/predict_proba.
    X,
    y_pu=None,                        # Supplied labeled positives are excluded from queries.
    y_true=None,                      # Enables supervised truth metrics when supplied.
    min_confidence=0.5,               # Selective-prediction confidence threshold.
    query_budget=0,                   # Maximum query count; 0 requests no queries.
    query_strategy="uncertainty",     # "uncertainty" / "shift_weighted" / "diverse_uncertainty".
    importance_weight=None,           # Required for shift_weighted; aligned with X rows.
    random_state=42,
)
```

Returns `PUUncertaintyReport`, a dataclass publicly exported from `pu_toolbox`:

| Field | Contents |
|---|---|
| `positive_probability` | Row-aligned positive probabilities or probability-like scores from the model |
| `uncertainty` | Row-aligned probability-margin uncertainty |
| `selective_predictions` | Row-aligned 0/1 predictions, with -1 for confidence below min_confidence |
| `query_indices` | Recommended review indices, ordered by query_strategy |
| `summary` | Sample/abstention counts, coverage, query candidate count, requested-query result count, and selective accuracy when truth is supplied |
| `provenance` | Method, score source, confidence threshold, query strategy, and label-exclusion settings |

`importance_weight` must be finite, nonnegative, nonzero, and aligned with the rows being
reviewed; source-row audit weights cannot replace a target-row vector.
Probability, uncertainty, and selective-prediction arrays are omitted from `to_json()`;
`query_indices`, summary, artifact names, and provenance remain. CSV preserves per-row
probabilities, uncertainty, selective predictions, and query flags.

### `JointShiftPUClassifier` (research)

```python
clf = JointShiftPUClassifier(
    alpha=0.1, target_mix=0.5, max_iter=5, tolerance=1e-3,
    n_cv_folds=3, probability_clip=1e-5, random_state=42,
)
clf.fit(X, y_pu, X_target=..., y_target_pu=..., class_prior=..., target_class_prior=...)
```

Import from `pu_toolbox.estimators.research`. Besides source X/y_pu, fit requires explicit
`X_target`, `y_target_pu`, `class_prior`, and `target_class_prior`. It is outside the stable
registry and Pipeline auto selection. `get_pu_metadata()["guarantee"]` is fixed to
`research_joint_shift_approximation`.

### `DynamicJointShiftPUClassifier` (research)

```python
clf = DynamicJointShiftPUClassifier(
    alpha=0.1, beta=0.5, hidden_dim=128, feature_dim=128,
    max_epochs=200, classifier_learning_rate=1e-4, weight_learning_rate=1e-3,
    weight_correction=True, classifier_correction=True,
    training_mode="dynamic", random_state=42, device="auto",
)
```

Import from `pu_toolbox.estimators.research`. This Torch clean-room path implements
Kumagai et al., AISTATS 2025, Equations (13), (19)–(23), and Algorithm 1: weight steps freeze
shared features and update bounded weight heads; classification steps freeze current weights
and update shared features/classifier head. `training_mode="two_step"` and both correction
switches support paper-style ablations. The authors have not released source; metadata is
`clean_room_paper_objective`, not `official_exact`. It stays outside the stable registry/auto.

`build_joint_shift_estimator(...)` builds `dynamic`, `trpu`, `tepu`, `fine_tune`, `mmd`,
`two_step`, and three correction ablations. Neural comparisons share feature/classifier-head
sizes and absolute-value-corrected PU risk.

## PUTuner

```python
tuner = PUTuner(
    classifier="nnpu",          # Explicit registry name required; "auto" raises an error.
    param_grid={...},           # Each combination expanded by ParameterGrid.
    scoring="pu_zero_one_risk", # Selection metric.
    higher_is_better=None,      # By default only pu_zero_one_risk is minimized.
    metrics=None,               # Reported metrics.
    **pipeline_params,          # cv / prior_estimator / random_state / architecture, etc.
)
result = tuner.fit(X, y_pu, y_true=None, class_prior=None)
```

Runs PU-stratified Pipeline CV for each `sklearn.model_selection.ParameterGrid` combination.
`pipeline_params` accepts `cv`, `prior_estimator`, `random_state`, `architecture`, and related
workflow settings. Trials use `refit=False`; only the selected combination receives full-data
refit and diagnostics.

Returns `TuningResult`:

| Field | Meaning |
|---|---|
| `best_params` / `best_score` | Best valid combination and CV mean |
| `best_report` | Selected combination's complete PipelineReport with full-data refit model |
| `trials` | Every TuningTrial with status ok / unavailable / failed |
| `scoring` | Canonical metric name |
| `higher_is_better` | Selection direction; by default only pu_zero_one_risk is minimized |

Raises `PipelineError` if no trial can compute the selection metric. Scoring with truth-based
metrics uses clean-label supervision; provide only permitted validation truth, keeping final
test truth separate. Full example: [model-tuning guide](../howto/model_tuning.md).

## PUModelComparator

```python
cmp = PUModelComparator(
    classifiers=["nnpu", "upu"], # At least two explicit registry names; "auto" raises an error.
    classifier_params=None,      # Per-model constructor arguments: {name: params}.
    scoring="pu_zero_one_risk",
    higher_is_better=None,
    metrics=None,
    **pipeline_params,
)
result = cmp.fit(X, y_pu, y_true=None, class_prior=None)
```

Compares at least two explicit registry names under the same PU-aware CV settings. Returns
`ModelComparisonResult` with `best_classifier`, `best_score`, per-model `trials`, and a
full-data-fitted `best_report`. Failed models are isolated and recorded; nonwinning models run
CV only.

## Progress and cancellation

`PUPipeline.fit_evaluate`, `PUTuner.fit`, and `PUModelComparator.fit` accept
`progress_callback=` and `cancellation_token=`.

```python
token = CancellationToken()

def on_progress(update: ProgressUpdate):
    print(update.stage, update.fraction)   # Fields: stage/completed/total/message/fraction

report = pipe.fit_evaluate(..., progress_callback=on_progress, cancellation_token=token)
token.cancel()                             # Thread-safe cooperative cancellation signal
```

`ProgressUpdate` fields are `stage` / `completed` / `total` / `message` / `fraction` (a property).
`CancellationToken.cancel()` sets the signal; the next safe-boundary check raises
`RunCancelledError`. An individual model `fit` already in progress is not forcibly interrupted.

## build_encoder

The shared encoder builder for deep classifiers is defined in
`pu_toolbox/estimators/deep/vision.py` and publicly exported as
`from pu_toolbox import build_encoder`. PUPipeline calls it internally for
`architecture="cnn"`; callers can also pass its result to a compatible classifier.

```python
encoder = build_encoder(
    architecture,             # "mlp" | "cnn"
    *,
    backbone="cnn13",         # "cnn13" / "cnn13_no_bn" / "resnet18" / "resnet50" (CNN only)
    in_channels,              # Image channel count, e.g. RGB=3
    normalization_mean=None,  # Per-channel means; default 0.5
    normalization_std=None,   # Per-channel standard deviations; default 0.5
)
```

- `"mlp"` returns `None`, selecting the classifier's built-in MLP path for tabular data.
- `"cnn"` returns a `build_wconpu_backbone(...)` image backbone for 4-D NCHW input,
  including channel normalization.
- An invalid `architecture` raises `ValueError`.

## profile_pu_data

Usage: [data profiling](../howto/data_profiling.md). Executable example:
[07_data_profiling.py](../../../examples/minimal/07_data_profiling.py).

```python
report = profile_pu_data(
    X,
    y_pu,
    y_true=None,               # Supplying truth enables identifying audit evidence.
    class_prior=None,
    input_sampling="os",      # Same-pool OS or independent P/population-U TS.
    min_labeled_positives=30,
    max_unlabeled_to_positive=100.0,
    low_variance_threshold=1e-12,
    scar_auc_threshold=0.65,
    cv=5,
    random_state=42,
)
```

Every observation with `y_pu == 1` must also be positive in supplied `y_true`;
otherwise the interface rejects the input. `input_sampling="os"` selects same-pool OS
semantics; `"ts"` selects independent P and
population-mixture U. Only OS interprets the labeled fraction divided by the population
`class_prior` as an implied labeling frequency. TS records the extra-P table fraction
and sets `implied_label_frequency=None`.

### Returned PUDataProfile

| Field | Contents |
|---|---|
| `summary` | Sample and feature counts, labeled-positive and unlabeled counts, PU ratio, sparsity, class prior, and implied labeling frequency |
| `feature_statistics` | Missing/infinite-value counts and indices of constant and low-variance columns |
| `selection_diagnostic` | AUC, status, evidence source, identifiability, actual fold count, and evaluation sample count |
| `issues` | Issues with stable `code`, severity, explanation, and suggested action |
| `assumption_hints` | Notes on SCAR/SAR, prior dependence, and interpretation limits |

Helpers: `report.has_errors` / `report.has_warnings` / `report.format_text()` / `report.to_dict()`.

`selection_diagnostic["status"]` is `plausible` when CV AUC does not exceed the threshold,
`at_risk` when AUC exceeds it, or `inconclusive` when a group has fewer than two samples or
features contain nonfinite values. Without audited truth, observed-mixture evidence does not
identify SCAR versus SAR; `plausible` is not proof of SCAR.

### Issue codes

| Code | Severity | Suggested action |
|---|---|---|
| `no_labeled_positives` | error | Check label encoding or collect trusted positives |
| `no_unlabeled_samples` | error | Use supervised learning or add unlabeled population data |
| `missing_features` / `infinite_features` | error | Clean or impute within a pipeline fitted only on the training fold |
| `few_labeled_positives` | warning | Use repeated validation and confidence intervals; collect more positives where possible |
| `extreme_pu_imbalance` | warning | Use PU-stratified splits and imbalance-aware metrics |
| `constant_features` | warning | Remove constant columns within the training pipeline |
| `high_dimensional_data` | warning | Use regularized models and avoid preprocessing on all data |
| `inconsistent_class_prior` | warning | Review the prior, sampling population, and label definitions |
| `sar_signal` | warning | Audited positives support SAR; prioritize PUSB/LBE evaluation and sensitivity analysis |
| `observed_selection_signal` | info | Evidence is non-identifying; obtain an audit or labeling-policy information |
| `low_variance_features` | info | Review feature scaling; low-variance columns may carry little signal |
| `selection_diagnostic_inconclusive` | info | Evaluation groups are too small or features are nonfinite; obtain an audit or trusted positive labels |

## recommend_methods / recommend_from_profile

Usage and selection rationale: [method selection](../concepts/method_selection.md).

```python
result = recommend_methods(
    X, y_pu,
    scenario=None,       # Scenario or a string, e.g. "case_control"
    assumption=None,     # Assumption or a string, e.g. "scar"
    class_prior=None,    # Known P(Y=1); None excludes methods requiring a prior.
    has_gpu=False,
    top_k=5,
    random_state=42,
    config=None,         # ScoringConfig; defaults to DEFAULT_CONFIG exported by pu_toolbox.advisor.
)

result = recommend_from_profile(
    profile,             # Existing PUDataProfile; skips repeated profiling.
    scenario=None, assumption=None, class_prior=None,
    class_prior_source=None,  # e.g. "user"/"estimated"; affects global-warning wording, not provenance.
    has_gpu=False, top_k=5, config=None,
)
```

Returns `RecommendationResult` with `candidates` (`MethodCandidate` fields:
name/score/rank/reasons/warnings/metadata), `filters_applied`, `global_warnings`, and
`provenance`. Export with `result.to_json()` / `to_markdown()` / `save(path)`.

### ScoringConfig: configurable scoring weights

The `config` argument of `recommend_methods` / `recommend_from_profile` defaults to
`DEFAULT_CONFIG`. The seven `*_max` fields are the scoring dimensions' anchors; their sum
`max_raw_score` is the denominator in `raw / max_raw_score * 100`:

| Field | Default | Meaning |
|---|---|---|
| `assumption_max` | `30.0` | Maximum assumption-match score |
| `maturity_max` | `20.0` | Maximum maturity score |
| `source_max` | `15.0` | Maximum source-status score |
| `scale_max` | `20.0` | Maximum data-scale score |
| `gpu_max` | `5.0` | Maximum GPU-availability score |
| `labeled_pos_max` | `10.0` | Maximum labeled-positive sufficiency score |
| `cost_max` | `10.0` | Maximum training-cost score |
| `maturity_scores` | `{stable:20, research:12, experimental:5}` | Scores by maturity level |
| `source_scores` | `{official_exact:15, official_bundle:12, official_related:9, third_party_only:6, not_found:3, unknown:1}` | Scores by source status |
| `small_data_threshold` | `1000` | Small-data threshold |
| `large_data_threshold` | `10000` | Large-data threshold |

`max_raw_score` is the sum of the seven `*_max` fields. `__post_init__` checks that
`maturity_scores` / `source_scores` values do not exceed their corresponding maxima,
keeping scores within the 0–100 contract.

## Data generation

Import these functions from `pu_toolbox.preprocessing`. Usage:
[SCAR/SAR simulation](../howto/sar_simulation.md) and [OS/TS sampling](../howto/pu_sampling.md).
Examples: [06](../../../examples/minimal/06_sar_simulation.py) and
[15](../../../examples/minimal/15_pu_sampling.py).

| Function | Signature and return value |
|---|---|
| `make_sar_propensity` | `(X, y_true, *, mechanism="linear", label_frequency=0.5, strength=1.0, feature_weights=None) -> propensity`: per-sample labeling propensity `P(S=1|Y,X)`; true-negative entries are fixed at 0 |
| `make_sar_labels` | `(X, y_true, *, mechanism="linear", label_frequency=0.5, strength=1.0, feature_weights=None, ensure_labeled=True, random_state=None, return_propensity=False) -> y_pu`; returns `(y_pu, propensity)` when `return_propensity=True` |
| `make_sar_dataset` | `(n_samples=1000, *, n_features=5, class_prior=0.3, separation=2.0, mechanism=None, label_frequency=0.5, strength=1.0, feature_weights=None, ensure_labeled=True, random_state=None) -> (X, y_pu, y_true, propensity)` |
| `make_scar_labels` | `(y_true, c=0.5, random_state=None) -> y_pu`: same-pool fixed-count uniform sampling of positives without replacement, `max(1, round(c*n_pos))`; returns all U if there are no positives |
| `make_fixed_count_labels` | `(y_true, n_labeled, random_state=None) -> y_pu`: same-pool uniform sampling of positives without replacement, preserving all input row order |
| `make_pu_labels` | `(y_true, mechanism="scar", c=None, n_labeled=None, random_state=None) -> y_pu`: scar uses c; fixed_count uses n_labeled |
| `make_case_control_labels` | `(y_true, n_labeled, random_state=None) -> y_pu`: compatibility name for fixed-count labeling; emits FutureWarning and is scheduled for removal in 2.0 |
| `make_scar_dataset` | `(n=100, c=0.5, n_features=5, separation=4.0, random_state=None) -> (X, y_pu, y_true)`: n samples per class; used by CLI `make-demo-data` |
| `make_gaussian_case_control_data` | `(n_p=50, n_u=100, *, class_prior=0.5, n_features=5, separation=2.0, random_state=None) -> (X, y_pu, y_true, metadata)`: independent TS Gaussian P and population-mixture U |
| `make_gaussian_pu_data` | `(n_p=50, n_u=100, n_features=5, separation=2.0, random_state=None) -> (X, y_pu, class_prior)`: all generated positives are visible and U comes from negatives; the third value is the true positive proportion of the concatenated table |

`make_fixed_count_labels` accepts one-dimensional true binary labels. `n_labeled` must be
an integer in `[1,n_positive]`; `numpy.integer` is accepted, while bool and floating-point
values are rejected. All-positive input can also receive fixed-count labeling.
Legacy aliases retain two-class input validation; dispatcher `mechanism="case_control"`
uses the same compatibility entry point.

For `make_gaussian_case_control_data`, `n_p/n_u/n_features` are positive integers excluding
bool. `class_prior` is a finite real in `[0,1]`, and `separation` is a finite nonnegative real;
both reject bool and strings. Each U sample's true class is drawn independently, then its
features are drawn from the corresponding Gaussian conditional distribution.
Output rows are ordered P then U; `y_true` supports synthetic audits and evaluation.
`metadata` contains `input_sampling="ts"`, `sampling_policy="independent_generative"`,
`population_class_prior`, `unlabeled_class_prior`, `training_class_prior`,
`n_positive`, `n_unlabeled`, `source_distribution`, `sample_ids`, and `random_state`.
`source_distribution` records family, positive_mean, negative_mean, identity covariance,
and n_features. `sample_ids` are draw identities `positive:i` / `population:i`, unique within
this output. `random_state` accepts an integer, RandomState, or None; metadata records a
numeric seed only when an integer was supplied.

- `make_sar_dataset` accepts `mechanism="scar"` / `"linear"` / `"nonlinear"`; see
  [SCAR/SAR](../concepts/scar_sar.md) for definitions. Omitting it selects `"linear"`
  (feature-dependent SAR) and emits `UserWarning`. Class-prior estimators assume SCAR;
  explicitly use `mechanism="scar"` for that setting.
- `label_frequency` is the calibrated target mean positive-class propensity; compute the
  realized labeling rate from the sampled labels.
- `make_sar_labels` defaults to `ensure_labeled=True`: if a small-sample draw labels no
  positive, it labels the highest-propensity true positive so downstream training can proceed.
- CLI `make-demo-data` defaults to separation 1.0; direct `make_scar_dataset` defaults to 4.0.

## analyze_pu_sensitivity

Usage: [sensitivity analysis](../howto/sensitivity_analysis.md). Executable example:
[09_sensitivity_analysis.py](../../../examples/minimal/09_sensitivity_analysis.py).

```python
analysis = analyze_pu_sensitivity(
    y_valid,
    y_pred,                              # Must use {0,1}.
    scores=None,                         # Finite 1-D array; zero is the PU zero-one risk threshold.
    class_priors=None,                   # Each value in (0, 1).
    label_propensities=None,             # Each value in (0, 1].
)
```

Supply at least one of `class_priors` and `label_propensities`; `y_pu` must contain both
labeled-positive and unlabeled groups. Returns `PUSensitivityAnalysis`, with each point containing:

| Field | Meaning |
|---|---|
| `axis`, `value` | Parameter being scanned and its assumed value |
| `class_prior` | Assumed class prior or value implied by `q/c̄` |
| `label_propensity` | Assumed mean propensity or value implied by `q/π` |
| `is_consistent` | Whether probability bounds and the observational identity hold |
| `consistency_reason` | Direct explanation of an inconsistency |
| `pu_estimated_precision` | `min(π·Recall̂_P/P̂(Ŷ=1), 1)` |
| `pu_zero_one_risk` | `2π·FNR̂_P + mean_all(score>0) − π` (OS) |

`metric_ranges` summarizes only consistent points with available metrics. Export with
`analysis.to_frame(axis=...)` / `analysis.save(path)` (strict JSON without NaN/Infinity).

## build_diagnostic_report

Usage: [diagnostic reports](../howto/diagnostic_reports.md). Executable example:
[08_diagnostic_report.py](../../../examples/minimal/08_diagnostic_report.py). Three input modes:

`input_sampling="os"` (default) uses the complete input table for population risk and precision;
`"ts"` uses native U, including for oracle metrics. `profile_pu_data` also accepts
`input_sampling`; TS sets `implied_label_frequency` to `None`.

Direct `pu_zero_one_risk(..., input_sampling="ts")` retains its native-U default formula;
`pu_estimated_precision(..., input_sampling="os")` retains its complete-table default.
Workflows and diagnostics pass the same sampling design explicitly.

```python
# Data only: runs profile_pu_data; prediction-based metrics have basis=unavailable.
report = build_diagnostic_report(X, y_pu, class_prior=0.30, random_state=42)

# Fitted estimator: calls predict, preferring decision_function and falling back to predict_proba[:, 1].
# Does not call fit; an unfitted estimator raises ValueError.
report = build_diagnostic_report(X_valid, y_valid, estimator=classifier,
                                 class_prior=0.30)

# Explicit outputs: estimator and y_pred/scores are mutually exclusive.
report = build_diagnostic_report(X_valid, y_valid, y_pred=predictions,
                                 scores=decision_scores, class_prior=0.30)
```

Optional `y_true` enables identifying selection-dependence checks and computes
`accuracy`/`f1`/`roc_auc`, labeled `supervised_oracle`. It is not passed to the estimator.

### Metric evidence levels

| `basis` | Metric inputs | Additional assumptions |
|---|---|---|
| `pu_observed` | PU labels and model predictions | No true labels required |
| `class_prior_dependent` | PU labels, predictions, and class prior | Values change with the prior |
| `supervised_oracle` | True labels and predictions/scores | Only for evaluation with truth |
| `unavailable` | Insufficient inputs or invalid numeric values | `reason` explains the missing value |

The fixed metrics are `labeled_positive_recall` / `unlabeled_negative_rate` /
`predicted_positive_rate` (`pu_observed`); `pu_estimated_precision` / `pu_zero_one_risk`
(`class_prior_dependent`); and `accuracy` / `f1` / `roc_auc` (`supervised_oracle`).

### Report issue codes: model-output checks

| Code | Severity | Meaning |
|---|---|---|
| `constant_predictions` | warning | Every sample is assigned the same class; review the threshold, prior, and convergence |
| `nonfinite_scores` | error | Scores contain `NaN/inf`; supervised AUC is marked unavailable |
| `constant_scores` | warning | The model provides no ranking information; check training and feature variation |

`PUDiagnosticReport` currently has top-level `schema_version="1.0"`. `save()` infers format
from `.json`/`.md` suffixes. JSON encoding is strict, converting undefined values to `null`.

## Experiment layer (experiment)

### Four-role execution and strategy injection

For an introduction and examples extending generators, trainers, and selection strategies,
see [custom experiments](../howto/custom_experiments.md) and the executable
[14_diy_experiment.py](../../../examples/minimal/14_diy_experiment.py).

```python
ExperimentRunner(
    *, seed=0, generator=None, protocols=None, class_prior=None,
    threshold_candidates=None, manifest_path=None, config=None,
)
runner.fit(model, train, pu_val, clean_val, test) -> RunResult
```

`train`, `pu_val`, `clean_val`, and `test` are `DatasetPart` objects with `view="clean"`,
true 0/1 labels, and disjoint global sample indices. Set `for_selection=False` for test.
The default generator is `SCARGenerator()`; the default selectors are
`[ProtocolPA(), ProtocolOA()]`. `generator` and `protocols` accept strategy instances.
Inject a trainer instance through `config["trainer"]`; the default is `DeepFitTrainer()`.
Supply candidate parameters through `config["candidates"]`, default `[{}]`.
PA receives the population prior from `class_prior` or
`config["split_ref"]["class_prior"]["population"]`.

Import extension interfaces `Generator`, `Trainer`, and `SelectionProtocol` from
`pu_toolbox.experiment.protocols`. Import trajectory/selection records `EpochRecord`,
`RunTrajectory`, `SelectionArtifact`, and `RunResult` from `pu_toolbox.experiment.tracking`.
An ordinary custom selector receives generated pu_val; one inheriting `ProtocolOA` receives
clean_val. The returned `SelectionArtifact.protocol` must match the strategy's `name`.

`RunResult` provides `selections`, `test_metrics`, `manifest`, `failures`, `resources`, and
`selected_models`. Continued prediction with a classical model must reuse the selected threshold
and validation-side score transform; the example's `predict_selected` implements that process.

### Method-specific training adapters

The independent exploratory adapters do not modify the locked Survey/Pipeline/CLI contracts.
`pu_toolbox.experiment.cvir_adapter.CVIRKnownPriorTrainer` keeps CVIR's explicit α_U and its
source separate from PA's population π and rejects calibrated TS/SAR.
`CVIRClassifier.input_sampling="ts"` means directly supplied native independent P and marginal U;
it does not request runner risk calibration.
`CVIRPopulationPriorTrainer(population_prior_source=...)` independently supports engineering
SCAR OS and TS-compatible paths. Constructor α_U/source must be unset: the adapter converts
the population prior using actual PU counts for OS risk, while calibrated TS binds π.
It does not pass population π to estimator `fit`. Actual priors, formulas, counts, and sources
enter the manifest; constructor parameters are restored after fitting. A checkpoint byte bound
is mandatory; formal Survey/SAR runs are rejected. See the
[CVIR method card](../../research/method_cards/CVIR.md) for optional Chinese background on
calibration wiring and boundaries.

`pu_toolbox.experiment.pulns_adapter.PULNSExtraSupportTrainer` requires independent labeled
support. Before training it audits actual four-role, prior-estimation, and support IDs plus
applicable grouping; it records the extra supervision and permits only terminal OA.
It also requires support_origin/original_role_counts when carved, plus
`support_source_verified` / `support_source_receipt`. Per-role original/remaining/reduced pool effects are recorded;
train reduction must match support_removed_from_train. Source provenance defaults to an
unverified caller declaration. Missing provenance, budgets, identities, or ADNI group evidence
cause rejection; validation/test data are not silently repurposed.

`pu_toolbox.experiment.clean_support_adapters.SelfPUCleanSupportTrainer` and
`LaGAMCleanSupportTrainer` reuse the same six-role, grouping, and extra-label-budget audits,
explicitly connecting the existing Self-PU clean-meta and LaGAM meta-label paths. Their
constructor arguments match the PULNS adapter above. Self-PU uses support only for meta
reweighting and internal teacher selection; external OA still uses independent clean_val.
The exploratory path supports SCAR OS and terminal OA.

Explicit checkpoint variants are
`pu_toolbox.experiment.support_checkpoints.SelfPUCleanSupportCheckpointTrainer` /
`LaGAMCleanSupportCheckpointTrainer`, with the same constructor audit arguments as the
terminal adapters. Their default risk view is SCAR OS, and they require a positive integer
runner `checkpoint_bytes_per_component`. Selection uses
`pu_toolbox.experiment.strategies.ProtocolPACleanSupport` (output `PA+clean-support`)
and/or `ProtocolOA`; ordinary PA is rejected. Exploratory runs capture completed meta-active
epochs. Versioned Survey runs also capture warmup snapshots according to their bound protocol;
the adapter records the policy basis and does not grant formal admission. Self-PU retains both
teacher streams for independent selection. Existing terminal OA and
Self-PU's default ablation remain unchanged. The exploratory CLI enables this only through
explicit `--selection-scope meta_checkpoints`; terminal selection remains the default.

`SelfPUTSCleanSupportCheckpointTrainer` additionally exposes an explicit SCAR TS-compatible
full-meta path. It requires `os_or_ts="ts"`, rejects bound Survey/recipe configurations, and
retains native-U identity pools. Its calibration applies only to the negative-marginal row-mass
blend; it does not verify population-mixture restoration or grant formal admission. The CPU/MLP
entry point `scripts/run_clean_support_experiment.py` selects this trainer only for
`--method self_pu --selection-scope meta_checkpoints --training-view ts`; other methods and
the default OS/terminal path remain unchanged.

`PULNSCleanSupportCheckpointTrainer` reuses those audits and both selection paths. Its
candidates are classifiers after completed episodes, excluding reward probes; it records
`captured_episodes`. Enable it with the same CLI's
`--method pulns --selection-scope meta_checkpoints`.

`pu_toolbox.experiment.support_data.prepare_support_dataset` prepares independent support
from reviewed original arrays under an explicit budget and refits preprocessing on remaining
train. `prepare_support_from_split` carves directly from a frozen four-role split for text
and ADNI, retaining pinned SBERT text features and raw uint8 images.
`save_support_dataset` writes only to a new directory. `load_support_dataset` validates
the five roles, sampling, files, and receipt. All eight datasets are supported.
`support_seed` must equal the split seed, recorded explicitly as `support_seed_policy`.
ADNI accepts only `group_policy="unverifiable"`, recording
`subject_isolation="not_checkable"`, known risks, and owner approval. This does not establish
subject isolation; callers remain responsible for source authenticity.

Artifacts also retain support_origin, complete original-role counts, and any unverified
source declaration. Experiment entry points forward all four provenance arguments according
to trainer constructor signatures. Artifact-layer validation is distinct from native adapter
validation; older adapters may reject newer provenance contracts.

The experiment layer exposes four-role PU research protocols through `pu_toolbox.experiment`.
For the public Survey split protocol, see
[pu_survey_protocol.md](../../research/pu_survey/pu_survey_protocol.md) §2.3–§2.4
(currently Chinese). [experiment_layer.md](../../dev/experiment_layer.md) §1 is optional
Chinese design background. The table lists public symbols and their purpose; docstrings
define parameter contracts and behavior.

| Symbol | Purpose |
|---|---|
| `ExperimentRunner` | Template Method workflow: validate data and model capabilities → generate PU views → train candidates → offline PA/OA selection → independent test → manifest recording; `config["architecture"]` can explicitly declare `"mlp"`/`"cnn"`, with strategies injected at each extension point |
| `DatasetBundle` | Four-role contract: `train` / `pu_val` / `clean_val` / `test` |
| `DatasetPart` | One partition with explicit label view (`view="clean"` for true labels / `view="pu"` for PU observations) and global sample indices |
| `validate_bundle` | Validate the four-role contract: views, true binary labels, disjoint indices, and `test.for_selection=False` |
| `SurveyDatasetSpec` | Read-only Survey dataset specification for modality, positive/negative class mappings, and official-test policy |
| `survey_dataset_catalog` | Return the locked MNIST/F-MNIST/CIFAR-10/ADNI/IMDB/20News/Spambase/Connect-4 catalog |
| `binaryize_survey_labels` | Produce true binary labels using protocol-locked mappings and reject classes outside the catalog |
| `prepare_survey_dataset` | Deterministically create 90% train/5% PU-val/5% clean-val plus official or derived test from caller-supplied arrays, returning a bundle and split manifest; optional `source_indices` defaults to `np.arange`, and official-test `test_indices` defaults to subsequent indices. Explicit indices must be a one-dimensional array of distinct finite JSON scalars (`None`/`bool`/`int`/`float`/`str`) matching the data length. Derived-test datasets (Spambase/Connect-4/ADNI) reject `test_indices`; test inherits `source_indices` |
| `fit_survey_image_preprocessing` | Fit channel statistics only on train images; lock NCHW, channels, scaling, and ResNet-18 settings; return unit-interval arrays and auditable specifications |
| `transform_survey_images` | Reuse train-frozen shape/scaling contracts for validation/test and reject cross-partition scaling drift |
| `build_survey_image_encoder` / `build_survey_image_augmentation` | Build a randomly initialized ResNet-18 from frozen specifications; augmentation is returned only for `train`, with PU/clean validation and test fixed to `None` |
| `SurveyImagePreprocessing` | JSON manifest specification for image size, first layer, normalization, augmentation, training data, and configuration SHA-256 |
| `calibrate_ts_os_batch` | **Survey boundary adapter**, delegating role construction to `pu_toolbox.core.training_views`: produce OS or TS-compatible loss views for a training batch; TS includes P in both positive and U loss inputs. Native-TS ledger and train-only gates stay in this layer |
| `TSOSBatchView` | Converted P/U loss arrays, original indices, run view, and calibration manifest; `run_view` is `"OS"` / `"TS-compatible"`, historical vocabulary confined to this layer's output |
| `adapt_image_bundle_to_features` | Extract four-role 2-D features in eval/no-grad mode with a fixed CNN or one declared as train-only fitted; verify unchanged encoder state and record weight/feature/split hashes |
| `LeaderboardRunSpec` / `partition_fair_leaderboard_runs` | Validate common dataset splits, seeds, epoch/batch/tuning budgets, and representations within a path; enforce grouping by training path (`native_2d`, native CNN, `cnn_feature_adapter`), marking adapter paths `benchmark-adapted` |
| `encode_survey_texts` | Produce 384-D embeddings using fixed `all-MiniLM-L6-v2`; require a recorded revision, address the cache by text content and normalization options, and validate SHA-256 before reuse |
| `SBERT_MODEL_NAME` / `SBERT_EMBEDDING_DIMENSION` | Model-identity and output-dimension constants locked by the Survey text protocol |
| `SCARGenerator` | Fixed-count uniform positive labeling without replacement, `n_L = round(c·n₊)`, recording `c_realized` |
| `SARLBEAGenerator` | SAR-LBE-A: `p ∝ scores^k` (k=10) with 0.9/0.1 smoothing (PU-Bench 2d95a19) |
| `SARLBEBGenerator` | SAR-LBE-B: `p ∝ (1.5 + shrink_coef − scores)^k`, clipping negatives and using uniform fallback for all-zero weights |
| `CleanLabelGenerator` | PN oracle view: pass true labels unchanged (`n_L = n₊`), declare `output_view="clean"`; PA is structurally rejected by view validation. Record `c` without applying it; oracle behavior is constant over c |
| `ProtocolPA` | Select run/epoch/threshold using only PU validation, with true labels structurally inaccessible; maximize proxy accuracy (Wang et al. 2026 Def. 1, OS branch; see `proxy_accuracy` below) on a min-max-normalized threshold grid; missing required `class_prior` raises an error |
| `ProtocolOA` | Oracle comparison: select run and threshold by true-label validation accuracy after min-max normalization |
| `FitTrainer` | Single-point training for classical estimators without epochs; forward `class_prior` only if accepted |
| `DeepFitTrainer` | Prefer `pu_validation_data` without true-label leakage, otherwise probe `validation_data`; convert `history_` into selection trajectories. Successful validation-aware fitting without history produces one point without fitting again |
| `SupervisedTrainer` | PN oracle: supervised baseline trained on true labels, declaring `trains_on_real_labels=True`. Use with `CleanLabelGenerator` and `protocols=[ProtocolOA()]`; the runner's default PU generation rejects this trainer before training |
| `aggregate_resource_usage` | Aggregate all candidate-tuning costs and the overall peak GPU memory across seed manifests |
| `select_threshold` | Maximize accuracy over candidate thresholds; ties choose the lowest candidate |

`proxy_accuracy` is not exported at package level, so it is absent from the table. It is defined
in `pu_toolbox.experiment.strategies` (`pu_toolbox/experiment/strategies.py`) and used by `ProtocolPA`.

The three PU labeling generators (`SCARGenerator` / `SARLBEAGenerator` / `SARLBEBGenerator`)
share generation-metadata fields in manifest `generation.train` / `generation.pu_val`:
`generator` (class name), `mechanism`, `c_requested` (unclamped request),
`c_realized` (realized fraction after clamping), `n_positive`, `n_labeled_requested`
(the **unclamped** protocol formula `round(c·n₊)`), `n_labeled` (actual count),
`generation_seed`, and `label_view_sha256` (label-view digest).

Generation metadata also records `input_sampling="os"` and `sampling_policy`:
`fixed_count_uniform_without_replacement` for SCAR and
`fixed_count_weighted_without_replacement` for LBE-A/B. `CleanLabelGenerator` records
`input_sampling="single_clean_pool"` and `sampling_policy="all_real_labels"`.

Requested and realized values can differ for small `n₊` or `c`: `_n_labeled` clamps to at
least one, labeling a positive even when `round(c·n₊) == 0`. Both fields are needed to audit
this difference.

`label_view_sha256` supports checking protocol §2.4 item 4: all methods share identical P/U
labels at the same seed and c. PN oracle `CleanLabelGenerator` retains its established fields
(`mechanism` / `c_realized` / `n_labeled` / `c_requested`) as a separate c-independent generator.

Training strategies declare expected `fit` label semantics with `trains_on_real_labels`:
PU trainers default to `False` (0 means unlabeled); PN oracle trainers use `True` (0 means
true negative). Before training, the runner requires that declaration to match the generated
view, raising `ValueError` for either mismatch. It rejects a trainer class in place of an
instance in `config["trainer"]`, and requires the generator's declared view to match its
reported `mechanism` (`pn_oracle` implies true labels and therefore a clean view).

The runner also requires estimator main-`fit` `label_semantics` to match the generated view:
`"pu"` for PU, `"pn"` for clean. `"pnu"` is not treated as binary PU.
Consequently `SupervisedTrainer` with a PU risk estimator fails before training. A bare sklearn
classifier without a `"pn"` declaration cannot directly serve as a supervised oracle.
Versioned pilot supervised models and the ordinary script's `OracleMLP` explicitly declare
`"pn"`; runtime data alone cannot verify the truth of that declaration.

Each test-protocol result from `ExperimentRunner` contains `accuracy`, `auc`, and
`auc_unavailable_reason`. With a single true test class, `auc` is `NaN` and the reason is text;
when AUC is computable, the reason is `null`. Errors in model `decision_function` or metric
implementations are not converted to `NaN`.

Candidate-training exceptions, nonfinite epoch metrics, or nonfinite validation scores trigger
one automatic retry with a fresh clone and the same seed. A candidate succeeding on retry is
`recovered`; one failing again is `excluded` from PA/OA selection. Manifest `candidate_runs`
maps candidates to usable trajectories, and `failures` records exception types/messages.

Current versioned Survey execution uses script `--protocol survey-v3.2 --dataset ...`, resolving
to the shipped `survey_protocol_v3_2.json` matrix. Retired version names are rejected by the
current resolver; historical results retain the protocol identities in their own manifests.
`config["survey_protocol"]` specifies the matrix path and execution unit. The runner reloads
and validates the actual model, budget parameters, generator/selectors, and image representation.
Locked backbone/budget/training_path fields cannot be overridden at runtime.

Recipe binding is explicit opt-in: versioned entry points accept
`--recipe-registry <locked.json>`. The runner receives loaded `config["recipe_registry"]` and
optional `expected_recipe_registry_sha256`. Before execution it requires a locked registry/profile
and checks protocol, scope, candidates, and constructor parameters; drafts are rejected.
Binding identities enter the manifest root, candidate_runs, and every selection.
Single-run CLI output gains a `recipe_<registry digest>/` level. Pilot subprocess digests are
pinned, and resumption requires complete expected identities, so legacy results cannot be reused.
Binding v1 supports only a single outer candidate, including PUSB internal CV. Omitting the
registry retains legacy behavior and does not backfill historical manifests.
Batch auditing/summary uses the external `survey-batch-roots-2` allowlist: every batch declares
`recipe_policy`; bound batches also pin the registry path and canonical digest. Deleting all
binding fields from a manifest cannot downgrade it to legacy.
`recipe_audit` is a separately versioned extension, leaving historical A01–A17 unchanged.
Failure produces no numeric rows; success still runs the original gates.
A report does not mix registries or bound/legacy namespaces; methods within one registry may
use different profiles. JSON rows/costs and extended CSV retain recipe identities. The standalone
legacy aggregation CLI directs bound results to the v2 audit/summary entry point.

Explicit `recipe_binding_schema_version="survey-recipe-binding-2"` or CLI/pilot
`--recipe-binding-schema-version survey-recipe-binding-2` records candidate-pool identities.
The run root records the complete pool; candidate/PA/OA records retain their actual identities
and validate trajectory indices after failed-candidate exclusion. Output uses the independent
`recipe_pool_<SHA256>/` directory. The external batch allowlist must declare the same schema.
Pool format still requires a locked registry, forbids parameter-level candidate overrides, and
does not change the frozen Pilot's single-candidate budget. JSON/CSV retain candidate identities
per seed and selection criterion, rather than attributing an assumed common selected candidate
to the whole pool or row.

Additional provenance includes `protocol_version`, `protocol_sha256`, `execution_unit`,
`backbone`, `budget`, `representation` (including adapter manifest), `comparability_group`,
`protocol_deviation`, and `formal_eligible`/`formal_blockers`. If the protocol declares
`review_items`, the run also records `protocol_pending_review_items`: IDs still awaiting
collaborator review, without changing formal-eligibility determination.
DIY runs without protocol configuration remain supported and are marked `technical_smoke`.
`survey_protocol.validate_comparable_manifests` defaults to rejecting non-formal results and
inconsistent paths, budgets, representations, or labels. Constructor/candidate overrides of locked
fields fail before training; bound-run protocol preflight failures also write refusal manifests.

Through the built-in runner, nnPU/Dist-PU/Self-PU/PULDA/PAN/Holistic-PU/GEN-PU and Torch MLP/CNN
oracle models supporting `epoch_callback(epoch, self)` use per-epoch weight snapshots by default.
`RunTrajectory.checkpoints` records epochs/components; `SelectionArtifact.checkpoint_index`
identifies selected weights. `RunResult.selected_models` returns independently restored
inference models. With `manifest_path`, weights default to the adjacent `checkpoints` directory;
set `config['checkpoint_dir']` to choose a root or `config['capture_epoch_checkpoints']=False`
to retain the legacy single-point path, which still cannot satisfy formal checkpoint acceptance.

`checkpoints.load_selected_checkpoint(selection, template, device=...)` verifies weight digests,
loads in weights-only mode, and restores the validation-side threshold. Callers must explicitly
supply a template with the same architecture. It does not restore optimizer/RNG state for
continued training. Without persistence, the manifest cannot provide reproducible checkpoint
file paths.
Estimators declaring `checkpoint_prediction_batch_size` record it as `prediction_batch_size`
in checkpoint references. Restoration retains it to avoid float32 rounding changes caused by
different CNN operation shapes. PULDA uses `unlabeled_batch_size`; PAN, GradPU, Robust-PU,
Split-PU, Holistic-PU, and GenPU use `batch_size`. Older references lacking this field retain
the historical default 256; frozen snapshots are not reinterpreted.

The versioned runner records `adaptation_level="benchmark-adapted"`; formal eligibility is
determined separately from current protocol checks and recorded blockers. Built-in PA uses proxy accuracy as described
above. Passing these engineering checks does not establish numerical reproduction of paper results.

If every candidate fails, the runner writes a manifest when a path is configured, then raises
`RuntimeError`.

Required manifest `resources` keeps three cost definitions separate:
`single_configuration_costs` lists every attempt, successful attempt, and total candidate time;
`tuning` is total time for all candidates, including failed retries, within the runner seed
(sum it across seed manifests); `peak_gpu_memory_bytes` is maximum CUDA allocated memory over
the whole process.

PU-label generation time is separate. Shared preprocessing not performed by the runner is
explicitly outside its scope. `environment` records Python, NumPy, scikit-learn, PyTorch,
CUDA/cuDNN, GPU, and driver information. `aggregate_resource_usage(manifests)` combines
multi-seed artifacts into the protocol's complete tuning cost.

### Training views (`pu_toolbox.core.training_views`)

Role construction lives in the neutral core because estimators import it and cannot depend
on the Survey experiment layer. Dependency direction is `estimators → core`; the reverse is
forbidden and checked by subprocess import-boundary tests in
`tests/unit/core/test_training_view_contract.py`. Core knows neither method ledgers/names nor
manifests. The experiment layer's ledger gates and `resolve_training_view` decide whether a
method may use calibration, then pass the resolved choice to core.

| Symbol | Purpose |
|---|---|
| `TrainingView` | Three roles (`positive` / `native_unlabeled` / `loss_unlabeled`) and source indices for one training partition; every role is nonempty. Owned arrays (three position arrays and `source_indices`) are read-only; large feature/label arrays are borrowed. A view is valid only during the training step requesting it and cannot be reused after reordering, in-place normalization, or relabeling |
| `build_training_view` | Pure constructor receiving an already resolved `requested_view` (`"os"`/`"ts"`) and `role`. TS loss-unlabeled is `D_U ∪ D_P`, native U first and positives second; OS uses `D_U`. P and U must both be nonempty for every `role`. Non-train roles exist in the signature to permit explicit rejection of train-only calibration, not to model validation/test partitions; calibration is restricted to train |

### Result reading, statistics, and export (`results` / `reporting`)

Result APIs read caller-supplied manifest files or result directories directly. They do not
depend on experiment-stage documents, approval text, batch names, queues, historical execution
configurations, or comparison matrices. Inputs are read-only; generated reports go to a separate
output directory.

```python
from pu_toolbox.experiment.results import build_summary
from pu_toolbox.experiment.reporting import export, write_summary

report = build_summary(
    ["results/current"],
    expected_seeds=(0, 1, 2, 3, 4),
    excluded_paths=(),
    pending_paths=(),
)
write_summary(report, "reports/current")
export(report, "reports/current")
```

| Symbol | Purpose |
|---|---|
| `results.build_summary(manifest_paths, *, expected_seeds=(0, 1, 2, 3, 4), excluded_paths=(), pending_paths=())` | Read an explicit scope, validate run/selection identities, and summarize comparable configurations over the specified seeds; explicit pending/excluded inputs retain source status and are not observations |
| `reporting.write_summary(report, out_dir)` | Write `summary.json/md/csv`, `coverage.csv`, and `refusals.json` |
| `reporting.export(report, out_dir, *, literature=(), native_view=None)` | Write `survey_results.xlsx` with four result-sheet groups, diagnostic cards, and `conclusions.md/json`; requires the `report` extra; literature/method facts are optional inputs |

New runs record `manifest['research_settings']`, schema `1.0`, after validating generation
and training contracts. Saved manifests also retain it when all candidates fail training.

| Setting field | Recorded content |
|---|---|
| `sampling` | Generator-declared `input_sampling` and `split_ref` |
| `labeling` | Complete train/pu_val generation metadata, including requested/realized counts, mechanism parameters, and score sources |
| `risk` | Requested `os_or_ts`, estimator default, and corresponding loss pool; algorithm-specific pools use `estimator_defined` |
| `prior` | Base prior input/target declaration, effective candidate inputs and actual training provenance, and selection's population prior |
| `supervision` | Training label view, Trainer, actual validation role for every selector, and extra supervision |
| `representation` | Input dimensionality, recorded training path/representation identity, and adapter/image metadata |
| `summary` | Readable string combining these settings |

`prior.training.value` takes the runner-supplied argument first, then estimator `class_prior`.
`declared_target` retains the estimator declaration. For `native_unlabeled` under a `ts` risk
view, `target` is `risk_pool`. Candidate overrides are in `prior.candidate_parameters`, effective
inputs in `prior.candidates`, and existing actual-training provenance in
`prior.candidate_training`. That provenance and summary expose specialized Trainer conversions
and CVIR risk-pool priors. Custom generators declare sampling design/policy through returned
metadata; missing fields display as `unspecified`.

Summary rows retain original per-seed settings as `research_settings_by_seed` and deduplicated
text as `research_settings_summaries`. CSV, Markdown, workbooks, and diagnostic cards display
that text. Historical results lacking the field are identified as historical records read
according to their original manifest settings; setting values remain `null`. Result identity,
protocol version, seed statistics, and costs retain their original meanings.

Grouping preserves dataset, method/profile, training path, actual view, mechanism, c, selector,
and recipe identity. PN oracle is c-independent; `PA+clean-support` is separate from purely PU PA.
PA/OA for the same experiment share a `cost_id`, counting fit cost once. Actual tree, fold,
multi-stage, and epoch work units are retained.

For each configuration, compute mean and sample standard deviation (n−1) over explicit expected
seeds. Missing seeds/metrics, failures, and numeric risks remain visible; no standard deviation
is invented for a single seed. Corrupted evidence is rejected, including an unreconstructible
recipe pool, unknown candidates, or disagreement between selection records and checkpoint
digests. Review status follows the original record. Expected scope covers rows in the current
inputs.

Underlying `survey_summary` retains statistical tools including `aggregate_metric`, `sample_std`,
`summarize_costs`, and selector/c-token normalization. `SummaryError` means result structure,
identity, or statistical inputs violate the contract. Diagnostics and conclusions distinguish
literature facts, experimental observations, and inference. Missing evidence retains its reason;
interpretation does not replace observation. For reproduction commands and output structure,
see the [runbook](../../research/pu_survey/runbook.md) (currently Chinese).

## Errors and exceptions

The core toolbox exceptions below inherit from `PULearningError`
(`pu_toolbox.core.exceptions`) and can be caught together with `except PULearningError`.
Other APIs can also raise standard Python errors, including `ValueError` subclasses such as
`SummaryError`. Code defines authoritative exception-message text; this table maps scenarios
to the core exception types.

| Exception | Defined in | Example trigger |
|---|---|---|
| `PULearningError` | `core/exceptions.py` | Base class for core toolbox exceptions; use for catching rather than as a one-off error type |
| `ValidationError` | `core/exceptions.py` | Invalid `y` encoding, no positives, fewer positives than `MIN_POSITIVE_SAMPLES` or CV folds, or invalid `sample_weight` shape/values |
| `NotFittedError` | `core/exceptions.py` | `predict`/`decision_function`/`score_samples` before `fit`; compatible with sklearn |
| `RegistryError` | `core/exceptions.py` | Unknown registered name/alias, duplicate registration, or invalid parameters such as inconsistent prior semantics |
| `PipelineError` | `core/exceptions.py` | Workflow composition errors: invalid classifier/prior name, method unavailable for automatic construction, unresolved required prior, `sample_weight` declared `ignored`/`not_implemented`, or incompatible CNN method |
| `RunCancelledError` | `core/exceptions.py` | Cooperative cancellation detected at the next safe boundary |

These exceptions are defined centrally in `core/exceptions.py`. `PipelineError` and
`RunCancelledError` retain their public re-export paths through `pu_toolbox.workflows` and
`pu_toolbox.progress`; this does not change behavior or their `PULearningError` base.

API error/issue-code navigation:

| API | Error scenarios | Issue codes |
|---|---|---|
| `PUPipeline` | [Error scenarios](#error-cases) | — |
| `profile_pu_data` | Input rejection, e.g. truth missing labeled positives | [Issue codes](#issue-codes): `no_labeled_positives` / `sar_signal` / … |
| `build_diagnostic_report` | An unfitted estimator raises `ValueError`; `estimator` and `y_pred`/`scores` are mutually exclusive | Model-output issue codes above: `constant_predictions` / … |
| `PUTuner` / shift workflows | No usable selection metric or a failed coverage gate raises `PipelineError` | — |

For environment/installation matters, version matrices, CI responsibilities, and Python support,
see [compatibility](../../dev/compatibility.md).
