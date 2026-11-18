# Classifiers and prior estimators

[API reference index](index.md)

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
to population π, or formal numerical qualification. The [BBE method card](../../../research/method_cards/BBE.md)
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
must be positive integers, excluding booleans. The [P3Mix method card](../../../research/method_cards/P3MIX.md)
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
multi-seed GPU runs, and owner qualification remain pending. The [TEDn method card](../../../research/method_cards/TEDn.md)
provides optional Chinese research background.

Classifiers follow the shared `fit(X, y)` + `predict(X)` + `decision_function(X)` +
`get_params()`/`set_params()` contract; class-prior estimators implement `fit` + `estimate()`.

Registered classifiers must declare `label_semantics`: for `"pu"`, 0 is unlabeled; for `"pn"`,
0 is a true negative; `"pnu"` accepts three main-input roles, positive/negative/unlabeled.
The experiment runner and PU workflows conservatively treat undeclared third-party estimators
as `"pu"`; supervised oracles must explicitly declare `"pn"`. The runner checks the generated
view. Check each estimator's declared main-input label contract before fitting.

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
- Optional Chinese research background: [class_prior_estimation method card](../../../research/method_cards/class_prior_estimation.md).
  Example: [05_recpe_pipeline.py](../../../../examples/minimal/05_recpe_pipeline.py).

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

- Optional Chinese research background: [ReCPE method card](../../../research/method_cards/ReCPE.md).
  Example: [05_recpe_pipeline.py](../../../../examples/minimal/05_recpe_pipeline.py).

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

- Construct with `variant="km1"` or `variant="km2"`.
- `estimate_target=estimation_pool="native_unlabeled"`; returns the U mixture proportion.
- Optional Chinese research background: [Kernel_Mean method card](../../../research/method_cards/Kernel_Mean_Class_Prior.md).

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
ElkanNotoClassifier(base_estimator=None, calibration_method="sigmoid", internal_cv_folds=3,
                    eps=1e-12, mode="weighted_retraining", random_state=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_estimator` | sklearn estimator | `None` (`LogisticRegression()`) | Base binary classifier; must implement `predict_proba` |
| `calibration_method` | `"sigmoid"` / `"isotonic"` | `"sigmoid"` | Calibration for base estimators other than LogisticRegression; LogisticRegression is used directly |
| `internal_cv_folds` | `int` | `3` | Stratified folds for estimating `c`; must be at least 2. Requires at least this many labeled positives and this many unlabeled samples |
| `eps` | `float` | `1e-12` | Clipping threshold against division by zero in weights |
| `mode` | `"probability_correction"` / `"weighted_retraining"` | `"weighted_retraining"` | Probability correction (`g(x)/c`) or weighted retraining; the toolbox defaults to weighted retraining (ADR-0016) |
| `random_state` | `int \| None` | `None` | K-fold split seed |

To estimate `c`, stratify on the observed PU labels, holding out both labeled positives and
unlabeled samples in each fold. Average the predictions on the held-out labeled positives
only. For base estimators that require calibration, the calibration fold count is capped by
`internal_cv_folds`, 5, and the smaller label count in the calibration input. Calibration
requires at least two samples of each label in that input, including within each propensity
training fold.

- Optional Chinese research background: [Elkan_Noto method card](../../../research/method_cards/Elkan_Noto.md).
  Example: [01_elkan_noto.py](../../../../examples/minimal/01_elkan_noto.py).

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

- Optional Chinese research background: [Convex uPU method card](../../../research/method_cards/Convex_Formulation_for_PU_DATA_Learning.md).
  Example: [02_upu.py](../../../../examples/minimal/02_upu.py).

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

- Optional Chinese research background: [nnPU method card](../../../research/method_cards/nnpu.md).
  Example: [03_nnpu.py](../../../../examples/minimal/03_nnpu.py).

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

- Optional Chinese research background: [PNU method card](../../../research/method_cards/PNU.md).
  Example: [04_pnu.py](../../../../examples/minimal/04_pnu.py).

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

- Optional Chinese research background: [PUET method card](../../../research/method_cards/PUET.md).

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

- Optional Chinese research background: [LDCE method card](../../../research/method_cards/LDCE.md).

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

- Optional Chinese research background: [KLDCE method card](../../../research/method_cards/KLDCE.md).

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
- Optional Chinese research background: [LLSVM method card](../../../research/method_cards/LLSVM.md).

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
  The [Dist-PU method card](../../../research/method_cards/Dist-PU.md) provides optional Chinese
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
The [VPU method card](../../../research/method_cards/VPU.md) provides optional Chinese research background.

`cpu_stream` keeps full training/validation pools on CPU, sending at most `batch_size` rows to
the device at once; normalization scans, diagnostic validation, and prediction are batched too.
Validation risk combines the **global** log-sum-exp and positive mean in float64 on CPU, rather
than averaging per-batch risks; last-digit floating-point differences are possible. Original
resident reductions remain unchanged. `data_device_policy_` / `training_data_device_` record
the actual policy and data residency. Per-epoch snapshots include encoder, head, normalization,
and inference batch size; optimizer resumption is unsupported.
Compatible classifiers can accept a shared encoder. Optional
Chinese research background: [VPU method card](../../../research/method_cards/VPU.md).

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
SGD, and warm-start timing. The [source review](../../../research/method_cards/CVIR.md) provides
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
sigmoid scores, not independently calibrated probabilities. The [PULDA method card](../../../research/method_cards/PULDA.md)
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
preregistration. The [method card](../../../research/method_cards/Holistic-PU.md) provides optional
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
calibration. The [GEN-PU method card](../../../research/method_cards/GEN-PU.md) provides optional
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
Direct calls require isolated
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

- Optional Chinese research background: [PUSB method card](../../../research/method_cards/PUSB.md).

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
- Optional Chinese research background: [PUSB method card §7.3](../../../research/method_cards/PUSB.md).

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

`LBEClassifier` fits two logistic branches by EM. `fit(class_prior=...)` sets the initial
positive posterior for unlabeled rows. After fitting:

- `predict_proba(X)` returns the fitted class probabilities in negative, positive order.
  Its positive column is the class branch's estimate of `P(Y=1 | X)`.
- `predict_label_proba(X)` returns one fitted observed-label probability per row:
  `P(S=1 | X) = P(Y=1 | X) × P(S=1 | Y=1, X)`. The second factor is the labeling
  propensity among true positives.
- `n_iter_` and `converged_` record EM iterations and posterior stopping.
  `observed_log_likelihood_history_` tracks the sum of observed log likelihoods at each iteration.

The observed-label likelihood depends on the product of the two branches and is unchanged
when the branches are exchanged. Interpreting their fitted outputs separately as the true
class posterior and labeling propensity requires an identifying assumption or additional
supervision. See the [LBE method card](../../../research/method_cards/LBE.md) for source and
implementation details.

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
batching, and CNN support are adaptations. The [LBE method card](../../../research/method_cards/LBE.md)
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
immediately. Non-None `sample_weight` raises an error. The [LaGAM method card](../../../research/method_cards/LaGAM.md)
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
adaptations; the [source review](../../../research/method_cards/LaGAM.md) is optional Chinese background.

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
The [Split-PU method card](../../../research/method_cards/Split-PU.md) provides optional Chinese background.

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
the formal Survey matrix. The [Robust-PU method card](../../../research/method_cards/Robust-PU.md)
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

- Optional Chinese research background: [GradPU method card](../../../research/method_cards/GradPU.md).

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
- Optional Chinese background: [Self-PU method card](../../../research/method_cards/Self-PU.md). Example: [10_self_pu.py](../../../../examples/minimal/10_self_pu.py); it applies the training-partition OS prior conversion explained in the [guide](../../howto/self_pu.md).

#### `InfoMaxPUClassifier` (registered as `infomax_pu`)

Deep PU classifier: PURL representation learning → prior estimation → nnPU classification.
Configure its constructor parameters directly.

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
- Optional Chinese research background: [InfoMax-PU method card](../../../research/method_cards/InfoMax-PU.md).

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

- Optional Chinese research background: [WConPU method card](../../../research/method_cards/WConPU.md).

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

- Optional Chinese research background: [DGPU method card](../../../research/method_cards/DGPU.md).
