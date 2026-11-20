# Tools and workflows

[API reference index](index.md)

## Distribution-shift API

Usage and interpretation: [distribution-shift guide](../../howto/distribution_shift.md).
Examples: [11_distribution_shift.py](../../../../examples/minimal/11_distribution_shift.py),
[12_shift_decision_tools.py](../../../../examples/minimal/12_shift_decision_tools.py), and
[13_dynamic_joint_shift.py](../../../../examples/minimal/13_dynamic_joint_shift.py).

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
    input_sampling="os",
    source_class_prior=None, target_class_prior=None,
    prior_estimator="pen_l1",
    prior_shift_threshold=0.05, propensity_shift_threshold=0.05,
    sensitivity_radius=0.05, bootstrap_replicates=0,
    confidence=0.95, random_state=42,
)
```

Accepts OS source/target features and PU labels with optional population priors. Missing
priors are estimated independently per domain; native-U estimates are converted using the
domain's observed label rate. Passing `input_sampling="ts"` raises an error because the
P/U mixing ratio is set by sampling. Returns `DomainAssumptionReport`, with conclusion
`stable`, `class_prior_shift`, `labeling_mechanism_shift`, `both_shift`, or `inconclusive`.
Mean labeling propensity averages over positives within a domain.

With `bootstrap_replicates>=2`, each domain receives row-level nonparametric resampling and
prior re-estimation per replicate using its own observed label rate for any conversion.
Percentile intervals propagate to priors, labeling rates, mean propensities, and the three
domain differences. `uncertainty` records requested,
successful, and failed replicates; this reflects sampling/estimator variation, not bias from
identifiability assumptions.

### `analyze_pu_uncertainty`

Builds a selective-prediction and active human-review plan for a fitted model.

```python
report = analyze_pu_uncertainty(
    estimator,                        # Fitted model; predict(X) supplies binary classes.
    X,
    y_pu=None,                        # Supplied labeled positives are excluded from queries.
    y_true=None,                      # Enables supervised truth metrics when supplied.
    min_confidence=0.5,               # Selective-prediction confidence threshold.
    query_budget=0,                   # Maximum query count; 0 requests no queries.
    query_strategy="uncertainty",     # "uncertainty" / "shift_weighted" / "diverse_uncertainty".
    importance_weight=None,           # Required for shift_weighted; aligned with X rows.
    random_state=42,
    decision_threshold=0.0,           # Reference for decision_function scores only.
)
```

Review keeps the class from `estimator.predict(X)` for every accepted row. It requires one
0/1 prediction per row. Scores control abstention and review priority, never the class:

- If `predict_proba(X)` is available, use the class-1 column of its finite two-column output.
  Use `classes_` when exposed; otherwise columns must be ordered `[0, 1]`. Clip the chosen
  column to `[0, 1]` so probability-like outputs such as Elkan–Noto's can still be used.
- Otherwise, require one finite `decision_function(X)` score per row, larger meaning more
  positive. Apply `sigmoid(score - decision_threshold)` regardless of the batch range. For a
  nonzero decision boundary, pass its value explicitly (`review --decision-threshold` in the
  CLI). The sigmoid rescales the margin; it does not calibrate it as a probability.

For either source, confidence is `2 * abs(scaled_score - 0.5)` and uncertainty is
`1 - confidence`. On the probability path this measures distance from 0.5, not distance from
a model's custom classification threshold. Confidence is not the probability of a correct
prediction.

Returns `PUUncertaintyReport`, a dataclass publicly exported from `pu_toolbox`:

| Field | Contents |
|---|---|
| `positive_probability` | Row-aligned scores scaled to [0, 1] as described above |
| `uncertainty` | Row-aligned score-margin uncertainty |
| `selective_predictions` | Model predictions (0/1), with -1 for confidence below `min_confidence` |
| `query_indices` | Recommended review indices, ordered by query_strategy |
| `summary` | Sample/abstention counts, coverage, query candidate count, requested-query result count, and selective accuracy when truth is supplied |
| `provenance` | Prediction and score sources, thresholds, query strategy, and label-exclusion settings |

`importance_weight` must be finite, nonnegative, nonzero, and aligned with the rows being
reviewed; source-row audit weights cannot replace a target-row vector.
Score, uncertainty, and selective-prediction arrays are omitted from `to_json()`;
`query_indices`, summary, artifact names, and provenance remain. CSV preserves per-row
scores, uncertainty, selective predictions, and query flags.

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
registry. `get_pu_metadata()["guarantee"]` is fixed to
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
`clean_room_paper_objective`, not `official_exact`. It stays outside the stable registry.

`build_joint_shift_estimator(...)` builds `dynamic`, `trpu`, `tepu`, `fine_tune`, `mmd`,
`two_step`, and three correction ablations. Neural comparisons share feature/classifier-head
sizes and absolute-value-corrected PU risk.

## build_encoder

The shared encoder builder for deep classifiers is defined in
`pu_toolbox/estimators/deep/vision.py` and publicly exported as
`from pu_toolbox import build_encoder`. Pass its result to a compatible classifier.

```python
from pu_toolbox import build_encoder

encoder = build_encoder("cnn", backbone="cnn13", in_channels=3)
```

Set `in_channels` to the number of input channels: 3 for RGB in this example.
`in_channels`, `backbone`, `normalization_mean`, and `normalization_std` are keyword-only.

- `"mlp"` returns `None`, selecting the classifier's built-in MLP path for tabular data.
- `"cnn"` builds an image encoder for 4-D NCHW input. Available backbones are `cnn13`
  (default), `cnn13_no_bn`, `resnet18`, and `resnet50`. The normalization mean and
  standard deviation each default to 0.5 per channel.
- An invalid `architecture` raises `ValueError`.

## profile_pu_data

Usage: [data profiling](../../howto/data_profiling.md). Executable example:
[07_data_profiling.py](../../../../examples/minimal/07_data_profiling.py).

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

## Explicit method discovery and selection

Use `pu-toolbox list-methods` to inspect registered capabilities and select candidates explicitly.
Use independent validation or [ExperimentRunner](../../howto/custom_experiments.md) for controlled single-model parameter grids or
comparison with authorized validation evidence. Profile OS and TS data with the correct
`input_sampling` in [profile_pu_data](#profile_pu_data); a TS table's positive fraction is not
a population labeling rate. The advisor ranking APIs are retired; no replacement
ranking or compatibility layer is specified ([ADR 0023](../../../adr/0023-explicit-method-selection.md)).

## Data generation

Import these functions from `pu_toolbox.preprocessing`. Usage:
[SCAR/SAR simulation](../../howto/sar_simulation.md) and [OS/TS sampling](../../howto/pu_sampling.md).
Examples: [06](../../../../examples/minimal/06_sar_simulation.py) and
[15](../../../../examples/minimal/15_pu_sampling.py).

| Function | Signature and return value |
|---|---|
| `construct_pu_dataset` | `(source, labeling, *, positive_classes, negative_classes, excluded_classes=()) -> PUConstruction`: map and label one already-loaded `SourcePartition` without reshuffling or downloading |
| `simulate_pu_dataset` | `(source, labeling) -> PUConstruction`: generate a clean binary `GaussianSourceConfig` source and apply the same OS labeling |
| `make_sar_propensity` | `(X, y_true, *, mechanism="linear", label_frequency=0.5, strength=1.0, feature_weights=None) -> propensity`: per-sample labeling propensity `P(S=1|Y,X)`; true-negative entries are fixed at 0 |
| `make_gaussian_case_control_data` | `(n_p=50, n_u=100, *, class_prior=0.5, n_features=5, separation=2.0, random_state=None) -> (X, y_pu, y_true, metadata)`: independent TS Gaussian P and population-mixture U |
| `make_gaussian_pu_data` | `(n_p=50, n_u=100, n_features=5, separation=2.0, random_state=None) -> (X, y_pu, class_prior)`: all generated positives are visible and U comes from negatives; the third value is the true positive proportion of the concatenated table |

**Configuration-driven source acquisition and reading:**

| Function | Signature and return value |
|---|---|
| `read_dataset` | `(config, root, *, partition) -> SourcePartition`: read the named local partition; never download |
| `fetch_dataset` | `(config, root) -> dict`: explicitly acquire configured artifacts, verify checksums and write provenance |

`config` is a dictionary or a path to an external JSON configuration. `root` is the exact
dataset directory; the source ID is not appended. The configuration contains:

- `source_id` and `version`, used in position IDs such as `mnist@idx-v1:train:0`.
- `format` selects a shared decoder. `partitions` maps each declared source partition
  to the input options for that decoder. All configured paths are relative to `root`.

  | Format | Options and per-partition fields |
  |---|---|
  | `idx_images` | `options.image_shape`; each partition supplies `features`, `labels` and `max_rows`. Use raw or `.gz` filenames explicitly. |
  | `npz_arrays` | Each partition supplies `file`, `features_key` and `labels_key`. Optional `options.archive_keys` fixes the allowed keys; `image_shape` requests uint8 NCHW images. |
  | `delimited_numeric` / `delimited_categorical` | Each partition supplies `file`, ordered `feature_columns` and a separate `label_column` (indices or names with a header). Options: `delimiter` (default `","`), `has_header` (default `false`), `label_dtype` (`"str"`, `"int"`, `"float"`) and optional `column_count`. Categorical tables accept optional `feature_values` and return raw tokens, not one-hot values. Both read UTF-8, including BOM. |
  | `text_documents` | `options.classes` sets raw category order, with `encoding`, optional `decode_errors` and `suffix`. A partition specifies `directory` or `archive` plus `member_root`. Directories use filename order; TAR files preserve member order within each class. |
  | `cifar_batches` | `options.variant="cifar10"`; each partition specifies `train` as a boolean. The torchvision reader is explicitly offline and returns NCHW images. |
  | `archive_images` | `options.classes` and `suffixes`; the partition specifies `archives` and `expected_counts` by class. The reader returns the encoded image bytes in class and member-name order, without extracting archives. |

- Optional `options.label_values` constrains raw labels for IDX, NPZ and delimited inputs.
- Optional `artifacts`, keyed by relative filename, with `url` and `checksum` containing
  `algorithm` (`sha256` or `md5`) and `digest`. Explicit fetch verifies the digest, does not
  overwrite existing data files, and writes `provenance.json` with source version and checksums.

`pu_toolbox/preprocessing/source_configs` contains profiles for the eight prepared Survey
sources. IDX, text, ZIP and delimited reads do not need torchvision; the CIFAR profile does.
A local read preserves partition order and raw labels, but does not authenticate the files;
row-position IDs are not proof of identity across copies. Text, categorical tokens and encoded
images must be represented as numeric features before PU construction. Reading a partition
neither assigns an experiment role nor chooses positive or negative classes. See the
[mapping example](../../howto/pu_sampling.md#configure-acquisition-and-local-reading).
Survey uses the same reader, but still owns the frozen NPZ input route, role splits,
preprocessing and integer IDs.

`SourcePartition` takes features, original labels, a stable `source_id`, a named `partition`,
and optional `sample_ids`. `PULabelingConfig` takes `mechanism`, `sampling_policy` and an
explicit integer seed or `RandomState`. Use SCAR `fixed_count` with exactly one of `c` or
`n_labeled`; an explicit quota must fit the available positives. Use SCAR or SAR `bernoulli`
with `c`; SAR additionally takes `propensity_rule` (`linear` or `nonlinear`) and supports
`strength` and `feature_weights` for 2-D features. Bernoulli top-up is opt-in via
`ensure_labeled`. The returned `PUConstruction` separates `observations` (features, PU labels,
IDs), `audit` (true and original labels, optional pre-top-up propensity), and `manifest`
(source, mapping, pool sizes, random states, requested and realized budget). The empirical
priors are audit facts, not automatic fit parameters. OS constructors reject independent TS.

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

- Specify the mechanism and sampling policy explicitly; there is no implicit SAR default.
- For Bernoulli sampling, `c` is the target mean positive-class propensity. The realized
  labeling rate fluctuates. `ensure_labeled=False` by default; opting in tops up an empty draw.
- CLI `make-demo-data` defaults to separation 1.0.

## analyze_pu_sensitivity

Usage: [sensitivity analysis](../../howto/sensitivity_analysis.md). Executable example:
[09_sensitivity_analysis.py](../../../../examples/minimal/09_sensitivity_analysis.py).

This analysis is defined for **OS inputs only**. It interprets the observed labeled fraction as
`q = P(S=1) = π · c̄` and computes both precision and risk using the full OS pool. It does not
detect TS inputs. Do not pass a combined TS table: its fraction of separately sampled labeled
positives is set by the sampling design, not by the population labeling rate, so the implied
priors, propensities, and metrics do not have the stated interpretation.

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

Usage: [diagnostic reports](../../howto/diagnostic_reports.md). Executable example:
[08_diagnostic_report.py](../../../../examples/minimal/08_diagnostic_report.py). Three input modes:

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
see [custom experiments](../../howto/custom_experiments.md) and the executable
[14_diy_experiment.py](../../../../examples/minimal/14_diy_experiment.py).

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
`config["split_ref"]["class_prior"]["population"]`. For uPU with the built-in fit trainers
and OS input, the runner converts its supplied population prior to the native-U prior
using the realized labeled fraction in `train` when the estimator uses its `os` view.
The `ts` view uses the complete OS pool and receives the population prior unchanged.
Without a runner override, uPU's constructor or candidate prior is used as given;
a split-only population prior supplies PA, not an override to uPU fit. Custom trainers
still receive the original runner input and are responsible for their own prior routing.
The manifest records the fit input and PA prior separately in `research_settings.prior`.
Built-in OA selects on `clean_val` accuracy and has no selection prior. Use
`protocols=[ProtocolOA()]` for SAR. The runner refuses built-in PA when `pu_val`
is declared SAR or independently sampled TS; the PA formula is for OS under SCAR.

`config["os_or_ts"]` requests a training view, not an input sampling design. Allowed values
are `None`, `"os"`, and `"ts"` (case-sensitive); `None` leaves the estimator default in place.
If the generator declares independent TS (`input_sampling="ts"`) for `train`, the runner
rejects the `ts` reconstruction view rather than adding the extra positive sample to its
already marginal U pool.

Import extension interfaces `Generator`, `Trainer`, and `SelectionProtocol` from
`pu_toolbox.experiment.protocols`. Import trajectory/selection records `EpochRecord`,
`RunTrajectory`, `SelectionArtifact`, and `RunResult` from `pu_toolbox.experiment.tracking`.
An ordinary custom selector receives generated pu_val; one inheriting `ProtocolOA` receives
clean_val. The returned `SelectionArtifact.protocol` must match the strategy's `name`.

`RunResult` provides `selections`, `test_metrics`, `manifest`, `failures`, `resources`, and
`selected_models`. Continued prediction with a classical model must reuse the selected threshold
and validation-side score transform; the example's `predict_selected` implements that process.

### Method-specific training adapters

LaGAM and PULNS require an independent clean support partition. Use the method's
support adapter through `config["trainer"]`; the default trainer cannot supply support.
A custom support trainer must expose its clean `support` as a `DatasetPart` with
`for_selection=False`, independently sourced sample IDs disjoint from the four roles,
and pass its features and labels to the estimator. `clean_val` is not support.

The independent exploratory adapters do not modify the locked Survey contracts.
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
[CVIR method card](../../../research/method_cards/CVIR.md) for optional Chinese background on
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
[pu_survey_protocol.md](../../../research/pu_survey/pu_survey_protocol.md) §2.3–§2.4.
[experiment_layer.md](../../../dev/experiment_layer.md) §1 is optional
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
see the [runbook](../../../research/pu_survey/runbook.md).
