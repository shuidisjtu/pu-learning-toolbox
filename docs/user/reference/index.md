# API reference index

Find public APIs by module. [api.md](api.md) defines signatures, parameters, and return
contracts; this page provides navigation. Docstrings are the behavioral source of truth
under ADR-0013.

## Classifiers and prior estimators

See [classifiers and estimators](api.md#classifiers-and-estimators) for the registry table,
method-specific contracts, and prior estimators. Discover the current registered names with
`pu-toolbox list-methods`; the reference also covers standalone research interfaces.

- **Class-prior estimation:** `class_prior_estimation`, `recpe`, and the `km1` / `km2`
  variants of `KernelMeanPriorEstimator`.
- **Classifiers:** classic, risk-estimation, bias-aware, and deep methods, with explicit
  input, prior, support-data, and architecture requirements.
- **Standalone mixture estimation:** `BBEMixtureEstimator` and `best_bin_estimate`.

## Pipelines and orchestration

- [`PUPipeline`](api.md#pupipeline): end-to-end PU analysis.
- [`PUTuner`](api.md#putuner): PU-aware parameter search.
- [`PUModelComparator`](api.md#pumodelcomparator): compare models.
- [Progress and cancellation](api.md#progress-and-cancellation): `ProgressUpdate` and
  `CancellationToken`.
- [`build_encoder`](api.md#build_encoder): shared image encoder construction.

## Distribution shift

See the [distribution-shift APIs](api.md#distribution-shift-api):

- `analyze_pu_shift`, `ShiftAwarePUPipeline`, and `PUShiftMonitor`.
- `analyze_domain_assumptions` and `analyze_pu_uncertainty`.
- Research-only joint-shift classifiers and their interpretation limits.

## Data profiling and recommendations

- [`profile_pu_data`](api.md#profile_pu_data): data-quality and assumption evidence.
- [`recommend_methods` / `recommend_from_profile`](api.md#recommend_methods--recommend_from_profile):
  method recommendations.

## Diagnostics and evaluation

- [`analyze_pu_sensitivity`](api.md#analyze_pu_sensitivity): class-prior and labeling-propensity
  sensitivity.
- [`build_diagnostic_report`](api.md#build_diagnostic_report): structured diagnostic reports.

## Research experiments

See the [experiment layer](api.md#experiment-layer-experiment) for `ExperimentRunner`,
`DatasetBundle` / `DatasetPart`, SCAR/SAR generators, PA/OA selection, trainers, support-data
adapters, checkpoints, and result/reporting interfaces. Start with the
[custom experiment guide](../howto/custom_experiments.md) for a complete workflow.

## Data generation

See [data generation](api.md#data-generation):

- `make_sar_propensity`, `make_sar_labels`, `make_sar_dataset`, and `make_scar_dataset`.
- `make_fixed_count_labels`, `make_pu_labels`, and `make_gaussian_case_control_data`
  for same-pool labeling and independent TS sampling.

## Errors and exceptions

See [errors and exceptions](api.md#errors-and-exceptions) for `PULearningError`,
`ValidationError`, `NotFittedError`, `RegistryError`, `PipelineError`, and
`RunCancelledError`.
