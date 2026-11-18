# Errors and exceptions

[API reference index](index.md)

`PULearningError` (`pu_toolbox.core.exceptions`) is the base for the custom exceptions in the
first table. `ValidationError` also inherits from `ValueError`, so either handler catches it.
A plain `ValueError`, or another `ValueError` subclass such as `SummaryError`, is not necessarily
a `PULearningError`. Other libraries and APIs may still raise their own exceptions.

| Exception | Defined in | Example trigger |
|---|---|---|
| `PULearningError` | `core/exceptions.py` | Base class for core toolbox exceptions; use for catching rather than as a one-off error type |
| `ValidationError` | `core/exceptions.py` | Invalid input or configuration, including invalid labels, CV folds, sample weights, or algorithm metadata |
| `NotFittedError` | `core/exceptions.py` | `predict`/`decision_function`/`score_samples` before `fit`; compatible with sklearn |
| `RegistryError` | `core/exceptions.py` | Unknown name or alias, conflicting registration names, or an invalid estimator class passed to `register_method` |
| `PipelineError` | `core/exceptions.py` | Workflow composition errors: invalid classifier/prior name, method unavailable for automatic construction, unresolved required prior, `sample_weight` declared `ignored`/`not_implemented`, or incompatible CNN method |
| `RunCancelledError` | `core/exceptions.py` | Cooperative cancellation detected at the next safe boundary |

These exceptions are defined centrally in `core/exceptions.py`. `PipelineError` and
`RunCancelledError` retain their public re-export paths through `pu_toolbox.workflows` and
`pu_toolbox.progress`; this does not change behavior or their `PULearningError` base.

## Exceptions by entry point

| Entry point | Scenario | Outcome |
|---|---|---|
| `PUPipeline.fit_evaluate` | Labeled positives are fewer than the requested `evaluation_cv` folds | `ValidationError` |
| `ElkanNotoClassifier.fit` | Labeled positives are fewer than `internal_cv_folds` | `ValidationError` |
| `PUPipeline.fit_evaluate` | Invalid `sample_weight` shape or values, e.g. negative weights | `ValidationError` |
| `ElkanNotoClassifier.fit` | Invalid `sample_weight` shape or values | `ValidationError` |
| `AlgorithmMetadata(...)` | Invalid metadata values, e.g. an unrecognized `class_prior_target` | `ValidationError` |
| `PUTuner.fit` / `PUModelComparator.fit` | No candidate produces a usable selection score | `PipelineError` |
| `ShiftAwarePUPipeline.fit_evaluate` | Adaptation requested and the coverage gate fails, without `allow_unstable` | `PipelineError` |
| `ShiftAwarePUPipeline.compare` | Coverage gate fails without `allow_unstable`, and baseline evaluation succeeds | Returns a comparison with `weighted=None` and `recommendation="collect_target_data"` |

## Error and issue-code navigation

| API | Error scenarios | Issue codes |
|---|---|---|
| `PUPipeline` | [Error scenarios](tools.md#error-cases) | — |
| `profile_pu_data` | Input rejection, e.g. truth missing labeled positives | [Issue codes](tools.md#issue-codes): `no_labeled_positives` / `sar_signal` / … |
| `build_diagnostic_report` | An unfitted estimator raises `ValueError`; `estimator` and `y_pred`/`scores` are mutually exclusive | [Model-output issue codes](tools.md#report-issue-codes-model-output-checks): `constant_predictions` / … |

For environment/installation matters, version matrices, CI responsibilities, and Python support,
see [compatibility](../../../dev/compatibility.md).
