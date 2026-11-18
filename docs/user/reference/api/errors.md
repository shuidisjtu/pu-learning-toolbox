# Errors and exceptions

[API reference index](index.md)

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
| `PUPipeline` | [Error scenarios](tools.md#error-cases) | — |
| `profile_pu_data` | Input rejection, e.g. truth missing labeled positives | [Issue codes](tools.md#issue-codes): `no_labeled_positives` / `sar_signal` / … |
| `build_diagnostic_report` | An unfitted estimator raises `ValueError`; `estimator` and `y_pred`/`scores` are mutually exclusive | [Model-output issue codes](tools.md#report-issue-codes-model-output-checks): `constant_predictions` / … |
| `PUTuner` / shift workflows | No usable selection metric or a failed coverage gate raises `PipelineError` | — |

For environment/installation matters, version matrices, CI responsibilities, and Python support,
see [compatibility](../../../dev/compatibility.md).
