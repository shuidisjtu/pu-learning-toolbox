# End-to-End Training and Evaluation with PUPipeline

> Prerequisite: complete the [quickstart](../quickstart.md).
> Concepts: [the PU problem and the roles of π](../concepts/pu_problem.md),
> [SCAR/SAR](../concepts/scar_sar.md), and [prior targets](../concepts/class_prior_semantics.md).

`PUPipeline` combines data validation, prior resolution, profiling, explicit classifier configuration,
training, PU-stratified cross-validation, and evaluation in one workflow. Supply `(X, y_pu)` and
an explicit method name or configured instance, to obtain a report recording its inputs and decisions.

## Quick example

Use the `X, y_pu` arrays loaded in the quickstart and its existing `results/` directory:

```python
from pu_toolbox import PUPipeline

pipe = PUPipeline(classifier="upu")  # Select a classifier explicitly.
report = pipe.fit_evaluate(X, y_pu)  # X: (n, d); y_pu: {1, 0} PU labels.

print(report.summary())
report.to_markdown()
report.to_json()  # Strict JSON without NaN literals.
report.save("results/pipeline.json")
```

`report.final_model` is refitted on all supplied samples and supports `predict` /
`decision_function` directly.

## The five workflow stages

1. **Validation:** `validate_pu_X_y` normalizes labels and checks shapes, minimum positive
   counts, and whether enough labeled positives exist for the CV fold count.
2. **Prior resolution:** resolve a supplied value or estimate once on the full supplied PU data
   when needed. This step **never uses `y_true`**.
3. **Profiling:** call `profile_pu_data` once for labeling-mechanism evidence and data-quality
   checks; choose the classifier separately.
4. **Cross-validation:** the default is `PUStratifiedKFold(5, shuffle=True, random_state=...)`.
   Each fold fits an independent estimator, computes metrics, and contributes to their mean
   and standard deviation.
5. **Final model and diagnostics:** refit on all supplied data, then call
   `build_diagnostic_report`. Metrics and profile issues enter `PipelineReport`.

The full-data prior estimate is shared across CV folds; the workflow converts it separately for
training consumption in each fold. Use an independent test split for final assessment after model
selection. The [four-role experiment guide](custom_experiments.md) covers explicitly isolated
selection and test roles.

## Class-prior resolution

Identify the distribution represented by the value first; see
[prior semantics](../concepts/class_prior_semantics.md). `input_sampling="os"` (default) means
one-sided labeling in one source pool. `"ts"` means independently acquired positives and
population-mixture U. A call-level `class_prior` targets the population. Native-U estimates
from KM/penL1/ReCPE are resolved to population values according to the input design, then
converted at training boundaries to the classifier's target.

For example, with OS labeled fraction 0.2 and estimated U prior 0.25, `report.prior.value` is
0.4; the default uPU risk consumes a U prior of 0.25 for that complete training partition.

`report.prior` records `input_value/input_target`, `estimation_pool`, `input_sampling`,
`conversion`, and `training_consumption`. Each fold uses its own P fraction for conversion,
and final refit consumption is recorded separately.
[Example 16](../../../examples/minimal/16_prior_targets.py) demonstrates known/estimated
priors for OS and TS.

Resolution follows explicit call input → classifier constructor input → automatic estimation →
no prior when unnecessary. Constructor values follow the classifier's `class_prior_target`;
see the [API reference](../reference/api/tools.md#class-prior-resolution-precedence) for exact priority rules.

- `prior_estimator` accepts `"pen_l1"` (default), `"recpe"`, `"km1"`, `"km2"`, or a configured
  estimator instance. KM names select `KernelMeanPriorEstimator` variants.
- With an explicit classifier, disabling estimation through `prior_estimator=None` while omitting
  a required prior raises `PipelineError`. Its message gives alternatives and states that
  `y_true` is never used for estimation.
- OS profiling audits population-prior consistency with the labeled fraction. Converting a
  population prior to native U requires it to be at least the labeled fraction of that training
  fold. TS profiling records the extra-P concatenation fraction separately.

## Classifier selection

Specify a registered classifier name, for example `classifier="upu"`, or a configured
estimator instance. Names and aliases are resolved case-insensitively; use
`classifier_params` for named constructors. Instances are cloned per fold. Supply required
method-specific arguments and any required extra supervision explicitly. Discover capabilities
with `pu-toolbox list-methods`; use explicit candidates with `PUTuner` or `PUModelComparator`
for empirical selection. Omitted classifiers and `classifier="auto"` are retired;
the decision is recorded in [ADR 0023](../../adr/0023-explicit-method-selection.md).

### Deep methods and architecture selection

Install `pip install "pu-toolbox[torch]"` for torch methods. CNN support is determined by the
registered capabilities and the constructor's `encoder` parameter, rather than a fixed method
list. Examples include nnPU, WConPU, and InfoMax PU; additional methods may declare support.
Self-PU can be instantiated by name but has no native CNN encoder-injection path.

| Parameter | Default | Meaning |
|---|---|---|
| `architecture` | `"mlp"` | `"mlp"` for tabular input or `"cnn"` for 4-D NCHW images |
| `backbone` | `"cnn13"` | `"cnn13"`, `"cnn13_no_bn"`, `"resnet18"`, or `"resnet50"`; used for CNN input |
| `device` | `None` (auto) | Torch device for classifiers instantiated by the workflow; `None` / `"auto"` resolves to CUDA when available, otherwise CPU |

- CNN requires an explicit deep classifier with an `encoder` constructor parameter and matching
  capability declarations. The pipeline builds and injects a CNN through `build_encoder`.
  Unsupported CNN paths, including Self-PU's, raise `PipelineError` during construction.
- CNN input must be 4-D NCHW; 2-D tabular input with `"cnn"` and 4-D input with `"mlp"` are rejected.
  CLI image input uses `.npy` arrays.
- Deep training can be costly: WConPU defaults to 100 epochs. `max_epochs` is forwarded when a
  named classifier's constructor accepts it; other epoch parameter names are not rewritten.
  User-supplied instances retain their own constructor configuration. With CV, expect one
  training run per fold plus final refit. Reduce the fold count (minimum 2) or the appropriate
  method-specific epoch budget for a small smoke run.

See [custom encoders](custom_encoders.md) for capability checks, injection, and persistence.

## Metrics and availability

The default is
`DEFAULT_METRICS = ("pu_zero_one_risk", "pu_recall", "pu_estimated_precision", "pu_auc_roc")`.
Other metrics include `average_precision`, `balanced_accuracy`, `brier_score`, and
`expected_calibration_error`. Their dependencies and evidence levels (`pu_observed`,
`class_prior_dependent`, `supervised_oracle`, `probability_calibration`) are defined in the
[API reference](../reference/api/tools.md#metrics).

Population risk and estimated precision use the complete validation table for OS and native U
for TS. TS oracle metrics likewise use population-representative U rows. `pu_recall` and
`pu_negative_rate` describe prediction behavior on observed P and U, respectively.

Missing truth, continuous scores, probabilities, or priors makes affected metrics unavailable
rather than aborting the workflow. `CVMetric.available=False` indicates no usable result;
`reason` explains the cause when all folds are skipped. A fold-specific error, such as single-class
AUC, skips only that fold. Means and standard deviations aggregate the computed folds.

## Errors

Unresolvable classifier/prior names and missing required constructor arguments fail early with
`PipelineError`. Input validation rejects data such as no positive samples or fewer positives
than CV folds with `ValidationError`. Invalid parameter combinations can also raise `ValueError`
or `TypeError`; consult the [API reference](../reference/api/tools.md#error-cases) for exact exception and fallback
contracts.

## Compared with a manual workflow

The [manual example](../../../examples/minimal/05_recpe_pipeline.py) assembles profiling, prior
estimation, CV, and evaluation explicitly. `PUPipeline` combines them and records classifier
resolution, prior provenance, skipped candidates, seeds, and architecture/backbone/device/encoder
information. To search several parameter combinations, use `PUTuner`; see [model tuning](model_tuning.md).

## Next steps

- [Diagnostic reports](diagnostic_reports.md).
- [Class-prior and propensity sensitivity](sensitivity_analysis.md).
- [API reference](../reference/api/tools.md#pupipeline).
