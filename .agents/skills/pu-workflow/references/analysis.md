# Profile, train, and diagnose existing PU data

Record input sampling, risk pool, prior target and source, permitted
supervision, and evaluation population. See [sampling and priors](sampling-and-priors.md)
and [representations](representations.md).

## Profile and select an estimator

For OS tabular data (headered feature CSV and one-column PU-label CSV):

```bash
uv run pu-toolbox profile --data X.csv --labels y_pu.csv --out-dir work/profile
uv run pu-toolbox list-methods
uv run pu-toolbox list-priors
```

Read `profile.json` for issues, sample counts, and selection-mechanism evidence.
Use `profile_pu_data(..., input_sampling="ts")` for independent TS. Choose the
estimator explicitly using its assumptions, capabilities and required supervision.

## Train on the permitted training data

Resolve the estimator with `registry.get_algorithm` or import its class.
For OS training data with population prior 0.4, default uPU consumes the
native-U prior computed using the labeled fraction of the training data:

```python
from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.prior import population_to_native_unlabeled

q = (y_train_pu == 1).mean()
pi_u = population_to_native_unlabeled(0.4, q, input_sampling="os")
model = UPUClassifier(risk_prior=pi_u, random_state=42).fit(X_train, y_train_pu)
```

For independent TS, native U already represents the population; do not add
extra P to U or apply an OS conversion. Fit any prior estimator on allowed data
and check its `estimate_target` and `estimation_pool`. Save the original
estimate, any conversion and the value consumed by the model.

Apply standalone diagnostic/metric APIs to the intended assessment data.
Label training-table metrics as such; do not use test truth for selection.
The caller owns candidate validation on a single PU table. For four clean
roles use the [research experiment](research-experiments.md) route.

## Fixed-output sensitivity and refitting

For an OS assumption sweep on a fitted model's fixed predictions:

```python
from pu_toolbox.diagnostics import analyze_pu_sensitivity

predictions = fitted_model.predict(X_os)
analysis = analyze_pu_sensitivity(
    y_pu_os, predictions, population_priors=[0.35, 0.4, 0.45],
)
analysis.save("work/fixed-output-sensitivity.json")
```

Here `fitted_model`, `X_os`, and `y_pu_os` are the saved model and aligned OS data
chosen for the audit. Read `analysis_type`, `observed_label_rate`, `points`, and
`metric_ranges`. The CLI `sensitivity` command fits its chosen prior-free classifier
(default `elkan_noto`) before auditing fixed outputs. Record that model identity.

For training sensitivity, refit the selected method for each allowed prior with
the same split and seed settings; retain a separate report for every run. Report
the prior grid, training-consumption values, selection settings, and metric changes.

## Report results and save artifacts

Read the requested profile, diagnostic and sensitivity artifacts. State the
actual prior, risk view, evaluation population, available metrics, and limiting
assumptions. For `review`, save the fitted estimator as a trusted local pickle
only when requested. Load only artifacts from a trusted source.
