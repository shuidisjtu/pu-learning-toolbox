# Analysis, tuning, and method comparison

Read this reference for an existing PU dataset, a single-model run, hyperparameter
search, or a comparison of registered methods. For sampling design and prior
targets, read [sampling-and-priors.md](sampling-and-priors.md). For image input,
read [representations.md](representations.md).

## Choose the execution route

Record the input design, allowed supervision, prior source, requested method,
scoring rule, seed, compute budget, and output directory. Reuse the user's choices.

- OS tabular analysis: use the CLI sequence below or `PUPipeline`.
- Independent TS: use Python APIs with `input_sampling="ts"`.
- Explicit train/validation/test roles, PA/OA, or custom research components: use
  [research-experiments.md](research-experiments.md).

The current `profile`, `recommend`, and `run` CLI commands use the OS default.
Query the installed CLI before selecting version-specific options:

```bash
pu-toolbox run --help
pu-toolbox list-methods
pu-toolbox list-priors
```

Use `uv run pu-toolbox` in the project's uv environment; use `pu-toolbox` in an
activated installed environment. Run saved Python examples with that environment's
interpreter. All paths below are relative to the user's working directory.

## OS analysis through the CLI

Inputs are a finite numeric feature CSV with a non-numeric header and a one-column
PU-label CSV with a header. Labels are `1` for labeled positive and `0` for U.
An optional one-column true-label CSV supplies classification evaluation labels.

For a supplied population prior of 0.4:

```bash
pu-toolbox profile --data X.csv --labels y_pu.csv --out-dir work/profile
pu-toolbox recommend --profile work/profile/profile.json --class-prior 0.4 --out-dir work/recommend
pu-toolbox run --data X.csv --labels y_pu.csv --class-prior 0.4 --classifier auto --cv 3 --seed 42 --out-dir work/run
```

Read `profile.json` for `selection_diagnostic`, `issues`, and input counts.
Read `recommendation.json` for ranked `candidates`, their `reasons`/`warnings`,
`global_warnings`, and `filters_applied`. The model selected for training is recorded
in the final report's `provenance`.

For authorized automatic prior estimation, let the Pipeline resolve the estimator
target, recommendation, and training values together:

```bash
pu-toolbox run --data X.csv --labels y_pu.csv --classifier auto --prior-estimator pen_l1 --cv 3 --seed 42 --out-dir work/estimated
```

Carry a requested estimator or explicit prior into the run. For a selected method
that supports an omitted external prior, `--prior-estimator none` selects that
policy. In auto mode the report records the selected method and any prior-estimation
degradation, together with the candidates considered.

The standalone `recommend --prior-estimator` path reads the estimator's scalar
output directly. Use an explicitly resolved population value for recommendations
that require that target; [sampling-and-priors.md](sampling-and-priors.md) gives
the shared conversion API. Pipeline reports preserve the complete estimation chain.

## Python analysis with explicit input sampling

This complete CPU example generates independent P and population U, fits uPU under
TS, and writes the report. Adapt the loading step to the user's arrays.

```python
from pathlib import Path

from pu_toolbox import PUPipeline
from pu_toolbox.preprocessing import make_gaussian_case_control_data

out = Path("work/ts")
out.mkdir(parents=True, exist_ok=True)
X, y_pu, y_true, metadata = make_gaussian_case_control_data(
    n_p=30, n_u=180, class_prior=0.4, n_features=3, random_state=42,
)
report = PUPipeline(
    classifier="upu",
    classifier_params={"loss": "logistic", "max_iter": 500},
    input_sampling="ts",
    cv=3,
    random_state=42,
).fit_evaluate(X, y_pu, class_prior=metadata["population_class_prior"])
report.save(out / "report.json")
report.save(out / "report.md")
```

For OS, set `input_sampling="os"` and use same-pool PU observations. Configure
`prior_estimator="pen_l1"`, `"recpe"`, `"km1"`, or `"km2"` for estimated-prior
analysis, and omit the `fit_evaluate` population-prior argument. A configured
classifier's prior requirement determines whether estimation is needed. Auto mode
resolves a prior before ranking candidates. Methods needing additional constructor
arguments use `classifier_params`, or a configured estimator instance.

## Tune a specified method

Choose the allowed scoring evidence and search budget before fitting. With PU
validation labels and a population prior, `pu_zero_one_risk` is minimized. Oracle
scores use authorized validation truth. Explicit test data is evaluated after
selection through the research route.

The following block uses `X`, `y_pu`, and `out` from the preceding example:

```python
import json

from pu_toolbox.model_selection import PUTuner

tuned = PUTuner(
    classifier="upu",
    param_grid={"loss": ["logistic"], "reg_lambda": [0.01, 0.1]},
    scoring="pu_zero_one_risk",
    input_sampling="ts",
    cv=3,
    random_state=42,
).fit(X, y_pu, class_prior=0.4)
(out / "tuning.json").write_text(
    json.dumps(tuned.to_dict(), indent=2, allow_nan=False), encoding="utf-8",
)
tuned.best_report.save(out / "tuned-report.json")
```

Read `best_params`, `best_score`, and every trial's `status`, `score`, and `error`.
Trials run CV; the selected combination receives a final full-data fit. The fitted
model is `tuned.best_report.final_model`.

## Compare specified methods

Use a shared split policy, representation, prior source, scoring rule, and seed.
Select methods whose declared label semantics and input dimensions match the data.

```python
from pu_toolbox.model_selection import PUModelComparator

compared = PUModelComparator(
    classifiers=["upu", "pusb"],
    scoring="pu_zero_one_risk",
    input_sampling="ts",
    cv=3,
    random_state=42,
).fit(X, y_pu, class_prior=0.4)
(out / "comparison.json").write_text(
    json.dumps(compared.to_dict(), indent=2, allow_nan=False), encoding="utf-8",
)
compared.best_report.save(out / "compared-report.json")
```

Read `best_classifier`, `best_score`, and all trial outcomes. Use the best report's
actual prior-consumption records when explaining the selected method.

For existing OS run configurations, `run --config config.json` supports
`tuning_grid` or `comparison_classifiers` with `scoring`. Fixed constructor values
can be supplied through repeated `--classifier-param KEY=VALUE`; prior-estimator
parameters use repeated `--prior-param KEY=VALUE`. Explicit CLI values take
precedence over configuration values. Read the emitted `tuning.json` or
`comparison.json` alongside `report.json`.

## Fixed-output sensitivity and refitting

For an OS assumption sweep on a fitted model's fixed predictions:

```python
from pu_toolbox.diagnostics import analyze_pu_sensitivity

predictions = fitted_model.predict(X_os)
analysis = analyze_pu_sensitivity(
    y_pu_os, predictions, class_priors=[0.35, 0.4, 0.45],
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

## Finish with artifacts and conclusions

Read `report.json` as schema `1.1`: `prior`, `cv_metrics`, `provenance`, and `issues`.
For each metric report `available`, `mean`, `std`, and `n_computed`; use `reason`
to explain an unavailable score. State diagnostic findings and the actions taken.
For Chinese summaries use [interpret.zh-CN.md](interpret.zh-CN.md).

Completion: the requested analysis/search/comparison has readable artifacts, its
method and prior provenance are recorded, and the summary cites the actual results.
