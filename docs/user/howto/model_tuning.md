# Configuring Models and Searching Hyperparameters

`PUPipeline` accepts constructor parameters for a classifier selected by registry name. `PUTuner`
compares combinations using the same PU-stratified validation, prior handling, and metric
semantics. Results retain the configurations and final selection for review.

Examples below use `X, y_pu` from the [quickstart](../quickstart.md). Their `class_prior=0.4` is
an explicit illustrative population-prior assumption, not a value computed from hidden labels.
Use a known or independently estimated prior appropriate to your own data.

`class_prior` targets the population. Default OS input converts an estimator's declared native-U
result to a population value; `input_sampling="ts"` uses independent population U. `PUTuner` and
`PUModelComparator` pass this setting into each trial's Pipeline. Reports retain the population
value in `prior.value` and actual per-fold/final targets and values in `training_consumption`.

## Set fixed model parameters

```python
from pu_toolbox import PUPipeline

report = PUPipeline(
    classifier="upu",
    classifier_params={
        "loss": "logistic",
        "reg_lambda": 0.01,
        "max_iter": 2000,
    },
    cv=5,
).fit_evaluate(X, y_pu, class_prior=0.4)
```

Methods with extra required constructor arguments can also be selected by name:

```python
pipe = PUPipeline(
    classifier="ldce",
    classifier_params={"flip_probability": 0.2},
)
```

The pipeline manages `class_prior`, `random_state`, and the CNN `encoder`; do not put these in
`classifier_params`. Use `fit_evaluate(class_prior=...)`, `random_state=...`, and the workflow's
`architecture` / `backbone` arguments instead.

The CLI accepts repeated `--classifier-param KEY=VALUE`. Values support JSON numbers, booleans,
lists, and objects; plain string values need no JSON quotation marks. This multi-line command
uses POSIX shell continuations; in PowerShell, put it on one line or use PowerShell continuations.

```bash
pu-toolbox run \
  --data demo/X.csv --labels demo/y_pu.csv --out-dir results/ \
  --classifier upu --class-prior 0.4 \
  --classifier-param loss=logistic \
  --classifier-param reg_lambda=0.01 \
  --classifier-param max_iter=2000
```

## PU-aware grid search

```python
from pu_toolbox.model_selection import PUTuner

tuner = PUTuner(
    classifier="upu",
    param_grid={
        "loss": ["double_hinge", "logistic"],
        "reg_lambda": [0.001, 0.01, 0.1],
    },
    scoring="pu_zero_one_risk",
    cv=5,
    random_state=42,
)
result = tuner.fit(X, y_pu, class_prior=0.4)

print(result.best_params)
print(result.best_score)
best_model = result.best_report.final_model
```

The default minimizes `pu_zero_one_risk` and maximizes other metrics; override explicitly with
`higher_is_better=` when needed. Failed combinations and combinations with unavailable metrics
remain in `result.trials`. If none produces a usable score, the tuner raises `PipelineError`.

Each search trial computes PU-stratified CV only. After choosing the best combination, the tuner
performs one full-data refit and builds diagnostics. `result.best_report.final_model` is therefore
ready for prediction without fitting a final model for every candidate.

Scoring with `pu_auc_roc`, `pu_accuracy`, or `pu_f1` requires `y_true` in `fit`. Supplying it makes
selection use clean-label supervision; use only labels permitted by your study's validation
budget, and keep final test truth separate. Without clean labels, consider `pu_zero_one_risk`
and inspect the reliability of its prior and assumptions. Assess the selected configuration
on independent test data after fixing the choice.

## Graphical interface

The [UI](ui.md) provides fixed-parameter and grid-search panels without Python code.

## Compare multiple models

`PUModelComparator` compares registered models using a common population prior, input sampling
design, PU-stratified CV, and metric, then refits the best model on all supplied training data.

**Example A: compatible PU label semantics.**

```python
from pu_toolbox.model_selection import PUModelComparator

comparison = PUModelComparator(
    classifiers=["upu", "pusb"],
    scoring="pu_zero_one_risk",
    cv=5,
    random_state=42,
).fit(X, y_pu, class_prior=0.4)

print(comparison.best_classifier, comparison.best_score)
best_model = comparison.best_report.final_model
```

**Example B: preflight rejection of PNU label semantics.**

```python
# y_pu contains {1, 0} PU labels; PNU requires {+1, -1, 0} for P/N/U.
# Construction rejects this combination before any model is trained.
try:
    PUModelComparator(classifiers=["upu", "pnu", "pusb"], cv=5)
except ValueError as error:
    print(error)  # declares label_semantics='pnu', ... requires 'pu'
```

PNU requires three-way P/N/U labels and cannot be used directly in this PU-only workflow. The
[PNU method card](../../research/method_cards/PNU.md) provides optional Chinese research background.
After semantic preflight, runtime optimizer/backend failures remain isolated: the comparator
records a failed method and continues. It selects `best_classifier` only from successful methods,
and raises `PipelineError` if every method fails or has unavailable scoring.

See the [pipeline guide](pipeline.md) for workflow boundaries and the
[model-selection concepts](../concepts/pu_model_selection.md) for independent PA/OA experiments.
