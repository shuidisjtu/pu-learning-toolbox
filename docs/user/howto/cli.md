# Using the command-line interface

> Prerequisite: complete the [quickstart](../quickstart.md).
> The CLI wraps library functionality. `sensitivity` and `recommend` directly call estimator
> `fit`; other commands delegate learning logic to the library. The `run` command also
> merges configuration, assembles searches, and writes artifacts: `report.json` and
> `report.md` are always produced; `tuning.json`, `comparison.json`, and `model.pkl`
> depend on the selected search/comparison branch and `--save-model`.

## Quickstart workflow

```bash
# 1. Generate SCAR demo data: X.csv, y_pu.csv, and y_true.csv.
pu-toolbox make-demo-data --out-dir demo/ --n 200 --seed 42

# 2. Train and evaluate; auto mode selects a method.
pu-toolbox run --data demo/X.csv --labels demo/y_pu.csv --out-dir results/

# 3. Inspect results/report.md and results/report.json.
# The JSON report uses strict encoding without NaN literals.
```

## Commands and parameters

### run

| Parameter | Required | Default | Meaning |
|---|---|---|---|
| `--data X.csv` | Yes | — | Features, one sample per row, with a header. Also accepts finite 4-D NCHW image arrays in `.npy` files with `--architecture cnn` |
| `--labels y_pu.csv` | Yes | — | Headered single-column PU labels `{1, 0}` |
| `--out-dir results/` | Yes | — | Output directory for report.json and report.md |
| `--true-labels` | No | — | Single-column truth `{0, 1}` for oracle metrics such as AUC |
| `--class-prior` | No | — | Explicit population prior in (0, 1), bypassing estimation |
| `--prior-estimator` | No | `pen_l1` | `pen_l1` / `recpe` / `km1` / `km2` / `none`, or registered names such as `class_prior_estimation` (alias `cpe`; `pe` is deprecated) |
| `--prior-param` | No | — | Repeatable estimator constructor parameters, e.g. `--prior-param sigma=3.0 --prior-param n_centers=100`; values accept JSON literals |
| `--classifier` | No | `auto` | Registered method name or automatic recommendation |
| `--classifier-param` | No | — | Repeatable constructor parameters accepting JSON numbers, booleans, lists, and objects, e.g. `--classifier-param reg_lambda=0.01` |
| `--config` | No | — | JSON run configuration shared with the UI; explicitly supplied nondefault options override configured values |
| `--architecture` | No | `mlp` | `mlp` for tables or `cnn` for 4-D NCHW images with an explicit supported classifier |
| `--backbone` | No | `cnn13` in CNN mode | `cnn13` / `cnn13_no_bn` / `resnet18` / `resnet50`; rejected in MLP mode |
| `--device` | No | `auto` | Torch device: `auto` / `cpu` / `cuda`; auto selects CUDA when available |
| `--max-epochs` | No | Estimator default | Epoch cap for constructors accepting `max_epochs`, e.g. WConPU, Self-PU, or nnPU |
| `--cv` | No | `5` | Number of CV folds |
| `--metrics` | No | Four default metrics | Comma-separated names or aliases, e.g. `pu_risk,recall,auc` |
| `--seed` | No | `42` | Random seed |
| `--save-model` | No | Off | Save the final model as `model.pkl` using pickle |
| `--quiet` | No | Off | Suppress the summary; errors still print |

The first CSV row must contain column names. A numeric first row is rejected to avoid silently
discarding the first sample. PU label 0 means unlabeled, not a known negative. The current CLI
uses OS input semantics; independent TS analyses use the Python API with `input_sampling="ts"`.

### Supporting commands

- `list-methods`: lists registered classifiers in eight columns: name, family, prior requirement,
  implementation status, Input dimensionality, native Arch, main-input Labels, and Auto-inst.
  Capabilities come from registry metadata. Labels can be pu/pn/pnu; Auto-inst describes
  construction, not eligibility for PU workflows. `run`, tuning/comparison, and `sensitivity`
  require PU main-input semantics, so PNU is excluded from these binary-PU entry points.
  Registered methods appear automatically.
- `list-priors`: lists available prior estimators. `km1` and `km2` map to
  `KernelMeanPriorEstimator(variant=...)`.
- `make-demo-data --out-dir demo/ [--n 200] [--c 0.5] [--n-features 5] [--separation 1.0] [--seed 42]`:
  uses `make_scar_dataset` to write demo CSVs. There are `n` samples per true class, 2n total.
  `--c` requests a positive-class labeling fraction under uniform fixed-count SCAR selection.
  The default separation 1.0 avoids the stronger prior-underestimation behavior of highly
  separated demo classes.
- `profile --data X.csv --labels y_pu.csv [--true-labels y_true.csv] [--out-dir .]`:
  profiles data and labeling-mechanism evidence, writing `profile.json`.
- `recommend --profile profile.json [--data X.csv --labels y_pu.csv] [--class-prior 0.3 | --prior-estimator recpe] [--top-k 5] [--has-gpu] [--out-dir .]`:
  recommends methods and writes `recommendation.json`. Prior estimation requires data and labels.
- `sensitivity --data X.csv --labels y_pu.csv [--classifier elkan_noto] [--class-priors 0.1,...,0.9] [--seed 42] [--out-dir .]`:
  scans class-prior/propensity assumptions and writes `sensitivity.json`.
- `audit-benchmark --result-dir results/ [--output audit.json]`: checks required artifacts,
  configuration hashes, trial/seed completeness, duplicate rows, finite metrics, and official-data
  PU split overlap and target-prior consistency. Failed checks exit with code 1;
  `paper_claim=false` and a dirty worktree remain warnings.
- `shift-audit --source-data source.csv --source-labels source_labels.csv --target-data target.csv [--target-labels target_labels.csv] --out-dir shift_results/`:
  uses an OOF domain classifier to audit source/target differences. Writes `shift_report.json`,
  `shift_report.md`, and `source_importance_weights.csv`. Detection and adaptation are separate;
  marginal weights have covariate-shift guarantees. See [distribution shift](distribution_shift.md).
- `shift-run --source-data source.csv --source-labels source_labels.csv --target-data target.csv --target-labels target_labels.csv [--target-true-labels y_true.csv | --target-class-prior 0.3] --out-dir shift_results/`:
  compares weighted and unweighted models on the same target set. Only oracle or prior-dependent
  metrics drive automatic recommendations. Outputs comparison/shift reports, source weights,
  and target predictions from both arms.
- `shift-monitor --reference-data source.csv --reference-labels source_labels.csv --window-data current.csv [--window-labels current_labels.csv] --window-id 2026-08 [--history previous_history.json] --out-dir monitor/`:
  appends a deployment window and writes alerts, a window shift report, source weights,
  and resumable `shift_history.json`.
- `review --model trusted_model.pkl --data current.csv [--labels current_labels.csv] --query-budget 20 --query-strategy uncertainty --out-dir review/`:
  produces selective predictions and an active human-review queue. Load only models you created
  or trust; do not load unknown pickle files.
- `skill install [--force] [--dest agent-home/]`: installs the pu-workflow entry and six
  reference files into `~/.claude/skills/` and `~/.agents/skills/`. `--dest` changes their
  shared root. Existing installations are retained unless `--force` is used.
  The skill chooses CLI or Python APIs by task; see [the skill guide](using_skill.md).

## Deep methods and image data

Install `pip install "pu-toolbox[torch]"` for deep learning. The `run` command supports
deep classifiers whose current declarations and constructor signatures allow CNN encoder
injection; nnPU, WConPU, and InfoMax PU are examples. Discover the current set through
`pu-toolbox run --help` and the capability columns in `pu-toolbox list-methods`.

- Tables use `--architecture mlp`, for example with `--classifier wconpu`.
- Images use finite 4-D NCHW float arrays in `.npy` files with
  `--architecture cnn --backbone cnn13` (or another supported backbone).
- CNN mode requires an explicit compatible classifier. `auto`, traditional methods, and
  methods without encoder injection, such as Self-PU, cannot use this pipeline CNN path.
  A method may impose additional backbone constraints.
- Deep training can be expensive. WConPU defaults to 100 epochs; use `--max-epochs` where
  supported and reduce `--cv` when appropriate.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Input/runtime error, such as a missing file, invalid labels, or method; stderr contains `error: ...` |
| 2 | Argument syntax error from argparse |

`audit-benchmark` validates artifact consistency and traceability. It does not establish equality
to official source, verify the paper protocol or data authorization, or review statistical
conclusions. Those require their own evidence before upgrading a reproduction claim.

Domain AUC in `shift-audit` detects observable domain differences without identifying the
shift type. Without target PU labels, only auditing is available and `adaptation_ready=false`.

## Prior resolution order

Explicit values take precedence:

1. `--class-prior 0.5`: the report records `source: user`; this value takes precedence over
   `--prior-estimator` when both are supplied.
2. A classifier constructor prior (Python API), recorded as `source: constructor` and interpreted
   by that classifier's declared target.
3. Automatic estimation through `--prior-estimator`, recorded as `source: estimated`.

`--prior-param KEY=VALUE` only configures the estimator in step 3. For example,
`--prior-param sigma=3.0` changes penL1's kernel width. Invalid numeric settings such as
`sigma=abc` fail before estimation rather than silently degrading.

The CLI call-level prior and report `prior.value` target the population. The Pipeline resolves
an estimator's declared target and converts the prior for each training consumer; see
[prior semantics](../concepts/class_prior_semantics.md).

## Relationship to Python APIs

A plain `pu-toolbox run` delegates to
`PUPipeline(classifier=..., prior_estimator=..., cv=..., metrics=..., random_state=...).fit_evaluate(X, y_pu, y_true=..., class_prior=...)`.
Pass a UI-exported configuration with `run --config`. A parameter grid selects the `PUTuner`
branch and produces `tuning.json`; a multi-model configuration selects `PUModelComparator`
and produces `comparison.json`. Use the Python APIs for custom searches or CV splitters.
See [model tuning](model_tuning.md), or use the [UI](ui.md).

## Next steps

- Python workflow, resolution, degradation, and metric evidence: [pipeline](pipeline.md).
- Exact parameter contracts: [API reference](../reference/api.md), currently Chinese.
