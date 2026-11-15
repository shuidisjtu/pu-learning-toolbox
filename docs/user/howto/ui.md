# Using the graphical interface

The Streamlit interface is an optional dependency, separate from the core installation.

The UI accepts PU main-training labels: 1 means labeled positive and 0 means unlabeled,
not a true negative. Optional uploaded truth is for evaluation and does not switch training
to PN. Model details show `label_semantics`; PN/PNU methods are excluded from PU choices.
Imported configurations and direct execution calls also pass the pre-training semantics gate.

The current runtime interface uses Chinese labels. This English guide gives translations
alongside the relevant controls; UI localization is a separate task.

## Install and launch

```bash
pip install "pu-toolbox[ui]"
pu-toolbox-ui
```

The command starts a local web service and opens a browser. The toolbox processes uploaded
data without initiating transfers to external services. On shared servers, the operator owns
file visibility, authentication, and network access policies.

## Page workflow

1. Upload a feature CSV or 4-D NCHW `.npy` image array.
2. Upload single-column `{1, 0}` PU labels and, optionally, true binary labels.
3. Inspect sample, feature, labeled-positive, and unlabeled counts.
4. Use automatic recommendation, select one model manually, or compare several under the same CV settings.
5. In the advanced panel, select parameters to edit and set their numeric, boolean, or categorical
   values. Use the additional fixed-parameter JSON field for complex values, or enable
   “compare hyperparameter combinations” for a grid.
6. Start analysis and inspect metrics, diagnostic messages, and tuning trials.
7. Download JSON/Markdown reports, prediction CSVs, models, and tuning records.
8. Use the deployment monitoring/review panel to upload a target window and inspect shift
   alerts, prediction coverage, and rows proposed for human review.

The sidebar imports/exports `pu-run-config.json`. It contains model choices, fixed parameters,
parameter grids, priors, CV, metrics, and seeds, but not uploaded data. Reuse it from the CLI:

```bash
pu-toolbox run --data X.csv --labels y_pu.csv --out-dir results/ \
  --config pu-run-config.json
```

Explicit nondefault CLI options and `--classifier-param` values override corresponding
configuration values. A configured parameter grid also runs through the CLI and produces
`tuning.json`.

Model-comparison mode needs at least two eligible methods with no additional required constructor
parameters. It applies the same PU-stratified CV to each. Runtime failures remain in the
comparison record while other methods continue; only the winning model receives a final
full-data refit. Download the record as `comparison.json`. A comparison configuration passed
to the CLI produces the same artifact name.

Training runs in a background thread with stage/progress updates. The cancel control
(`取消运行`) takes effect at the next safe boundary: workflow stages, CV folds, tuning
trials, or comparison items. An estimator `fit` already in progress finishes its current call.
The sidebar keeps start/end times, mode, status, and summaries for the service process's most
recent 20 runs. History survives page refreshes but is cleared when the Streamlit process
restarts. Sessions in a shared service process share this history.

After training, the deployment panel uses the current training data as a fixed reference domain.
Upload a target window's features and PU labels to:

- Start a new window ID or continue from an uploaded `shift_history.json`.
- Inspect domain AUC, alert level, prediction coverage, and review count.
- Choose uncertainty or diversity-aware uncertainty sampling.
- Download updated history, window shift JSON, and per-row review CSVs.

Deployment checks do not retrain the model. Users must save and re-upload monitoring histories;
the server does not retain them across processes.

## Input and model eligibility

CSV headers must be nonnumeric column names, and features must be finite numeric values.
Image mode uses deep methods that declare encoder capability and meet the current UI
requirements. Examples include InfoMax PU, WConPU, nnPU, PAN, PULDA, and Holistic-PU;
the list is derived from registry capabilities and UI eligibility. Available backbones include
CNN13, ResNet-18, and ResNet-50, subject to method-specific restrictions. Install torch as well:

```bash
pip install "pu-toolbox[ui,torch]"
```

DGPU requires a Python-object `generator` protocol that cannot be constructed safely from a
JSON form, so it is absent from the model selector. Configure it through the Python API.

PULNS and LaGAM require an independent clean reward/support set. The current UI, CLI, and ordinary
Pipeline do not expose that additional role, so these methods are excluded from manual/comparison
choices. Importing an incompatible configuration fails explicitly without altering the original
configuration. Optional evaluation truth cannot substitute for training support labels. PULNS's
native CNN capability remains registered; use its estimator API with explicitly isolated
`support_data`.

## Parameter examples

For common scalar and categorical parameters, choose fields under “parameters to adjust”
(`需要调整的参数`). The UI derives numeric controls, switches, and dropdowns from constructor
types. Unselected optional parameters retain model defaults.

For complex values or bulk entry, use “additional fixed parameters (advanced JSON)”
(`额外固定参数（高级 JSON）`):

```json
{"loss": "logistic", "max_iter": 2000}
```

Enable “compare hyperparameter combinations” (`比较多组超参数`) and enter a grid:

```json
{"reg_lambda": [0.001, 0.01, 0.1]}
```

Fixed parameters apply to every trial. A parameter cannot occur in both the fixed mapping and
the search grid.

Next: [model configuration and tuning](model_tuning.md) and the
[API reference](../reference/api.md).
