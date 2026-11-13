---
name: pu-workflow
description: >-
  Run positive-unlabeled learning tasks with pu-learning-toolbox:
  analyze PU data, construct OS/TS and SCAR/SAR studies, estimate class
  priors, tune or compare methods, run PA/OA experiments, and configure
  image encoders. Also handles requested drift, monitoring, review, and
  benchmark audits.
license: MIT
metadata:
  author: shuidisjtu
  version: 1.11.0
---

# pu-workflow

Turn the user's PU learning goal into an executable analysis or study,
then explain the saved results in the user's language. Choose the route
that matches the task and load its reference when needed.

## Establish the task

Reuse the user's choices and existing authorization. Establish the goal,
input paths and shapes, label availability, sampling design, prior target
and source, available validation/support data, device, execution budget,
and output directory. Ask for missing information when it changes the
statistical design or the permitted work. For a complete analysis request,
use an appropriate automatic prior estimator when required and record it.
Carry diagnostic findings into method selection and reporting while
continuing the authorized work.

## Environment and capability discovery

Use Python >=3.10 and pu-toolbox >=1.11.0 with the capabilities selected below.
Work in the user's data/output directory or a source checkout. Resolve
references below relative to this skill directory. Prefer `uv run` in a
project environment; an installed environment can invoke `pu-toolbox`
and its Python interpreter directly.

```bash
uv run pu-toolbox --help
uv run pu-toolbox list-methods
uv run pu-toolbox list-priors
```

Check the installed `pu_toolbox.__version__`, selected command's `--help`,
and Python signatures for the chosen route. Research APIs are available
in the current source build; feature discovery selects the matching
installed capabilities. Use the project's torch extras for image and
encoder tasks. Report the package version and relevant capabilities.

CSV feature inputs have a non-numeric header row; PU labels are a headered
single column of 1 (labeled positive) and 0 (unlabeled). Raw image inputs
use finite 4-D NCHW `.npy` arrays. Python APIs accept validated arrays.
Inspect input shape, label counts and source identities before training.

## Select the route

| User task | Read | Execution and completion |
|---|---|---|
| Analyze existing PU data; tune or compare methods | [Analysis](references/analysis.md) | CLI/Pipeline or tuner/comparator; read profiles, selected settings and metric reports |
| Construct PU observations; identify OS/TS; estimate or convert a prior | [Sampling and priors](references/sampling-and-priors.md) | Preprocessing and target-aware prior APIs; save sampling/source metadata and actual prior consumption |
| Run four-role research, PA/OA or custom strategies | [Research experiments](references/research-experiments.md) | `ExperimentRunner`; read separate protocol selections, checkpoints, thresholds and test results |
| Train images, inject an encoder or use frozen features | [Representations](references/representations.md) | Discover registered capabilities; save representation identity, training scope and model results |
| Audit drift, monitor windows, create a review queue or audit a benchmark | [Extensions](references/extensions.md) | Run the requested extension and read its structured artifacts |

A task can combine routes: construct data, run an experiment and interpret
its results. Load the references for those stages as they become relevant.

## Shared execution semantics

- Record input sampling, labeling policy, training risk view, prior target,
  validation supervision and representation as separate settings.
- The current CLI profile/recommend/run path uses OS input semantics.
  Explicit TS analyses use Python APIs with `input_sampling="ts"`.
- Pipeline's call-level `class_prior` and report `prior.value` target the
  population. Read estimator target declarations and recorded conversions
  when estimating; training consumers receive their required pool priors.
- For raw images, select an explicit CNN-capable classifier and specify
  prior information appropriate to the image/feature route.
- Fit preprocessing and trainable representations on the permitted train
  data. Validation selects candidates, checkpoints and thresholds; test
  evaluates the selected result. Record extra clean-support supervision.
- Honor user-selected methods and budgets. Resolve CLI/input errors using
  the actual error and command help; resume after a justified correction.
  When progress requires a user decision, present the concrete choice.

## Report the result

Read the saved JSON and model-selection artifacts. State the settings,
executed budget, findings, available metrics, and actionable next steps.
For Chinese summaries, load [the interpretation guide](references/interpret.zh-CN.md).
State the conclusion supported by the analysis and its applicable
conditions directly. Use actual artifact values and explain each metric
in relation to its evaluation population.
Keep each paragraph focused on the question the computation answers.
For a fixed-prediction sweep, report the assumption grid and observed
metric changes. Express additional information needs as concrete actions,
such as collecting validation truth for supervised evaluation.

Complete the task when the requested artifacts exist, their key fields
have been checked, and the user has a clear account of the result and
any concrete follow-up work.
