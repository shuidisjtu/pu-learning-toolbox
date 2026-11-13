[English](README.md) | [中文](README.zh-CN.md)

# PU Learning Toolbox

**Positive-Unlabeled learning in Python** -- registered PU algorithms, research joint-shift adaptation, SCAR & SAR support.

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![Status](https://img.shields.io/badge/status-1.11.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Registered PU algorithms** from recent research, all native clean-room implementations, plus an isolated research joint-shift solver ([method cards](docs/research/method_cards/)); `pu-toolbox list-methods` prints the current set
- **sklearn-compatible API** -- `fit(X, y)` / `predict(X)` / `decision_function(X)`, works with pipelines and cross-validation
- **SCAR & SAR** -- constant and instance-dependent labeling mechanisms, with a data simulator
- **Data profiling + recommender** -- automatic quality checks, SCAR/SAR evidence, and a 7-dimension scoring recommender that picks the method for your data
- **Auditable pipeline** -- one-call `PUPipeline` (profile -> prior -> train -> PU-stratified CV -> evaluate) plus structured diagnostic reports and prior/propensity sensitivity analysis
- **Composable PU research** -- OS and independent TS sampling, explicit prior targets, injectable experiment strategies, custom encoders, and recorded research settings ([guide](docs/user/howto/custom_experiments.md))
- **Distribution-shift guardrails** -- OOF source/target drift audit, bounded covariate weights, overlap diagnostics, and a guarded `ShiftAwarePUPipeline`
- **Deployment monitoring** -- resumable window alerts, coverage/rejection controls, and active-review exports in CLI and UI
- **CLI** -- `pu-toolbox` turns the whole pipeline into terminal commands
- **Model tuning** -- unified classifier parameters plus PU-aware grid search
- **Evaluation metrics** -- PU-native risk plus supervised ranking, balanced-accuracy, and probability-calibration metrics with explicit availability contracts
- **Reproducible traditional-PU benchmark** -- locked seven-method protocols, leakage preflight, resumable trials, paired comparisons, and tuning evidence
- **Graphical UI** -- upload data, configure/compare models, inspect diagnostics, and download results

## Quick Start

```bash
pip install pu-toolbox                # core dependencies (Python >= 3.10)
pip install "pu-toolbox[torch]"       # + PyTorch-based methods (nnPU, Dist-PU, Self-PU, ...)
pip install "pu-toolbox[text]"        # + fixed SBERT preprocessing for survey text datasets
pip install "pu-toolbox[ui]"          # + Streamlit graphical interface
```

### Installation environments

Any Python interpreter >= 3.10 works: the package is a pure-Python universal
wheel with no compiled extensions, so the interpreter source does not matter.
Notes per environment:

- **venv / uv** (recommended): standard isolated environments, nothing special.
- **System Python** (python.org / Ubuntu / Homebrew): must be >= 3.10.
  Ubuntu 22.04+ and Debian 12+ block `pip install` into the system environment
  (PEP 668) -- create a venv instead.
- **Anaconda / Miniconda**: `pip install pu-toolbox` inside a conda env
  (the package is PyPI-only; `conda install` will not find it). If you already
  installed torch via conda, a plain `pip install pu-toolbox` (without the
  `[torch]` extra) still enables the PyTorch-based methods -- torch is an
  optional dependency loaded lazily.

### Hello World

```python
from pu_toolbox.preprocessing import make_scar_dataset
from pu_toolbox import PUPipeline

# Synthetic OS data with SCAR labeling: each positive has the same labeling
# probability. Observed labels are 1 (labeled positive) and 0 (unlabeled).
X, y_pu, y_true = make_scar_dataset(
    n=500, c=0.5, n_features=8, separation=1.0, random_state=42,
)

# One call: profile -> class prior -> train -> PU-stratified CV -> evaluate
report = PUPipeline().fit_evaluate(X, y_pu, y_true=y_true)
print(report.summary())
```

Full docs (Chinese): [docs/README.md](docs/README.md). More runnable examples: [`examples/minimal/`](examples/minimal/).

`PUPipeline` uses OS input sampling by default; independent P and population U
use `input_sampling="ts"`. A supplied `class_prior` targets the population;
the report records its source, conversions, and the value consumed by each
training fold. See [prior semantics](docs/user/concepts/class_prior_semantics.md)
and the [pipeline guide](docs/user/howto/pipeline.md).

## Command Line

The `pu-toolbox` console command wraps the full pipeline. Full guide: [`docs/user/howto/cli.md`](docs/user/howto/cli.md).

```bash
# 1. Generate SCAR demo data (X.csv / y_pu.csv / y_true.csv)
pu-toolbox make-demo-data --out-dir demo/ --n 200 --seed 42

# 2. One-shot full pipeline run (auto mode picks the algorithm)
pu-toolbox run --data demo/X.csv --labels demo/y_pu.csv --out-dir results/

# 3. Inspect results
#    results/report.md     full Markdown report
#    results/report.json   strict JSON (no NaN), machine-readable
```

## Graphical UI and model tuning

```bash
pip install "pu-toolbox[ui]"
pu-toolbox-ui
```

The UI supports automatic recommendations, manual model parameters,
PU-stratified grid search, metric charts, diagnostics, and report/model
downloads. See the [model tuning guide](docs/user/howto/model_tuning.md) and
[UI guide](docs/user/howto/ui.md).

## Documentation

Docs are split by audience; the full index is [`docs/README.md`](docs/README.md).

| Entry | Content |
|----------|---------|
| [`docs/user/quickstart.md`](docs/user/quickstart.md) | 5-minute start (CLI + Python) |
| [`docs/user/concepts/`](docs/user/concepts/) | Sampling, SCAR/SAR, prior targets, PA/OA selection, representations, method choice |
| [`docs/user/howto/`](docs/user/howto/) | Simulation, pipeline, DIY experiments, encoders, CLI, reports, sensitivity, distribution shift |
| [`docs/user/reference/api.md`](docs/user/reference/api.md) | Precise API contract |
| [`docs/dev/`](docs/dev/) | Contributor docs: architecture, structure, compatibility |
| [`docs/dev/project_structure.md`](docs/dev/project_structure.md) | Source/data boundaries, local artifact retention, and scripts grouped by task |
| [`docs/research/method_cards/`](docs/research/method_cards/) | Per-paper research cards |

### Composable PU research

From a repository checkout, explore the research workflow through these CPU examples:

| Task | Guide | Runnable example |
|---|---|---|
| Construct OS and independent TS data | [Sampling](docs/user/howto/pu_sampling.md) | [15_pu_sampling.py](examples/minimal/15_pu_sampling.py) |
| Trace known and estimated priors into training | [Prior targets](docs/user/concepts/class_prior_semantics.md) | [16_prior_targets.py](examples/minimal/16_prior_targets.py) |
| Compare Bernoulli and fixed-count SAR policies | [SAR simulation](docs/user/howto/sar_simulation.md) | [17_sar_policies.py](examples/minimal/17_sar_policies.py) |
| Train a custom encoder or use frozen features | [Custom encoders](docs/user/howto/custom_encoders.md) | [18_custom_encoders.py](examples/minimal/18_custom_encoders.py) |
| Inject experiment strategies and select independently with PA/OA | [DIY experiments](docs/user/howto/custom_experiments.md) | [14_diy_experiment.py](examples/minimal/14_diy_experiment.py) |

Experiment manifests describe sampling, labeling, risk pools, prior inputs and
training consumption, supervision, and representations. Summary files, workbooks,
and diagnostic cards carry readable settings; historical results retain their
recorded identities. See the [reporting workflow](docs/research/pu_survey/runbook.md).

### PU Survey experiments

Survey scripts run the repository's selected research settings through
`ExperimentRunner`, with four data roles, independent PA/OA selection, and final
test evaluation. Prepare reproducible four-way splits and inspect the runner.

The LBE diagnostic uses `scripts/_lbe_diagnostic_reference.py` as a separate
SCAR-constrained reference implementation.
Inspect the commands with:

```bash
uv run python scripts/prepare_survey_splits.py --help
uv run python scripts/run_survey_experiment.py --help
uv run python scripts/run_lbe_comparison.py --help
uv run python scripts/run_lbe_diagnostics.py --help
uv run python scripts/prepare_clean_support.py --help
uv run python scripts/run_clean_support_experiment.py --help
uv run python scripts/run_genpu_recipe.py --help
uv run python scripts/run_candidate_image_experiment.py --help

# SCAR row (PA + OA selection)
uv run python scripts/run_survey_experiment.py path/to/splits \
    --method lbe --c 0.1,0.3,0.5

# SAR pressure test (LBE-A/LBE-B labeling, OA only, c in {0.05, 0.5})
uv run python scripts/run_survey_experiment.py path/to/splits \
    --method lbe --labeling-mechanism sar_lbe_a --c 0.05,0.5
```

Methods whose registry entry requires a prior take `--class-prior` at the run
boundary and constructor arguments via `--model-params`. Interpret each parameter
using its [prior target](docs/user/concepts/class_prior_semantics.md) and recorded
training view; runnable configurations are in the
[runbook](docs/research/pu_survey/runbook.md). `--labeling-mechanism` selects how the
PU label view is generated (SCAR or the SAR LBE variants) and is orthogonal to
`--method`: the mechanism is the experiment's independent variable. SAR rows
report OA for methods supporting the selected mechanism.

Results carry their execution mode and eligibility status. The protocol,
execution status, and reporting workflow are in
[`docs/research/pu_survey/`](docs/research/pu_survey/).

Supporting scripts (see each script's docstring and the Survey docs for scope and limits):

| Stage | Scripts |
|---|---|
| Data and splits | `collect_survey_public_data.py` (collect missing public raw data; excludes access-controlled ADNI), `survey_splits_archive.py` (split transfer) |
| Runs | `run_survey_pilot.py` (scope planning, execution and resume) |
| Reporting | `summarize_survey_results.py`, `export_survey_results.py` (workbook, diagnostics and conclusions; needs the `report` extra), `_source_provenance.py` (internal code and environment identity helper) |
| Recipes | `prepare_survey_recipe_registry.py`, `check_survey_recipe_registry.py` (validate a registry; CI checks the frozen lock) |
| Independent image runs | `run_candidate_image_experiment.py` (PULDA/GradPU four-role PA/OA smoke, measured snapshots and replay) |
| Independent classical runs | `run_classical_candidate_experiment.py` (PUET/RP/KLDCE four-role training with explicit parameters and actual-cost receipts) |
| Independent support runs | `run_clean_support_experiment.py` (Self-PU/LaGAM/PULNS extra-label audit and explicit Self-PU TS meta-checkpoints) |

These scripts record technical checks and measurements. Reporting reads recorded
results and writes artifacts to the selected `--out-dir`.

## AI workflow skill

`pu-workflow` (Agent Skills open standard) routes natural-language tasks
to CLI or Python APIs: PU analysis, OS/TS and SCAR/SAR construction,
prior resolution, tuning/comparison, four-role PA/OA experiments,
CNN/encoder workflows, drift, monitoring, review, and benchmark audits.
It carries the user's settings and budget through execution and reads
the saved artifacts to explain the results.

The wheel bundles the entrypoint and six task references. Install with
`pu-toolbox skill install` into Claude Code / Cursor (`.claude/skills/`)
and Codex / Gemini CLI / Windsurf (`.agents/skills/`); use `--dest` for a
custom root and `--force` when choosing to update an existing install.
The revised skill is validated with the current source-built 1.11.0 wheel;
routes use the installed package's discovered capabilities. See
[How to enable and use the skill](docs/user/howto/using_skill.md).

## Development

```bash
git clone https://github.com/shuidisjtu/pu-learning-toolbox.git
cd pu-learning-toolbox
pip install -e ".[dev,torch]"   # development install
uv run pytest tests/ -v -m "not slow and not e2e"   # fast tests (e2e runs nightly)
uv run ruff check pu_toolbox/               # lint
uv run ruff format --check pu_toolbox/      # format check

# Quality gates
uv run python scripts/check_test_quality.py
uv run python scripts/check_doc_links.py
uv run python scripts/check_api_docs.py       # api.md covers every public symbol (anti-drift)
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
uv run python scripts/check_baseline_configs.py    # baseline config vs source defaults
uv run python scripts/check_format.py        # ruff check + format --check (full scope)
uv run python scripts/check_comment_quality.py  # comment hygiene: legacy markers / bare TBD; trailing comments are errors inside migrated (strict) partitions, advisory elsewhere (self-reported on pass)
uv run python scripts/check_survey_recipe_registry.py  # P4.1 recipe registry gate; CI runs it against the frozen registry with --require-present --require-locked
uv run python scripts/generate_structure.py --check    # structure document consistency (--update to regenerate)
uv run python scripts/generate_layer_deps.py --check  # architecture.md §2.1 layer table (--update to regenerate)
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## License

MIT
