[English](README.md) | [中文](README.zh-CN.md)

# PU Learning Toolbox

**Positive-Unlabeled learning in Python** -- 19 registered algorithms, research joint-shift adaptation, SCAR & SAR support.

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![Status](https://img.shields.io/badge/status-1.11.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **19 registered algorithms** from recent PU learning research, all native clean-room implementations, plus an isolated research joint-shift solver ([method cards](docs/research/method_cards/))
- **sklearn-compatible API** -- `fit(X, y)` / `predict(X)` / `decision_function(X)`, works with pipelines and cross-validation
- **SCAR & SAR** -- constant and instance-dependent labeling mechanisms, with a data simulator
- **Data profiling + recommender** -- automatic quality checks, SCAR/SAR evidence, and a 7-dimension scoring recommender that picks the method for your data
- **Auditable pipeline** -- one-call `PUPipeline` (profile -> prior -> train -> PU-stratified CV -> evaluate) plus structured diagnostic reports and prior/propensity sensitivity analysis
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
import numpy as np
from pu_toolbox.preprocessing import make_scar_dataset
from pu_toolbox import PUPipeline

# Synthetic SCAR data (labeling independent of features — the premise of
# every class-prior estimator): some positives are labeled (1), the rest
# are unlabeled (0). For SAR data use make_sar_dataset(mechanism="linear").
X, y_pu, y_true = make_scar_dataset(
    n=500, c=0.5, n_features=8, separation=1.0, random_state=42,
)

# One call: profile -> class prior -> train -> PU-stratified CV -> evaluate
report = PUPipeline().fit_evaluate(X, y_pu, y_true=y_true)
print(report.summary())
```

Full docs (Chinese): [docs/README.md](docs/README.md). More runnable examples: [`examples/minimal/`](examples/minimal/).

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
| [`docs/user/concepts/`](docs/user/concepts/) | PU problem, SCAR/SAR, method selection |
| [`docs/user/howto/`](docs/user/howto/) | Task guides: simulation, profiling, pipeline, CLI, reports, sensitivity, distribution shift |
| [`docs/user/reference/api.md`](docs/user/reference/api.md) | Precise API contract |
| [`docs/dev/`](docs/dev/) | Contributor docs: architecture, structure, compatibility |
| [`docs/research/method_cards/`](docs/research/method_cards/) | Per-paper research cards |

### PU Survey experiments

Repository research experiments use dedicated scripts rather than the general
`PUPipeline` workflow. Prepare reproducible four-way splits and inspect the
Survey runner with:

```bash
uv run python scripts/prepare_survey_splits.py --help
uv run python scripts/run_survey_experiment.py --help

# SCAR row (PA + OA selection)
uv run python scripts/run_survey_experiment.py path/to/splits \
    --method lbe --c 0.1,0.3,0.5

# SAR pressure test (LBE-A/LBE-B labeling, OA only, c in {0.05, 0.5})
uv run python scripts/run_survey_experiment.py path/to/splits \
    --method lbe --labeling-mechanism sar_lbe_a --c 0.05,0.5
```

Methods that require the population class prior take `--class-prior` for the
gate plus their constructor arguments via `--model-params` (see
`--help` and the script docstring for a uPU example).

`--labeling-mechanism` picks how the PU label view is generated (SCAR or the
SAR LBE variants) and is orthogonal to `--method`: the mechanism is the
experiment's independent variable, so the SAR rows run any survey method and
report OA only. See [`docs/research/pu_survey/`](docs/research/pu_survey/) for
the protocol, current execution status, and reporting boundaries. Until all
protocol gates are satisfied, results must be identified as a
`pilot / partial benchmark`.

A pilot is several hundred runs, and the single-unit entry point stops at the
first failure — so `run_survey_pilot.py` drives the whole matrix instead. It
expands the protocol, asks the run manifests what is already finished, and
runs only what is left. A batch is one unit and mechanism, coalesced only when
the pending runs are that unit's complete seed-by-token grid: the unit script
runs the grid's cross product with no per-cell check of its own, so a
half-finished unit is split per seed rather than re-running cells that are
already on disk.

"Finished" is read from the manifest, not from the directory existing — a
directory is created before a run either succeeds or fails, and a run whose
candidates were *all* excluded writes a manifest too, with an empty selection
beside its failures. A run also only counts for the split it ran on, so
rebuilding the splits invalidates the runs made against the old ones.

Most methods need the population class prior, which protocol §3.1 defines as
data-generation metadata — the pool's positive rate before the stratified
split, not something to back-infer from a subset — so the driver reads it from
the splits that recorded it and refuses to start if it has neither that nor
`--class-prior`. A `--class-prior` that *contradicts* a recorded one is an
error, not a warning: the protocol fixes one constant per dataset, and nothing
downstream compares priors, so a typo would go unnoticed to the end of the
pilot. `--device` is passed through for the same reason: the unit script
defaults to CPU, and the image rows are not worth running there.

`--dry-run` prints the plan and the checkpoint storage the whole pilot implies
without running anything. Checkpoints accumulate, since nothing deletes them
and offline selection needs them afterwards, so that figure sizes a host
rather than describing one run. It is an upper bound for the image rows that
train an adapter head rather than the ResNet their row names — the protocol's
per-component constant does not distinguish them — so treat it as a ceiling to
plan against, not as the bytes that will land.

```bash
uv run python scripts/run_survey_pilot.py --dry-run
uv run python scripts/run_survey_pilot.py --device cuda
```

Aggregating a run tree forces the leaderboard-separation gates and reports
each comparability group separately; non-formal results are refused unless
`--diagnostic` asks for them:

```bash
uv run python scripts/aggregate_survey_runs.py results/survey
uv run python scripts/aggregate_survey_runs.py results/survey --diagnostic
```

The prepared splits are not distributed with the repository, so moving them to
another machine is a manual step with a tool at each end: `pack` writes one
deterministic archive per dataset plus an index describing every file in it,
and `verify` runs where the archives land and reports every disagreement
between the unpacked tree and the index, in both directions — a split the
index describes but the tree lacks, a split the tree holds but the index does
not, and whether the `.npz` indices still hash to the value each manifest
records for them. It reports rather than raising, so a damaged delivery yields
the whole list; corruption that predates packing is not among the things it
can catch, since `X` is pinned by file digest alone. Record the index's
archive digests in the repository before sending: a digest that travelled with
the bytes it describes proves the trip was faithful, not that what was sent was
right. Nothing runs `verify` automatically.

```bash
uv run python scripts/survey_splits_archive.py pack \
    --root data/splits --out-dir dist/p1.4-splits
uv run python scripts/survey_splits_archive.py verify \
    --root unpacked --index dist/p1.4-splits/split_artifacts_index.json \
    --archive-dir dist/p1.4-splits
```

## AI workflow skill

`pu-workflow` (Agent Skills open standard) drives the full PU analysis
workflow — profiling, assumption diagnosis, method recommendation,
training, and result interpretation — from natural language. Loaded
natively by Claude Code / Cursor (`.claude/skills/`) and Codex / Gemini
CLI / Windsurf (`.agents/skills/`). The skill ships inside the PyPI
wheel: `pip install "pu-toolbox>=1.2" && pu-toolbox skill install` —
see [How to enable and use the skill](docs/user/howto/using_skill.md).

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
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
uv run python scripts/check_baseline_configs.py    # baseline config vs source defaults
uv run python scripts/check_format.py        # ruff check + format --check (full scope)
uv run python scripts/generate_structure.py --check    # structure document consistency (--update to regenerate)
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## License

MIT
