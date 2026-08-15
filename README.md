[English](README.md) | [中文](README.zh-CN.md)

# PU Learning Toolbox

**Positive-Unlabeled learning in Python** -- sklearn-compatible API, 17 research paper methods, SCAR & SAR support.

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![Status](https://img.shields.io/badge/status-1.5.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **17 algorithms** from recent PU learning research, all native clean-room implementations ([method cards](docs/research/method_cards/))
- **sklearn-compatible API** -- `fit(X, y)` / `predict(X)` / `decision_function(X)`, works with pipelines and cross-validation
- **SCAR & SAR** -- constant and instance-dependent labeling mechanisms, with a data simulator
- **Data profiling + recommender** -- automatic quality checks, SCAR/SAR evidence, and a 7-dimension scoring recommender that picks the method for your data
- **Auditable pipeline** -- one-call `PUPipeline` (profile -> prior -> train -> PU-stratified CV -> evaluate) plus structured diagnostic reports and prior/propensity sensitivity analysis
- **CLI** -- `pu-toolbox` turns the whole pipeline into terminal commands
- **Model tuning** -- unified classifier parameters plus PU-aware grid search
- **Graphical UI** -- upload data, configure/compare models, inspect diagnostics, and download results

## Quick Start

```bash
pip install pu-toolbox                # core dependencies (Python >= 3.10)
pip install "pu-toolbox[torch]"       # + PyTorch-based methods (nnPU, Dist-PU, Self-PU, ...)
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
| [`docs/user/howto/`](docs/user/howto/) | Task guides: simulation, profiling, pipeline, CLI, reports, sensitivity |
| [`docs/user/reference/api.md`](docs/user/reference/api.md) | Precise API contract |
| [`docs/dev/`](docs/dev/) | Contributor docs: architecture, structure, compatibility |
| [`docs/research/method_cards/`](docs/research/method_cards/) | Per-paper research cards |

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
uv run python scripts/check_format.py        # ruff check + format --check (full scope)
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## License

MIT
