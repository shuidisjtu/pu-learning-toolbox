[English](README.md) | [中文](README.zh-CN.md)

# PU Learning Toolbox

**Positive-Unlabeled learning in Python** -- sklearn-compatible API, 16 research paper methods, SCAR & SAR support.

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![Status](https://img.shields.io/badge/status-0.1.0--dev-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **16 algorithms** from recent PU learning research, all native clean-room implementations ([method cards](docs/research/method_cards/))
- **sklearn-compatible API** -- `fit(X, y)` / `predict(X)` / `decision_function(X)`, works with pipelines and cross-validation
- **SCAR & SAR** -- constant and instance-dependent labeling mechanisms, with a data simulator
- **Data profiling + recommender** -- automatic quality checks, SCAR/SAR evidence, and a 7-dimension scoring recommender that picks the method for your data
- **Auditable pipeline** -- one-call `PUPipeline` (profile -> prior -> train -> PU-stratified CV -> evaluate) plus structured diagnostic reports and prior/propensity sensitivity analysis
- **CLI** -- `pu-toolbox` turns the whole pipeline into terminal commands

## Quick Start

> **Note**: Not yet published on PyPI. Install from source.

```bash
git clone https://github.com/shuidisjtu/pu-learning-toolbox.git
cd pu-learning-toolbox
pip install -e .          # core dependencies
pip install -e ".[torch]" # + PyTorch-based methods (nnPU, Dist-PU, Self-PU, ...)
```

### Hello World

```python
import numpy as np
from pu_toolbox.preprocessing import make_sar_dataset
from pu_toolbox import PUPipeline

# Synthetic PU data: some positives are labeled (1), rest are unlabeled (0)
X, y_pu, y_true, _ = make_sar_dataset(
    n_samples=1000, n_features=8, class_prior=0.3, random_state=42,
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

## Documentation

Docs are split by audience; the full index is [`docs/README.md`](docs/README.md).

| Entry | Content |
|----------|---------|
| [`docs/user/quickstart.md`](docs/user/quickstart.md) | 5-minute start (CLI + Python) |
| [`docs/user/concepts/`](docs/user/concepts/) | PU problem, SCAR/SAR, method selection |
| [`docs/user/howto/`](docs/user/howto/) | Task guides: simulation, profiling, pipeline, CLI, reports, sensitivity |
| [`docs/user/reference/api.md`](docs/user/reference/api.md) | Precise API contract |
| [`docs/dev/`](docs/dev/) | Contributor docs: architecture, structure, roadmap, compatibility |
| [`docs/research/method_cards/`](docs/research/method_cards/) | Per-paper research cards |

## Development

```bash
uv run pytest tests/ -v -m "not slow"       # run tests
uv run ruff check pu_toolbox/               # lint
uv run ruff format --check pu_toolbox/      # format check

# Quality gates
uv run python scripts/check_test_quality.py
uv run python scripts/check_doc_links.py
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## License

MIT
