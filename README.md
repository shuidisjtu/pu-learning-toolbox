[English](README.md) | [中文](README.zh-CN.md)

# PU Learning Toolbox

**Positive-Unlabeled learning in Python** -- sklearn-compatible API, 15 research paper methods, SCAR & SAR support.

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![Status](https://img.shields.io/badge/status-0.1.0--dev-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **15 algorithms** from recent PU learning research, all with native implementations
- **sklearn-compatible API** -- `fit(X, y)` / `predict(X)` / `decision_function(X)`, works with pipelines and cross-validation
- **SCAR & SAR support** -- both Selected Completely At Random and Selected At Random labeling mechanisms
- **Data profiling** -- automatic data quality checks and labeling mechanism diagnostics
- **Algorithm recommender** -- match your data profile to the best-suited methods with customizable scoring
- **Diagnostics & sensitivity** -- structured reports, assumption sensitivity analysis, JSON/Markdown export

## Quick Start

> **Note**: Not yet published on PyPI. Install from source.

```bash
git clone https://github.com/shuidisjtu/pu-learning-toolbox.git
cd pu-learning-toolbox
pip install -e .          # core dependencies
pip install -e ".[torch]" # + PyTorch-based methods (nnPU, Dist-PU, Self-PU, etc.)
```

### Hello World

```python
import numpy as np
from pu_toolbox.estimators.classic import ElkanNotoClassifier

# Synthetic PU data: some positives are labeled (1), rest are unlabeled (0)
rng = np.random.RandomState(42)
X = rng.randn(500, 5)
y_true = (X[:, 0] + X[:, 1] > 0).astype(int)
y_pu = y_true * (rng.rand(500) < 0.5)  # SCAR labeling with c=0.5

# Train and predict
clf = ElkanNotoClassifier(random_state=42)
clf.fit(X, y_pu)
predictions = clf.predict(X)
print(f"Accuracy: {np.mean(predictions == y_true):.3f}")
```

More examples: [`examples/minimal/`](examples/minimal/) (10 self-contained scripts).

## Supported Algorithms

### Class Prior Estimation

| Method | Backend | Maturity | Requires class prior |
|--------|---------|----------|---------------------|
| penL1 / KM1 / KM2 | numpy | stable | No |
| ReCPE | numpy | stable | No |

### Classic & Calibration

| Method | Backend | Maturity | Requires class prior |
|--------|---------|----------|---------------------|
| Elkan-Noto | sklearn | stable | No |

### Risk Estimation

| Method | Backend | Maturity | Requires class prior |
|--------|---------|----------|---------------------|
| uPU | numpy | stable | No |
| nnPU | torch | stable | Yes |
| PNU | numpy | research | Yes |
| LDCE / KLDCE | numpy | research | No |
| LLSVM | numpy | research | Yes |
| Dist-PU | torch | research | Yes |

### Bias-Aware (SAR)

| Method | Backend | Maturity | Requires class prior |
|--------|---------|----------|---------------------|
| PUSB | sklearn | research | No |
| LBE | sklearn | research | No |

### Deep PU

| Method | Backend | Maturity | Requires class prior |
|--------|---------|----------|---------------------|
| Self-PU | torch | research | Yes |
| InfoMax PU | torch | research | No |
| WConPU | torch | research | Yes |
| DGPU | torch | experimental | Yes |

## Key Modules

### Algorithm Recommender

```python
from pu_toolbox.advisor import recommend_methods, ScoringConfig

result = recommend_methods(X, y_pu, class_prior=0.3, has_gpu=True)
for c in result.candidates:
    print(f"{c.rank}. {c.name} (score={c.score:.1f})")

# Custom scoring weights
config = ScoringConfig(assumption_max=40.0, maturity_max=10.0)
result = recommend_methods(X, y_pu, config=config)
```

### Data Profiling

```python
from pu_toolbox.preprocessing import profile_pu_data

report = profile_pu_data(X, y_pu, class_prior=0.3)
print(report.format_text())
# Checks: label balance, feature quality, SCAR/SAR evidence
```

### Diagnostic Report

```python
from pu_toolbox.diagnostics import build_diagnostic_report

report = build_diagnostic_report(X, y_pu, estimator=clf, class_prior=0.3)
print(report.to_markdown())  # or report.to_json()
```

### Sensitivity Analysis

```python
from pu_toolbox.diagnostics import analyze_pu_sensitivity

analysis = analyze_pu_sensitivity(
    y_pu, clf.predict(X),
    class_priors=[0.2, 0.3, 0.4],
    label_propensities=[0.3, 0.5, 0.8],
)
print(analysis.to_frame())  # pandas DataFrame
```

### End-to-End Pipeline

```python
from pu_toolbox import PUPipeline

# One call: profile -> class prior -> train -> PU-stratified CV -> evaluate
pipe = PUPipeline()                        # classifier="auto" picks the method
report = pipe.fit_evaluate(X, y_pu)        # y_pu: {1, 0} PU labels

print(report.summary())                    # metric table + issues
report.save("results/pipeline.json")       # strict JSON / Markdown export
```

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/user/data_profiling.md`](docs/user/data_profiling.md) | Data profiling guide |
| [`docs/user/diagnostic_reports.md`](docs/user/diagnostic_reports.md) | Diagnostic report guide |
| [`docs/user/sensitivity_analysis.md`](docs/user/sensitivity_analysis.md) | Sensitivity analysis guide |
| [`docs/user/sar_simulation.md`](docs/user/sar_simulation.md) | SCAR/SAR data simulation |
| [`docs/user/self_pu.md`](docs/user/self_pu.md) | Self-PU usage guide |
| [`docs/method_selection.md`](docs/method_selection.md) | Algorithm selection guide |
| [`docs/research/method_cards/`](docs/research/method_cards/) | Per-method research cards |
| [`docs/architecture.md`](docs/architecture.md) | Architecture design |

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
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## License

MIT
