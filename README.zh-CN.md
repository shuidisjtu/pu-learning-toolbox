[English](README.md) | [中文](README.zh-CN.md)

# PU Learning Toolbox

**正例-无标记学习 Python 工具箱** -- 兼容 sklearn API，15 篇论文方法，支持 SCAR 与 SAR。

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![Status](https://img.shields.io/badge/status-0.1.0--dev-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 特性

- **15 个算法**，来自近年 PU 学习研究论文，全部为 native 实现
- **兼容 sklearn API** -- `fit(X, y)` / `predict(X)` / `decision_function(X)`，支持 Pipeline 和交叉验证
- **SCAR & SAR 支持** -- 同时支持完全随机标记（SCAR）和随机标记（SAR）机制
- **数据画像** -- 自动检查数据质量，诊断标记机制
- **算法推荐** -- 根据数据特征自动匹配最适合的方法，支持自定义评分权重
- **诊断与敏感性分析** -- 结构化报告、假设敏感性分析、JSON/Markdown 导出

## 快速开始

> **注意**：尚未发布到 PyPI，需要从源码安装。

```bash
git clone https://github.com/shuidisjtu/pu-learning-toolbox.git
cd pu-learning-toolbox
pip install -e .          # 核心依赖
pip install -e ".[torch]" # + 基于 PyTorch 的方法（nnPU、Dist-PU、Self-PU 等）
```

### Hello World

```python
import numpy as np
from pu_toolbox.estimators.classic import ElkanNotoClassifier

# 合成 PU 数据：部分正例被标记（1），其余为无标记（0）
rng = np.random.RandomState(42)
X = rng.randn(500, 5)
y_true = (X[:, 0] + X[:, 1] > 0).astype(int)
y_pu = y_true * (rng.rand(500) < 0.5)  # SCAR 标记，c=0.5

# 训练与预测
clf = ElkanNotoClassifier(random_state=42)
clf.fit(X, y_pu)
predictions = clf.predict(X)
print(f"准确率: {np.mean(predictions == y_true):.3f}")
```

更多示例：[`examples/minimal/`](examples/minimal/)（10 个独立可运行脚本）。

## 支持的算法

### 类先验估计

| 方法 | 后端 | 成熟度 | 需要类先验 |
|------|------|--------|-----------|
| penL1 / KM1 / KM2 | numpy | stable | 否 |
| ReCPE | numpy | stable | 否 |

### 经典校准

| 方法 | 后端 | 成熟度 | 需要类先验 |
|------|------|--------|-----------|
| Elkan-Noto | sklearn | stable | 否 |

### 风险估计

| 方法 | 后端 | 成熟度 | 需要类先验 |
|------|------|--------|-----------|
| uPU | numpy | stable | 否 |
| nnPU | torch | stable | 是 |
| PNU | numpy | research | 是 |
| LDCE / KLDCE | numpy | research | 否 |
| LLSVM | numpy | research | 是 |
| Dist-PU | torch | research | 是 |

### 偏差感知（SAR）

| 方法 | 后端 | 成熟度 | 需要类先验 |
|------|------|--------|-----------|
| PUSB | sklearn | research | 否 |
| LBE | sklearn | research | 否 |

### 深度 PU

| 方法 | 后端 | 成熟度 | 需要类先验 |
|------|------|--------|-----------|
| Self-PU | torch | research | 是 |
| InfoMax PU | torch | research | 否 |
| WConPU | torch | research | 是 |
| DGPU | torch | experimental | 是 |

## 核心模块

### 算法推荐

```python
from pu_toolbox.advisor import recommend_methods, ScoringConfig

result = recommend_methods(X, y_pu, class_prior=0.3, has_gpu=True)
for c in result.candidates:
    print(f"{c.rank}. {c.name} (score={c.score:.1f})")

# 自定义评分权重
config = ScoringConfig(assumption_max=40.0, maturity_max=10.0)
result = recommend_methods(X, y_pu, config=config)
```

### 数据画像

```python
from pu_toolbox.preprocessing import profile_pu_data

report = profile_pu_data(X, y_pu, class_prior=0.3)
print(report.format_text())
# 检查：标签平衡、特征质量、SCAR/SAR 证据
```

### 诊断报告

```python
from pu_toolbox.diagnostics import build_diagnostic_report

report = build_diagnostic_report(X, y_pu, estimator=clf, class_prior=0.3)
print(report.to_markdown())  # 或 report.to_json()
```

### 敏感性分析

```python
from pu_toolbox.diagnostics import analyze_pu_sensitivity

analysis = analyze_pu_sensitivity(
    y_pu, clf.predict(X),
    class_priors=[0.2, 0.3, 0.4],
    label_propensities=[0.3, 0.5, 0.8],
)
print(analysis.to_frame())  # pandas DataFrame
```

## 文档

| 文档 | 说明 |
|------|------|
| [`docs/user/data_profiling.md`](docs/user/data_profiling.md) | 数据画像指南 |
| [`docs/user/diagnostic_reports.md`](docs/user/diagnostic_reports.md) | 诊断报告指南 |
| [`docs/user/sensitivity_analysis.md`](docs/user/sensitivity_analysis.md) | 敏感性分析指南 |
| [`docs/user/sar_simulation.md`](docs/user/sar_simulation.md) | SCAR/SAR 数据模拟 |
| [`docs/user/self_pu.md`](docs/user/self_pu.md) | Self-PU 使用指南 |
| [`docs/method_selection.md`](docs/method_selection.md) | 算法选择指南 |
| [`docs/research/method_cards/`](docs/research/method_cards/) | 各方法研究卡片 |
| [`docs/architecture.md`](docs/architecture.md) | 架构设计 |

## 开发

```bash
uv run pytest tests/ -v -m "not slow"       # 运行测试
uv run ruff check pu_toolbox/               # Lint 检查
uv run ruff format --check pu_toolbox/      # 格式检查

# 质量门禁
uv run python scripts/check_test_quality.py
uv run python scripts/check_doc_links.py
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
```

贡献指南见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 许可证

MIT
