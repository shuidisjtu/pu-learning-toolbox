# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

PU Learning Toolbox — Positive-Unlabeled Learning Python 工具箱。在只有"已标记正样本 + 无标记样本"的条件下，提供数据校验、类先验估计、模型训练、评估、算法选择与论文复现。目标：sklearn-compatible API、可扩展框架、尊重论文作者源码、差异化支持 SAR/Instance-Dependent PU。

## 当前状态与下一步

**Phase 0（规划阶段）— 尚无 Python 代码。** `docs/` 中有完整设计文档，是开发的权威参考。所有模块目录、API 契约、元数据 schema 已在文档中定义。

### 已就绪
- 设计文档完备（`docs/` 下 6 份核心文档）
- `.gitignore` 已配置
- 全局 `E:\Project\CLAUDE.md` 已配置（环境、MCP、工作流规则）

### Phase 0 待完成任务（按优先级）

1. **创建 `pyproject.toml`** — 项目元数据、core/torch/research 分层依赖、`[project.optional-dependencies]`
2. **创建 `pu_toolbox/` 包骨架** — 所有模块目录 + `__init__.py`（目录结构以 `docs/project_structure.md` 为权威来源）
3. **实现 Core 基类** — `core/base.py`（BasePUClassifier, BasePriorEstimator, BasePropensityEstimator, BasePULoss）、`core/labels.py`（normalize_pu_labels）、`core/validation.py`（validate_pu_X_y）、`core/exceptions.py`
4. **实现 Registry 系统** — `registry/registry.py`（get_algorithm, register_method）、`registry/metadata.py`（AlgorithmMetadata）、`registry/source_metadata.py`（SourcePolicy）
5. **实现 SourceAdapter 基类** — `source_adapters/base.py`（BaseSourceAdapter, is_available, build_estimator）
6. **搭建测试框架** — `tests/` + pytest 配置 + `tests/test_import.py`
7. **为 15 篇论文方法注册 `api_only` metadata** — 占位，无训练逻辑

> **Phase 0 验收标准**：项目可 `pip install -e .`、core 包可 import、pytest 可运行、所有基类 API 清晰、registry 可注册 `api_only` 算法。

### Phase 0→1 策略

Framework-first：先完成框架与 API 契约，用 mock estimator 跑通 registry → advisor → metrics 链路，再逐个集成论文算法。有作者源码的方法优先做 adapter，不急于重写。

## 环境

全局环境配置见 `E:\Project\CLAUDE.md`。本项目关键约束：

- Python 包管理: `uv`（默认），Anaconda base (`E:\Anaconda2025`, Python 3.13) 备选
- Bash 中必须用完整路径调用 python（Git mingw64 PATH 不含 Anaconda）
- 项目初始化后，Python 代码中 Windows 路径用原始字符串 `r'...'` 或正斜杠，禁止裸反斜杠
- 网络：查询库文档优先 Context7；网页搜索用 Bing（MCP open-websearch）

## 开发命令

项目尚未初始化，以下命令为 Phase 0 设计规范，实施时以此为蓝本。

```bash
# === Phase 0 初始化（按顺序执行，尚未运行） ===
# 1. 创建 pyproject.toml（参考 docs/development_roadmap.md 和下方依赖分层）
# 2. 创建包目录结构（参考 docs/project_structure.md）
# 3. 初始化环境
uv venv
uv pip install -e ".[dev]"

# === 日常开发 ===
pytest tests/ -v                          # 全部测试
pytest tests/test_validation.py -v        # 单文件测试
pytest tests/ -v -k "test_normalize"      # 按名称筛选

ruff check pu_toolbox/                    # Lint
ruff format pu_toolbox/                   # Format

python -m build                           # 构建分发包

# === 分层安装 ===
pip install -e .                  # Core only (numpy, scipy, pandas, sklearn)
pip install -e ".[torch]"         # + PyTorch
pip install -e ".[research]"      # + torchvision, lightning
```

### pyproject.toml 依赖设计（Phase 0 创建时参考）

| Extra | 依赖 |
|---|---|
| `dev` | pytest, pytest-cov, ruff |
| `torch` | torch >= 2.0 |
| `research` | torchvision, lightning, tqdm |
| `all` | 以上全部 |

## 核心架构

### 分层依赖（自下而上）

| 层 | 依赖 | 内容 |
|---|---|---|
| Core | numpy, scipy, pandas, sklearn | 标签处理、校验、经典算法、类先验、指标、CV、registry |
| Torch Extension | torch | uPU, nnPU, Dist-PU, 深度表征学习 |
| Research Extension | torchvision, lightning | 图像 benchmark、生成式模型、论文复现 |

Core 必须轻量；深度学习依赖放入 optional extension。

### 模块分层

```
Core (core, preprocessing, registry)     → 稳定 API、标签规范、校验、注册
Estimation (prior, propensity, losses)   → 类先验、标记倾向、PU 损失
Algorithms (estimators)                  → 具体 PU 分类器（classic/risk/bias_aware/deep）
Source Integration (source_adapters)     → 作者源码 adapter
Evaluation (metrics, model_selection)    → 评估、诊断、切分
User Layer (advisor, examples, docs)     → 推荐、报告、教程
```

完整目录结构以 `docs/project_structure.md` 为权威来源。

### 关键基类与 API 契约

- `BasePUClassifier(BaseEstimator, ClassifierMixin)` — 所有 PU 分类器基类
- `BasePriorEstimator(BaseEstimator)` — 类先验估计器
- `BasePropensityEstimator(BaseEstimator)` — 标记倾向估计器
- `BasePULoss` — PU 损失函数
- `BaseSourceAdapter` — 论文作者源码包装器

所有分类器必须: `fit(X, y_pu)` + `predict(X)` + `decision_function(X)` 或 `score_samples(X)`。`predict_proba(X)` 可选。必须支持 `get_params()`/`set_params()`（兼容 sklearn Pipeline/GridSearchCV）。

### 算法注册与元数据

所有算法通过 registry 注册元数据（`family`, `scenario`, `assumption`, `requires_class_prior`, `backend`, `maturity`, `implementation_status`, `source_status`）。advisor 依赖元数据推荐算法，不直接依赖具体实现。

`implementation_status` 枚举: `api_only`（占位，无训练逻辑）| `native`（clean-room 实现）| `official_adapter`（调作者源码）| `official_aligned_native`（参考官方做原生实现+对齐测试）| `experimental`

## 开发原则

### Framework-First
先定义稳定 API 契约与框架，用 mock estimator 跑通 registry、advisor、metrics、benchmark，再逐个集成论文算法。

### 源码集成优先级
```
official_exact → adapter 优先
official_bundle → 参考实现
official_related → 参考
third_party_only → 仅参考接口
not_found → clean-room reimplementation
```
有作者源码的算法优先做 adapter，不急于重写。adapter 不改变统一 estimator API。

### SAR / Instance-Dependent PU
本项目中长期差异化重点。与 source adapter 体系共同构成核心竞争力。

## 数据场景与假设

| 场景 | 含义 |
|---|---|
| Single-training-set | 从总体采样一批数据，部分正样本打标 |
| Case-control | 正样本与无标记集合分别采样 |
| Selection-biased PU | 已标记正样本不代表全部正样本 |

| 假设 | 含义 | P(s=1\|y=1) |
|---|---|---|
| SCAR | 标记与 x 无关 | 常数 c |
| SAR / Instance-Dependent | 标记依赖 x | c(x) |

## 15 篇论文覆盖

按 `docs/resources_optimized.md`：8 篇 `official_exact`、3 篇 `official_bundle`/`official_related`、1 篇仅第三方、3 篇 `not_found`。算法族：Class-Prior Estimation、Classic & Calibration、Risk Estimation、Bias-Aware PU、Deep PU。优先实现优先级和版本路线见 `docs/development_roadmap.md`。

## 版本路线

```
0.1.0  MVP — 经典 PU + 类先验 + registry 占位
0.2.0  uPU / nnPU / PNU 风险模块
0.3.0  官方源码 adapter + paper-like benchmark
0.4.0  推荐器 + 诊断报告
0.5.0  SAR / selection-biased PU
0.6.0  Self-PU, Dist-PU
1.0.0  API 稳定
```

## 文档索引

| 文档 | 内容 | 何时查阅 |
|---|---|---|
| `docs/README.md` | 项目总览、定位、使用示例 | 了解项目全貌 |
| `docs/architecture.md` | 包结构、基类 API、数据流、adapter 设计 | 实现基类、定义接口 |
| `docs/project_structure.md` | 完整目录结构（权威来源） | 创建目录/文件 |
| `docs/method_selection.md` | 算法分类、选型逻辑、推荐器设计 | 实现 advisor |
| `docs/development_roadmap.md` | Phase 0–6 任务拆分、MVP 清单 | 规划工作 |
| `docs/resources_optimized.md` | 论文源码状态、URL、集成策略 | 集成论文方法 |

## 关键约束速查

- **Windows 路径**: Python 中必须 `r'E:\...'` 或 `'E:/...'`，禁止裸反斜杠
- **Python 调用**: Bash 中必须全路径（如 `E:\Anaconda2025\python.exe`），不可直接写 `python`
- **包管理**: 默认 `uv`，备选 Anaconda base
- **测试**: `pytest tests/ -v`，单文件 `pytest tests/test_xxx.py -v`，筛选 `-k "pattern"`
- **Lint**: `ruff check pu_toolbox/`，Format: `ruff format pu_toolbox/`
- **API 契约**: 所有分类器必须 `fit(X, y_pu)` + `predict(X)` + `decision_function(X)` 或 `score_samples(X)` + `get_params()`/`set_params()`
- **实现状态**: 未实现的算法注册为 `api_only`（仅占位，无训练逻辑），不可假装可用
- **源码优先**: 有 `official_exact` 源码的论文 → adapter 优先，不急于重写
