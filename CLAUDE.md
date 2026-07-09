# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

PU Learning Toolbox — 面向 Positive-Unlabeled Learning 的 Python 工具箱。在只有"已标记正样本 + 无标记样本"的条件下，提供数据校验、类先验估计、模型训练、评估、算法选择与论文复现。

目标：sklearn-compatible API、可扩展框架、尊重论文作者源码、差异化支持 SAR/Instance-Dependent PU。

## 当前状态

项目处于 **Phase 0（规划阶段）**，尚无代码。`docs/` 中有完整的设计文档，是开发的权威参考。所有模块目录、API 契约、元数据 schema 已在文档中定义，尚未创建实际文件。

## 开发命令

项目尚未初始化，以下为规划中的命令：

```bash
# 安装（计划）
pip install -e .                # 核心安装
pip install -e ".[torch]"       # + PyTorch 扩展
pip install -e ".[research]"    # + 研究扩展

# 测试
pytest tests/ -v
pytest tests/test_validation.py -v

# Lint / Format
ruff check pu_toolbox/
ruff format pu_toolbox/

# 构建
python -m build
```

## 核心架构

### 分层依赖
| 层 | 依赖 | 内容 |
|---|---|---|
| Core | numpy, scipy, pandas, sklearn | 标签处理、数据校验、经典算法、类先验、指标、CV、registry |
| Torch Extension | torch | uPU, nnPU, Dist-PU, 深度表征学习 |
| Research Extension | torchvision, lightning | 图像 benchmark、生成式模型、论文复现 |

Core 必须保持轻量，深度学习依赖放入 optional extension。

### 模块分层
```
Core (core, preprocessing, registry)     → 稳定 API、标签规范、校验、注册
Estimation (prior, propensity, losses)   → 类先验估计、标记倾向、PU 损失
Algorithms (estimators)                  → 具体 PU 分类器
Source Integration (source_adapters)     → 作者源码 adapter
Evaluation (metrics, model_selection)    → 评估、诊断、切分
User Layer (advisor, examples, docs)     → 推荐、报告、教程
```

### 关键基类
- `BasePUClassifier(BaseEstimator, ClassifierMixin)` — 所有 PU 分类器的基类
- `BasePriorEstimator(BaseEstimator)` — 类先验估计器
- `BasePropensityEstimator(BaseEstimator)` — 标记倾向估计器
- `BasePULoss` — PU 损失函数
- `BaseSourceAdapter` — 论文作者源码包装器

### 输出接口
所有分类器必须实现 `fit(X, y_pu)` + `predict(X)` + `decision_function(X)` 或 `score_samples(X)`。`predict_proba(X)` 可选。必须支持 `get_params()`/`set_params()`（兼容 sklearn Pipeline/GridSearchCV）。

## 开发原则

### Framework-First
先定义稳定、可测试的 API 契约与框架，用 mock estimator 跑通 registry、advisor、metrics、benchmark，再逐个集成论文算法。

### 源码集成优先级
```
official_exact 作者源码 → adapter 优先
official_bundle 官方工具包 → 参考实现
official_related 作者相关实现 → 参考
third_party_only → 仅参考接口
not_found → clean-room reimplementation
```
有作者源码的算法优先做 adapter，不急于重写。adapter 不改变统一 estimator API。

### implementation_status 枚举
- `api_only` — 仅 API 占位，无训练逻辑（MVP 中未实现方法的默认状态）
- `native` — Toolbox 内部 clean-room 实现
- `official_adapter` — 通过 adapter 调作者源码
- `official_aligned_native` — 参考官方实现后原生实现，有对齐测试
- `experimental` — 研究版，API 可能变化

### 注册表
所有算法必须通过 registry 注册元数据（family, scenario, assumption, requires_class_prior, backend, maturity, implementation_status, source_status）。advisor 依赖元数据推荐算法，不直接依赖具体实现。

## 数据场景与假设

| 场景 | 含义 |
|---|---|
| Single-training-set | 从总体采样一批数据，部分正样本打标 |
| Case-control | 正样本与无标记集合分别采样 |
| Selection-biased PU | 已标记正样本不代表全部正样本 |

| 假设 | 含义 | P(s=1|y=1) |
|---|---|---|
| SCAR | 标记与 x 无关 | 常数 c |
| SAR / Instance-Dependent | 标记依赖 x | c(x) |

SAR / Instance-Dependent PU 与 source adapter 体系是本项目的中长期差异化重点。

## 15 篇论文覆盖

按 `docs/resources_optimized.md`：8 篇有 `official_exact` 源码、3 篇有 `official_bundle`/`official_related`、1 篇仅第三方、3 篇 `not_found`。

算法族：Class-Prior Estimation、Classic & Calibration、Risk Estimation、Bias-Aware PU、Deep PU。

## 版本路线

```
0.1.0  MVP — 经典 PU 模型（Elkan-Noto, PU Bagging, Biased SVM, Weighted LR）+ 类先验 + registry 占位
0.2.0  uPU / nnPU / PNU 风险模块
0.3.0  官方源码 adapter + paper-like benchmark
0.4.0  推荐器 + 诊断报告
0.5.0  SAR / selection-biased PU
0.6.0  Self-PU, Dist-PU
1.0.0  API 稳定
```

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/README.md` | 项目总览、定位、使用示例 |
| `docs/architecture.md` | 包结构、基类 API、数据流、adapter 设计 |
| `docs/project_structure.md` | 完整目录结构（权威来源） |
| `docs/method_selection.md` | 算法分类、选型逻辑、推荐器设计 |
| `docs/development_roadmap.md` | Phase 0–6 任务拆分、MVP 清单 |
| `docs/resources_optimized.md` | 论文源码状态、URL、集成策略 |
