# 架构设计

## 1. 设计决策

设计决策与代价已迁移至 [docs/adr/](../adr/README.md)。本文档只描述当前架构。

**与 `project_structure.md` 的分工**：本文档解释"为什么这样组织"（模块划分、
依赖方向、数据流、算法推荐与注册约定）；文件清单与目录结构以 [`project_structure.md`](project_structure.md) 为权威来源。
**API 契约**：类的精确签名与方法语义以 [`user/reference/api.md`](../user/reference/api.md) 为权威来源，本文档不重复。

## 2. 模块分层

> 分层为概念归类；"模块"列给出对应 `pu_toolbox/` 子包；职责为模块级边界。
> 文件级职能（每个文件的核心功能与公开入口）见 [`project_structure.md`](project_structure.md)
> 目录树旁注——本表不重复，避免真相源分裂。

### Core — 地基层：无上层依赖，被其余层引用

| 模块 | 核心职责 | 详情来源 |
|---|---|---|
| `core/` | PU 基类、标签语义规范、输入校验与规范化、设备/随机源统一、异常与 tags 语义 | [`../../pu_toolbox/core/__init__.py`](../../pu_toolbox/core/__init__.py) |
| `preprocessing/` | SCAR/SAR 标签与数据生成、结构化数据画像 | [`../../pu_toolbox/preprocessing/__init__.py`](../../pu_toolbox/preprocessing/__init__.py)、[`画像指南`](../user/howto/data_profiling.md) |
| `registry/` | 算法注册、元数据与内置方法发现 | [`../../pu_toolbox/registry/__init__.py`](../../pu_toolbox/registry/__init__.py)、[`内置方法表`](../../pu_toolbox/registry/builtin_methods.py) |
| `advisor/` | 数据画像驱动的算法推荐 | [`../../pu_toolbox/advisor/__init__.py`](../../pu_toolbox/advisor/__init__.py)、[`选型原理`](../user/concepts/method_selection.md) |
| `utils/` | 共享工具（非公开 API，可随小版本变化） | [`../../pu_toolbox/utils/__init__.py`](../../pu_toolbox/utils/__init__.py) |

### Estimation — 类先验与 PU 损失，被 Algorithms 使用

| 模块 | 核心职责 | 详情来源 |
|---|---|---|
| `prior/` | 类先验估计器（ReCPE / penL1 / KM） | [`../../pu_toolbox/prior/__init__.py`](../../pu_toolbox/prior/__init__.py)、[`先验方法卡`](../research/method_cards/class_prior_estimation.md) |
| `losses/` | PU 风险/损失函数 | [`../../pu_toolbox/losses/__init__.py`](../../pu_toolbox/losses/__init__.py) |

### Algorithms — 具体 PU 分类器实现

| 模块 | 核心职责 | 详情来源 |
|---|---|---|
| `estimators/` | 全部 PU 分类器，按 `classic/`、`risk/`、`bias_aware/`、`deep/`、`research/` 分包 | [`../../pu_toolbox/estimators/__init__.py`](../../pu_toolbox/estimators/__init__.py) |

算法↔模块落点与实现状态（NATIVE / api_only / `official_exact` adapter）
以注册表[`内置方法表`](../../pu_toolbox/registry/builtin_methods.py)（`_BUILTINS`）
为真相源；各算法原理见[`方法卡`](../research/method_cards/)。

### Evaluation — 结果评估与诊断，被 Orchestration 调用

| 模块 | 核心职责 | 详情来源 |
|---|---|---|
| `metrics/` | PU 评估指标（PU-only + 标准监督指标包装） | [`../../pu_toolbox/metrics/__init__.py`](../../pu_toolbox/metrics/__init__.py)、[`指标契约`](../research/traditional_pu_metric_contract.md) |
| `model_selection/` | PU 分层切分、模型调优与比较 | [`../../pu_toolbox/model_selection/__init__.py`](../../pu_toolbox/model_selection/__init__.py)、[`调优指南`](../user/howto/model_tuning.md) |
| `diagnostics/` | 结构化诊断报告与假设敏感性分析 | [`../../pu_toolbox/diagnostics/__init__.py`](../../pu_toolbox/diagnostics/__init__.py)、[`报告指南`](../user/howto/diagnostic_reports.md) |

### Orchestration — 端到端编排与 CLI

| 模块 | 核心职责 | 详情来源 |
|---|---|---|
| `workflows/` | PUPipeline 端到端编排与漂移感知工作流 | [`../../pu_toolbox/workflows/__init__.py`](../../pu_toolbox/workflows/__init__.py)、[`流水线指南`](../user/howto/pipeline.md)、[`漂移指南`](../user/howto/distribution_shift.md) |
| `cli/` | 命令行薄封装（子命令一览） | [`命令行指南`](../user/howto/cli.md)、[`../../pu_toolbox/cli/__init__.py`](../../pu_toolbox/cli/__init__.py) |

### User Layer — 用户入口（教学 / 图形界面 / 复现工具）与 agent 包装

| 模块 | 核心职责 | 详情来源 |
|---|---|---|
| `examples/` | 教程与最小示例 | [`../../examples/`](../../examples/) |
| `ui/` | 图形界面（可选 ext，核心安装不导入 streamlit） | [`../../pu_toolbox/ui/__init__.py`](../../pu_toolbox/ui/__init__.py)、[`图形界面指南`](../user/howto/ui.md) |
| `benchmarks/` | 论文复现基准（runner / 官方数据 / 产物管理；仓库根，官方数据由执行方提供） | [`../../benchmarks/`](../../benchmarks/) |
| `scripts/pu_workflow/` | 兼容包装（委托 CLI 子命令） | [`../../scripts/pu_workflow/`](../../scripts/pu_workflow/) |
| pu-workflow skill | agent 端到端流程（触发词驱动，内部走 CLI） | [`../../.claude/skills/pu-workflow/SKILL.md`](../../.claude/skills/pu-workflow/SKILL.md) |

### 2.1 模块依赖关系

**层间调用方向**（指向被依赖方；基座层被所有上层引用，不允许反向）：

| 层 | 使用/调用 |
|---|---|
| Orchestration（`workflows/`、`cli/`） | Evaluation、Algorithms、Estimation、Core |
| Algorithms（`estimators/`） | Estimation、Core |
| Estimation（`prior/`、`losses/`） | Core |
| Evaluation（`metrics/`、`model_selection/`、`diagnostics/`） | —（指标与切分为无层内依赖的纯计算，供编排层调用） |
| User Layer（`ui/`、`benchmarks/`） | Orchestration（经 CLI / workflows 消费工具箱能力） |

**模块级依赖链**（代表性，全部为单向防环设计）：

- **标签语义链**：`core/labels.py`（纯元语：标签约定识别与重映射）→ `core/validation.py`（组装层：标签规范化 + X/y 一致性 + 样本量门槛与告警，返回值已是规范形）→ 各估计器 `fit` 入口
- **数据画像链**：`preprocessing/profiling.py`（统计元语：`pu_data_summary`/`pnu_data_summary`/`scar_diagnostic`，向后兼容）→ `preprocessing/data_profiler.py`（聚合编排：`PUDataProfile` + 可行动 issues）→ `workflows`（pipeline 首步）/ `diagnostics`（报告）/ `advisor`（推荐）
- **字段语义**：`core/tags.py` 是 registry 元数据字段与枚举的权威来源，registry/advisor 均以其为准
- **设备与随机源入口**：`core/device.py` 的 `resolve_device`、`core/random.py` 的 `check_random_state` 是全工具箱唯一的设备/seed 归一化入口，避免各调用点语义漂移

> 分层为代表性概览，细粒度依赖以 [`project_structure.md`](project_structure.md)
> 目录树为准。

## 3. 数据流

一条主链：`PUPipeline.fit_evaluate` 从数据到报告；方括号为对应模块
（与 §2 分层表术语一致），后续文字只解释图中说不清的部分。

```text
用户输入 (X, y_pu[, y_true])
  ↓ 校验 + 标签规范化      [core/validation + labels]
数据画像 profile            [preprocessing/data_profiler]
  ↓ 画像 issues 含 error → fail-fast 停止
推荐器选方法（auto 默认路径） [advisor/recommend_from_profile]
  ↓ 显式 classifier 时跳过；推荐理由写入 provenance.classifier
类先验估计（按需）         [prior/*]
  ↑ 仅 requires_class_prior=True 的算法执行；不需要 π 的跳过
  （如 PUSB/LBE/Elkan-Noto/InfoMax PU）
训练（含 CV 切分）         [estimators/* + model_selection/split]
  ↓ 输出 predict / decision_function / predict_proba
评估 + 诊断 → 报告         [metrics + diagnostics → workflows/report]
```

- **输入契约**：`X` / `y_pu` 是用户整理好的 PU 数据——抽样（选择哪些样本）与
  标签标记（哪些正例被标注）由研究者在工具箱之外完成，工具箱将其视为已给输入；
  数据模拟器（`make_sar_*`/`make_scar_labels`/demo）只用于合成研究，不替代真实采样。
  可选的 `y_true` 仅用于监督指标与审计，不参与训练。
- **画像语义**：`PUDataProfile` 含基础统计、特征质量、问题级别、行动建议和标记机制证据。
  无审计 `y_true` 时 SCAR/SAR 提示明确标记为非识别性筛查；提供 `y_true` 时仅在真实
  正例内部评估 selection dependence，避免把类别可分性误认为 SAR。
- **双域路径**（独立于单域入口）：`analyze_pu_shift` 用 OOF 域分类器估计可观测边际
  漂移和相对密度比；`ShiftAwarePUPipeline` 在覆盖门禁通过后把源域权重逐折传给
  `PUPipeline`。该路径保证固定为 covariate-shift-only，不把边际权重描述为联合
  `p_target(x,y)/p_source(x,y)` 适配。
- **报告组装**：`build_diagnostic_report`（`diagnostics`）只读画像、已拟合 estimator
  和指标接口，不训练模型；将观测 PU、类先验依赖、监督 oracle 和不可用指标分别标记，
  输出稳定 schema 的 JSON/Markdown 报告。

## 4. 算法注册与推荐

每个算法在 `registry` 注册元信息（name/aliases/family/scenario/assumption/
requires_class_prior/backend/maturity/source_status/implementation_status 与 4 个
架构能力字段 native_architectures / input_ndims / encoder_parameter /
trains_encoder）；字段语义与枚举以 `pu_toolbox/core/tags.py` 为权威，内置方法与
算法↔模块落点、实现状态见 `pu_toolbox/registry/builtin_methods.py`。能力字段以
估算器类属性为权威、注册时经 `_SYNC_FIELDS` 镜像进 registry（语义与消费点见
`dual_architecture_plan.md` §3-§4）。

`advisor` 把数据画像与 registry 元数据匹配后推荐方法：硬过滤（trainable、
scenario、sparse、class_prior 可用性）→ 软评分（assumption 匹配/成熟度/可信度/
规模/成本/GPU/标记充足度）→ 风险提示；权重外化为可定制的 `ScoringConfig`。
推荐结果的结构与序列化契约见 [`user/reference/api.md`](../user/reference/api.md)，
用户侧选型决策原理见 [`method_selection.md`](../user/concepts/method_selection.md)。

### 与 `model_selection` 的分界

推荐器只做**元数据推理**（数据画像 × registry 元数据的过滤/评分/排序），
不训练任何模型；选定算法后的**实证选择**——超参搜索（`PUTuner`）、跨配置
比较（`PUModelComparator`）、CV 折结构（`PUStratifiedKFold`）——由
`model_selection` 承担并真实训练。二者是同一"选择"决策链的两段：推荐器
回答"该用哪个算法"（秒级、无 GPU），`model_selection` 回答"选定后哪个
配置在数据上最好"（实证评估）。串联默认：`PUPipeline(classifier="auto")`
消费推荐结果选定方法，再经 `model_selection` 的切分/调参进入训练；调参为
显式叠加，不自动执行。用户侧决策原理见
[`method_selection.md`](../user/concepts/method_selection.md)。

## 5. 评价与切分

| 能力 | 架构要点 | 详情 |
|---|---|---|
| PU 分层切分 | `PUStratifiedKFold` / `PUStratifiedShuffleSplit`：保证每个训练折含 labeled positive，保留 P/U 比例 | [`调优指南`](../user/howto/model_tuning.md) |
| PU 评估指标 | PU-only（`pu_zero_one_risk`/`pu_recall`/`pu_estimated_precision`/`pu_negative_rate`，不需真实标签）；有真值时用标准监督包装（AUC/F1/Accuracy 等） | [`指标契约`](../research/traditional_pu_metric_contract.md) |
| SCAR/SAR 证据 | 无审计真值时仅作非识别性筛查；提供 `y_true` 时在真实正例内检查 selection dependence | [`画像指南`](../user/howto/data_profiling.md) |
| 结构化报告 | `build_diagnostic_report` 只读画像、已拟合估计器与指标接口，不训练；输出稳定 schema 的 JSON/Markdown | [`报告指南`](../user/howto/diagnostic_reports.md) |
| 假设敏感性 | `analyze_pu_sensitivity` 固定模型输出扫 类先验×平均标记倾向 网格相容性；不承担 propensity 识别 | [`敏感性指南`](../user/howto/sensitivity_analysis.md) |
| Selection-bias 模拟 | `make_sar_*` 支持常数/线性/非线性 propensity；`y_true/propensity` 对用户隐藏、仅供 benchmark | [`SAR 指南`](../user/howto/sar_simulation.md) |

## 6. 论文方法到实现的索引

每个方法的模块落点见 `project_structure.md` 目录树；论文公式、源码状态与
复现风险见各方法卡 [`../research/method_cards/`](../research/method_cards/)
（§8 源码状态与复现风险）。
