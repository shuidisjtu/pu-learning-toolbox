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
| `metrics/` | PU 评估指标（PU-only + 标准监督指标包装） | [`../../pu_toolbox/metrics/__init__.py`](../../pu_toolbox/metrics/__init__.py)、[`指标契约`](../research/traditional_pu/traditional_pu_metric_contract.md) |
| `model_selection/` | PU 分层切分、模型调优与比较 | [`../../pu_toolbox/model_selection/__init__.py`](../../pu_toolbox/model_selection/__init__.py)、[`调优指南`](../user/howto/model_tuning.md) |
| `diagnostics/` | 结构化诊断报告与假设敏感性分析 | [`../../pu_toolbox/diagnostics/__init__.py`](../../pu_toolbox/diagnostics/__init__.py)、[`报告指南`](../user/howto/diagnostic_reports.md) |

### Orchestration — 端到端编排与 CLI

| 模块 | 核心职责 | 详情来源 |
|---|---|---|
| `workflows/` | PUPipeline 端到端编排与漂移感知工作流 | [`../../pu_toolbox/workflows/__init__.py`](../../pu_toolbox/workflows/__init__.py)、[`流水线指南`](../user/howto/pipeline.md)、[`漂移指南`](../user/howto/distribution_shift.md) |
| `cli/` | 命令行薄封装（子命令一览） | [`命令行指南`](../user/howto/cli.md)、[`../../pu_toolbox/cli/__init__.py`](../../pu_toolbox/cli/__init__.py) |

### Experiment — 实验/研究协议层

| 模块 | 核心职责 | 详情来源 |
|---|---|---|
| `experiment/` | 面向研究者/实验用户的协议化实验编排：四路数据角色（train/pu_val/clean_val/test）、PA/OA 双协议离线选模、独立 test 评测、版本化留痕；`ExperimentRunner` 固定编排骨架，数据生成/训练/选模为可注入策略（`Generator`/`Trainer`/`SelectionProtocol`） | [`../../pu_toolbox/experiment/__init__.py`](../../pu_toolbox/experiment/__init__.py)、[`实验层设计`](experiment_layer.md)、[`ADR-0018`](../adr/0018-experiment-layer-public-api.md) |

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
| Experiment（`experiment/`） | Core、utils（直接）；registry（延迟，获取算法类）；estimators（经 `fit(model, …)` 注入实例，不静态 import）；不被任何层依赖（叶子入口） |
| User Layer（`ui/`、`benchmarks/`） | Orchestration（经 CLI / workflows 消费工具箱能力） |

**模块级依赖链**（代表性，全部为单向防环设计）：

- **标签语义链**：`core/labels.py`（纯元语：标签约定识别与重映射）→ `core/validation.py`（组装层：标签规范化 + X/y 一致性 + 样本量门槛与告警，返回值已是规范形）→ 各估计器 `fit` 入口
- **数据画像链**：`preprocessing/profiling.py`（统计元语：`pu_data_summary`/`pnu_data_summary`/`scar_diagnostic`，向后兼容）→ `preprocessing/data_profiler.py`（聚合编排：`PUDataProfile` + 可行动 issues）→ `workflows`（pipeline 首步）/ `diagnostics`（报告）/ `advisor`（推荐）
- **字段语义**：`core/tags.py` 是 registry 元数据字段与枚举的权威来源，registry/advisor 均以其为准
- **设备与随机源入口**：`core/device.py` 的 `resolve_device`、`core/random.py` 的 `check_random_state` 是全工具箱唯一的设备/seed 归一化入口，避免各调用点语义漂移
- **实验层注入链**：`experiment/runner.py`（固定编排骨架）→ 注入的 `model` 实例（estimators，调用方经 `registry.get_algorithm` 获取，非静态 import）+ 策略 ABC（`protocols.py` 的 `Generator`/`Trainer`/`SelectionProtocol`）→ 生成/训练/选模/留痕各由可替换策略承担

  「非静态 import」指 `ExperimentRunner` 的 import 列表不含任何 estimator 或 registry，只经
  `fit(model, …)` 接收实例；`model` 由**调用方**按方法名字符串 `registry.get_algorithm(method)`
  查表取算法**类**、实例化后注入。按名字查表使实验层与算法谱系解耦——新增算法只需在 registry
  注册，实验层的 runner/脚本/选模链零感知、零改动。

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

### 3.1 实验层数据流（与 PUPipeline 主链平行）

实验层是**面向研究者/实验用户**的第二条入口，数据带角色 + 显式选模协议，不经
advisor/prior 推荐链路（与 §3 主链的 PUPipeline 非专家路径互补，零改动共存）：

```text
四路数据 train/pu_val/clean_val/test（clean 视图）  [bundle/DatasetPart + core/validation]
  ↓ 校验（索引两两不重叠、视图语义、test 不参与选模）
生成 PU 视图（SCAR/SAR/oracle，按需）              [strategies/Generator]
  ↓ Trainer/PA 路径结构性接收不到真实标签（view=="pu" 校验）
候选训练 → RunTrajectory                           [Trainer → 注入的 estimator]
  （训练视图：OS 原生；原生 TS 方法经 TS-OS 校准，仅 train）
  ↓
PA/OA 离线选模（逐 checkpoint，阈值网格，π 必传）   [strategies/SelectionProtocol]
  ↓
独立 test 评测 + 留痕（manifest）                   [runner → manifest]
```

- **视图语义**：bundle 输入的四份数据（含 `pu_val`）都存**真实标签**（clean 视图，
  `validate_bundle` 强制）——PU 标签是训练/选模前由 `Generator.generate` 从真实标签按
  SCAR/SAR 机制生成的，不预先落盘；生成产物 `view=="pu"`，使 Trainer/PA 路径结构性
  接收不到真实标签（防止泄漏）。`pu_val`/`clean_val` 的命名指各数据**选模时用的视图**
  （PA 用 PU 视图、OA 用 clean 视图），而非存储时的标签。
- **抽样视图（OS/TS）**：基础数据始终为 OS（Single-Training-Set，单一 i.i.d. 样本）；
  原生 TS（Two-Sample / Case-Control）方法不重新独立抽样，而是在训练期对每 mini-batch 做
  TS-OS 校准 `D_U^k ∪ D_P^k`（正例批次并入未标记损失输入，仅限 train，验证/测试保持 OS）。
  语义与门禁见协议 §2.3。
- **选模协议**：PA/OA 各自独立——PA 用 proxy accuracy（OS 分支，π 必传），OA 用
  clean_val 真实 Accuracy；逐 checkpoint 阈值网格选择，平手取最早候选（`selection_spec`）。
- **版本化留痕**：manifest 固化 seed / split_ref / generation / selection /
  test_results / elapsed / failures / resources 八类键，支持可审计复现。

## 4. 算法注册与推荐

每个算法在 `registry` 注册元信息（name/aliases/family/scenario/assumption/
requires_class_prior/backend/maturity/source_status/implementation_status 与 4 个
架构能力字段 native_architectures / input_ndims / encoder_parameter /
trains_encoder）；字段语义与枚举以 `pu_toolbox/core/tags.py` 为权威，内置方法与
算法↔模块落点、实现状态见 `pu_toolbox/registry/builtin_methods.py`。能力字段以
估算器类属性为权威、注册时经 `_SYNC_FIELDS` 镜像进 registry（语义与消费点见
`dual_architecture_plan.md` §3-§4）。

### registry 与实验层 method_ledger.json 的分工

`registry` 是**代码侧真相源**：`get_algorithm(name)` 按名字返回算法**类**（供脚本实例化训练）、
`get_metadata(name)` 返回元数据（供能力/先验门禁与 advisor 推荐）。实验层另有
`pu_toolbox/experiment/method_ledger.json`（survey 方法台账），是**实验侧结果标注真相源**：记录每个
方法的论文出处、原生采样假设、OS/TS 校准、适配级别与复核结论。脚本读它的两处用途：① `--method`
必须落在台账内（survey 范围外的方法 fail-loud）；② 把该 entry 复制到 run 目录旁
（`method_ledger_entry.json`）标注结果。二者不重复：台账里与 registry 重叠的字段（能力 4 字段、
`source_status`、`requires_class_prior`↔`prior_semantics`）由
`tests/contract/test_ledger_registry_consistency.py` 合同测试绑定，台账镜像 registry、不得漂移。
台账语义与字段见 [pu_survey_protocol.md](../research/pu_survey/pu_survey_protocol.md) §4 与
[survey_execution_plan.md](../research/pu_survey/survey_execution_plan.md) P1.3a。

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
| PU 评估指标 | PU-only（`pu_zero_one_risk`/`pu_recall`/`pu_estimated_precision`/`pu_negative_rate`，不需真实标签）；有真值时用标准监督包装（AUC/F1/Accuracy 等） | [`指标契约`](../research/traditional_pu/traditional_pu_metric_contract.md) |
| SCAR/SAR 证据 | 无审计真值时仅作非识别性筛查；提供 `y_true` 时在真实正例内检查 selection dependence | [`画像指南`](../user/howto/data_profiling.md) |
| 结构化报告 | `build_diagnostic_report` 只读画像、已拟合估计器与指标接口，不训练；输出稳定 schema 的 JSON/Markdown | [`报告指南`](../user/howto/diagnostic_reports.md) |
| 假设敏感性 | `analyze_pu_sensitivity` 固定模型输出扫 类先验×平均标记倾向 网格相容性；不承担 propensity 识别 | [`敏感性指南`](../user/howto/sensitivity_analysis.md) |
| Selection-bias 模拟 | `make_sar_*` 支持常数/线性/非线性 propensity；`y_true/propensity` 对用户隐藏、仅供 benchmark | [`SAR 指南`](../user/howto/sar_simulation.md) |

## 6. 论文方法到实现的索引

每个方法的模块落点见 `project_structure.md` 目录树；论文公式、源码状态与
复现风险见各方法卡 [`../research/method_cards/`](../research/method_cards/)
（§8 源码状态与复现风险）。
