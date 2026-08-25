# 传统 PU 七类分类器单域指标契约

```yaml
schema_version: 1
status: design_contract
scope:
  - elkan_noto
  - upu
  - nnpu
  - pnu
  - ldce
  - kldce
  - llsvm
purpose: toolbox_performance_improvement
implemented_metrics:
  - pu_zero_one_risk
  - pu_recall
  - pu_estimated_precision
  - pu_auc_roc
  - pu_accuracy
  - pu_f1
  - pu_negative_rate
planned_metrics:
  - average_precision
  - balanced_accuracy
  - brier_score
  - expected_calibration_error
```

## 1. 目的与声明边界

本契约定义传统 PU 分类器的评价、调参与基线构建规则。其目标是评估和改进
本工具箱当前实现的可靠性能；它**不是**论文复现协议，也不得据此声明论文结果
已复现。

`implemented_metrics` 表示当前统一 `PUPipeline` 评价入口已经支持的指标；
`planned_metrics` 表示本契约要求、但尚未全部接入统一入口的后续能力。当前文档是
设计契约，不应被解释为所有规划指标已经可调用。

适用对象为 Elkan--Noto、uPU、nnPU、PNU、LDCE、KLDCE 和 LLSVM。类先验估计器、
表征学习器及其他分类器不在本契约范围内。分布漂移审计、跨域适配、部署监控、
拒绝预测、主动复核以及 JointShift/DynamicJointShift 研究模型分别由独立契约或
研究协议规定，不得套用本单域契约。

所有指标必须同时声明可用条件、统计方向和使用目的。缺少前提时，运行器必须将
指标标记为不可用并保存原因，不能以替代值或静默降级掩盖该事实。

## 2. 数据协议与隔离

### 2.1 PU 协议

Elkan--Noto、uPU、nnPU、LDCE、KLDCE 和 LLSVM 使用共同的二元 PU 协议：

- 主比较场景为 SCAR；linear SAR 仅用于鲁棒性与失效边界报告，不用于算法主排名；
- 类先验为 $\pi \in \{0.1, 0.3, 0.5\}$；
- 每个场景包含小样本和中等样本两个规模档位；第一版固定特征维度，不同时扫描维度；
- 数据生成器保留隐藏真值 `y_true`，但该值只能用于最终评测；
- 开发期种子固定为 `0..4`，确认期种子固定为 `100..119`，两个集合不得重叠。

调参只能使用训练折中的 PU 信息。禁止使用测试折或确认集的 `y_true` 进行参数选择、
阈值选择或早停。

### 2.2 PNU 协议

PNU 使用独立的 $\{+1,-1,0\}$ 三元标签协议，不得与仅有 $\{+1,0\}$ 标签的 PU
运行混用。其数据配置必须显式记录正例 P、可信负例 N、未标记样本 U 的数量和比例，
并至少覆盖 `1:1:4`、`1:2:4`、`1:1:8` 三种 P:N:U 比例。

PNU 不参与纯 PU 分类器的横向排名；其验证也不得使用隐藏 `y_true` 选参。

### 2.3 LDCE 与 KLDCE 的翻转率

LDCE/KLDCE 的 `flip_probability=h` 定义为真阳性被翻转为观测负/未标记的概率。基线
数据生成器必须保存真实 `h`，并将同一值传递给这两个分类器；不得把任意
`label_frequency` 当作 `h`。

当 $|1 - 2\pi h|$ 接近零时，LDCE 问题会病态。此类单元必须排除或明确标记为病态，
不能与正常单元混合汇总。主基线使用真实 `h`；另行报告 `h` 扰动的敏感性实验。

## 3. 指标可用性与用途

| 指标 | 方向 | 所需输入 | 可用场景 | 用途 |
|---|---:|---|---|---|
| `pu_zero_one_risk` | 越低越好 | PU 标签、连续 score、类先验 $\pi$ | SCAR，且 $\pi$ 可靠 | PU 算法的主选参指标 |
| `pu_recall` | 越高越好 | PU 标签、预测标签 | PU | 诊断观测正例覆盖率 |
| `pu_estimated_precision` | 越高越好 | PU 标签、预测标签、$\pi$ | SCAR，且 $\pi$ 可靠 | 诊断；不单独决定优劣 |
| ROC-AUC | 越高越好 | `y_true`、连续 score | 隐藏真值可用 | 最终排序评测 |
| AP | 越高越好 | `y_true`、连续 score | 隐藏真值可用 | 最终正类排序评测 |
| balanced accuracy | 越高越好 | `y_true`、预测标签 | 隐藏真值可用 | 最终阈值分类评测 |
| F1 | 越高越好 | `y_true`、预测标签 | 隐藏真值可用 | 最终正类决策评测 |
| Brier score | 越低越好 | `y_true`、真实概率 | 概率输出可用 | 最终概率质量评测 |
| ECE | 越低越好 | `y_true`、真实概率 | 概率输出可用 | 最终校准评测 |
| `elapsed_seconds` | 越低越好 | 计时数据 | 全部 | 计算成本 |
| `success_rate` | 越高越好 | trial 状态 | 全部 | 数值可靠性 |

上表中的 AP、balanced accuracy、Brier score 和 ECE 属于 `planned_metrics`；在它们
尚未接入统一评价入口前，只能由明确声明的 benchmark 运行器单独计算，不能假定
`PUPipeline` 或 `PUTuner` 已支持这些名称。

连续 score 只能用于 ROC-AUC/AP 等排序指标。Brier/ECE 只能使用分类器真实的
`predict_proba` 概率输出；不得将 `decision_function` 的任意分数转换为伪概率。
ECE 使用 10 个等宽概率桶，并额外保存每个桶的样本数、平均置信度和经验准确率。

默认决策评测使用模型原生阈值。以验证集 PU 信息选择阈值的结果必须作为独立实验，
不得替代默认基线。

## 4. 算法级指标契约

| 算法 | 调参主指标 | 隐藏真值确认指标 | 必须保存的诊断 |
|---|---|---|---|
| Elkan--Noto | SCAR 下的 `pu_zero_one_risk` | ROC-AUC、AP、F1、balanced accuracy、Brier、ECE | $c$ 估计、校准桶、耗时 |
| uPU | SCAR 下的 `pu_zero_one_risk` | ROC-AUC、AP、F1、balanced accuracy | $\pi$ 敏感性、优化状态、耗时 |
| nnPU | SCAR 下的 `pu_zero_one_risk` | ROC-AUC、AP、F1、balanced accuracy | 负风险修正触发比例、训练曲线、耗时 |
| PNU | P/N/U 分层验证中的可观测 PNU 风险或验证损失 | ROC-AUC、AP、F1、balanced accuracy | P:N:U 比例、耗时 |
| LDCE | SCAR 下的 `pu_zero_one_risk` | ROC-AUC、AP、F1、balanced accuracy | `converged_`、`n_iter_`、条件数、耗时 |
| KLDCE | SCAR 下的 `pu_zero_one_risk` | ROC-AUC、AP、F1、balanced accuracy | `converged_`、`n_acs_iter_`、QP/ACS 状态、条件数、耗时 |
| LLSVM | SCAR 下的 `pu_zero_one_risk` | ROC-AUC、AP、F1、balanced accuracy | 最优 epoch、早停、种子方差、耗时 |

在 SAR 场景，PU-only 指标可保留作诊断，但必须标记为不可作为最终优劣结论的依据。
最终报告不折算跨算法总分；排序、阈值分类、概率校准和数值稳定性不可互换。

## 5. 基线定义

基线必须使用源码构造器默认参数。uPU 的既有 D6 配置
`loss=double_hinge, reg_lambda=0.01` 只能作为名为 `existing_tuned_reference` 的附加
变体，不得代替默认参数基线。"源码默认参数"以基线运行时源码为准：若默认值此后
演进，重跑基线必须锁定当时的默认值（或在 config 中显式声明）。已知演进：LDCE
默认 `max_iter` 100 → 10000（100 轮在 SCAR 网格几乎全域不收敛，见 M6 判定；
放宽容限不改变解语义），v1 基线（`confirmation_v1`）仍以 100 运行。

后续演进记录（2026-08-25）：KLDCE 内层求解器重写（原生 SMO + 单调 ACS 回滚 +
`inner_tol` 1e-8→1e-6，提交 `cd53c17`）；v1 基线仍以修复前默认参数运行
（全域 0 success）。修复与 LDCE `max_iter` 写回均落入源码默认后，v2 基线
（`configs/seven_methods_pu_baseline_v2.json`，`results/baseline_v2`）以**显式锁定
参数**运行：`methods` 逐键钉住当前构造器默认值（runner 注入的
`random_state`/`class_prior`/`flip_probability` 除外），配置带
`locks_source_defaults: true` 标记，由 `scripts/check_baseline_configs.py` 门禁持续
校验——源码默认参数再演进时门禁报警，基线不再静默漂移。v1 配置保持空 `{}` 的
历史形态，仅作历史快照，不得在当前源码下重跑。

正式结果按以下目录契约保存：

```text
benchmarks/traditional_pu/
  configs/
    seven_methods_pu_baseline_v1.json
    pnu_baseline_v1.json
  results/<run-name>/
    resolved_config.json
    run_manifest.json
    trials.csv
    summary.csv
```

`trials.csv` 至少具有统一字段：`algorithm`、`scenario`、`seed`、`status`、
`elapsed_seconds`、`warning_count`、`failure_reason`、参数快照及全部可用指标。算法
专属训练诊断保存为结构化 JSON 或专属列，不因统一表格而丢失。

正式 20-seed 基线提交配置、manifest、原始 trial 表与摘要；模型权重、缓存和冗长日志
保留在仓库外。manifest 必须记录代码 commit、依赖版本、种子集合、配置哈希和数据来源。

## 6. 统计与失败规则

每个配置至少报告均值、样本标准差、95% 置信区间、成功数、失败数、未收敛率，以及
平均与 P95 耗时。性能比较必须基于同一 seed 的配对差值及其 95% 置信区间，而非仅比较
两个均值。

以下情况必须标记为非成功 trial：超时、异常、NaN/Inf、显式未收敛 warning，或达到最大
迭代次数但未满足容差。非成功 trial 产生的分数可保留供诊断，但不得进入主性能均值；
`success_rate` 必须包含它们。

首次运行先记录算法实际耗时分布，再冻结算法特异的超时阈值。在阈值冻结前，采用统一且
宽松的保护上限，并完整保存超时原因。

后续调优只有同时满足以下条件，才可标记为 `confirmed_improvement`：

1. 确认种子 `100..119` 上的主指标配对 95% CI 支持改善；
2. 成功率和未收敛率未发生不可接受恶化；
3. 平均/P95 耗时未超过预先声明的预算；
4. 数据协议、评测阈值和指标口径与基线完全一致。

## 7. 实施顺序与兼容性

1. 实现 `planned_metrics`（AP、balanced accuracy、Brier、ECE），以及各指标的可用性原因；
2. 保持现有 `PUPipeline` 默认指标和报告语义不变，新能力均为 additive 变更；
3. 新建 `benchmarks/traditional_pu/`，不得将 PNU 特例塞入现有 `assigned_methods` runner；
4. 为指标计算、不可用条件、数据隔离、失败统计和结果审计添加单元/集成测试；
5. 先运行 5-seed 开发基线并修复协议问题，再冻结配置并运行 20-seed 正式基线；
6. 合成基线冻结后，另行锁定公开真实数据集、版本哈希及标签映射，作为外部确认线。

本契约为版本 1。任何改变指标含义、可用条件、主指标、数据协议或统计规则的改动都必须
提升 `schema_version` 并重新生成受影响的基线。
