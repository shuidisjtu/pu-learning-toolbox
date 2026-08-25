# 传统 PU 七算法调优设计方案

```yaml
schema_version: 1
status: design_only
scope:
  - elkan_noto
  - upu
  - nnpu
  - pnu
  - ldce
  - kldce
  - llsvm
baseline: benchmarks/traditional_pu/results/baseline_v2
paper_claim: false
```

## 1. 目标与边界

本方案用于改进当前工具箱实现的可靠性、分类质量和计算成本，不用于声称论文结果已复现。
优化目标采用分层规则：先满足数值可靠性，再比较分类质量，最后约束计算成本。不得把三者
压缩为一个未经声明的综合分数。

本阶段只覆盖 Elkan--Noto、uPU、nnPU、PNU、LDCE、KLDCE 和 LLSVM。其他算法可作为外部
参考，但不进入本阶段排名。`baseline_v2` 是冻结对照，不得被候选参数覆盖或静默重跑。

## 2. 固定实验协议

- 主协议沿用 `benchmarks/traditional_pu/` 的 SCAR 网格：
  `pi ∈ {0.1, 0.3, 0.5}`、small/mid 两个规模、确认种子 `100..119`。
- linear SAR 只作鲁棒性诊断，不用于主排名。
- 开发种子为 `0..4`，确认种子与开发种子不得重叠。
- 调参和早停只能使用开发种子（`0..4`）数据的 PU 信息；不得读取确认集或测试折的 `y_true`。
- PNU 继续使用独立的 `{+1,-1,0}` P/N/U 协议，不与二元 PU 结果混排。
- 所有实验必须先通过 [数据泄露审计门禁](../dev/data_leakage_audit_design.md)。

## 3. 指标与判定

### 3.1 主指标和约束

候选参数的主选参指标为 `pu_zero_one_risk`。它是依赖类先验的 PU 估计风险，可能出现负值，
不得按普通有界误差解释。

每个候选必须同时记录：

- success rate、未收敛率、异常/超时率；
- `pu_zero_one_risk`、`pu_recall`、`pu_estimated_precision`；
- 预测阳性率及 `degenerate_prediction` 标记；
- 平均耗时、P95 耗时和算法专属收敛诊断。

`degenerate_prediction` 标记与预测阳性率列为 runner 前置扩展（当前 trials.csv 仅有
`pu_negative_rate`，可推导阳性率但无退化标记），须在第 8 节第 4 步完成后方可作为晋级判据。

### 3.2 隐藏真值确认

参数选择完成并冻结后，才在确认集使用隐藏真值报告：ROC-AUC、AP、balanced accuracy、F1。
模型确实提供 `[0,1]` 概率时，额外报告 Brier score 和 ECE；无合法概率时必须记录不可用原因，
不得把 decision score 伪装成概率。

### 3.3 退化预测

- 预测阳性率为 0 或 1 时，标记 `degenerate_prediction`；
- 退化 trial 可保留用于诊断，但不能作为有效分类质量证据；
- 是否淘汰候选必须在实验前声明，不能根据结果临时调整。本方案默认淘汰规则（对全部七算法
  生效）：候选在开发阶段任一成功单元出现退化预测，即淘汰该候选，不进入多种子确认；个别算法
  的例外必须在实验配置中显式声明并记录理由；
- 低先验场景尤其检查 KLDCE/LDCE 的全负预测。

## 4. 调优顺序与参数簇

每一轮只搜索一个参数簇，先进行小范围筛选，再对前 3--5 个候选进行多种子确认，避免全参数
笛卡尔积造成不可解释的交互和过度搜索。筛选准则固定为：按开发种子（`0..4`）全部单元的
`pu_zero_one_risk` 均值升序排序；仅开发阶段 success rate 为 100% 且无退化预测的候选可进入
多种子确认（取前 3--5 个）。

| 顺序 | 算法 | 首轮参数簇 | 重点风险 |
|---:|---|---|---|
| 1 | KLDCE | `covariance_ridge`、`reg_strength`、`centroid_radius` | 低先验全负预测、病态协方差、ACS/QP 收敛 |
| 2 | LDCE | `reg_strength`、`covariance_ridge`、`centroid_radius`、`max_iter` | 收敛预算、矩估计稳定性 |
| 3 | nnPU | `beta/gamma`、`batch_size`、`max_epochs`、`patience` | 负风险修正、种子方差 |
| 4 | uPU | loss、`reg_lambda`、basis、kernel width | 风险估计和正则敏感性 |
| 5 | LLSVM | 学习率、正则、`gamma`、patience/min epochs | 非凸 SGD、训练成本 |
| 6 | Elkan--Noto | calibration method、CV 折数、`eps` | `c` 估计、概率合法性和校准 |
| 7 | PNU | `eta`、`reg_lambda`、`basis`、`kernel_width` | P/N/U 比例和三元风险 |

参数簇说明：KLDCE 的 `inner_tol/max_inner_iter` 已随实现修复（提交 `cd53c17`）确定为
`1e-6` / `2000`，首轮不再搜索。nnPU 的优化器/学习率不在首轮参数簇内——构造器只暴露
`optimizer: null`，JSON config 无法表达 torch optimizer 对象；如需搜索学习率，须先扩展
`ESTIMATOR_FACTORY` 支持 optimizer spec 或另立代码级实验，两者均属 runner 扩展。PNU 构造器
无 `loss` 参数（三元风险由 `eta` 加权），参数簇以构造器签名为准。

## 5. 晋级规则

候选只有同时满足以下条件，才能标记为 `confirmed_improvement` 并考虑写回默认值：

1. 确认种子上的主指标配对 95% CI 支持改善（与契约 §6 条件 1 一致；`compare.py` 口径为严格
   改善：lower-is-better 指标 CI 上界 < 0，反之上界 > 0，"至少不劣"不构成晋级）；
2. 成功率与未收敛率未发生不可接受恶化（口径与 `compare.py` 一致：成功率相对冻结 baseline
   恶化不超过 5 个百分点；未收敛计入非成功，已被成功率覆盖）；
3. 退化预测率不增加；
4. 平均和 P95 耗时不超过预先声明的预算；
5. 数据协议、切分、阈值和指标口径完全一致；
6. 结果在未参与筛选的确认种子上复现。

只有 oracle 指标改善而 PU 主指标没有改善的候选，标记为 `oracle_only_improvement`，不得
宣称 PU 调优成功。该分类需要 `compare.py` 扩展（对 oracle 指标同样计算配对 95% CI），
未实现前不得手工判读。实现修复、超参数调优和指标/协议变更必须使用不同结果类型。

## 6. 必须保存的产物

每次开发或确认运行都必须保存：

```text
resolved_config.json
run_manifest.json
trials.csv
summary.csv
findings.md
data_leakage_audit.json
```

`run_manifest.json` 必须记录源码 commit、依赖环境、seed 集合、配置哈希和数据协议。trial 至少
记录算法、场景、seed、参数快照、状态、失败原因、收敛诊断、全部可用指标和耗时。
`data_leakage_audit.json` 由泄露审计 preflight 生成；preflight 实现前，实验报告必须标注
`audit_design_only`（见审计设计 §6），不得声称已通过完整泄露门禁。

## 7. 结果类型

| 类型 | 含义 | 是否可直接更新默认值 |
|---|---|---|
| `implementation_fix` | 改变求解器、收敛判定或数值实现 | 需单独回归和基线重锁 |
| `hyperparameter_tuning` | 只选择已有构造参数 | 满足第 5 节后可申请；写回默认值必须同步重锁基线（更新 v2 锁定值或升 v3）并在确认种子重跑确认，否则 `check_baseline_configs` 门禁报警 |
| `metric_or_protocol_change` | 改变指标、数据或切分口径 | 必须更新契约并重跑受影响基线 |

## 8. 第一阶段交付顺序

1. 建立数据泄露审计门禁和故意泄露负向测试（实施范围按审计设计 §7 分阶段裁剪：先做
   `y_true` 路径约束、trial 列写入门禁、manifest 审计状态与审计函数负向单测）；
2. 同步指标契约中 implemented/planned 状态（已完成：AP、balanced accuracy、Brier、ECE 已
   移入 `implemented_metrics`）；
3. 冻结并复核 baseline_v2（已完成：1200/1200 success）；
4. 补齐 runner/compare 前置扩展：trials 增加 `degenerate_prediction` 与预测阳性率列；compare
   增加 `oracle_only_improvement` 分类；
5. 按 KLDCE → LDCE → nnPU → uPU → LLSVM → Elkan--Noto → PNU 顺序开展调优；
6. 仅在候选通过晋级规则后，另行讨论是否写回源码默认值（写回必须同步重锁基线并重跑确认）。

