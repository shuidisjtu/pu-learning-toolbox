# uPU 来源与行为核查（2026-10-19）

范围：核对 du Plessis et al. ICML 2015 论文、官方关联 `pywsl` 包说明、工具箱 uPU loss/estimator、TS 校准与测试范围。属于单方技术审计，不是方法负责人签署、正式准入或完整实验复现。

## 来源及版本边界

- 原论文：[PMLR v37 paper page/PDF](https://proceedings.mlr.press/v37/plessis15.html)，提出通过正例与 U 使用不同损失的凸 PU 风险，并特别提出 double-hinge（C-DH）；摘要明确 naive P/U separation 有偏，本文不能把任意 P/U 分类器都称为 uPU。
- 官方关联代码：[t-sakai-kure/pywsl](https://github.com/t-sakai-kure/pywsl)。其 README 声明包含该 ICML 2015 uPU，但本地 `benchmarks/assigned_methods/configs/official_sources.lock.json` 没有 `pywsl` immutable commit/发布包摘要。因此本核查能确认官方项目/能力声明，**不能把具体代码行为归因到某个锁定版本**；方法卡将论文作为公式权威，并记录该仓库只提供 squared-loss 闭式实现。

## 论文目标与 toolbox 路径

| 项目 | 论文/官方范围 | toolbox 实现与核查 |
|---|---|---|
| 风险结构 | SCAR、总体先验 π 已知；正例项为 `−π E_P[g]`，U 项为 `E_U[ℓ(−g)]`；凸性要求论文指定的 linear-odd/convex 损失关系 | `UPUClassifier` 与 `UPULoss` 分别对 P/U 风险按各自样本数取均值；先验必填。数学 golden/梯度测试覆盖 squared/logistic。 |
| 损失分支 | 论文重点为 double hinge；pywsl 官方 bundle 只提供 squared 闭式分支 | toolbox 另实现 C-DH 的 SLSQP QP 及 C-LL 的 L-BFGS，属论文推导 clean-room 扩展，不是官方代码复刻。默认 `loss="squared"`，方法卡与台账记录其与论文主推 C-DH 不同，pilot recipe 必须显式选型。 |
| 正则与偏置 | 正则作用于模型系数；intercept 不惩罚 | 代码对 `α` 正则、不惩罚 intercept；方法卡列出的线性/RBF basis、solver 和数值参数是工程规格，不应读作论文全部固定。 |
| 视图校准 | 原文 case-control U 为总体边缘；项目 TS-OS 规则把承担 U 风险角色的数据由 `D_U` 扩为 `D_U∪D_P` | `fit(os_or_ts="ts")` 只替换 U 风险角色；π 与 P 项不变。RBF center pool 跟随 view，但中心数按校准前 `n_U` 截断；sample_weight 被忽略，因此不需互斥门。TS 是项目协议层适配，不是原文独立算法结论。 |
| 选模与验证 | 原文实验用论文指定的 PU zero-one CV 和其设置 | 本 estimator 无内置 validation/test 接口；项目正式候选/选模流程由 protocol 控制。方法卡记录 loss 默认及超参方案与论文原始实验并不完全一致。 |

## 验证证据与缺口

- `tests/unit/losses/test_upu_loss.py` 覆盖风险手算 golden、先验作用、logistic/squared 梯度及有限差分。
- `tests/unit/estimators/test_upu.py` 与 `test_upu_ts_view.py` 覆盖拟合、solver/参数、视图角色、RBF centers 与校准前后容量约束。
- 测试证明本实现与登记公式及工程不变量一致，不证明 `pywsl` 某个特定源码版本的逐行一致性，也不证明完整 ICML 2015 数据/调参/结果已复现。
- 应补的来源治理项是给 `pywsl` 记录 immutable commit 或 release/archive hash；当前官方 README 是浮动分支信息。完成 pin 后可再做文件级代码比较。本轮不因没有 pin 而推断官方实现不存在，也不伪造版本号。

结论：`official_bundle` 应理解为官方关联软件中有 uPU 实现，不等于 toolbox 三种 loss 都有官方源码对应。Squared 分支、C-DH/C-LL 适配、TS calibration 需分层披露。该核查不改算法参数、冻结结果、准入或正式 recipe。
