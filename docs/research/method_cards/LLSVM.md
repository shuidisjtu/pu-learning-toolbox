# Method Card: LLSVM（Large-Margin Label-Calibrated SVM）

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Large-Margin Label-Calibrated Support Vector Machines for Positive and Unlabeled Learning |
| Authors | Chen Gong, Tongliang Liu, Jian Yang, Dacheng Tao |
| Venue | IEEE Transactions on Neural Networks and Learning Systems, 30(11), 3471-3482 |
| Year | 2019 |
| DOI | `10.1109/TNNLS.2019.2892403` |
| Setting | 二分类 P/U；线性实值判别函数 $`f_\omega(x)=\omega^\top \bar{x}`$ |
| Requires class prior | `True`：未标记集中的正类先验 $`\pi`$，用于 $`t=2\pi-1`$ |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| GPU required | `False`（SGD 张量实现可支持 GPU） |
| Source status | `official_exact`：论文 + 官方 MATLAB 代码包已审阅；**实现以代码为准**（见 §6 ADR-0014 #8） |

### Assumptions

```math
X_P\sim p(x\mid y=+1),\qquad
X_U\sim p(x)=\pi p(x\mid y=+1)+(1-\pi)p(x\mid y=-1).
```

其中 $`0<\pi<1`$。论文的目标还隐含 P 与潜在负类存在可利用的低密度间隔/聚类结构；这不是标准无偏 PU 风险估计的分布无关假设。

---

## 2. 问题设定与符号

P/U 协议：$`X_P`$ 的标签为 $`+1`$，$`X_U`$ 为未标记；不需要已知负样本，也不需要 propensity。目标是学得线性判别函数 $`f_\omega(x)=\omega^\top\bar{x}`$，使 P 位于正侧、U 远离决策边界，并以 U 的平均软标签均值约束校正“全判正”的偏置。

| 符号 | 含义 |
|---|---|
| $`P,U`$ | 正样本集、未标记集 |
| $`p,u,n=p+u`$ | P、U、总样本数 |
| $`\bar{x}`$ | 末尾增广常数 1 的特征 |
| $`f_\omega(x)=\omega^\top\bar{x}`$ | 实值 score；符号决定类别 |
| $`\alpha`$ | 正样本平方 hinge 权重 |
| $`\beta`$ | 未标记 Gaussian-like hat 权重 |
| $`\gamma`$ | 标签校准权重 |
| $`\pi`$ | U 中正类先验 |
| $`t=2\pi-1`$ | U 的平均标签上界 |
| $`A`$ | 压缩函数缩放参数（论文取 2，代码取 10） |
| $`\Phi_A(z)=\frac{A}{\pi}\arctan z`$ | 将 score 压缩到 $`[-A/2, A/2]`$；论文写为 $`\frac{2}{\pi}\arctan z`$ |

---

## 3. 核心公式

### 3.1 原始建模动机

论文先以正样本的 hinge、U 上的 hat loss 和 U 平均软标签约束建模。hat loss 为

```math
h(z)=\max(1-|z|,0),
```

它惩罚 $`z\in[-1,1]`$，推动未标记点远离边界。校准约束是

```math
\frac1u\sum_{x\in U}\Phi(f_\omega(x))\le t+\eta,\qquad \eta\ge0.
```

这解释了 $`t`$：若 U 中正类比例为 $`\pi`$，真实标签均值为 $`\pi-(1-\pi)=2\pi-1`$。原始形式含非光滑、跨 U 样本耦合的项，**不应直接作为 minibatch 实现目标**。

### 3.2 实际训练目标（论文式 9）

用平方 hinge、Gaussian-like 近似和 Jensen 上界后，最小化：

```math
J(\omega)=\frac12\lVert\omega\rVert_2^2
+\frac{\alpha}{p}\sum_{x\in P}[\max(1-f_\omega(x),0)]^2
+\frac{\beta}{u}\sum_{x\in U}\exp[-3f_\omega(x)^2]
+\frac{\gamma}{u}\sum_{x\in U}[\max(\Phi(f_\omega(x))-t,0)]^2.
```

- 第一项：$`\ell_2`$ 正则。
- 第二项：让标记正样本 score 至少为 1。
- 第三项：在 score 为 0 时取最大值 1，促使 U 离开边界；它使目标非凸。
- 第四项：逐个约束 U 的软标签均值上界的可分上界，校准“全判正”的偏置。

对 U，第三和第四项的梯度（未含正则）分别为：

```math
\nabla_\omega\frac{\beta}{u}e^{-3f^2}
=-\frac{6\beta}{u}f e^{-3f^2}\bar{x},
```

```math
\nabla_\omega\frac{\gamma}{u}[\max(\Phi(f)-t,0)]^2
=\frac{4\gamma}{\pi u(1+f^2)}\max(\Phi(f)-t,0)\bar{x}.
```

实现时优先让自动微分计算梯度，并用上述式子做小批量数值梯度校验。

> 论文式 (9) 与官方代码存在 5 项偏差（指数系数、压缩函数、P/U 项归一化、增广常数），实现以代码为准——见 ADR-0014 #8。

### 3.3 类先验与阈值

论文以 $`\mathrm{penL1}`$（du Plessis, Niu, Sugiyama, 2015）在网格 $`\{0.05,0.10,\ldots,0.95\}`$ 上估计 $`\pi`$，随后设 $`t=2\pi-1`$。

---

## 4. 算法概要

1. 校验 P/U 均非空，标签仅为 $`\{+1,0\}`$，且 $`0<\pi<1`$。
2. 若未提供 `class_prior`，调用先验估计器在 P/U 上估计 $`\pi`$；令 $`t=2\pi-1`$。
3. 初始化线性参数（含或不含截距）；固定随机种子后打乱训练索引。
4. 对每个 epoch 和 minibatch，按式（9）计算 P 与 U 项的批量估计、反向传播并更新参数。
5. 收敛早停：全数据目标值在 trailing-window（`patience` epoch）内的相对变化低于 `tol` 且已训练满 `min_epochs` 时停止（无验证集；损失前期非单调，禁用逐 epoch 比较）。
6. 保存 `class_prior_`、`calibration_target_`、最终目标分量和训练历史，提供可诊断输出。

论文固定步长 $`\tau=0.01`$、$`N=40`$ 个 minibatch，未固定 epoch 数（3000 仅官方代码设定）；官方代码实际使用步长 $`5\times10^{-6}`$、$`N=20`$ 个 minibatch、$`3000`$ epochs，仅在训练开始时 shuffle 一次。

---

## 5. 论文边界

- 方法只适用于 P/U：$`X_P`$ 的标签为 $`+1`$，$`X_U`$ 为未标记。它不需要已知负样本，也不需要 propensity。
- 核心归纳偏置是：P 与隐藏负类在特征空间中形成可分簇，未标记样本应远离决策边界。若类别高度重叠、特征不具备聚类/间隔结构，hat 项可能造成过度自信预测。
- 仅使用正样本 hinge 项和未标记 hat 项会把所有训练点推向正类；标签校准项是防止该退化的必要部分，不能省略。
- 训练目标非凸；SGD 只保证得到局部解。需要固定随机种子、保存最优验证 checkpoint，并报告多随机种子方差。
- 论文用 $`\mathrm{penL1}`$ 估计类先验；先验误差会直接影响 $`t`$ 和边界位置，论文的泛化界不覆盖该估计误差。
- **论文结论的可用边界**：论文给出在特征范数有界时的 margin 泛化界：界随 P、U 样本量增大而降低，且依赖 $`\alpha+\beta+\gamma t^2`$、margin $`\rho`$ 和样本数。它支持“更多 P/U 数据有助于泛化”的定性判断，但不提供超参数默认值、全局最优保证或先验误差保证。
- 论文实验在 synthetic DoubleGaussian、4 个 OpenML 数据集、CIFAR cat-vs-dog 特征和 GermanCredit 上，大多数设置优于其比较基线；作者也明确指出 $`\alpha,\beta,\gamma`$ 敏感，仍需调参。这些结论应作为 benchmark 假设，而非性能承诺。

---

## 6. 实现注记

> 见 ADR-0014 #8（论文式 (9) 与官方代码 5 项偏差：`exp[-5f²]`、`A=10`、P/U 项不除样本数、增广常数 10；含代码实际训练目标、U 项梯度与正则化建议）

- **[状态]** native 实现完成 2026-07-23；`penL1` 已实现并作为默认类先验估计器。官方代码已审阅，实现以代码为准（见上）。
- **早停（2026-08-05）**：`early_stopping` 默认开启，基于 trailing-window 相对损失变化 + `min_epochs` 下限。论文未固定 epoch（仅官方代码用 3000），默认值校准于双高斯 PU 数据（`tol=5e-4`，60×5 实测停 440–1979 epoch，120×5 停 ~1271）。
- **默认参数**：以官方代码为准（`lr=5e-6`、20 minibatch、3000 epoch），论文值仅作参考；工程实现默认每 epoch shuffle 并使其可配置（论文仅在训练开始时 shuffle 一次）。
- 式 (9) 第一项 $`\ell_2`$ 正则：若使用独立 `intercept`，建议不正则化截距。
- 先验估计：复用现有 `penL1`，LLSVM 只消费最终 `class_prior_`，不复制一套先验估计实现；允许用户传入可信先验以跳过估计。

---

## 7. 论文实验参考

| 项目 | 论文设置 |
|---|---|
| 合成数据 | DoubleGaussian（两簇可分） |
| 真实数据 | 4 个 OpenML 数据集、CIFAR cat-vs-dog 特征、GermanCredit |
| 对照方法 | 论文比较的基线（WSVM 等） |
| 结论 | 大多数设置下优于比较基线；$`\alpha,\beta,\gamma`$ 敏感，仍需调参；论文数值不能直接视为本项目复现结果 |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact`：论文 + 官方 MATLAB 代码包已审阅 |
| Official code | `LLSVM_TNNLS19.rar`（MATLAB，作者实际运行实验的版本） |
| 复现基准 | **实现以官方代码为准**，与论文式 (9) 的偏差清单见 ADR-0014 #8 |
| 参考实现 | `penL1`（du Plessis et al., 2015）已实现，作为默认类先验估计器 |
| 复现风险 | 官方代码与论文式 (9) 有 5 项偏差，跨数据集时 P/U 项归一化尺度需注意；目标非凸，需固定种子并报告多随机种子方差 |
