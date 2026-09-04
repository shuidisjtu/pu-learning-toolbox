# Method Card: Regrouping Class-Prior Estimation (ReCPE)

> 参数契约（签名、参数表、返回结构）以 [API 参考](../../user/reference/api.md) 为权威；本文档只记论文研究内容、实现边界与复现状态。

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Rethinking Class-Prior Estimation for Positive-Unlabeled Learning |
| Authors | Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Gang Niu, Masashi Sugiyama, Dacheng Tao |
| Venue | arXiv preprint，arXiv:2002.03673v2 |
| Year | 2022 |
| Setting | case-control；也可适配单一 PU 数据集，但需保证正例抽样条件 |
| Requires class prior | `False`，方法输出类别先验估计 |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| GPU required | `False` |

### Assumptions

```math
S_p=\{x_i^p\}_{i=1}^{n_p}\overset{i.i.d.}{\sim}P_p,
\qquad
S_u=\{x_j^u\}_{j=1}^{n_u}\overset{i.i.d.}{\sim}P_u
```

```math
P_u=(1-\pi)P_n+\pi P_p,
\qquad 0<\pi<1
```

其中 $`P_p`$ 是正类条件分布，$`P_n`$ 是负类条件分布，$`P_u`$ 是未标记边缘分布。

---

## 2. 问题设定与符号

给定正类样本和未标记样本，传统的 distributional-assumption-free CPE 方法通常估计未标记分布中正类分布的最大混合比例。它们隐含依赖 irreducibility assumption：正类分布的 support 不能被负类分布完全包含。当该假设不成立时，传统方法会系统性高估类别先验。

ReCPE 不直接估计原始 $`P_p`$ 在 $`P_u`$ 中的最大比例，而是从 $`P_u`$ 中找出最像正类的一小部分样本，构造辅助正类分布 $`P_p'`$，再调用已有 CPE 方法。

| 论文符号 | 含义 |
|---|---|
| $`P_p`$ | 正类条件分布 |
| $`P_n`$ | 负类条件分布 |
| $`P_u`$ | 未标记边缘分布 |
| $`\pi`$ | 真实类别先验 $`P(y=1)`$ |
| $`S_p`$ | 正类样本集 |
| $`S_u`$ | 未标记样本集 |
| $`A`$ | 被 regrouping 的小样本集合 |
| $`p`$ | 复制比例 |
| $`P_p'`$ | regrouping 后的辅助正类分布 |
| $`\pi'`$ | 辅助问题中的新类别先验 |
| $`q(C=1\mid x)`$ | 样本属于正类样本来源的后验 |
| $`q(C=0\mid x)`$ | 样本属于未标记来源的后验 |

---

## 3. 核心公式

### 3.1 传统 CPE 的偏差

传统 distributional-assumption-free CPE 方法通常估计：

```math
\kappa^*=\sup\{\kappa:P_u=\kappa P_p+(1-\kappa)Q\}
```

当 $`P_n`$ 对 $`P_p`$ 可约（reducible）时，令：

```math
\beta^*=\inf_{S:P_p(S)>0}\frac{P_n(S)}{P_p(S)}>0
```

则：

```math
\kappa^*=\pi+(1-\pi)\beta^*>\pi
```

因此直接估计 $`\kappa^*`$ 会产生正向偏差。

### 3.2 Regrouping 后的分布

选取集合 $`A`$，将负类分布在 $`A`$ 上的概率质量转移到正类中：

```math
\pi'=\pi+(1-\pi)P_n(A)
```

```math
P_n'=\frac{P_n^{A^c}}{P_n(A^c)}
```

```math
P_p'=\frac{(1-\pi)P_n^A+\pi P_p}{(1-\pi)P_n(A)+\pi}
```

于是：

```math
P_u=(1-\pi')P_n'+\pi'P_p'
```

论文证明 regrouping 后的 $`P_n'`$ 和 $`P_p'`$ 满足 anchor set assumption，$`\pi'`$ 可以被已有 MPE/CPE 方法识别。当 $`P_n(A)`$ 很小时，$`\pi'`$ 接近原始 $`\pi`$。

### 3.3 选择集合 $`A`$

论文给出的理想目标是最小化集合中“像负类”的质量与“像正类”的质量之比：

```math
A^*=\arg\min_{A\in\mathcal S}
\frac{\mathbb E_q[\mathbf 1_A(X)q(C=0\mid X)]}
{\mathbb E_q[\mathbf 1_A(X)q(C=1\mid X)]}
```

实际使用分类器后验进行近似，因此选择 $`q(C=1\mid x)`$ 最大的样本。

### 3.4 辅助分布的样本近似

论文实际使用复制样本近似 $`P_p'`$：

```math
\widetilde P_p'
=\frac{P_u^A+P_p}{P_u(A)+1}
```

当 $`P_u(A)`$ 较小时，$`\widetilde P_p'`$ 与理论上的 $`P_p'`$ 接近。$`p`$（复制比例）控制 $`A`$ 的大小。

---

## 4. 算法概要

由于 $`P_n`$ 不可观测，实际 regrouping 用 $`P_u`$ 中最像正类的样本近似集合 $`A`$：

```text
输入：正类样本 Sp，未标记样本 Su，复制比例 p，底层 CPE 算法 A

1. 用 Sp 标记为来源正类、Su 标记为来源未标记，训练分类器 h。
2. 对 Su 中每个样本计算 h 的 positive probability。
3. 选择 positive probability 最大的前 ceil(p * |Su|) 个样本 A_hat。
4. 将 A_hat 复制到 Sp，形成辅助正类集合 Sp'。
5. 用 Su 和 Sp' 调用底层 CPE 算法 A。
6. 输出底层算法得到的类别先验估计 pi_hat'。
```

---

## 5. 论文边界

- ReCPE 是一个**外层 regrouping 方法**，论文并未限定唯一的底层 CPE；最终效果依赖底层估计器和正类/未标记分类器。
- 论文中的 $`S_p`$ 应来自 $`P_p=p(x\mid y=1)`$，$`S_u`$ 应来自边缘分布 $`P_u=p(x)`$。如果数据来自 single-training-set，已标记正例需要能够代表完整正类分布。
- 论文希望缓解 irreducibility 失效导致的正向偏差，但不等于对任意数据分布都能无偏估计类别先验。
- 复制比例太小会使底层 CPE 对 regrouping 不敏感；太大则会显著改变辅助正类分布。论文实验统一使用 `p=10%`。
- 论文的 regrouped 构造保证 $`P_n'`$ 与 $`P_p'`$ 满足 anchor set assumption 而可识别；当 $`P_n(A)`$ 很小时估计接近原始 $`\pi`$，但该方法不承诺任意分布上的无偏性。

---

## 6. 实现注记

> 见 ADR-0014 #11（2026-08-10 默认 base CPE 改 `KernelMeanPriorEstimator(variant="km2")`；原 1% 分位 + 未校准 LR 坍缩 0.036 vs 0.5，KM2 升至 ~0.40）
>
> 见 ADR-0014 #12（v1.2.1 边界声明：全部 base 变体常规 SCAR 系统性低估 0.08–0.19，修复目标仅“不坍缩”≥0.1）

- 当前默认排序器为 sklearn `LogisticRegression`，不是论文实验中的两层神经网络；因此当前实现是算法逻辑对齐，而不是完整实验数值复现。
- 工程侧 classifier-based baseline（`_DensityRatioCPE`）保留为显式可选，不应与论文对比的 KM、AlphaMax 等方法混同。
- 当前自动化验证：`pytest -q`，133 项测试通过。
- clean-room 合成结果已归档于 `benchmarks/assigned_methods/results/clean_room_multiseed/`（以 benchmarks 产物为准）。默认 Logistic Regression 排序器和 density-ratio CPE 后端得到 prior MAE `0.2715 ± 0.0227`，明显差于同组 penL1。该负结果被保留为替换底层 CPE 和接入官方 FCNet/KM1/KM2 的回归基线，不代表论文 ReCPE 的数值表现。

---

## 7. 论文实验参考

| 项目 | 论文设置 |
|---|---|
| UCI 数据 | 数据集名称、类别映射、样本数和预处理必须从论文附录/官方仓库生成 manifest，不凭 sklearn 同名数据集猜测版本；每个 split 先保留真实标签，再只在训练数据上构造 $`S_p`$/ $`S_u`$ |
| 来源分类器 | 论文的两层神经网络路径；层宽、激活、优化器、epoch 和验证策略从官方配置逐项抄录并固定 |
| 复制比例 | `p=0.10` 作为论文主设置，其他比例只进入敏感性分析 |
| 底层 CPE | KM1、KM2、AlphaMax 等底层方法必须在原始与 regrouped 输入上成对运行，使用相同 split |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact` |
| Upstream URL | https://github.com/a5507203/Rethinking-Class-Prior-Estimation-for-Positive-Unlabeled-Learning |
| License | registry 中记录为 MIT；重新分发前应核验上游许可证 |
| Framework | 论文实验使用 Python 神经网络；本项目默认使用 NumPy + scikit-learn |
| 当前实现范围 | ReCPE regrouping 核心流程 + 可注入底层 CPE + 默认 baseline |
| 尚未完全复现 | 论文中的神经网络结构、验证集选择、UCI 全量实验和所有基线方法 |
| 复现风险 | 论文数值只有在官方网络、底层方法和数据 manifest 全部对齐后才能标记为 `paper_like`；当前 LR 排序器和默认 CPE 只进入 `clean_room_baseline` 组，不得命名为论文结果 |
