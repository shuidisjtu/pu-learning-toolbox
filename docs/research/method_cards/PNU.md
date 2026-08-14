# Method Card: PNU Semi-Supervised Classification

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Semi-Supervised Classification Based on Classification from Positive and Unlabeled Data |
| Authors | Tomoya Sakai, Marthinus Christoffel du Plessis, Gang Niu, Masashi Sugiyama |
| Venue | ICML / PMLR 70 |
| Year | 2017 |
| Setting | 有标注正集 $`X_P`$、负集 $`X_N`$ 与未标记集 $`X_U`$ 的二分类 |
| Requires class prior | `True`：$`\theta_P`$（$`\theta_N=1-\theta_P`$） |
| Requires propensity | `False` |
| Requires negative samples | `True` |
| GPU required | `False` |

### Assumptions

```math
X_P\overset{i.i.d.}{\sim}p_P(x)=p(x\mid y=+1),\quad
X_N\overset{i.i.d.}{\sim}p_N(x)=p(x\mid y=-1),\quad
X_U\overset{i.i.d.}{\sim}p(x)=\theta_Pp_P(x)+\theta_Np_N(x).
```

不要求 cluster / manifold / low-density separation 等传统半监督分布假设。

---

## 2. 问题设定与符号

PNU 通过组合 PN、PU、NU 三种经验风险训练二分类器：PN 项需要正、负标签，PU/NU 项把未标记样本分别按负类/正类方向使用，从而在 $`R_{PN}`$ 与 $`R_{PU}`$、$`R_{NU}`$ 之间以参数 $`\eta`$ 插值。

| 论文符号 | 含义 |
|---|---|
| $`g(x)`$ | 实值决策函数，按 $`\mathrm{sign}(g(x))`$ 分类 |
| $`\ell(m)`$ | margin loss，$`m=yg(x)`$ |
| $`R_P,R_N`$ | P/N 条件风险 |
| $`R_{U,P},R_{U,N}`$ | U 上取 $`\ell(g)`$ / $`\ell(-g)`$ 的风险 |
| $`R_{PN},R_{PU},R_{NU}`$ | PN、PU、NU 总风险 |
| $`\widetilde\ell(m)`$ | composite loss：$`\ell(m)-\ell(-m)`$ |
| $`\theta_P,\theta_N`$ | 正/负类先验，和为 1 |
| $`\eta\in[-1,1]`$ | PNU 取舍参数 |
| $`\gamma\in[0,1]`$ | PNPU / PNNU 中的权重 |

---

## 3. 核心公式

### 3.1 基础 PN、PU、NU 风险

```math
R_{PN}(g)=\theta_PR_P(g)+\theta_NR_N(g).
```

定义复合损失 $`\widetilde\ell(m)=\ell(m)-\ell(-m)`$。

```math
R_{PU}(g)=\theta_P\mathbb E_P[\widetilde\ell(g(x))]+\mathbb E_U[\ell(-g(x))].
```

```math
R_{NU}(g)=\theta_N\mathbb E_N[\widetilde\ell(-g(x))]+\mathbb E_U[\ell(g(x))].
```

它们与 $`R_{PN}`$ 有相同的总体风险；实现时将每个期望替换为对应样本均值。注意 composite loss 落在 P/N 条件项上，不是在 U 项上。

### 3.2 PNU 风险

```math
R_{PNPU}^{\gamma}(g)=(1-\gamma)R_{PN}(g)+\gamma R_{PU}(g),
```

```math
R_{PNNU}^{\gamma}(g)=(1-\gamma)R_{PN}(g)+\gamma R_{NU}(g).
```

```math
R_{PNU}^{\eta}(g)=
\begin{cases}
R_{PNPU}^{\eta}(g), & \eta\ge0,\\
R_{PNNU}^{-\eta}(g), & \eta<0.
\end{cases}
```

端点退化：$`\eta=-1,0,+1`$ 分别退化为 NU、PN、PU。

### 3.3 首选：凸实现（平方损失）

选择满足 $`\ell(m)-\ell(-m)=-m`$ 的凸 surrogate；论文实验训练使用平方损失 $`\ell_S(m)=(1-m)^2/4`$。此时：

```math
R_{C\text{-}PU}(g)=\theta_PR_P^L(g)+R_{U,N}(g),\qquad
R_P^L(g)=\mathbb E_P[-g(x)],
```

```math
R_{C\text{-}NU}(g)=\theta_NR_N^L(g)+R_{U,P}(g),\qquad
R_N^L(g)=\mathbb E_N[g(x)].
```

将上式代入 §3.2，即可得到可微、凸（在线性模型 + $`\ell_2`$ 正则下）的 PNU 目标。训练目标：

```math
\min_w\ \widehat R_{PNU}^{\eta}(g_w)+\lambda\lVert w\rVert_2^2.
```

### 3.4 非凸实现（可选）

若损失满足 $`\ell(m)+\ell(-m)=1`$，可用 ramp loss：

```math
\ell_R(m)=\tfrac12\max(0,\min(2,1-m)).
```

对应 PU/NU 风险可改写为无偏非凸目标，需 CCCP 求局部解（实现状态见第 6 节）。

### 3.5 $`\eta`$ 的理论启发

设 $`\psi_P=\theta_P^2\sigma_P^2(g)/n_P`$、$`\psi_N=\theta_N^2\sigma_N^2(g)/n_N`$，在 $`n_U\to\infty`$、固定 $`g`$ 下：

```math
\gamma_{N\text{-}PNPU}^*=\frac{\psi_N-\psi_P}{\psi_P+\psi_N},\qquad
\gamma_{N\text{-}PNNU}^*=\frac{\psi_P-\psi_N}{\psi_P+\psi_N}.
```

仅在相应 $`\gamma^*\in[0,1]`$ 的分支使用；论文的大型实验取 $`\sigma_P(g)=\sigma_N(g)`$ 作为近似，再以五折 CV 的 PNU 零一风险选择超参数。

---

## 4. 算法概要

1. 校验 $`X_P,X_N,X_U`$ 均非空，$`0<\theta_P<1`$。
2. 选择 `eta_grid`（至少含 `[-1, 0, 1]`）和模型正则化参数；对每组参数做分层交叉验证。
3. 对每个 $`\eta`$，构造 $`\widehat R_{PNPU}^{\eta}`$ 或 $`\widehat R_{PNNU}^{-\eta}`$，并最小化加正则项的经验风险。
4. 用独立验证集/折的同一 PNU 风险选最优参数；重训最终模型。
5. 输出 classifier；同时保存所用先验、$`\eta`$ 与各风险分量，便于诊断。

---

## 5. 论文边界

- PNU 不是“仅有 P 与 U”的 PU 分类器：训练必须同时有正、负、未标记样本。
- 论文假设三个集合分别独立同分布于 $`p(x\mid y=+1)`$、$`p(x\mid y=-1)`$、$`p(x)`$。若数据只有 `+1/0` 标签，不能直接使用；必须额外提供带负标签数据。
- 该方法需要正类先验 $`\theta_P`$，论文实验中将其视为已知或先估计；先验估计误差不在本文保证范围内。
- 无偏风险在灵活模型下可出现负经验风险/过拟合；论文结论指出未来工作可结合 nnPU 非负校正。
- 不要求 cluster / manifold / low-density separation 等传统半监督分布假设。

---

## 6. 实现注记

- **[状态]** 已实现为 native 分类器（NumPy 闭式解，2026-07-18），接口遵循 `BasePUClassifier` 契约；`eta` 的交叉验证选择未实现——论文方差公式仅在特定条件下给出理论依据，不能作为无验证数据时的通用默认值。
- **ramp loss + CCCP（v1 不实现）**：优化复杂、可重复性和维护成本高，且平方损失已覆盖论文主实验路径。
- **`eta` 选择**：对实现而言，优先直接在验证折上搜索 `eta`，$`\gamma^*`$ 公式只可作为候选网格中心或初始化。
- **[项目适配]** 当前 toolbox 已有 nnPU，“PNU + 非负校正”留为后续扩展，不能宣称本文已给出该公式。
- **论文实验设定**：Gaussian-kernel 设定（中心为 $`X_P\cup X_N`$，带宽候选为 `{1/8, 1/4, 1/2, 1, 3/2, 2} × median_pairwise_distance`）可作为 benchmark，但不应写死到通用实现。

---

## 7. 论文实验参考

| 项目 | 论文设置 |
|---|---|
| 实验模型 | Gaussian kernel（中心为 $`X_P\cup X_N`$，带宽候选 `{1/8, 1/4, 1/2, 1, 3/2, 2} × median_pairwise_distance`） |
| 损失 | 平方损失（凸实现路径） |
| 超参选择 | 大型实验取 $`\sigma_P(g)=\sigma_N(g)`$ 作为近似，再以五折 CV 的 PNU zero-one 风险选择超参数 |
| 端点设定 | $`\eta\in\{-1,0,+1\}`$ 分别对应 NU、PN、PU 退化情形 |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact` |
| Official code | [`t-sakai-kure/pywsl`](https://github.com/t-sakai-kure/pywsl)（Sugiyama Lab，MIT） |
| 实现状态 | native 实现（backend NUMPY，squared loss 闭式解） |
| 复现参考 | 以论文公式为数学权威，以 `pywsl` 为源码级复现参考；不得再将 PNU 记为仅论文依据的方法 |
| 复现风险 | 无偏风险在灵活模型下可出现负风险/过拟合；$`\eta`$ 需验证数据选择，论文方差公式仅在特定条件下成立 |
