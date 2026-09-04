# Method Card: Kernel Mean Class-Prior Estimation（KM1 / KM2）

> 参数契约（签名、参数表、返回结构）以 [API 参考](../../user/reference/api.md) 为权威；本文档只记论文研究内容、实现边界与复现状态。

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Mixture Proportion Estimation via Kernel Embedding of Distributions |
| Authors | Harish G. Ramaswamy, Clayton Scott, Ambuj Tewari |
| Venue | ICML 2016（PMLR 48） |
| Year | 2016 |
| Setting | 从可靠正样本与未标记混合样本估计正类先验 |
| Requires class prior | `False`，方法输出先验估计 |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| GPU required | `False` |

---

## 2. 问题设定与符号

设未标记分布为 $`F`$，可靠正类分布为 $`H`$，未知负类分布为 $`G`$：

```math
F=(1-\kappa)G+\kappa H,\qquad 0\le\kappa<1.
```

输入样本满足：

```math
X_U\overset{i.i.d.}{\sim}F,\qquad X_P\overset{i.i.d.}{\sim}H.
```

该方法只估计 $`\pi=P(Y=1)`$，不直接训练分类器。

| 论文符号 | 含义 |
|---|---|
| $`\kappa`$ | 混合比例（正类先验） |
| $`\lambda`$ | $`1/(1-\kappa)`$ 重参数化 |
| $`\mu_Q`$ | 分布 $`Q`$ 的 RKHS 均值嵌入 |
| $`C`$ | 概率分布均值嵌入集合 |
| $`d(\lambda)`$ | 候选负类均值到 $`C`$ 的距离 |
| $`K`$ | Gram matrix |
| $`u_\lambda`$ | 目标混合向量 |
| $`v`$ | probability simplex 上的权重向量 |
| $`\tau`$ | 斜率阈值 |

---

## 3. 核心公式

### 3.1 RKHS 距离曲线

令 $`\phi(x)`$ 为核 $`k`$ 对应的 RKHS feature map，分布均值嵌入为 $`\mu_Q=E_Q[\phi(X)]`$。重参数化：

```math
\lambda=\frac{1}{1-\kappa},\qquad
\kappa=\frac{\lambda-1}{\lambda}.
```

则候选负类均值为：

```math
\lambda\mu_F+(1-\lambda)\mu_H.
```

算法计算该点到所有概率分布均值嵌入集合 $`C`$ 的距离：

```math
d(\lambda)=
\inf_{w\in C}
\left\|\lambda\mu_F+(1-\lambda)\mu_H-w\right\|_{\mathcal H}.
```

当 `lambda` 仍处于可行 mixture 区域时，距离接近零；越过真实拐点后，距离近似线性增长。KM1/KM2 通过距离曲线斜率超过阈值的位置估计拐点。

### 3.2 经验 QP

拼接 $`n_U+n_P=N`$ 个样本，记 Gram matrix 为 $`K`$。对候选 `lambda` 构造：

```math
u_\lambda=
\left[
\frac{\lambda}{n_U}\mathbf 1_{n_U},
\frac{1-\lambda}{n_P}\mathbf 1_{n_P}
\right].
```

经验距离通过 probability simplex 上的二次规划获得：

```math
\widehat d(\lambda)^2
=\min_{v\ge0,\ \mathbf 1^Tv=1}
(u_\lambda-v)^T K(u_\lambda-v).
```

### 3.3 KM1 与 KM2 阈值

算法用有限差分估计斜率：

```math
\widehat d'(\lambda)
\approx
\frac{\widehat d(\lambda+\epsilon/2)-\widehat d(\lambda)}{\epsilon/2}.
```

KM1 使用理论阈值：

```math
\tau_{KM1}=\frac{1}{\sqrt{\min(n_U,n_P)}}.
```

KM2 使用启发式阈值，与作者代码一致，默认由初始斜率和经验 RKHS 分布距离加权：

```math
\tau_{KM2}=0.8s_{initial}+0.2\|\widehat\mu_F-\widehat\mu_H\|_{\mathcal H}.
```

在 $`[1, \lambda_{upper\_bound}]`$ 上二分搜索首个超过阈值的位置，最后转换为 $`\kappa=(\lambda-1)/\lambda`$。

### 3.4 RBF 核与复杂度

实现采用 RBF kernel：

```math
k(x,x')=\exp\left(-\frac{\|x-x'\|^2}{2\sigma^2}\right).
```

Gram matrix 的时间和内存均为 $`O(N^2)`$；每个候选宽度需要一次 kernel matrix 评估；每个二分点包含两次 simplex QP。全量大数据运行前应估算内存。

---

## 4. 算法概要

1. 拼接 $`n_U+n_P=N`$ 个样本，计算核矩阵 $`K`$（每个候选宽度一次）。
2. 对候选 `lambda` 构造 $`u_\lambda`$，在 probability simplex 上求解 QP 得到 $`\widehat d(\lambda)^2`$。
3. 用有限差分估计斜率 $`\widehat d'(\lambda)`$。
4. KM1 用理论阈值 $`1/\sqrt{\min(n_U,n_P)}`$；KM2 用启发式阈值 $`0.8s_{initial}+0.2\|\widehat\mu_F-\widehat\mu_H\|`$。
5. 在 $`[1, \lambda_{upper\_bound}]`$ 上二分搜索首个斜率超过阈值的位置。
6. 转换为 $`\kappa=(\lambda-1)/\lambda`$；实现同时计算 KM1/KM2，由所选变体决定最终输出。

---

## 5. 论文边界

- 该方法只估计 $`\pi=P(Y=1)`$，不直接训练分类器，也不输出 posterior probability。
- **可识别性**：仅凭两个分布不能无条件识别任意 mixture proportion。论文的可辨识结论依赖 $`G`$ 相对于 $`H`$ 的不可约性；若负类分布本身含有可分解出的 $`H`$ 成分，算法估计的是最大可行 mixture proportion，而不一定是生成过程写下的名义比例。
- $`H=p(x\mid Y=1)`$ 仅在标准 SCAR/case-control PU 设定下成立；若可靠正例受选择偏差影响，$`H`$ 不再等于总体正类条件分布，此估计量的目标也随之改变。
- 输出被裁剪到 $`[0,1]`$；下游分类器仍要求严格位于 $`(0,1)`$，退化估计会被拒绝。
- 拐点由斜率阈值启发式判定：KM1 阈值有理论依据，KM2 阈值含经验加权项（$`0.8s_{initial}+0.2\|\widehat\mu_F-\widehat\mu_H\|`$），后者是作者代码的启发式约定，不是分布无关保证。

---

## 6. 实现注记

> 见 ADR-0014 #4（`width_selection="relative"` 默认（2026-08-10）；`mmd_grid` 系统性偏选宽带宽低估 km1 0.30→0.59、km2 0.31→0.46，真值 0.5；论文协议复现须显式 pin `width_selection="mmd_grid"`）

- **数值后端**：作者 Python 2.7 代码使用 CVXOPT；本项目为避免新增重型 solver 依赖，使用带精确线搜索的 Frank-Wolfe 算法求同一个 simplex QP，并记录每次求解的迭代数与最终 dual gap。该变化是数值后端差异，不是目标函数变化；正式复现仍应检查 QP gap 和容差敏感性。
- **standardize**：`standardize=False` 与作者 kernel routine 的直接输入行为一致。若上游没有稳定尺度，应在训练数据内标准化并记录统计量。
- **InfoMax-PU 复现差距**：InfoMax-PU 论文在表示学习后使用 kernel-mean estimator，但没有说明使用 KM1 还是 KM2，paper-protocol 配置暂时显式锁定 KM1，并保留为复现差距；其 20-seed paper-protocol 尚未实际执行，现阶段不能声称复现论文数值。
- 降采样（`max_samples_per_group`）会改变统计估计量，必须进入实验配置和结果记录。

---

## 7. 论文实验参考

| 项目 | 论文设置 |
|---|---|
| 评估设置 | 原卡未记录论文实验细节；以论文原文与作者软件页面为准（见 §8） |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact`（论文 + 作者代码；作者代码为 Python 2.7 + CVXOPT） |
| 论文页面 | <https://proceedings.mlr.press/v48/ramaswamy16.html> |
| 作者软件页面 | <https://web.eecs.umich.edu/~cscott/code.html#kmpe> |
| 实现方式 | NumPy/sklearn 原生实现，Frank-Wolfe simplex QP（替代作者 CVXOPT，见 §6） |
| 复现差距 | `width_selection` 默认 `"relative"` 与作者 max-MMD 5 档宽度搜索不同，论文协议须 pin `"mmd_grid"`（见 ADR-0014 #4）；InfoMax 表示空间的 20-seed paper-protocol 未执行 |
| 复现风险 | Gram matrix 为 $`O(N^2)`$ 时间与内存；降采样改变统计估计量，必须进入实验配置和结果记录；KM2 阈值含经验加权项，需按作者代码对齐 |

参考资料：

1. Ramaswamy, Scott, Tewari. *Mixture Proportion Estimation via Kernel Embedding of Distributions*. ICML 2016.
2. 论文页面：<https://proceedings.mlr.press/v48/ramaswamy16.html>
3. 作者软件页面：<https://web.eecs.umich.edu/~cscott/code.html#kmpe>
4. InfoMax-PU：<https://arxiv.org/abs/1710.05359>
