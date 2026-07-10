# Method Card: Class-Prior Estimation for PU Learning

## 1. 待办与注意

### 1.1 待办

- **实现 penL1**：`PenL1Estimator(BasePriorEstimator)`，放入 `pu_toolbox/prior/pen_l1.py`。
- **实现 L1**：`L1PriorEstimator(BasePriorEstimator)`，optional。QP 求解使用开源替代（如 `scipy.optimize`），论文用 Gurobi。
- **设计 CV protocol**：$\sigma$、$\lambda$ 无默认值，论文只说通过 CV 选择，需自行设计具体 CV 流程。
- **输入标准化**：Gaussian kernel 对特征尺度敏感，计算前做标准化。
- **single-training-set 适配**：论文假设 positive set 直接从 $p(x\mid y=1)$ 抽样。若项目输入是 single-training-set `y_pu`，需保证已标注正样本能无偏代表 $p(x\mid y=1)$。
- **`confidence_interval()`**：返回 `NotImplemented`。论文未给出 CI / bootstrap / asymptotic normality。
- **写测试**：`tests/test_pen_l1.py`，覆盖合成实验（§8.1）和边界条件。

### 1.2 注意

- 必须同时有 positive set 和 unlabeled set，缺任一无法计算 $\beta_\ell$。
- **MATLAB 源码 ≠ 2017 论文**：MATLAB 是 L₂-RKHS + 矩阵求逆，论文是 penalized L₁ + 闭式解 thresholding。penL1 核心机制（$c=\infty$ → 闭式解）在 MATLAB 中不存在，必须从论文公式直接翻译。
- L1 每个候选 $\theta$ 都要求解 QP，计算代价远高于 penL1，不适合大规模数据或密集 $\theta$ 网格。
- PE、EN、SB 是本文对比基线，来自其他论文，不在此实现。
- 论文未使用 SCAR 术语，未分析 SCAR 违反时的偏差方向。

---

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Class-prior Estimation for Learning from Positive and Unlabeled Data |
| Authors | Marthinus C. du Plessis, Gang Niu, Masashi Sugiyama |
| Venue | MLJ |
| Year | 2017 |
| Family | `class_prior_estimation` |
| Setting | Two datasets: positive set $X \sim p(x\mid y=1)$，unlabeled set $X' \sim p(x)$ |
| Requires class prior | `False`，该方法输出 $\hat{\pi}$ |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| GPU required | `False` |

### Assumptions

$$
X=\{x_i\}_{i=1}^{n} \overset{i.i.d.}{\sim} p(x\mid y=1),
\qquad
X'=\{x'_j\}_{j=1}^{n'} \overset{i.i.d.}{\sim} p(x)
$$

$$
p(x)=\pi p(x\mid y=1)+(1-\pi)p(x\mid y=-1)
$$

---

## 3. 符号与记号

| 论文符号 | 含义 | 开发侧对应 |
|---|---|---|
| $\pi=p(y=1)$ | 真实类别先验 | `class_prior` |
| $\hat{\pi}$ | 估计出的类别先验 | estimator output |
| $\theta$ | 候选类别先验 | candidate prior |
| $X=\{x_i\}_{i=1}^n$ | positive samples | `P` |
| $X'=\{x'_j\}_{j=1}^{n'}$ | unlabeled samples | `U` |
| $p(x\mid y=1)$ | 正类条件分布 | positive density |
| $p(x\mid y=-1)$ | 负类条件分布 | negative density |
| $p(x)$ | unlabeled marginal density | unlabeled density |
| $f(t)$ | divergence generator | divergence function |
| $\tilde f(t)$ | penalized $f$-divergence generator | penalized divergence |
| $f^*(z)$ | Fenchel conjugate | conjugate |
| $r(x)$ | Fenchel dual function | scoring function (linear model) |
| $\phi_\ell(x)$ | non-negative basis function | Gaussian kernel basis |
| $\alpha_\ell$ | basis coefficient | coefficient |
| $\beta_\ell$ | empirical coefficient used in objective | precomputed statistic |
| $\lambda$ | $\ell_2$ regularization coefficient for $\alpha$ | regularization |
| $c$ | penalty parameter ($\infty$ for penL1, $1$ for L1) | penalty control |

---

## 4. 核心公式

### 4.1 Penalized $f$-divergence 框架

Partial distribution matching:

$$
\theta=\arg\min_{0\le\theta\le1} \mathrm{Div}_f(\theta),
\qquad
\mathrm{Div}_f(\theta) = \int f\left(\frac{\theta p(x\mid y=1)}{p(x)}\right)p(x)\,dx
$$

Penalized $f$-divergence（惩罚 $\theta p(x\mid y=1)>p(x)$ 的区域）：

$$
\tilde f(t)=
\begin{cases}
f(t), & 0\le t\le 1,\\
\infty, & t>1.
\end{cases}
$$

Fenchel dual 样本均值形式（用 $\tilde f^*$ 替代 $f^*$ 即得 penalized 版本）：

$$
\widehat{\mathrm{Div}}_f(\theta) \ge \sup_r \left[
\frac{\theta}{n}\sum_{i=1}^{n}r(x_i)
- \frac{1}{n'}\sum_{j=1}^{n'}f^*(r(x'_j))
\right]
$$

### 4.2 共享模型

线性模型 + Gaussian kernel basis：

$$
r(x)=\sum_{\ell=1}^{b}\alpha_\ell\phi_\ell(x)-1
$$

$$
\phi_\ell(x) = \exp\left(-\frac{\|x-c_\ell\|^2}{2\sigma^2}\right),
\qquad 0<\phi_\ell(x)\le 1
$$

实验中 Gaussian centers 取全部训练样本，$b=n+n'$。

对每个候选 $\theta$，先计算：

$$
\beta_\ell(\theta) = \frac{\theta}{n}\sum_{i=1}^{n}\phi_\ell(x_i) - \frac{1}{n'}\sum_{j=1}^{n'}\phi_\ell(x'_j)
$$

### 4.3 penL1（$c=\infty$）

内层闭式解：

$$
\hat{\alpha}_\ell(\theta) = \frac{1}{\lambda}\max(0,\beta_\ell(\theta))
$$

经验目标：

$$
\widehat{\mathrm{penL}}_1(\theta) = \frac{1}{\lambda}\sum_{\ell=1}^{b}\max(0,\beta_\ell(\theta))\beta_\ell(\theta) - \theta + 1
$$

最终估计：

$$
\hat{\pi} = \arg\min_{0\le\theta\le1} \widehat{\mathrm{penL}}_1(\theta)
$$

### 4.4 L1（$c=1$）

内层为带约束二次规划：

$$
\hat{\alpha}(\theta) = \arg\min_{\alpha} \left[ \frac{\lambda}{2}\sum_{\ell=1}^{b}\alpha_\ell^2 - \sum_{\ell=1}^{b}\alpha_\ell\beta_\ell(\theta) \right]
$$

subject to:

$$
\sum_{\ell=1}^{b}\alpha_\ell\phi_\ell(x'_j) \le 2 \quad (j=1,\ldots,n'),\qquad
\alpha_\ell \ge 0 \quad (\ell=1,\ldots,b)
$$

经验目标：

$$
\widehat L_1(\theta) = \hat{\alpha}(\theta)^\top\hat{\beta}(\theta) - \theta + 1
$$

最终估计：

$$
\hat{\pi} = \arg\min_{0\le\theta\le1} \widehat L_1(\theta)
$$

### 4.5 超参数

| 参数 | 含义 | 论文设置 |
|---|---|---|
| $\lambda$ | $\frac{\lambda}{2}\sum_\ell\alpha_\ell^2$ 正则化系数 | 无默认值；CV 选择 |
| $\sigma$ | Gaussian kernel width | 无默认值；CV 选择 |
| $b$ | basis functions 数量 | $b=n+n'$（全部训练样本为中心） |
| $c$ | penalty parameter | penL1: $\infty$，L1: $1$ |
| $\theta$ | candidate class prior | grid search $[0,1]$；论文未指定搜索方式 |

---

## 5. 算法概要

两种方法共享框架：对 $\theta$ 做 grid search，内层用 Gaussian kernel basis 计算 $\beta_\ell(\theta)$，以对应目标函数选最优 $\theta$。$\sigma$ 和 $\lambda$ 按论文要求通过 CV 选择（论文未给出具体 protocol）。

- **penL1**：内层闭式解 $\hat{\alpha}_\ell = \max(0, \beta_\ell) / \lambda$，$O(b)$。
- **L1**：内层每个候选 $\theta$ 解一次带 $n'$ 个线性约束的 QP，计算代价远高于 penL1。

---

## 6. 源码状态

| 字段 | 内容 |
|---|---|
| Source status | `official_related` |
| Upstream URL | http://www.mcduplessis.com/index.php/software/ |
| License | `needs_review` |
| Framework | MATLAB |
| 实际对应论文 | ICML 2012 (du Plessis & Sugiyama)，不是 2017 MLJ 论文 |
| 包含方法 | `LSDDPriorEstMedian.m`（L₂-distance）、`pe_prior_est_grid.m`（Pearson divergence） |

---

## 7. API 接口

| API | 约定 |
|---|---|
| `fit(X, y_pu)` | `y_pu` 中 `+1` 为 positive、`0` 为 unlabeled（项目约定，非论文符号） |
| `estimate()` | 返回标量 `float`，即 $\hat{\pi}$ |
| `confidence_interval(alpha)` | 返回 `NotImplemented` |
| `get_params()` / `set_params()` | 暴露 `theta_grid`（或一维搜索策略）、`sigma_candidates`、`lambda_candidates`、basis-center 策略 |

---

## 8. 测试参考

### 8.1 Synthetic overlap test

论文 1D 实验设置：

```text
p(x | y = 1)  = U(0, 1)
p(x | y = -1) = U(1 - gamma, 2 - gamma)
pi_true = 0.7
gamma in {0.25, 0.75}
```

### 8.2 MNIST one-vs-rest test

```text
For digit k:
    positive = digit k, negative = all others
    PCA to 4 dimensions
    evaluate class-prior squared error
```

### 8.3 Convergence sanity test

增大 $n$ 和 $n'$（保持数据分布不变），预期平均估计误差下降。tolerance 根据项目数据规模和随机种子单独设定。
