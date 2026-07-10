# Method Card: Class-Prior Estimation for PU Learning

## 1. Basic Info

| 字段 | 内容 |
|---|---|
| Paper | Class-prior Estimation for Learning from Positive and Unlabeled Data |
| Authors | Marthinus C. du Plessis, Gang Niu, Masashi Sugiyama |
| Source / Venue | MLJ |
| Year | 2017 |
| Family | `class_prior_estimation` |
| Setting | Positive and unlabeled data with two datasets: positive set \(X\) and unlabeled set \(X'\) |
| Requires class prior | `False`，该方法输出 \(\hat{\pi}\) |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| GPU required | `False` |

### Paper assumptions

论文设定为：

$$
X=\{x_i\}_{i=1}^{n} \overset{i.i.d.}{\sim} p(x\mid y=1),
\qquad
X'=\{x'_j\}_{j=1}^{n'} \overset{i.i.d.}{\sim} p(x)
$$

其中未标注分布满足：

$$
p(x)=\pi p(x\mid y=1)+(1-\pi)p(x\mid y=-1)
$$

**项目适配标注**：论文直接假设存在从 \(p(x\mid y=1)\) 抽样的 positive set。如果项目输入是 single-training-set 格式的 `y_pu`，且正样本来自选择机制，则需要额外假设这些已标注正样本能无偏代表 \(p(x\mid y=1)\)。论文没有使用 SCAR 术语，也没有分析 SCAR 违反时的偏差方向。

---

## 2. Problem Setup & Notation

| 论文符号 | 含义 | 开发侧对应 |
|---|---|---|
| \(\pi=p(y=1)\) | 真实类别先验 | `class_prior` |
| \(\hat{\pi}\) | 估计出的类别先验 | estimator output |
| \(\theta\) | 候选类别先验 | candidate prior |
| \(X=\{x_i\}_{i=1}^n\) | positive samples | `P` |
| \(X'=\{x'_j\}_{j=1}^{n'}\) | unlabeled samples | `U` |
| \(p(x\mid y=1)\) | 正类条件分布 | positive density |
| \(p(x\mid y=-1)\) | 负类条件分布 | negative density |
| \(p(x)\) | unlabeled marginal density | unlabeled density |
| \(q(x;\theta)=\theta p(x\mid y=1)\) | partial matching model | partial model |
| \(f(t)\) | divergence generator | divergence function |
| \(\tilde f(t)\) | penalized divergence generator | penalized divergence |
| \(f^*(z)\), \(\tilde f^*(z)\) | Fenchel conjugate | conjugate |
| \(r(x)\) | Fenchel dual function / model function | scoring function |
| \(\phi_\ell(x)\) | non-negative basis function | kernel basis |
| \(\alpha_\ell\) | basis coefficient | coefficient |
| \(\beta_\ell\) | empirical coefficient used in objective | precomputed statistic |
| \(\lambda\) | \(\ell_2\) regularization coefficient for \(\alpha\) | regularization |
| \(c\) | finite/infinite penalty parameter | penalty control |

---

## 3. Methods Overview

### 3.1 Core idea

已有 partial distribution matching 方法使用：

$$
\theta=\arg\min_{0\le\theta\le1}
\mathrm{Div}_f(\theta),
\qquad
\mathrm{Div}_f(\theta)
=
\int f\left(
\frac{\theta p(x\mid y=1)}{p(x)}
\right)p(x)\,dx
$$

论文指出：当正负类条件分布存在重叠时，这类未惩罚的 partial matching 方法会系统性高估 \(\pi\)。论文提出使用 penalized \(f\)-divergence：

$$
\tilde f(t)=
\begin{cases}
f(t), & 0\le t\le 1,\\
\infty, & t>1.
\end{cases}
$$

其目的在于惩罚 \(\theta p(x\mid y=1)>p(x)\) 的区域，从而修正高估问题。

### 3.2 Methods in the paper

| 方法 | 类型 | 核心思想 | 实现状态建议 |
|---|---|---|---|
| penL1 | proposed | penalized \(L_1\)-distance with \(c=\infty\)，内层 \(\alpha\) 有解析解 | 推荐作为主实现 |
| L1 | proposed | ordinary \(L_1\)-distance estimation，即 \(c=1\) 的有限约束版本 | 可作为小规模对照实现；计算慢 |
| EN | baseline | Elkan and Noto 方法；可解释为 partial matching | 仅作为基线 |
| PE | baseline | Pearson-divergence matching | 仅作为基线；重叠时可能高估 |
| SB | baseline | Neyman-Pearson / ROC endpoint 方法 | 仅作为基线；端点估计可能不稳定 |

---

## 4. Core Formulas

### 4.1 Penalized \(f\)-divergence direct estimation

通过 Fenchel duality，论文将 divergence estimation 写成样本均值形式：

$$
\widehat{\mathrm{Div}}_f(\theta)
\ge
\sup_r
\left[
\frac{\theta}{n}\sum_{i=1}^{n}r(x_i)
-
\frac{1}{n'}\sum_{j=1}^{n'}f^*(r(x'_j))
\right]
$$

对 penalized divergence 使用 \(\tilde f^*\) 替代 \(f^*\)。

### 4.2 Shared model for L1-based estimators

论文采用线性模型：

$$
r(x)=\sum_{\ell=1}^{b}\alpha_\ell\phi_\ell(x)-1
$$

其中 \(\phi_\ell(x)\) 是非负 basis function。实验中使用 Gaussian kernels，并以所有训练样本为中心：

$$
\phi_\ell(x)=
\exp\left(
-\frac{\|x-c_\ell\|^2}{2\sigma^2}
\right)
$$

论文理论部分假设 basis functions 有界且严格为正：

$$
0<\phi_\ell(x)\le 1
$$

---

## 5. Estimator: penL1

### 5.1 Objective

penL1 对应 \(c=\infty\)。对每个候选 \(\theta\)，先计算：

$$
\beta_\ell(\theta)
=
\frac{\theta}{n}
\sum_{i=1}^{n}
\phi_\ell(x_i)
-
\frac{1}{n'}
\sum_{j=1}^{n'}
\phi_\ell(x'_j)
$$

内层解析解为：

$$
\hat{\alpha}_\ell(\theta)
=
\frac{1}{\lambda}\max(0,\beta_\ell(\theta))
$$

经验目标函数为：

$$
\widehat{\mathrm{penL}}_1(\theta)
=
\frac{1}{\lambda}
\sum_{\ell=1}^{b}
\max(0,\beta_\ell(\theta))\beta_\ell(\theta)
-\theta+1
$$

最终估计：

$$
\hat{\pi}
=
\arg\min_{0\le\theta\le1}
\widehat{\mathrm{penL}}_1(\theta)
$$

### 5.2 Optimization

- 对固定 \(\theta\)，\(\hat{\alpha}_\ell\) 有闭式解，只需 `max` 操作。
- 外层选择使 \(\widehat{\mathrm{penL}}_1(\theta)\) 最小的 \(\theta\)。
- 论文没有指定外层 \(\theta\) 的数值搜索方式。
- 论文说明 \(\sigma\) 和 \(\lambda\) 对每个 \(\theta\) 通过 cross-validation 选择，但没有给出具体 CV protocol。

**项目适配标注**：实现时可以使用一维 grid search 或 bounded scalar optimization 搜索 \(\theta\)。若使用 grid search，应将网格密度作为项目超参数，而不是论文默认值。

### 5.3 Hyperparameters

| 参数 | 含义 | 论文给出的设置 |
|---|---|---|
| \(\lambda\) | \(\frac{\lambda}{2}\sum_{\ell=1}^b\alpha_\ell^2\) 的正则化系数 | 无默认值；通过 cross-validation 选择 |
| \(\sigma\) | Gaussian kernel width | 无默认值；通过 cross-validation 选择 |
| \(b\) | basis functions 数量 | 实验中 Gaussian kernels centered at all training samples，通常 \(b=n+n'\) |
| \(c\) | penalty parameter | penL1 固定为 \(c=\infty\) |
| \(\theta\) | candidate class prior | 搜索范围 \(0\le\theta\le1\) |

---

## 6. Estimator: L1

### 6.1 Objective

L1 是 \(c=1\) 的有限约束版本，即 ordinary \(L_1\)-distance estimation。为了与 penL1 区分，本文档记其目标为 \(\widehat L_1(\theta)\)。

对固定 \(\theta\)，计算同样的：

$$
\beta_\ell(\theta)
=
\frac{\theta}{n}
\sum_{i=1}^{n}
\phi_\ell(x_i)
-
\frac{1}{n'}
\sum_{j=1}^{n'}
\phi_\ell(x'_j)
$$

然后求解二次规划：

$$
\hat{\alpha}(\theta)
=
\arg\min_{\alpha}
\left[
\frac{\lambda}{2}\sum_{\ell=1}^{b}\alpha_\ell^2
-
\sum_{\ell=1}^{b}\alpha_\ell\beta_\ell(\theta)
\right]
$$

subject to：

$$
\sum_{\ell=1}^{b}\alpha_\ell\phi_\ell(x'_j)-1\le c,
\qquad j=1,\ldots,n'
$$

$$
\alpha_\ell\ge0,
\qquad \ell=1,\ldots,b
$$

对 L1，\(c=1\)，因此约束等价于：

$$
\sum_{\ell=1}^{b}\alpha_\ell\phi_\ell(x'_j)\le2,
\qquad j=1,\ldots,n'
$$

经验目标可写为：

$$
\widehat L_1(\theta)
=
\hat{\alpha}(\theta)^\top\hat{\beta}(\theta)
-\theta+1
$$

最终估计：

$$
\hat{\pi}
=
\arg\min_{0\le\theta\le1}
\widehat L_1(\theta)
$$

### 6.2 Optimization

- 对每个候选 \(\theta\)，需要求解一次带约束二次规划。
- 论文实验中使用 Gurobi 求解该二次规划。
- 由于每个候选 class prior 都要解 QP，论文指出 L1 在实践中非常慢。
- 因计算成本高，L1 没有用于 Sec. 5.3 的大规模 MNIST 实验。

---

## 7. Algorithm Pseudocode

### 7.1 penL1 pseudocode

```text
Input:
    X        : feature matrix
    y_pu     : +1 for positive, 0 for unlabeled   [project convention]
    theta_grid
    sigma_candidates
    lambda_candidates

Output:
    pi_hat

1. Split data:
    P = {x_i | y_pu_i = +1}
    U = {x'_j | y_pu_j = 0}
    n = |P|, n_prime = |U|

2. Initialize:
    best_score = +infinity
    pi_hat = None

3. For theta in theta_grid:
    3.1 Select sigma and lambda by cross-validation for this theta.
        Note: the paper says to use cross-validation, but does not specify the protocol.
    3.2 Set Gaussian centers C = P union U.
    3.3 For each center c_l in C, define:
        phi_l(x) = exp(-||x - c_l||^2 / (2 * sigma^2))
    3.4 For each basis l, compute:
        beta_l = theta / n * sum_{x in P} phi_l(x)
                 - 1 / n_prime * sum_{x in U} phi_l(x)
    3.5 Compute the closed-form inner solution:
        alpha_l = max(0, beta_l) / lambda
    3.6 Compute:
        score(theta) = (1 / lambda) * sum_l max(0, beta_l) * beta_l
                       - theta + 1
    3.7 If score(theta) < best_score:
        best_score = score(theta)
        pi_hat = theta

4. Return pi_hat
```

### 7.2 L1 pseudocode

```text
Input:
    X        : feature matrix
    y_pu     : +1 for positive, 0 for unlabeled   [project convention]
    theta_grid
    sigma_candidates
    lambda_candidates

Output:
    pi_hat

1. Split data:
    P = {x_i | y_pu_i = +1}
    U = {x'_j | y_pu_j = 0}
    n = |P|, n_prime = |U|

2. Initialize:
    best_score = +infinity
    pi_hat = None

3. For theta in theta_grid:
    3.1 Select sigma and lambda by cross-validation for this theta.
        Note: the paper does not specify the CV protocol.
    3.2 Set Gaussian centers C = P union U.
    3.3 Compute beta_l for all basis functions:
        beta_l = theta / n * sum_{x in P} phi_l(x)
                 - 1 / n_prime * sum_{x in U} phi_l(x)
    3.4 Solve QP:
        minimize_alpha:
            lambda / 2 * sum_l alpha_l^2 - sum_l alpha_l * beta_l
        subject to:
            alpha_l >= 0 for all l
            sum_l alpha_l * phi_l(x'_j) <= 2 for all x'_j in U
    3.5 Compute:
        score(theta) = alpha_hat^T beta - theta + 1
    3.6 If score(theta) < best_score:
        best_score = score(theta)
        pi_hat = theta

4. Return pi_hat
```

---

## 8. Theoretical Guarantees

论文对 penalized \(L_1\)-distance estimators 给出以下理论分析。

| 内容 | 结论 |
|---|---|
| Consistency for fixed \(\theta\), \(c=\infty\) | \(\|\hat{\alpha}_I-\alpha_I^*\|_2=O_p(1/\sqrt n+1/\sqrt{n'})\)，\(|\widehat{\mathrm{penL}}_1(\theta)-\mathrm{penL}_1^*(\theta)|=O_p(1/\sqrt n+1/\sqrt{n'})\) |
| Consistency for fixed \(\theta\), finite \(c\) | 一般情形下较复杂；在额外稳定条件下也可达到 \(O_p(1/\sqrt n+1/\sqrt{n'})\) |
| Stability | 给出 fixed-\(\theta\) deviation bound 和 uniform deviation bound |
| Estimation error | \(\hat{\theta}\) 相对于 \(\mathrm{penL}_1^*(\theta)\) 的 estimation error bound 具有 \(O(1/\sqrt n+1/\sqrt{n'})\) 量级 |

开发侧可使用的结论：positive 样本量 \(n\) 和 unlabeled 样本量 \(n'\) 增加时，理论上估计误差会下降。论文没有给出可直接转成标准误的 asymptotic normality 结果。

---

## 9. Confidence Interval

| 项目 | 结论 |
|---|---|
| 是否给出 \(\hat{\pi}\) 的置信区间 | 否 |
| 是否使用 bootstrap | 否 |
| 是否给出 asymptotic normality / standard error | 否 |
| 是否有可直接实现的 CI 公式 | 否 |

论文只给出一致性、稳定性和 estimation error bound，没有构造：

$$
\hat{\pi}\pm z_{\alpha/2}\mathrm{SE}(\hat{\pi})
$$

或：

$$
[\hat{\pi}_{lower},\hat{\pi}_{upper}]
$$

**项目适配标注**：若 `BasePriorEstimator.confidence_interval(alpha)` 是必须接口，建议返回 `NotImplemented` / `None` 并附带说明：该论文未提供置信区间构造方法。不要把 high-probability error bound 直接当作 \(\hat{\pi}\) 的置信区间。

---

## 10. Experimental Observations

### 10.1 Datasets

| 数据集 | 样本量 | 维度 | 备注 |
|---|---:|---:|---|
| Synthetic uniform distributions | 论文未给出 | 1 | \(p(x\mid y=1)=U(0,1)\)，\(p(x\mid y=-1)=U(1-\gamma,2-\gamma)\)；\(\gamma\in\{0.25,0.75\}\)，真实 \(\pi=0.7\) |
| MNIST one-vs-rest | 论文未给出 | PCA 降至 4 | 每次选一个数字为 positive class，其余数字为 negative class；降到 4 维用于增加类别重叠 |

### 10.2 Method behavior

| 方法 | 偏差 / 现象 | 方差 | 适合场景 |
|---|---|---|---|
| penL1 | 整体估计准确；MNIST 上 class-prior squared error 较低 | 论文未单独报告 | 首选；适合类别重叠、需要稳定估计并用于后续 PU 分类的场景 |
| L1 | \(\gamma=0.25\) 时表现合理；\(\gamma=0.75\) 时出现高估 | 论文未单独报告 | 小规模对照；不适合大规模候选 prior 搜索 |
| PE | 类别重叠时系统性高估；小 class prior 时误差较大 | 论文未单独报告 | baseline；类别重叠明显时不推荐 |
| EN | 与 partial matching 相关；MNIST 上小 class prior 时误差较大 | 论文未单独报告 | baseline |
| SB | 基于 Neyman-Pearson / ROC endpoint；论文指出 ROC 右端点在高维输入下可能不稳定 | 论文未单独报告 | baseline；实现复杂度和端点稳定性需注意 |

### 10.3 Key conclusions

1. penL1 是论文实验中最实用的选择：估计精度整体较好，且用于 PU classification 时，分类误差通常接近使用真实 class prior 的结果。
2. 类别重叠会导致未惩罚的 partial matching 方法高估 \(\pi\)。PE 在 \(\gamma=0.25\) 和 \(\gamma=0.75\) 下都高估；L1 在重叠更严重的 \(\gamma=0.75\) 下也会高估。
3. L1 比 PE 更稳一些，但计算代价高。论文明确指出 L1 对每个候选 class prior 都要解 QP，因此大规模实验未使用。
4. MNIST one-vs-rest 结果显示，EN 和 PE 在小 class prior 时误差更大，在较高 class prior 时相对更准确。
5. 有些实验中 PE / EN 的分类误差低于使用真实 prior 的情况，论文解释这可能是 density-ratio 低估与 prior 高估相互抵消，不能说明 PE / EN 的 prior 估计更准确。

---

## 11. Development Boundary Conditions

| 条件 | 论文依据 / 预期行为 | 建议严重程度 |
|---|---|---|
| 没有 positive set 或 unlabeled set | 方法定义依赖 \(X\) 和 \(X'\) 两个样本集；缺任一集合无法计算 \(\beta_\ell\) | High |
| 存在类别重叠 | PE / EN 等 partial matching baseline 可能高估；penL1 设计目标是修正该问题 | Medium |
| 重叠很严重 | 普通 L1 也可能高估；penL1 更推荐 | Medium |
| L1 用于大规模数据或密集 \(\theta\) 网格 | 每个候选 prior 都要解 QP，论文指出非常慢 | Medium |
| 未选择 \(\sigma,\lambda\) | 论文要求通过 cross-validation 选择；无默认值 | Medium |
| Gaussian kernel 输入特征尺度差异大 | 论文未讨论；但 kernel width 对尺度敏感 | Project decision: 建议标准化 |
| single-training-set PU 输入 | 论文未分析该数据生成过程 | Project decision: 需保证已标注正样本能代表 \(p(x\mid y=1)\) |

---

## 12. Source Status

| 字段 | 内容 |
|---|---|
| Source status | `official_related` |
| Upstream URL | http://www.mcduplessis.com/index.php/software/ |
| License | `needs_review` |
| Framework | MATLAB |
| 实际对应论文 | ICML 2012 *"Semi-supervised learning of class balance under class-prior change by distribution matching"* (du Plessis & Sugiyama)，**不是** 2017 MLJ 论文 |
| 包含方法 | ① `LSDDPriorEstMedian.m` — L₂-distance (LSDD) 方法；② `pe_prior_est_grid.m` — Pearson divergence (uLSIF) 方法 |
| Toolbox 建议 | **clean-room 实现** penL1 / L1，MATLAB 代码仅作架构参考（grid search + CV for σ,λ + Gaussian kernels 的整体框架可借鉴） |

### MATLAB 代码 vs 2017 论文 — 差异明细

| 维度 | MATLAB 代码 (ICML 2012) | 2017 MLJ 论文 |
|------|------------------------|--------------|
| 问题设定 | Semi-supervised class-prior change（标注数据覆盖全部类别） | PU Learning（仅有 positive + unlabeled） |
| 散度类型 | L₂-distance 或 Pearson divergence | **Penalized L₁-distance** |
| 惩罚机制 (c=∞) |  不存在 |  论文核心创新，用于修正类别重叠时的系统性高估 |
| 内层优化 | 解线性系统 `(H + λI)⁻¹h`（O(b³)，需矩阵求逆） | 闭式解 `α̂_ℓ = max(0, β_ℓ) / λ`（O(b)，无需求逆） |
| Basis 数量 | `b = min(300, n_total)`，随机子采样 | `b = n + n'`，全部训练样本 |
| Kernel 形式 | `exp(-‖x-c‖²/(4σ²))`（RKHS Gram 矩阵） | `exp(-‖x-c‖²/(2σ²))`（直接作为 basis φ_ℓ） |
| σ 选择 | 以数据点间 median distance 为中心搜索（启发式可用） | 论文只说"通过 CV 选择" |
| 输入 API | `(Xte, {x1, x2})` — 无标注集 + 各类别 cell array | `(X, X')` — positive set + unlabeled set |

**结论：penL1 的核心机制在 MATLAB 代码中完全不存在。** 论文取 c=∞ 后内层优化退化为逐元素 thresholding，这正是 penL1 比 L1 快的原因；MATLAB 代码走的是完全不同的 L₂-RKHS 路线。实现 penL1 必须从论文公式直接翻译，不可直接改编 MATLAB 代码。

---

## 13. Implementation Mapping

**项目适配标注**：以下类名和模块名不是论文内容，只是开发建议。

| 论文方法 | 建议类名 | 是否主实现 | 备注 |
|---|---|---:|---|
| penL1 | `PenL1Estimator` | Yes | 推荐作为主 estimator |
| L1 | `L1PriorEstimator` | Optional | 可选对照；需要 QP solver |

> 注：PE、EN、SB 是本文的对比基线，来自其他论文（PE → du Plessis & Sugiyama 2014；EN → Elkan & Noto 2008），其类名和模块归属见各自的方法卡。

### API notes

| API | 建议 |
|---|---|
| `fit(X, y_pu)` | 从 `y_pu` 中拆分 positive / unlabeled；`+1` 和 `0` 是项目约定，不是论文符号 |
| `estimate()` | 返回标量 `float`，即 \(\hat{\pi}\) |
| `confidence_interval(alpha)` | 不实现或返回 `None` / `NotImplemented`；论文没有 CI |
| `get_params()` / `set_params()` | 至少暴露 `theta_grid` 或一维搜索策略、`sigma_candidates`、`lambda_candidates`、basis-center 策略 |
| diagnostics | 建议返回 best score、selected \(\theta\)、selected \(\sigma,\lambda\)，便于调试；论文未要求 |

---

## 14. Test Design Hints

### 14.1 Synthetic overlap test

复现实验中的 1D 均匀分布：

```text
p(x | y = 1)  = U(0, 1)
p(x | y = -1) = U(1 - gamma, 2 - gamma)
pi_true = 0.7
gamma in {0.25, 0.75}
```

预期：

- \(\gamma=0.25\)：PE 可能高估；L1 和 penL1 应较合理。
- \(\gamma=0.75\)：PE 和 L1 都可能高估；penL1 应更稳健。

### 14.2 MNIST-style one-vs-rest test

论文设置：

```text
For digit k:
    positive class = digit k
    negative class = all other digits
    reduce features to 4 dimensions by PCA
    evaluate class-prior squared error and downstream PU classification error
```

### 14.3 Convergence sanity test

基于理论收敛结果设计 sanity test：

```text
Increase n and n_prime while keeping data-generating distribution fixed.
Expected: average estimation error decreases.
```

论文没有给出通用 tolerance，因此测试阈值需要根据项目数据规模、网格精度和随机种子单独设定。

---

## 15. Implementation Notes

**待实现：**

- `PenL1Estimator(BasePriorEstimator)` — 主实现，放入 `pu_toolbox/prior/pen_l1.py`
- `L1PriorEstimator(BasePriorEstimator)` — optional 对照实现
- `tests/test_pen_l1.py` — 覆盖 §14 中的合成实验和边界条件

**注意事项：**

- 论文使用 Gurobi 求解 L1 的 QP，Toolbox 需使用开源替代（如 `scipy.optimize`）。
- `max(0, β_ℓ)·β_ℓ` 等价于 `[max(0, β_ℓ)]²`，选可读性更好的写法即可。
