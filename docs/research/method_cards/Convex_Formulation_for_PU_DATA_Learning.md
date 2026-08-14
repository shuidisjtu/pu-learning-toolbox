# Method Card: Convex Formulation for PU Classification

> 标注约定：`【项目适配】` 表示依据当前项目模板/API 约定补充的内容，不是论文原文结论；`【实现建议】` 表示论文未规定、需要工程侧自行确定的实现选择。

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Convex Formulation for Learning from Positive and Unlabeled Data |
| Authors | Marthinus Christoffel du Plessis, Gang Niu, Masashi Sugiyama |
| Venue | ICML 2015，PMLR/JMLR W&CP Vol. 37 |
| Core method | Convex PU risk with different losses for P and U |
| Recommended variant | C-DH: convex PU classification with double hinge loss |
| Assumption | `SCAR`（论文未使用 SCAR 术语，但要求 P 为 $`p(x\mid y=1)`$ 的 i.i.d. 抽样） |
| Scenario | `case_control`（P 和 U 分别收集） |
| Setting | Two datasets: $`P\sim p(x\mid y=1)`$，$`U\sim p(x)`$ |
| Requires class prior | `True`，训练时需要 $`\pi=p(y=1)`$ |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| Optimization | Convex；C-DH 为 QP，C-LL 为光滑凸优化 |
| GPU required | `False` |
| Output | Binary classifier；论文 $`\hat y=\mathrm{sign}(g(x))\in\{+1,-1\}`$ |

### 1.1 适用场景标签

| 场景 | 论文中的含义 |
|---|---|
| `identification` | 用少量目标正样本从未标注集合中识别相似样本，如自动人脸标注 |
| `inlier_based_outlier_detection` | 已知 inlier 样本，从未标注集合中识别 outlier |
| `one_vs_rest` | negative class 过于多样，难以代表性采集 |
| `negative_class_shift` | negative 分布随时间变化，仅更新 unlabeled set 比持续重采 negative 更低成本 |

### Assumptions

```math
P=\{x_i^P\}_{i=1}^{n_P}\overset{i.i.d.}{\sim}p(x\mid y=1),
\qquad
U=\{x_j^U\}_{j=1}^{n_U}\overset{i.i.d.}{\sim}p(x)
```

```math
p(x)=\pi p(x\mid y=1)+(1-\pi)p(x\mid y=-1),
\qquad 0<\pi<1
```

分类器：

```math
g:\mathbb{R}^d\rightarrow\mathbb{R},
\qquad
\hat y=\mathrm{sign}(g(x))
```

---

## 2. 问题设定与符号

本文给出 PU 学习的凸风险框架：将 margin loss $`\ell`$ 替换为 composite loss $`\tilde\ell(z)=\ell(z)-\ell(-z)`$ 后，普通监督风险可只用 P、U 两个分布的期望表达；在 $`\tilde\ell`$ 凸（因而必为线性）的条件下得到凸目标，避免直接替换 surrogate 产生的 superfluous penalty。

| 论文符号 | 含义 |
|---|---|
| $`x\in\mathbb{R}^d`$ | 输入特征 |
| $`y\in\{1,-1\}`$ | 真实类别 |
| $`P=\{x_i^P\}_{i=1}^{n_P}`$ | 正类样本 |
| $`U=\{x_j^U\}_{j=1}^{n_U}`$ | 未标注样本 |
| $`\pi=p(y=1)`$ | unlabeled/目标分布中的正类先验 |
| $`\mathbb{E}_P`$ | 对 $`p(x\mid y=1)`$ 的期望 |
| $`\mathbb{E}_N`$ | 对 $`p(x\mid y=-1)`$ 的期望（训练中不可直接估计） |
| $`\mathbb{E}_U`$ / $`\mathbb{E}_X`$ | 对 $`p(x)`$ 的期望 |
| $`g(x)`$ | 判别函数 |
| $`\ell(z)`$ | ordinary margin loss（用于 U 项） |
| $`\tilde\ell(z)`$ | composite loss：$`\ell(z)-\ell(-z)`$（用于正类项） |
| $`\phi(x)\in\mathbb{R}^m`$ | basis vector |
| $`\alpha\in\mathbb{R}^m`$ | basis coefficients |
| $`b`$ | intercept |
| $`\lambda`$ | $`\ell_2`$ 正则系数 |
| $`\Phi_P,\Phi_U`$ | P/U 的 basis design matrices |
| $`\xi\in\mathbb{R}^{n_U}`$ | C-DH slack variables |

---

## 3. 核心公式

### 3.1 普通监督风险

对 margin loss $`\ell`$，普通二分类风险为：

```math
R_\ell(g)
=
\pi\mathbb{E}_P[\ell(g(X))]
+
(1-\pi)\mathbb{E}_N[\ell(-g(X))]
```

由于

```math
\mathbb{E}_U[\ell(-g(X))]
=
\pi\mathbb{E}_P[\ell(-g(X))]
+
(1-\pi)\mathbb{E}_N[\ell(-g(X))]
```

可消去不可直接估计的 negative expectation。

### 3.2 直接替换 surrogate loss 的偏差来源

从 zero-one PU 表达式直接替换为一般 surrogate loss 会得到：

```math
J_{\text{naive}}(g)
=
2\pi\mathbb{E}_P[\ell(g(X))]
+
\mathbb{E}_U[\ell(-g(X))]
-\pi
```

展开为：

```math
J_{\text{naive}}(g)
=
R_\ell(g)
+
\pi\mathbb{E}_P[\ell(g(X))+\ell(-g(X))]
-\pi
```

其中

```math
\pi\mathbb{E}_P[\ell(g(X))+\ell(-g(X))]
```

是 PU setting 特有的 superfluous penalty。只有 $`\ell(z)+\ell(-z)=1`$ 时该项才成为常数；满足该条件的典型 loss（如 ramp loss）是非凸的。

### 3.3 凸 PU 风险

使用另一种代数消元：

```math
R_\ell(g)
=
\pi\mathbb{E}_P[\ell(g(X))-\ell(-g(X))]
+
\mathbb{E}_U[\ell(-g(X))]
```

定义 composite loss：

```math
\tilde\ell(z)=\ell(z)-\ell(-z)
```

则：

```math
J(g)
=
\pi\mathbb{E}_P[\tilde\ell(g(X))]
+
\mathbb{E}_U[\ell(-g(X))]
```

论文定理：若 $`\tilde\ell`$ 是凸函数，则它必为线性函数。将可用 loss 归一化为：

```math
\tilde\ell(z)=-z
```

最终风险为：

```math
J(g)
=
-\pi\mathbb{E}_P[g(X)]
+
\mathbb{E}_U[\ell(-g(X))]
```

### 3.4 经验目标和模型

线性参数模型：

```math
g(x)=\alpha^\top\phi(x)+b
```

basis 可选 Gaussian、linear 或 polynomial。论文给出的 Gaussian basis 为：

```math
\phi_\ell(x)
=
\exp\left(
-\frac{\|x-c_\ell\|^2}{2\sigma^2}
\right)
```

论文示例以全部 P/U 样本作为 centers：

```math
\{c_\ell\}_{\ell=1}^{m}=P\cup U,
\qquad
m=n_P+n_U
```

通用正则化经验目标：

```math
\widehat J(\alpha,b)
=
-\frac{\pi}{n_P}\sum_{i=1}^{n_P}g(x_i^P)
+
\frac{1}{n_U}\sum_{j=1}^{n_U}\ell(-g(x_j^U))
+
\frac{\lambda}{2}\alpha^\top\alpha
```

论文不正则化 intercept $`b`$。

### 3.5 Loss 选择

| Loss | $`\ell(z)`$ | $`\tilde\ell(z)`$ | 优化 |
|---|---|---|---|
| Squared | $`\frac14(z-1)^2`$ | $`-z`$ | 凸；无 intercept 时可闭式求解 |
| Logistic | $`\log(1+\exp(-z))`$ | $`-z`$ | 光滑凸；quasi-Newton |
| Hinge | $`\frac12\max(0,1-z)`$ | 非线性且非凸 | 不满足本文凸条件 |
| Double hinge | $`\max\{-z,0,\frac12-\frac12z\}`$ | $`-z`$ | 凸 QP |

#### C-DH：Double hinge

```math
\ell_{\mathrm{DH}}(z)
=
\max\left\{
-z,\ 0,\ \frac{1-z}{2}
\right\}
```

因此：

```math
\ell_{\mathrm{DH}}(-g)
=
\max\left\{
g,\ 0,\ \frac{1+g}{2}
\right\}
```

经验目标：

```math
\widehat J_{\mathrm{DH}}(\alpha,b)
=
-\frac{\pi}{n_P}\mathbf{1}^\top\Phi_P\alpha
-\pi b
+
\frac{1}{n_U}\sum_{j=1}^{n_U}
\ell_{\mathrm{DH}}\left(-g(x_j^U)\right)
+
\frac{\lambda}{2}\alpha^\top\alpha
```

引入 slack $`\xi`$ 后的 QP：

```math
\min_{\alpha,b,\xi}
-\frac{\pi}{n_P}\mathbf{1}^\top\Phi_P\alpha
-\pi b
+\frac{1}{n_U}\mathbf{1}^\top\xi
+\frac{\lambda}{2}\alpha^\top\alpha
```

subject to：

```math
\xi\ge 0
```

```math
\xi
\ge
\frac12\mathbf{1}
+\frac12\Phi_U\alpha
+\frac12b\mathbf{1}
```

```math
\xi
\ge
\Phi_U\alpha+b\mathbf{1}
```

所有不等式逐元素成立。QP 为凸问题；$`\lambda>0`$ 时 $`\alpha`$ 部分强凸，但 $`b`$ 是否唯一仍取决于数据和约束。

#### C-LL：Logistic

```math
\ell_{\mathrm{LL}}(z)=\log(1+\exp(-z))
```

```math
\widehat J_{\mathrm{LL}}(\alpha,b)
=
-\frac{\pi}{n_P}\sum_{i=1}^{n_P}g(x_i^P)
+
\frac{1}{n_U}\sum_{j=1}^{n_U}\log(1+\exp(g(x_j^U)))
+
\frac{\lambda}{2}\alpha^\top\alpha
```

梯度：

```math
\nabla_\alpha\widehat J_{\mathrm{LL}}
=
-\frac{\pi}{n_P}\Phi_P^\top\mathbf{1}
+
\frac{1}{n_U}\Phi_U^\top\sigma(g_U)
+
\lambda\alpha
```

```math
\frac{\partial\widehat J_{\mathrm{LL}}}{\partial b}
=
-\pi
+
\frac{1}{n_U}\mathbf{1}^\top\sigma(g_U)
```

其中 $`\sigma(t)=1/(1+\exp(-t))`$（数值稳定性实现见第 6 节）。

#### Squared loss

```math
\widehat J_{\mathrm{S}}(\alpha,b)
=
-\frac{\pi}{n_P}\sum_{i=1}^{n_P}g(x_i^P)
+
\frac{1}{4n_U}\sum_{j=1}^{n_U}(g(x_j^U)+1)^2
+
\frac{\lambda}{2}\alpha^\top\alpha
```

省略 $`b`$ 时：

```math
\alpha
=
\left(
\frac{1}{2n_U}\Phi_U^\top\Phi_U+\lambda I
\right)^{-1}
\left[
\frac{\pi}{n_P}\Phi_P^\top\mathbf{1}
-
\frac{1}{2n_U}\Phi_U^\top\mathbf{1}
\right]
```

### 3.6 PU-CV 目标

论文使用式 (2) 的 zero-one PU 风险做交叉验证：

```math
\widehat R_{0\text{-}1}^{\mathrm{PU}}(g)
=
\frac{2\pi}{n_P^{\mathrm{val}}}
\sum_{i=1}^{n_P^{\mathrm{val}}}
\ell_{0\text{-}1}(g(x_i^P))
+
\frac{1}{n_U^{\mathrm{val}}}
\sum_{j=1}^{n_U^{\mathrm{val}}}
\ell_{0\text{-}1}(-g(x_j^U))
-\pi
```

其中：

```math
\ell_{0\text{-}1}(z)=\mathbf{1}[z\le0]
```

> **zero-one loss 疑似排版符号错误**：论文式 (1) 写成 $`\frac12\mathrm{sign}(z)+\frac12`$，但该式对应“分类正确指示量”而非待最小化的误分类损失。实现和 CV 应使用 $`\ell_{0\text{-}1}(z)=\mathbf{1}[z\le 0]`$，或忽略 $`z=0`$ 约定时等价的 $`\frac12-\frac12\mathrm{sign}(z)`$。该估计因有限样本可能小于 0，可用于模型排序，不应裁剪后再比较。

### 3.7 理论收敛

对 squared、logistic 和 double hinge，论文在以下条件下证明参数和目标值均达到：

```math
O_p\left(n_P^{-1/2}+n_U^{-1/2}\right)
```

关键条件：

- basis 数 $`m`$ 固定，不随 $`n_P,n_U`$ 增长；
- $`0\le\phi_j(x)\le1`$；
- 使用 $`\ell_2`$ 正则使最优参数有界；
- 理论推导为简化忽略 intercept $`b`$。

因此该结论不能直接外推到“每个训练样本一个 Gaussian center、$`m=n_P+n_U`$”的增长维度实现。

---

## 4. 算法概要

### 4.1 推荐实现路径：C-DH

```text
Input:
    P, U            # labeled positive / unlabeled sets
    class prior pi
    reg_lambda
    basis configuration

1. Split:  P = {x : y = +1}, U = {x : y = 0}
2. Validate:  P and U are non-empty; 0 < pi < 1; reg_lambda > 0
3. Build basis:  linear (phi(x) = x) 或 Gaussian/polynomial
4. Build Phi_P and Phi_U
5. Solve the convex QP in §3.5
6. Store alpha, b 及求解器诊断
7. Predict:  score = alpha^T phi(x) + b;  label = 1 if score >= 0 else 0
```

### 4.2 超参数（论文设置）

| 参数 | 论文设置 |
|---|---|
| $`\pi`$ | 训练时已知 |
| $`\lambda`$ | 通过 CV 选择 |
| basis type | Gaussian / linear / polynomial 均可 |
| $`\sigma`$ | Gaussian width；通过 CV 选择 |
| centers | 全部 P/U 样本 |
| intercept | 使用 $`b`$，且不正则化 |

### 4.3 复杂度与扩展性

设 basis 数为 $`m`$：

- 构建设计矩阵：约 $`O((n_P+n_U)m)`$。
- 全样本 Gaussian centers：$`m=n_P+n_U`$，设计矩阵内存约为 $`O((n_P+n_U)^2)`$。
- C-DH QP 变量数约为 $`m+1+n_U`$，约束数为 $`3n_U`$。
- C-LL 每次目标/梯度计算约为 $`O((n_P+n_U)m)`$。
- Squared closed form 朴素线性代数成本约为 $`O(m^3)`$。

---

## 5. 论文边界

- 输入必须同时包含 positive set 和 unlabeled set；不使用显式 negative samples。
- positive set 必须可视为从 $`p(x\mid y=1)`$ 独立同分布抽样，unlabeled set 从当前目标边缘分布 $`p(x)`$ 抽样。
- `【项目适配】` 若输入为 single-training-set `y_pu`，已标注正样本需能代表完整正类分布；这相当于项目侧需要满足 SCAR-like 的抽样条件。论文未使用 SCAR 术语，也未分析该条件失效时的偏差。
- **普通 Hinge 不能直接用于本文凸框架**：其 composite loss 不是线性函数，会使目标非凸。
- **普通加权 LogReg/Hinge 不是本文方法**：直接把 unlabeled 当作 negative 会引入 superfluous penalty；它们在论文中仅作为有偏基线。
- 论文只笼统提出在退化情形下可增加非负约束以避免数值问题，没有给出完整的稳定化算法。不要将后续 nnPU 的 non-negative risk correction 直接视为本文目标。

---

## 6. 实现注记

- **C-LL 数值稳定性**：实现时用稳定的 softplus 计算 `log(1+exp(g_u))`：

  ```python
  softplus_g = np.logaddexp(0.0, g_u)
  ```

- **Squared 闭式解**：实现应使用线性方程求解（如 `np.linalg.solve`），不显式计算矩阵逆。

---

## 7. 论文实验参考

| 项目 | 论文设置 |
|---|---|
| 合成可分数据 | $`p(x\mid y=+1)=\mathrm{Uniform}(0.1,1.0)`$，$`p(x\mid y=-1)=\mathrm{Uniform}(-1.1,-0.1)`$；线性模型 $`g(x)=wx+b`$，正确边界满足 $`-b/w\in[-0.1,0.1]`$ |
| 非凸性示意 | $`p(x\mid y=+1)=\mathrm{Normal}(2,1/2)`$，$`p(x\mid y=-1)=\mathrm{Normal}(-2,1/2)`$；$`n_P=10`$、$`n_U=20`$、$`\pi=0.5`$、$`\lambda=1\times10^{-3}`$ |
| MNIST one-vs-rest | 正类 digit 0，负类为 1..9 之一；PCA 2 维；$`n_P=200`$、$`n_U=400`$；$`\pi\in\{0.1,0.4,0.7\}`$；超参选择用式 (2) PU zero-one CV |
| 先验敏感性 | $`\pi_{\mathrm{used}}=\pi_{\mathrm{true}}+\delta`$，$`\delta\in\{-0.2,-0.1,0,+0.1,+0.2\}`$ |
| 结论 | C-DH 在论文实验中与非凸 ramp loss 精度相当，计算成本显著更低；naive Hinge/LogReg 仅为有偏基线 |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_bundle` |
| Upstream URL | https://github.com/t-sakai-kure/pywsl |
| License | MIT |
| 包含方法 | `pywsl` 由 Sugiyama Lab 维护；uPU 仅提供 Squared Loss 闭式解（NumPy + scipy），不含 double hinge（C-DH）和 logistic（C-LL）；nnPU / PNU / PU-SKC / PNU-AUC 等其他方法见仓库 README |
| Implementation basis | 论文式 (8)、式 (9)、Table 1、式 (2) CV |
| 参考实现 | 官方关联代码库 `pywsl` 作算法参考；本卡公式以论文原文为权威来源 |
| Reproduction risk | solver 选择、CV 网格、预处理、basis-center 策略均未由论文完整规定 |
