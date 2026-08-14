# Method Card: Class-Prior Estimation（penL1 / L1）

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Class-Prior Estimation for Learning from Positive and Unlabeled Data |
| Authors | Marthinus C. du Plessis, Gang Niu, Masashi Sugiyama |
| Venue | Machine Learning |
| Year | 2017 |
| Setting | `single_training_set`、`case_control`（取决于 P/U 抽样方式） |
| Requires class prior | `False`；输出 `pi_hat` |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| Source record | [作者软件页面](http://www.mcduplessis.com/index.php/software/) |
| DOI | [10.1007/s10994-016-5604-6](https://doi.org/10.1007/s10994-016-5604-6) |

---

## 2. 问题设定与符号

令 $`p_P(x)=p(x\mid Y=1)`$、$`p_N(x)=p(x\mid Y=-1)`$，未标记边缘分布为：

```math
p_U(x)=p(x)=\pi p_P(x)+(1-\pi)p_N(x),
\qquad
\pi=P(Y=1).
```

观察到两个独立样本集：

```math
X_P=\{x_i^P\}_{i=1}^{n_P}\sim p_P(x),
\qquad
X_U=\{x_j^U\}_{j=1}^{n_U}\sim p_U(x).
```

目标是从 $`X_P`$ 和 $`X_U`$ 中估计 `pi`，不需要观测 $`p_N`$ 或真实负标签。论文研究的是先验估计本身，不直接训练分类器。

| 论文符号 | 含义 |
|---|---|
| $`\pi`$ | 真实正类先验 |
| $`\theta`$ | 候选先验 |
| $`P`$ | 正类条件样本 |
| $`U`$ | 边缘未标记样本 |
| $`f`$ | divergence generator |
| $`r(x)`$ | Fenchel dual scoring function |
| $`\phi_l(x)`$ | 第 $`l`$ 个非负 basis |
| $`\alpha_l`$ | basis 系数 |
| $`\beta_l(\theta)`$ | 经验 basis 差异 |
| $`\lambda`$ | coefficient L2 regularization |
| $`\sigma`$ | Gaussian width |
| $`b`$ | basis 数 |

---

## 3. 核心公式

### 3.1 Partial distribution matching

对于候选比例 $`\theta`$，考虑将 $`\theta p_P`$ 与 $`p_U`$ 匹配。用 f-divergence 表示：

```math
D_f(\theta)=
\int f\left(\frac{\theta p_P(x)}{p_U(x)}\right)p_U(x)dx.
```

当 `theta` 不超过真实 mixture proportion 时，$`\theta p_P`$ 可以被 $`p_U`$ 包含；估计过程通过最小化 divergence 找到可行的最大比例。

### 3.2 Penalized f-divergence

为了惩罚 $`\theta p_P(x) > p_U(x)`$ 的区域，论文引入：

```math
\tilde f(t)=
\begin{cases}
f(t), & 0\le t\le 1,\\
\infty, & t>1.
\end{cases}
```

这使得违反 mixture 包含关系的候选比例代价变大。Fenchel dual 后，对固定 `theta` 求解一个关于 $`r`$ 的上界/经验目标，再在 `theta` 上搜索。

### 3.3 Gaussian basis

论文使用非负 Gaussian basis：

```math
\phi_l(x)=
\exp\left(-\frac{\|x-c_l\|^2}{2\sigma^2}\right),
\qquad 0<\phi_l(x)\le1.
```

```math
r_\alpha(x)=\sum_{l=1}^{b}\alpha_l\phi_l(x)-1.
```

论文实验可使用全部训练样本作为 centers，但大规模数据下会带来二次内存和计算开销。

### 3.4 经验 basis 差异

对每个 `theta` 和 basis $`l`$：

```math
\beta_l(\theta)=
\theta\frac{1}{n_P}\sum_{i=1}^{n_P}\phi_l(x_i^P)
-\frac{1}{n_U}\sum_{j=1}^{n_U}\phi_l(x_j^U).
```

注意 P 与 U 的分母必须分别是 $`n_P`$ 和 $`n_U`$；不能把两个集合拼接后统一平均。

### 3.5 penL1 闭式内层解

penL1 对系数采用非负约束和 L2 正则。固定 `theta` 后：

```math
\hat\alpha_l(\theta)=
\frac{1}{\lambda}\max(0,\beta_l(\theta)).
```

### 3.6 外层目标

代入闭式解后，penL1 经验目标为：

```math
\widehat J_{penL1}(\theta)=
\frac{1}{\lambda}
\sum_{l=1}^{b}\max(0,\beta_l(\theta))\beta_l(\theta)
-\theta+1.
```

最终估计为：

```math
\hat\pi=\arg\min_{\theta\in\Theta}\widehat J_{penL1}(\theta),
```

其中 $`\Theta`$ 是 $`[0,1]`$ 内候选网格。

### 3.7 解析解边界

论文由 penalized f-divergence 推导 penalized L1-distance。固定 `theta` 时，式 (11) 在非负 Gaussian basis 下化为：

```math
\min_{\alpha\ge0}
\frac{\lambda}{2}\|\alpha\|_2^2
-\alpha^T\beta(\theta),
```

唯一显式约束是逐坐标非负：

```math
\alpha_l\ge 0,\qquad l=1,\ldots,b.
```

由于目标对不同 $`\alpha_l`$ 完全解耦，其解正是 $`\alpha_l=\max(0,\beta_l)/\lambda`$——penL1 内层无需通用 QP solver，闭式解即论文算法。

---

## 4. 算法概要

```text
输入：X、y_pu、sigma、lambda、theta_grid、n_centers

1. 校验 P/U 均非空，转换为 P=X[y_pu==1]、U=X[y_pu==0]。
2. 对 X 做训练集标准化（可关闭）。
3. 选择 Gaussian centers，计算 Phi_P 和 Phi_U。
4. 对 theta_grid 中每个 theta：
   a. 计算 beta(theta)=theta*mean(Phi_P)-mean(Phi_U)；
   b. 计算 penL1 经验目标 J(theta)。
5. 选择 J 最小的 theta，作为 pi_hat 估计输出。
```

**复杂度**：构造 $`\Phi_P/\Phi_U`$ 的时间和内存约为 $`O((n_P+n_U)b d)`$（$`b`$ 为 centers 数，$`d`$ 为特征数）；当 $`b=n_P+n_U`$ 时接近二次规模。theta 搜索额外为 $`O(|\Theta|b)`$；penL1 内层为闭式解，不需要逐 theta 求解 QP。

---

## 5. 论文边界

- 论文研究的是先验估计 $`\pi=P(Y=1)`$，不是分类器训练，也不直接输出 posterior probability；`estimate()` 的输出不是分类器概率，也不是置信区间。
- 输入的 $`P`$ 应代表 $`p(x\mid y=1)`$；若已标记正例存在 selection bias，直接使用 penL1 会把 labeling bias 混入类先验估计。
- $`U`$ 必须来自边缘分布 $`p(x)`$。如果 U 是经过筛选的子集，论文的 mixture decomposition 不再直接成立。
- Gaussian basis 对特征尺度很敏感，正式复现必须记录标准化方式及是否在训练 fold 内计算统计量。
- $`\theta\_grid`$、$`\sigma`$ 和 $`\lambda`$ 是工程参数；论文要求通过交叉验证选择，但没有提供唯一默认网格。
- 关键分布条件：

| 条件 | 含义 |
|---|---|
| P 可靠 | `P` 中样本真实为正（由数据生成机制保证，代码无法验证） |
| U 为 mixture | `U ~ p(x)`（由任务采样协议保证） |
| `0 < pi < 1` | 非退化 mixture |
| 可计算密度 ratio | basis/regularization 足够表达 |

---

## 6. 实现注记

> 见 ADR-0014 #1（0.0380 MAE 为默认 `sigma=1.0` 时代的数字；2026-08-10 默认改为数据自适应 `sigma` 后 benchmark 复现须显式 pin `sigma`）

- **2026-08-10 变更**：默认 `sigma=None` 改为数据自适应——`0.6 × 标准化后数据的中位 pairwise 欧氏距离`（确定性、无真值泄漏）。跨分离度/先验回归已验证其落在验收带内，但对 `prior<0.5 + 低分离度` 有高估倾向，用户可用显式 `sigma` 覆盖。
- 论文源码页面中的 MATLAB 文件与 2017 MLJ 论文的 penL1 公式并不完全对应；实现以论文公式为数学权威。
- **文档纠错**：此前文档中“每个 U 样本还需满足上界约束、每个 theta 都需通用 QP solver”的描述不属于该论文，现已撤销。仍未对齐的是论文实验中的逐 `theta` 超参数 CV、全样本 centers、MNIST PCA 协议与不可变作者源码，而不是另一个 L1-QP 变体。
- clean-room 合成结果已归档于 `benchmarks/assigned_methods/results/clean_room_multiseed/`（以 benchmarks 产物为准）。seed `0..4` 的 penL1 prior MAE `0.0380 ± 0.0192` 不含 MNIST、论文逐 `theta` CV 和完整基线，不是论文表格复现；该数字为默认 `sigma=1.0` 时代的产物，当前默认下无意义（见 ADR-0014 #1）。

---

## 7. 论文实验参考

| 项目 | 论文设置 |
|---|---|
| 合成数据 | Gaussian mixture：$`P\sim N(\mu_p, I)`$、$`N\sim N(\mu_n, I)`$、U 先以概率 `pi` 采样 Y 再从对应的 P/N 分布采样 X；P 与 U 必须独立采样 |
| 真实数据 | MNIST one-vs-rest；具体正类数字、训练/测试划分和样本数必须由论文或官方源码清单锁定 |
| 参数选择 | 论文对每个 `theta` 做 “straightforward cross-validation” 选择 `sigma`/`lambda`；精确 fold/scoring 仍需作者实现证据 |
| PU 构造 | 先保留真实标签，仅用训练折构造 P/U；标准化、Gaussian centers 和所有超参数只在训练折拟合，测试标签不参与选择 `sigma`/`lambda` |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_related`；作者页面源码与本文 penL1 公式需分开核对 |
| 已实现 | Gaussian basis、penL1 闭式系数、先验网格搜索、统一 prior API（当前为 penL1 clean-room） |
| 未实现 | 论文逐 `theta` CV 的精确 protocol、MNIST/PCA 执行层、CI/bootstrap、paper-like benchmark（工具箱侧机制已就绪——preflight 审计、执行层、数据/源码锁定与 blocker 诊断；全量运行依赖外部官方数据与历史环境，官方数据不内置工具箱，由执行方提供，非工具箱缺口） |
| 主要风险 | basis 尺度、先验搜索网格、P/U 抽样偏差和有限样本误差都会显著影响 `pi_hat` |
| 解释边界 | 输出是 mixture proportion/class prior estimate，不是分类器概率，也不是置信区间 |
