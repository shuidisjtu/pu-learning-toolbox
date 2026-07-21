# Method Card: Class-Prior Estimation（penL1 / L1）

## 1. 待办与注意

### 1.1 待办

- [x] 定义 class-prior estimation 的输入为可靠正类集合 `P` 与边缘未标记集合 `U`。
- [x] 按论文 penL1 闭式公式实现 Gaussian basis、`beta_l(theta)` 和先验网格搜索。
- [x] 接入 `BasePriorEstimator`，注册为 `class_prior_estimation`，别名包含 `pen_l1`。
- [x] 编写边界、确定性和 `[0,1]` 范围测试。
- [ ] 实现论文 L1 变体的带约束 QP。
- [ ] 按论文 protocol 增加 `sigma/lambda` 的 nested CV，而不是依赖用户手工选择。
- [ ] 增加 confidence interval/bootstrap 和 paper-like benchmark。

### 1.2 注意

- 论文研究的是先验估计 `pi=P(Y=1)`，不是分类器训练，也不直接输出 posterior probability。
- 输入的 `P` 应代表 `p(x|y=1)`；若已标记正例存在 selection bias，直接使用 penL1 会把 labeling bias 混入类先验估计。
- `U` 必须来自边缘分布 `p(x)`。如果 U 是经过筛选的子集，论文的 mixture decomposition 不再直接成立。
- Gaussian basis 对特征尺度很敏感。当前实现默认标准化；正式复现必须记录标准化方式及是否在训练 fold 内计算统计量。
- `theta_grid`、`sigma` 和 `reg_lambda` 是工程参数。论文要求通过交叉验证选择，但没有为项目提供唯一默认网格。
- 论文源码页面中的 MATLAB 文件与 2017 MLJ 论文的 penL1 公式并不完全对应；实现以论文公式为数学权威。

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Class-Prior Estimation for Learning from Positive and Unlabeled Data |
| Authors | Marthinus C. du Plessis, Gang Niu, Masashi Sugiyama |
| Venue | Machine Learning |
| Year | 2017 |
| Family | `class_prior_estimation` |
| Scenario | `single_training_set`、`case_control`（取决于 P/U 抽样方式） |
| Requires class prior | `False`；输出 `pi_hat` |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| Backend | NumPy + SciPy/sklearn preprocessing |
| Source record | [作者软件页面](http://www.mcduplessis.com/index.php/software/) |
| Registry | `class_prior_estimation` / `pen_l1`，`NATIVE` |

## 3. 问题设定与目标

令：

```math
p_P(x)=p(x\mid Y=1),
\qquad
p_N(x)=p(x\mid Y=-1),
```

未标记边缘分布为：

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

目标是从 `X_P` 和 `X_U` 中估计 `pi`，不需要观测 `p_N` 或真实负标签。

### 3.1 关键分布条件

| 条件 | 含义 | 项目检查 |
|---|---|---|
| P 可靠 | `P` 中样本真实为正 | 由数据生成机制保证，代码无法验证 |
| U 为 mixture | `U ~ p(x)` | 由任务采样协议保证 |
| `0 < pi < 1` | 非退化 mixture | 输出网格默认避开端点 |
| 可计算密度 ratio | basis/regularization 足够表达 | 通过合成实验诊断 |

## 4. 符号与记号

| 论文符号 | 含义 | 开发侧对应 |
|---|---|---|
| `pi` | 真实正类先验 | 目标未知量 |
| `theta` | 候选先验 | `theta_grid` 中的值 |
| `P` | 正类条件样本 | `X[y_pu == 1]` |
| `U` | 边缘未标记样本 | `X[y_pu == 0]` |
| `f` | divergence generator | penL1 框架中的目标函数 |
| `r(x)` | Fenchel dual scoring function | Gaussian basis linear model |
| `phi_l(x)` | 第 `l` 个非负 basis | `exp(-distance/(2*sigma^2))` |
| `alpha_l` | basis 系数 | penL1 闭式系数 |
| `beta_l(theta)` | 经验 basis 差异 | 代码循环中的 `beta` |
| `lambda` | coefficient L2 regularization | `reg_lambda` |
| `sigma` | Gaussian width | `sigma` |
| `b` | basis 数 | `n_centers_` |

## 5. 核心推导

### 5.1 Partial distribution matching

对于候选比例 `theta`，考虑将 `theta p_P` 与 `p_U` 匹配。用 f-divergence 表示：

```math
D_f(\theta)=
\int f\left(\frac{\theta p_P(x)}{p_U(x)}\right)p_U(x)dx.
```

当 `theta` 不超过真实 mixture proportion 时，`theta p_P` 可以被 `p_U` 包含；估计过程通过最小化 divergence 找到可行的最大比例。

### 5.2 Penalized f-divergence

为了惩罚 `theta p_P(x) > p_U(x)` 的区域，论文引入：

```math
\tilde f(t)=
\begin{cases}
f(t), & 0\le t\le 1,\\
\infty, & t>1.
\end{cases}
```

这使得违反 mixture 包含关系的候选比例代价变大。Fenchel dual 后，对固定 `theta` 求解一个关于 `r` 的上界/经验目标，再在 `theta` 上搜索。

### 5.3 Gaussian basis

项目使用非负 Gaussian basis：

```math
\phi_l(x)=
\exp\left(-\frac{\|x-c_l\|^2}{2\sigma^2}\right),
\qquad 0<\phi_l(x)\le1.
```

```math
r_\alpha(x)=\sum_{l=1}^{b}\alpha_l\phi_l(x)-1.
```

当前实现将训练数据的前 `n_centers` 个样本作为 centers；`n_centers=None` 时使用全部样本。论文实验可使用全部训练样本作为 centers，但大规模数据下会带来二次内存和计算开销。

### 5.4 经验 basis 差异

对每个 `theta` 和 basis `l`：

```math
\beta_l(\theta)=
\theta\frac{1}{n_P}\sum_{i=1}^{n_P}\phi_l(x_i^P)
-\frac{1}{n_U}\sum_{j=1}^{n_U}\phi_l(x_j^U).
```

注意 P 与 U 的分母必须分别是 `n_P` 和 `n_U`；不能把两个集合拼接后统一平均。

## 6. penL1 算法

### 6.1 闭式内层解

penL1 对系数采用非负约束和 L2 正则。固定 `theta` 后：

```math
\hat\alpha_l(\theta)=
\frac{1}{\lambda}\max(0,\beta_l(\theta)).
```

### 6.2 外层目标

代入闭式解后，项目使用：

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

其中 `Theta` 是 `[0,1]` 内候选网格。当前代码默认 `0.01` 到 `0.99` 的 99 点网格，正式实验应显式传入网格或实现连续搜索敏感性分析。

## 7. L1 变体

论文还讨论 `c=1` 的 L1 版本。其固定 `theta` 的内层问题需要解带约束 QP：

```math
\min_{\alpha\ge0}
\frac{\lambda}{2}\|\alpha\|_2^2
-\alpha^T\beta(\theta),
```

并满足每个 U 样本上的非负函数约束，例如：

```math
\sum_l\alpha_l\phi_l(x_j^U)\le 2,
\qquad j=1,\ldots,n_U.
```

当前项目只实现 penL1。L1 不能通过把 `max(0,beta)/lambda` 改名得到；每个候选 `theta` 都需要一次 QP，复杂度和 solver 容差必须在方法卡和 benchmark 中单独记录。

## 8. 算法概要

```text
输入：X、y_pu、sigma、lambda、theta_grid、n_centers

1. 校验 P/U 均非空，转换为 P=X[y_pu==1]、U=X[y_pu==0]。
2. 对 X 做训练集标准化（可关闭）。
3. 选择 Gaussian centers，计算 Phi_P 和 Phi_U。
4. 对 theta_grid 中每个 theta：
   a. 计算 beta(theta)=theta*mean(Phi_P)-mean(Phi_U)；
   b. 计算 penL1 经验目标 J(theta)。
5. 选择 J 最小的 theta，保存 class_prior_ 和 objective_values_。
6. estimate() 返回 class_prior_。
```

## 9. 超参数与复杂度

| 参数 | 当前默认值 | 含义 | 选择建议 |
|---|---:|---|---|
| `sigma` | 1.0 | Gaussian width | 应在训练 fold 内 CV；对标准化尺度敏感 |
| `reg_lambda` | `1e-2` | alpha 的 L2 正则 | 应与 sigma 联合搜索 |
| `theta_grid` | 99 点 | 候选先验 | 数据量大时可先粗网格再局部细化 |
| `n_centers` | 200 | basis 中心数 | 小数据可设 `None`；大数据需限制 |
| `standardize` | True | 是否特征标准化 | 必须记录并避免验证/测试泄漏 |

若 `b` 为 centers 数量，构造 `Phi_P/Phi_U` 的时间和内存约为 `O((n_P+n_U)b d)`；当 `b=n_P+n_U` 时接近二次规模。theta 搜索额外为 `O(|Theta|b)`，penL1 内层不需要逐 theta 求解 QP。

## 10. API 接口与项目落点

### 10.1 构造函数

```python
class ClassPriorEstimator(BasePriorEstimator):
    def __init__(
        self,
        *,
        sigma=1.0,
        reg_lambda=1e-2,
        theta_grid=None,
        n_centers=200,
        standardize=True,
    ):
        ...
```

### 10.2 API 语义

| API / 属性 | 约定 |
|---|---|
| `fit(X, y_pu)` | `1` 为可靠正样本，`0` 为 U；返回 self |
| `estimate()` | 返回 `[0,1]` 内的 `float` |
| `confidence_interval(alpha)` | 当前返回 `None`，论文未提供项目可直接使用的 CI |
| `class_prior_` | 选中的 `theta` |
| `theta_grid_` | 实际使用的候选网格 |
| `objective_values_` | 每个候选 `theta` 的 penL1 目标 |
| `mean_`, `scale_` | 标准化统计量（standardize=True 时） |
| `n_centers_` | 实际 basis 数 |
| `get_params/set_params` | sklearn `BaseEstimator` 参数协议 |

### 10.3 模块落点

| 模块 | 责任 | 状态 |
|---|---|---|
| `pu_toolbox/prior/pen_l1.py` | `ClassPriorEstimator` / `PenL1Estimator` | ✅ penL1 |
| `pu_toolbox/prior/__init__.py` | 公开导出 | ✅ |
| `pu_toolbox/registry/builtin_methods.py` | class-prior metadata 和 binding | ✅ |
| `tests/unit/prior/test_pen_l1.py` | 闭式目标 smoke、确定性、边界测试 | ✅ |
| `benchmarks/paper_like/class_prior_estimation/` | 合成 overlap、MNIST one-vs-rest | ⏳ |

## 11. 测试与验收标准

### 11.1 API 与边界

- 没有正样本或没有 U 时拒绝。
- `sigma <= 0`、`reg_lambda <= 0` 时拒绝。
- `theta_grid` 必须是一维、非空且落在 `[0,1]`。
- `estimate()` 在 fit 前抛出 `NotFittedError`。
- 相同输入和参数时输出确定。
- 输出先验和目标数组均为有限值。

### 11.2 数学测试

使用人工 Gaussian basis 验证：

```text
beta(theta) = theta * mean(Phi_P, axis=0) - mean(Phi_U, axis=0)
alpha(theta) = maximum(beta(theta), 0) / lambda
J(theta) = dot(alpha(theta), beta(theta)) - theta + 1
```

测试必须分别验证 P/U 分母，不能只验证最终 `argmin`。

### 11.3 统计/论文复现测试

- 在已知 `pi` 的合成 mixture 上报告 bias、MAE 和标准差；
- 改变 overlap、P/U 样本量和 `pi`，画出估计误差曲线；
- 对 `sigma` 和 `lambda` 做 nested CV，避免用真实 `pi` 选择参数；
- 与 ReCPE、Elkan-Noto/其他 CPE baseline 对比时，明确不同方法的输入假设；
- 报告失败比例和估计落在边界的比例，不只报告均值。

## 12. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_related`；作者页面源码与本文 penL1 公式需分开核对 |
| Implementation status | `NATIVE`，当前为 penL1 clean-room |
| 已实现 | Gaussian basis、penL1 闭式系数、先验网格搜索、统一 prior API |
| 未实现 | L1-QP、论文完整 CV protocol、CI/bootstrap、paper-like benchmark |
| 主要风险 | basis 尺度、先验搜索网格、P/U 抽样偏差和有限样本误差都会显著影响 `pi_hat` |
| 解释边界 | `estimate()` 是 mixture proportion/class prior estimate，不是分类器概率，也不是置信区间 |
