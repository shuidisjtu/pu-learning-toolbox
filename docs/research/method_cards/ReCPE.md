# Method Card: Regrouping Class-Prior Estimation (ReCPE)

## 1. 待办与注意

### 1.1 待办

- [x] 实现 ReCPE 的 regrouping 流程：训练正类/未标记二分类器、选择最像正类的未标记样本、复制到正类集合，再调用底层 CPE。
- [x] 提供可替换的 `base_estimator` 接口，使 ReCPE 可以包裹 KM、AlphaMax、DEDPUL 等其他类先验估计器。
- [x] 提供默认的 classifier-based mixture-proportion baseline，保证没有其他 CPE 实现时仍可运行。
- [x] 接入 `BasePriorEstimator` 和 registry，注册名为 `recpe`，别名为 `re_cpe`、`rethinking_cpe`。
- [x] 编写合成数据、底层估计器注入、边界条件测试。
- [ ] 增加与论文神经网络分类器、KM1/KM2/AlphaMax 的 paper-like benchmark。

### 1.2 注意

- ReCPE 是一个**外层 regrouping 方法**，论文并未限定唯一的底层 CPE；最终效果依赖底层估计器和正类/未标记分类器。
- 论文中的 `S_p` 应来自 $`P_p=p(x\mid y=1)`$，`S_u` 应来自边缘分布 $`P_u=p(x)`$。如果项目的 `y_pu` 来自 single-training-set，已标记正例需要能够代表完整正类分布。
- 论文希望缓解 irreducibility 失效导致的正向偏差，但不等于对任意数据分布都能无偏估计类别先验。
- 复制比例 `copy_fraction` 太小会使底层 CPE 对 regrouping 不敏感；太大则会显著改变辅助正类分布。论文实验统一使用 `p=10%`，本实现默认 `0.1`。
- 当前默认排序器为 sklearn `LogisticRegression`，不是论文实验中的两层神经网络；因此当前实现是算法逻辑对齐，而不是完整实验数值复现。
- 当前默认 base CPE 是工程侧提供的 classifier-based baseline，不应与论文对比的 KM、AlphaMax 等方法混同。

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Rethinking Class-Prior Estimation for Positive-Unlabeled Learning |
| Authors | Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Gang Niu, Masashi Sugiyama, Dacheng Tao |
| Venue | arXiv preprint，arXiv:2002.03673v2 |
| Year | 2022 |
| Family | `class_prior_estimation` |
| Scenario | `case_control`；也可适配单一 PU 数据集，但需保证正例抽样条件 |
| Requires class prior | `False`，方法输出类别先验估计 |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| GPU required | `False` |

### 2.1 研究问题

给定正类样本和未标记样本，传统的 distributional-assumption-free CPE 方法通常估计未标记分布中正类分布的最大混合比例。它们隐含依赖 irreducibility assumption：正类分布的 support 不能被负类分布完全包含。

当该假设不成立时，传统方法会系统性高估类别先验。ReCPE 不直接估计原始 $`P_p`$ 在 $`P_u`$ 中的最大比例，而是从 $`P_u`$ 中找出最像正类的一小部分样本，构造辅助正类分布 $`P_p'`$，再调用已有 CPE 方法。

### 2.2 数据假设

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

## 3. 符号与记号

| 论文符号 | 含义 | 开发侧对应 |
|---|---|---|
| $`P_p`$ | 正类条件分布 | `X[y_pu == 1]` |
| $`P_n`$ | 负类条件分布 | 不可直接观测 |
| $`P_u`$ | 未标记边缘分布 | `X[y_pu == 0]` |
| $`\pi`$ | 真实类别先验 $`P(y=1)`$ | 未知目标 |
| $`S_p`$ | 正类样本集 | `positive_samples` |
| $`S_u`$ | 未标记样本集 | `unlabeled_samples` |
| $`A`$ | 被 regrouping 的小样本集合 | 从 `S_u` 中选择的样本 |
| $`p`$ | 复制比例 | `copy_fraction` |
| $`P_p'`$ | regrouping 后的辅助正类分布 | 复制样本后的 positive set |
| $`\pi'`$ | 辅助问题中的新类别先验 | `base_estimator_.estimate()` |
| $`q(C=1\mid x)`$ | 样本属于正类样本来源的后验 | 排序器的 positive probability |
| $`q(C=0\mid x)`$ | 样本属于未标记来源的后验 | `1 - positive_probability` |

## 4. 核心公式

### 4.1 传统 CPE 的偏差

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

### 4.2 Regrouping 后的分布

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

### 4.3 实际 regrouping 过程

由于 $`P_n`$ 不可观测，工程实现用 $`P_u`$ 中最像正类的样本近似集合 $`A`$：

1. 将 $`S_p`$ 标为来源类别 1，将 $`S_u`$ 标为来源类别 0，训练二分类器。
2. 对 $`x\in S_u`$ 计算 $`q(C=1\mid x)`$。
3. 选取 positive probability 最大的前 $`p|S_u|`$ 个未标记样本。
4. 将这些样本复制到 $`S_p`$，形成辅助正类样本集 $`S_p'`$。
5. 用 $`S_p'`$ 和原始 $`S_u`$ 调用底层 CPE，得到 $`\hat\pi'`$。

### 4.4 选择集合 $`A`$

论文给出的理想目标是最小化集合中“像负类”的质量与“像正类”的质量之比：

```math
A^*=\arg\min_{A\in\mathcal S}
\frac{\mathbb E_q[\mathbf 1_A(X)q(C=0\mid X)]}
{\mathbb E_q[\mathbf 1_A(X)q(C=1\mid X)]}
```

实际使用分类器后验进行近似，因此选择 $`q(C=1\mid x)`$ 最大的样本。

### 4.5 辅助分布的样本近似

论文实际使用复制样本近似 $`P_p'`$：

```math
\widetilde P_p'
=\frac{P_u^A+P_p}{P_u(A)+1}
```

当 $`P_u(A)`$ 较小时，$`\widetilde P_p'`$ 与理论上的 $`P_p'`$ 接近。`copy_fraction` 控制 $`A`$ 的大小。

## 5. 算法概要

```text
输入：正类样本 Sp，未标记样本 Su，复制比例 p，底层 CPE 算法 A

1. 用 Sp 标记为来源正类、Su 标记为来源未标记，训练分类器 h。
2. 对 Su 中每个样本计算 h 的 positive probability。
3. 选择 positive probability 最大的前 ceil(p * |Su|) 个样本 A_hat。
4. 将 A_hat 复制到 Sp，形成辅助正类集合 Sp'。
5. 用 Su 和 Sp' 调用底层 CPE 算法 A。
6. 输出底层算法得到的类别先验估计 pi_hat'。
```

当前实现中的默认选择：

| 项目 | 实现选择 |
|---|---|
| 排序分类器 | `StandardScaler + LogisticRegression` |
| 复制比例 | `copy_fraction=0.1` |
| 复制数量 | `max(1, ceil(copy_fraction * n_unlabeled))` |
| 底层 CPE | classifier-based density-ratio baseline |
| 自定义底层方法 | `base_estimator`，要求支持 `fit(X, y_pu)` 和 `estimate()` |

## 6. 源码状态

| 字段 | 内容 |
|---|---|
| Source status | `official_exact` |
| Upstream URL | https://github.com/a5507203/Rethinking-Class-Prior-Estimation-for-Positive-Unlabeled-Learning |
| License | registry 中记录为 MIT；重新分发前应核验上游许可证 |
| Framework | 论文实验使用 Python 神经网络；本项目默认使用 NumPy + scikit-learn |
| Integration mode | `native` |
| 当前实现范围 | ReCPE regrouping 核心流程 + 可注入底层 CPE + 默认 baseline |
| 尚未完全复现 | 论文中的神经网络结构、验证集选择、UCI 全量实验和所有基线方法 |

## 7. API 接口

### 7.1 构造函数

```python
class ReCPEEstimator(BasePriorEstimator):
    def __init__(
        self,
        copy_fraction=0.1,
        base_estimator=None,
        classifier=None,
        classifier_max_iter=1000,
    ):
        ...
```

| 参数 | 含义 |
|---|---|
| `copy_fraction` | 从未标记集复制到正类集的比例，必须在 `(0, 1)` 内 |
| `base_estimator` | 底层 CPE；需要实现 `fit(X, y_pu)` 和 `estimate()`；默认使用内置 baseline |
| `classifier` | 自定义来源分类器；需要支持 `fit()`、`predict_proba()`；默认使用 Logistic Regression |
| `classifier_max_iter` | 默认排序分类器的最大迭代次数 |

### 7.2 方法与拟合属性

| API / 属性 | 约定 |
|---|---|
| `fit(X, y_pu)` | `y_pu=1` 表示 positive，`y_pu=0` 表示 unlabeled；返回 `self` |
| `estimate()` | 返回标量 `float`，即 $`\hat\pi'`$ |
| `confidence_interval(alpha)` | 返回 `None`；论文未给出置信区间方法 |
| `get_params()` / `set_params()` | 由 sklearn `BaseEstimator` 提供 |
| `class_prior_` | `fit()` 后的类别先验估计 |
| `base_estimator_` | 实际拟合的底层 CPE 实例 |
| `classifier_` | 实际拟合的来源分类器 |
| `selected_indices_` | 原始未标记数组中被复制样本的索引 |
| `copy_count_` | 实际复制的样本数量 |
| `get_metadata()` | 返回方法名、复制比例、复制数量、估计值和底层估计器名称 |

### 7.3 底层 CPE 接口

```python
class BasePriorEstimator:
    def fit(self, X, y_pu):
        ...

    def estimate(self) -> float:
        ...
```

ReCPE 会在 regrouping 后构造新的 `X/y_pu`，其中被选中的未标记样本会被改为 `y_pu=1`，然后把数据交给底层 CPE。

## 8. Toolbox 集成映射

### 8.1 文件与注册

| 项目 | 内容 |
|---|---|
| Prior 模块 | `pu_toolbox/prior/recpe.py` |
| 类名 | `ReCPEEstimator(BasePriorEstimator)` |
| 导出 | `pu_toolbox/prior/__init__.py` |
| 注册名称 | `recpe` |
| 别名 | `re_cpe`、`rethinking_cpe` |
| Registry 状态 | `implementation_status=NATIVE` |
| 后端 | `Backend.NUMPY` |

### 8.2 类级元数据

```python
family = AlgorithmFamily.CLASS_PRIOR_ESTIMATION
assumption = (Assumption.SCAR,)
scenario = (Scenario.SINGLE_TRAINING_SET, Scenario.CASE_CONTROL)
requires_class_prior = False
implementation_status = ImplementationStatus.NATIVE
source_status = SourceStatus.OFFICIAL_EXACT
backend = Backend.NUMPY
maturity = Maturity.STABLE
```

### 8.3 使用示例

```python
from pu_toolbox.prior import ReCPEEstimator

estimator = ReCPEEstimator(copy_fraction=0.1)
estimator.fit(X, y_pu)
pi_hat = estimator.estimate()
```

接入自定义底层 CPE：

```python
recpe = ReCPEEstimator(
    copy_fraction=0.1,
    base_estimator=your_cpe_estimator,
)
recpe.fit(X, y_pu)
```

## 9. 测试参考

### 9.1 基础流程测试

- 输入合成正类和未标记数据，`fit()` 后 `estimate()` 返回 `[0, 1]` 内的浮点数。
- `selected_indices_` 长度等于 `copy_count_`。
- `copy_count_ = ceil(copy_fraction * n_unlabeled)`，至少复制一个样本。

### 9.2 底层估计器组合测试

- 传入自定义 `base_estimator`。
- 验证 regrouping 后正类数量增加。
- 验证 ReCPE 的最终估计值来自底层估计器，而不是硬编码值。

### 9.3 边界条件测试

- `copy_fraction <= 0` 或 `copy_fraction >= 1` 应抛出 `ValueError`。
- 未拟合时调用 `estimate()` 应抛出 `NotFittedError`。
- 缺少 positive 或 unlabeled 样本时应给出明确错误。
- 输入包含非法 PU 标签、非二维特征或非有限值时遵守项目 validation 约定。

### 9.4 论文复现实验

- 不可约高斯混合数据：比较 base CPE 与 ReCPE 的绝对估计误差。
- 可约高斯混合数据：验证 regrouping 不应显著恶化 base CPE。
- 复制比例扫描：比较 `p∈{0.05, 0.10, 0.15, 0.20}` 的估计误差。
- UCI 二分类数据：按论文的正例比例、样本量和重复实验协议评估。

当前自动化验证：`pytest -q`，133 项测试通过；针对 ReCPE 的测试位于 `tests/unit/prior/test_recpe.py`。

## 10. 开放问题

- 增加 KM1、KM2、AlphaMax、DEDPUL 等底层 CPE 后端，并建立统一 benchmark。
- 增加论文中的两层神经网络来源分类器和验证集选择流程。
- 明确 `single_training_set` 与 `case_control` 两种数据场景下的 API 标注和可识别性提示。
- 为复制比例和底层 CPE 增加 PU-CV 或 paper-like 调参协议。
