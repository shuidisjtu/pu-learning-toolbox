# Method Card: Convex Formulation for PU Classification

> 标注约定：`【项目适配】` 表示依据当前项目模板/API 约定补充的内容，不是论文原文结论；`【实现建议】` 表示论文未规定、需要工程侧自行确定的实现选择。

## 1. 待办与注意

### 1.1 待办

- **实现主方法 C-DH**：实现基于 double hinge loss 的凸 PU 分类器。`【项目适配】` 建议类名 `UPUClassifier`，默认 `loss="double_hinge"`。
- **实现可选方法 C-LL**：logistic loss 版本目标光滑，可作为更易实现的备选和回归对照。
- **接入类别先验**：该方法训练时必须已知 $`\pi=p(y=1)`$。`【项目适配】` 类别先验估计应由独立 estimator/pipeline 完成，不在分类器内部隐式估计。
- **实现 PU-CV**：按论文式 (2) 的 zero-one PU 风险选择 $`\lambda`$、basis/kernel 参数；论文未给出候选网格和折数。
- **确定 QP 后端**：C-DH 需要凸二次规划。`【实现建议】` 优先复用项目已有 QP solver；若无，使用可选后端，避免把重依赖设为基础安装必需项。
- **写测试**：覆盖可分数据决策边界、类别先验敏感性、MNIST one-vs-rest、求解器最优性、输入边界条件。
- **规模控制**：Gaussian basis 以全部 P/U 样本为中心时，basis 数 $`m=n_P+n_U`$，内存和求解成本较高；需提供中心下采样或 linear basis。

### 1.2 注意

- 输入必须同时包含 positive set 和 unlabeled set；不使用显式 negative samples。
- positive set 必须可视为从 $`p(x\mid y=1)`$ 独立同分布抽样，unlabeled set 从当前目标边缘分布 $`p(x)`$ 抽样。
- `【项目适配】` 若输入为 single-training-set `y_pu`，已标注正样本需能代表完整正类分布；这相当于项目侧需要满足 SCAR-like 的抽样条件。论文未使用 SCAR 术语，也未分析该条件失效时的偏差。
- **普通 Hinge 不能直接用于本文凸框架**：其 composite loss 不是线性函数，会使目标非凸。
- **普通加权 LogReg/Hinge 不是本文方法**：直接把 unlabeled 当作 negative 会引入 superfluous penalty；它们在论文中仅作为有偏基线。
- **zero-one loss 疑似排版符号错误**：论文式 (1) 写成 $`\frac12\mathrm{sign}(z)+\frac12`$，但该式对应“分类正确指示量”而非待最小化的误分类损失。实现和 CV 应使用
```math
  \ell_{0\text{-}1}(z)=\mathbf{1}[z\le 0]
```
  或忽略 $`z=0`$ 约定时等价的 $`\frac12-\frac12\mathrm{sign}(z)`$。
- 论文只笼统提出在退化情形下可增加非负约束以避免数值问题，没有给出完整的稳定化算法。不要将后续 nnPU 的 non-negative risk correction 直接视为本文目标。
- `【项目适配】` 官方关联代码库 [pywsl](https://github.com/t-sakai-kure/pywsl)（Sugiyama Lab，MIT）包含 uPU 实现，应作为算法参考；本卡公式以论文原文为权威来源。

---

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Convex Formulation for Learning from Positive and Unlabeled Data |
| Authors | Marthinus Christoffel du Plessis, Gang Niu, Masashi Sugiyama |
| Venue | ICML 2015，PMLR/JMLR W&CP Vol. 37 |
| Family | `risk_estimation` |
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
| Output | Binary classifier；论文 $`\hat y=\mathrm{sign}(g(x))\in\{+1,-1\}`$，`【项目适配】` predict 输出 $`\{0,1\}`$ |

### 2.1 适用场景标签

| 场景 | 论文中的含义 |
|---|---|
| `identification` | 用少量目标正样本从未标注集合中识别相似样本，如自动人脸标注 |
| `inlier_based_outlier_detection` | 已知 inlier 样本，从未标注集合中识别 outlier |
| `one_vs_rest` | negative class 过于多样，难以代表性采集 |
| `negative_class_shift` | negative 分布随时间变化，仅更新 unlabeled set 比持续重采 negative 更低成本 |

### 2.2 数据假设

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

## 3. 符号与记号

| 论文符号 | 含义 | 开发侧对应 |
|---|---|---|
| $`x\in\mathbb{R}^d`$ | 输入特征 | feature row |
| $`y\in\{1,-1\}`$ | 真实类别 | fully labeled evaluation only |
| $`P=\{x_i^P\}_{i=1}^{n_P}`$ | 正类样本 | `X_pos` |
| $`U=\{x_j^U\}_{j=1}^{n_U}`$ | 未标注样本 | `X_unlabeled` |
| $`\pi=p(y=1)`$ | unlabeled/目标分布中的正类先验 | `class_prior` |
| $`\mathbb{E}_P`$ | 对 $`p(x\mid y=1)`$ 的期望 | positive mean |
| $`\mathbb{E}_N`$ | 对 $`p(x\mid y=-1)`$ 的期望 | unavailable during training |
| $`\mathbb{E}_U`$ / $`\mathbb{E}_X`$ | 对 $`p(x)`$ 的期望 | unlabeled mean |
| $`g(x)`$ | 判别函数 | `decision_function` |
| $`\ell(z)`$ | ordinary margin loss | loss on unlabeled term |
| $`\tilde\ell(z)`$ | composite loss：$`\ell(z)-\ell(-z)`$ | positive-term loss |
| $`\phi(x)\in\mathbb{R}^m`$ | basis vector | transformed feature |
| $`\alpha\in\mathbb{R}^m`$ | basis coefficients | `coef_` |
| $`b`$ | intercept | `intercept_` |
| $`\lambda`$ | $`\ell_2`$ 正则系数 | `reg_lambda` |
| $`\Phi_P,\Phi_U`$ | P/U 的 basis design matrices | transformed matrices |
| $`\xi\in\mathbb{R}^{n_U}`$ | C-DH slack variables | QP auxiliary variables |

---

## 4. 核心公式

### 4.1 普通监督风险

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

### 4.2 直接替换 surrogate loss 的偏差来源

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

### 4.3 凸 PU 风险

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

### 4.4 经验目标和模型

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

### 4.5 Loss 选择

| Loss | $`\ell(z)`$ | $`\tilde\ell(z)`$ | 优化 | 开发结论 |
|---|---|---|---|---|
| Squared | $`\frac14(z-1)^2`$ | $`-z`$ | 凸；无 intercept 时可闭式求解 | optional；正确大 margin 仍会被惩罚 |
| Logistic | $`\log(1+\exp(-z))`$ | $`-z`$ | 光滑凸；quasi-Newton | optional，易实现 |
| Hinge | $`\frac12\max(0,1-z)`$ | 非线性且非凸 | 不满足本文凸条件 | **禁止作为本文实现** |
| Double hinge | $`\max\{-z,0,\frac12-\frac12z\}`$ | $`-z`$ | 凸 QP | **主实现** |

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

实现时用稳定的 softplus：

```python
softplus_g = np.logaddexp(0.0, g_u)
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

其中 $`\sigma(t)=1/(1+\exp(-t))`$。

#### Squared loss（optional）

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

实现应使用线性方程求解，不显式计算矩阵逆。

### 4.6 PU-CV 目标

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

`【实现建议】` 对 P 和 U 分别做 K-fold 划分，保持每折均有 P/U；用各折风险均值选参数。该估计因有限样本可能小于 0，可用于模型排序，不应裁剪后再比较。

### 4.7 理论收敛

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

## 5. 算法概要

### 5.1 推荐实现路径：C-DH

```text
Input:
    X, y_pu
    class_prior pi
    reg_lambda
    basis configuration
    QP solver configuration

1. Split:
       P = X[y_pu == +1]
       U = X[y_pu == 0]

2. Validate:
       P and U are non-empty
       0 < pi < 1
       reg_lambda > 0
       all features are finite

3. Fit feature preprocessing on training fold only.

4. Build basis:
       linear: phi(x) = x
       or Gaussian/polynomial basis

5. Build Phi_P and Phi_U.

6. Solve the convex QP in §4.5.

7. Store alpha, b, preprocessing state and solver diagnostics.

8. Predict:
       score = alpha.T @ phi(x) + b          # _decision_function
       label = 1 if score >= 0 else 0         # _predict: {0, 1} per BasePUClassifier
```

### 5.2 超参数

| 参数 | 论文设置 | 项目侧处理 |
|---|---|---|
| $`\pi`$ | 训练时已知 | required constructor/fit argument |
| $`\lambda`$ | 通过 CV 选择 | `reg_lambda_grid` |
| basis type | Gaussian / linear / polynomial 均可 | `【项目适配】` 默认 linear，RBF optional |
| $`\sigma`$ | Gaussian width；通过 CV 选择 | `kernel_width_grid` |
| centers | 全部 P/U 样本 | 提供 `all`、subsample、固定数量策略 |
| intercept | 使用 $`b`$ | 默认开启，不正则化 |
| solver tolerance | 未规定 | 暴露 `tol`、`max_iter` |

### 5.3 推荐默认

- `loss="double_hinge"`：论文实验中与非凸 ramp loss 精度相当，计算成本显著更低。
- `basis="linear"`：`【项目适配】` 作为可扩展默认；RBF 全样本 centers 仅用于小规模复现实验。
- `class_prior` 必填，不在分类器内自动估计。
- $`\lambda`$ 不设置“论文默认值”，必须通过 PU-CV 或由用户显式提供。
- C-LL 可作为 QP 后端不可用时的可选算法，但不要静默替换用户指定的 C-DH。

### 5.4 复杂度与扩展性

设 basis 数为 $`m`$：

- 构建设计矩阵：约 $`O((n_P+n_U)m)`$。
- 全样本 Gaussian centers：$`m=n_P+n_U`$，设计矩阵内存约为 $`O((n_P+n_U)^2)`$。
- C-DH QP 变量数约为 $`m+1+n_U`$，约束数为 $`3n_U`$。
- C-LL 每次目标/梯度计算约为 $`O((n_P+n_U)m)`$。
- Squared closed form 朴素线性代数成本约为 $`O(m^3)`$。

---

## 6. 源码状态

| 字段 | 内容 |
|---|---|
| Source status | `official_bundle` |
| Upstream URL | https://github.com/t-sakai-kure/pywsl |
| License | MIT |
| Framework | uPU 部分：Python (NumPy + scipy + scikit-learn)；nnPU 部分：Chainer |
| 包含方法 | `pywsl` 由 Sugiyama Lab 维护；uPU 仅提供 Squared Loss 闭式解（NumPy + scipy），不含 double hinge（C-DH）和 logistic（C-LL）；nnPU / PNU / PU-SKC / PNU-AUC 等其他方法见仓库 README |
| Implementation basis | 论文式 (8)、式 (9)、Table 1、式 (2) CV |
| 实现策略 | `【项目适配】` adapter (pywsl) + native 实现（见 `docs/resources_optimized.md`） |
| Reproduction risk | solver 选择、CV 网格、预处理、basis-center 策略均未由论文完整规定 |

---

## 7. API 接口

### 7.1 分类器 API

| API | 约定 |
|---|---|
| `fit(X, y_pu, *, class_prior=None, sample_weight=None)` | `【项目适配】` 签名匹配 `BasePUClassifier`；`y_pu=+1` 表示 labeled positive，`y_pu=0` 表示 unlabeled；`class_prior` 可覆盖构造参数；`sample_weight` 被接受但忽略（本方法不支持逐样本权重） |
| `_decision_function(X)` | 返回 $`g(x)`$，shape `(n_samples,)`；公共 `decision_function(X)` 由基类包装，自动检查 fitted 状态 |
| `_predict(X)` | `【项目适配】` 返回 `{0, 1}`：`(g(x) >= 0).astype(int)`；与论文的 $`\mathrm{sign}(g(x))`$ 等价但编码不同，遵循 `BasePUClassifier` 契约 |
| `predict_proba(X)` | 抛出 `NotImplementedError`；$`g(x)`$ 不是天然校准概率 |
| `pu_validation_risk(X, y_pu)` | 返回式 (2) 的 PU zero-one 风险，用于调参；`【项目适配】` 此为本分类器扩展方法，不在 `BasePUClassifier` 契约中 |
| `score_samples(X)` | 复用 `_decision_function`，无需覆盖 |
| `get_params()` / `set_params()` | 由 sklearn `BaseEstimator` 提供；暴露 loss、$`\pi`$、$`\lambda`$、basis、kernel width、centers 参数 |

### 7.2 建议构造参数

```python
class UPUClassifier(BasePUClassifier):
    def __init__(
        self,
        class_prior: float,
    loss: Literal["double_hinge", "logistic", "squared"] = "double_hinge",
    reg_lambda: float = 1e-3,
    basis: Literal["linear", "rbf"] = "linear",
    kernel_width: float | None = None,
    n_centers: int | None = None,
    fit_intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-6,
        random_state: int | None = None,
    ):
        super().__init__()
```

`reg_lambda=1e-3` 仅为 `【项目适配】` API 占位建议，不是论文推荐默认值；基准实验和正式使用应调参。

### 7.3 类别先验 pipeline

```text
PriorEstimator.fit(X, y_pu)
pi_hat = PriorEstimator.estimate()

UPUClassifier(class_prior=pi_hat, ...)
    .fit(X, y_pu)
```

避免分类器在同一 CV validation fold 上重新估计 $`\pi`$，否则可能产生信息泄漏。若联合调参，prior estimation 必须在每个 training fold 内完成。

---

## 8. Toolbox 集成映射

### 8.1 文件与注册

| 项目 | 内容 |
|---|---|
| Loss 模块 | `pu_toolbox/losses/upu.py` — uPU 风险函数，`UPULoss(BasePULoss)` |
| Estimator 模块 | `pu_toolbox/estimators/risk/upu.py` — 分类器，`UPUClassifier(BasePUClassifier)` |
| 注册名称 | `"upu"`（已实现，`implementation_status=NATIVE`） |
| 别名 | `["convex_pu", "unbiased_pu", "u-pu"]` |
| 导出 | `losses/__init__.py` 添加 `UPULoss`；`estimators/risk/__init__.py` 添加 `UPUClassifier` |

### 8.2 注册表更新

> ✅ 已完成。`upu` 条目已更新为 `NATIVE`，已绑定 `UPUClassifier`，已添加 lazy import。

### 8.3 类级元数据

```python
family = AlgorithmFamily.RISK_ESTIMATION
assumption = (Assumption.SCAR,)
scenario = (Scenario.CASE_CONTROL,)
requires_class_prior = True
implementation_status = ImplementationStatus.NATIVE
source_status = SourceStatus.OFFICIAL_BUNDLE
backend = Backend.NUMPY        # C-DH 使用 QP solver；见 §8.5 后端决策
maturity = Maturity.STABLE
```

### 8.4 双模块架构：Loss vs. Estimator

`【项目适配】` 本论文涉及两个层次的实现：

**Loss 模块** (`losses/upu.py`)：
- `UPULoss(BasePULoss)` 实现可微分的凸 PU 风险
- 接口：`__call__(positive_scores, unlabeled_scores, *, class_prior)` → scalar loss
- 仅 C-LL（logistic）和 Squared loss 可用此接口（可微分）
- 该 loss 可被 nnPU、PNU 等同族方法复用和扩展

**Estimator 模块** (`estimators/risk/upu.py`)：
- `UPUClassifier(BasePUClassifier)` 为完整分类器
- C-DH（double hinge）需要 QP solver，不适合通用 loss 接口，直接在 estimator 中实现
- C-LL 版本可选择调用 `UPULoss` + 梯度优化器，或独立实现

### 8.5 后端决策

`【项目适配】` 注册表当前声明 `Backend.TORCH`，但需分情况处理：

| Loss 变体 | 后端 | 理由 |
|---|---|---|
| C-DH (double hinge) | `Backend.NUMPY` | QP 求解，非梯度优化 |
| C-LL (logistic) | `Backend.NUMPY` 或 `Backend.TORCH` | 光滑凸，L-BFGS（numpy）或 SGD/Adam（torch）均可 |
| Squared | `Backend.NUMPY` | 闭式解 |

建议：
- `UPUClassifier` 默认 `Backend.NUMPY`（覆盖所有 loss 变体）
- `UPULoss` 可选 PyTorch 实现，供深度模型 estimator（如 `estimators/risk/nnpu.py`）使用
- 注册表的 `backend` 字段在实现时根据实际决策更新

### 8.6 参考实现

| 优先级 | 代码库 | 用途 |
|---|---|---|
| **主参考** | [`t-sakai-kure/pywsl`](https://github.com/t-sakai-kure/pywsl) | Sugiyama Lab 官方关联实现（MIT），包含 uPU/nnPU/PNU，用作算法验证基准 |
| 论文参考 | ICML 2015 论文 | 公式权威来源（式 (8)/(9)、Table 1、式 (2) CV） |

### 8.7 与 nnPU / PNU 的共享基础设施

`【项目适配】` uPU、nnPU、PNU 归属同一工作包 WP9，共享 PU 风险分解框架：

```math
\hat R_{\text{PU}} = \pi \hat R_P^{+} + \hat R_U^{-} - \pi \hat R_P^{-}
```

| 方法 | 与 uPU 的关系 | 共享组件 |
|---|---|---|
| nnPU (Kiryo 2017) | 在 uPU 基础上对 $`\hat R_U^{-} - \pi \hat R_P^{-}`$ 加非负约束 | `UPULoss` 的计算逻辑 |
| PNU | 在 uPU 基础上加入 PN 和 NU 风险的凸组合 | 风险分解函数 |

建议在 `losses/base.py` 或 `losses/upu.py` 中提取共用的风险分解工具函数，供 `nnpu.py` 和 `pnu.py` 调用，而非各自独立实现相同的分解逻辑。

---

## 9. 测试参考

### 9.1 Fully separable boundary test

论文设置：

```text
p(x | y = +1) = Uniform(0.1, 1.0)
p(x | y = -1) = Uniform(-1.1, -0.1)
```

对线性模型 $`g(x)=wx+b`$，正确边界满足：

```math
-\frac{b}{w}\in[-0.1,0.1]
```

测试要求：

- C-DH 和 C-LL 的平均边界落入或接近该区间；
- naive LogReg/Hinge 可作为预期失败基线；
- 在多个 $`\pi`$ 和随机种子下运行；
- 不以单次随机结果作为断言。

### 9.2 Local-minimum illustration

论文用于说明 ramp loss 非凸性的设置：

```text
p(x | y = +1) = Normal(mean=2, variance=1/2)
p(x | y = -1) = Normal(mean=-2, variance=1/2)
n_P = 10
n_U = 20
pi = 0.5
g(x) = w*x + b
lambda = 1e-3
```

C-DH 应从不同初始化得到相同目标值/决策边界（求解器容差内）。Ramp 仅在实现非凸对照时用于展示不同局部解，不是主模块必测依赖。

### 9.3 MNIST one-vs-rest benchmark

论文设置：

```text
positive class: digit 0
negative class: one digit from 1..9
PCA dimensions: 2
n_P: 200
n_U: 400
class_prior: 0.1, 0.4, 0.7
hyperparameter selection: PU zero-one CV, Eq. (2)
```

对比：

```text
C-DH
C-LL
naive Hinge
naive LogReg
optional Ramp baseline
```

验收重点：

- C-DH 精度应明显优于 naive Hinge/LogReg；
- C-DH 与 Ramp 的误差应处于相近量级；
- C-DH 运行时间应显著低于 CCCP Ramp；
- 不硬编码论文表格中的单个百分比，使用多随机种子和区间断言。

### 9.4 Convergence sanity test

固定 basis 维度 $`m`$，增大 $`n_P,n_U`$：

```text
n in {100, 200, 400, 800}
fixed feature map
fixed pi
multiple random seeds
```

预期参数误差或目标误差总体下降。若用大样本解作 proxy，可检查 log-log slope 接近 $`-1/2`$，但测试容差需宽松。

### 9.5 类别先验敏感性

分别训练：

```text
pi_used = pi_true + delta
delta in {-0.2, -0.1, 0, +0.1, +0.2}
```

预期 $`\delta=0`$ 附近表现最好或最稳定。该测试用于暴露 pipeline 中先验估计误差，不把单调性作为严格理论断言。

### 9.6 API 与边界条件

- P 为空或 U 为空：抛出 `ValueError`。
- `class_prior <= 0` 或 `>= 1`：抛出 `ValueError`。
- `reg_lambda <= 0`：抛出 `ValueError`。
- 输入含 NaN/Inf：抛出明确错误。
- `basis="rbf"` 且未给 kernel width：要求 CV 配置或明确默认策略。
- QP solver 未收敛：保留状态、迭代次数和消息，不返回伪成功模型。
- `predict_proba()`：明确 `NotImplementedError`。
- 固定随机种子时，center subsampling 和 CV split 可复现。
