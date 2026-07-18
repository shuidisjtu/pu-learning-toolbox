# Method Card: Elkan–Noto PU Learning

## 1. 待办与注意

### 1.1 待办

- 实现两阶段流程：先训练 `s=1` vs. `s=0` 的概率分类器 $`g(x)`$，再估计标注倾向 $`c`$ 并输出传统正类概率 $`f(x)`$。
- 训练 $`g`$ 时必须使用**概率校准**的分类器；默认基础学习器为 `LogisticRegression()`，若用户传入其他模型需确保输出可靠概率，必要时增加校准步骤（Platt scaling 或 isotonic regression）。
- 通过构造函数 `mode` 参数切换两种使用方式：`"probability_correction"`（概率校正，默认）和 `"weighted_retraining"`（对无标签样本复制并加权后重训）。
- 通过构造函数 `n_cv_folds=3` 控制内部分层 K-fold out-of-fold 预测来估计 $`c`$；不得用同一模型对其训练正例的 in-sample 预测直接估计 $`c`$。
- 对估计出的概率、权重和类别先验做 $`[0,1]`$ 范围检查；数值越界应报告校准/SCAR 假设可能失效，而非静默当作理论保证。
- 仅实现首选估计量 $`\hat c_1`$（验证正例上 $`g(x)`$ 的均值），不暴露 $`\hat c_2/\hat c_3`$ 切换参数。
- 实现策略：Native clean-room（以 `pulearn` 为算法参考，以本项目 `BasePUClassifier` 为 API 契约）。

### 1.2 注意

- 本文假设 **SCAR**：在真实正类内，被标注的概率与特征 $`x`$ 无关。若标注机制依赖 $`x`$（SAR），$`f(x)=g(x)/c`$ 和后续权重均会有系统偏差。
- 论文的类别先验估计只适用于**单一训练集**：样本先从总体 $`p(x,y,s)`$ 抽取，再只记录 $`(x,s)`$。若 $`P`$ 与 $`U`$ 是独立收集的 case-control 数据，$`p(y=1)`$ 不可由本文识别。
- 仅做排序时无需估计 $`c`$：$`f(x)`$ 是 $`g(x)`$ 的正比例变换，二者排序相同。
- 概率校正依赖 $`g(x)\approx p(s=1\mid x)`$，不是普通分类分数；AUC 高不代表概率可用于本方法。
- 原论文未给出置信区间、SCAR 检验、SAR 修正或现代深度模型的校准 protocol；这些不能作为本文保证的一部分。

---

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Learning Classifiers from Only Positive and Unlabeled Data |
| Authors | Charles Elkan, Keith Noto |
| Venue | KDD |
| Year | 2008 |
| Family | `pu_learning` |
| Setting | 单一训练集中的 PU 学习；也讨论 case-control 应用，但先验不可识别 |
| Requires class prior | `False`（仅单一训练集时可同时估计） |
| Requires propensity | `False`（方法估计常数 $`c`$） |
| Requires negative samples | `False` |
| GPU required | `False` |

### Assumptions

令 $`y\in\{0,1\}`$ 为真实类别，$`s\in\{0,1\}`$ 表示是否被标注。仅正例可被标注：

```math
p(s=1\mid x,y=0)=0.
```

SCAR 假设为：

```math
p(s=1\mid x,y=1)=p(s=1\mid y=1)=c,
\qquad 0<c\le 1.
```

训练数据从 $`p(x,y,s)`$ 随机抽样，但只观测 $`(x,s)`$。这一区别对类别先验可识别性至关重要。

---

## 3. 符号与记号

| 论文符号 | 含义 | 开发侧对应（建议） |
|---|---|---|
| $`x`$ | 特征样本 | `X` 的一行 |
| $`y\in\{0,1\}`$ | 真实正/负标签 | 不可完整观测 |
| $`s\in\{0,1\}`$ | 是否被标注 | `y_pu`：1=标注正例，0=无标签 |
| $`P`$ | 验证集或训练集中的 $`s=1`$ 样本 | labeled positives |
| $`U`$ | $`s=0`$ 的无标签样本 | unlabeled samples |
| $`g(x)`$ | 非传统分类器，$`p(s=1\mid x)`$ | `label_probability` |
| $`f(x)`$ | 传统分类器，$`p(y=1\mid x)`$ | `positive_probability` |
| $`c`$ | 正例被标注的常数概率 $`p(s=1\mid y=1)`$ | `propensity` |
| $`w(x)`$ | 无标签样本为真实正例的后验概率 | `unlabeled_positive_weight` |
| $`m`$ | 单一训练集总样本数 | `n_samples` |
| $`n`$ | 已标注正例数 | `n_labeled` |

---

## 4. 核心公式

### 4.1 中心引理：从 $`g`$ 到真实正类概率

以 $`s`$ 为目标训练概率分类器：

```math
g(x)\approx p(s=1\mid x).
```

在 SCAR 下：

```math
g(x)=p(s=1\mid x)=p(y=1\mid x)\,p(s=1\mid y=1)=c f(x),
```

因此：

```math
\boxed{f(x)=\frac{g(x)}{c}}.
```

论文的首选估计量（在独立验证集的标注正例集合 $`P_V`$ 上计算）：

```math
\hat c=\frac{1}{|P_V|}\sum_{x\in P_V} g(x).
```

论文还列出但不推荐作为首选的估计量：

```math
\hat c_2=\frac{\sum_{x\in P_V}g(x)}{\sum_{x\in V}g(x)},
\qquad
\hat c_3=\max_{x\in V} g(x).
```

原因：$`\hat c_1`$ 使用均值，通常比最大值方差更低，也避免 $`\hat c_2`$ 分母的额外方差。

### 4.2 概率校正与分类阈值

对测试样本：

```math
\hat f(x)=\frac{g(x)}{\hat c}.
```

若以真实正类概率阈值 $`\tau`$ 做二分类，则等价地在 $`g`$ 空间使用阈值 $`\hat c\tau`$；论文使用自然阈值 $`\tau=0.5`$，即 $`g(x)\ge 0.5\hat c`$。

### 4.3 无标签样本的软标签权重

对 $`s=0`$ 的样本：

```math
w(x)=p(y=1\mid x,s=0)
=\frac{(1-c)\,p(s=1\mid x)}{c\,[1-p(s=1\mid x)]}
=\frac{(1-c)g(x)}{c[1-g(x)]}.
```

将每个无标签样本复制两份：

| 副本标签 | 样本权重 |
|---|---:|
| 正类 $`y=1`$ | $`w(x)`$ |
| 负类 $`y=0`$ | $`1-w(x)`$ |

原有标注正例以标签 $`y=1`$、权重 1 参与训练。

### 4.4 类别先验（仅单一训练集）

```math
\widehat{p(y=1)}=\frac{1}{m}\left[n+\sum_{x\in U}w(x)\right].
```

等价的另一估计形式为：

```math
\widehat{p(y=1)}=\frac{n/m}{\hat c}.
```

**限制**：在 case-control 数据中不得使用这两个式子报告可识别的类别先验。

---

## 5. 算法概要

### 5.1 概率校正（推荐用于概率输出）

1. 将标注正例设为 $`s=1`$、无标签样本设为 $`s=0`$，训练并校准 $`g(x)\approx p(s=1\mid x)`$。
2. 用独立验证集的标注正例预测值均值计算 $`\hat c`$。
3. 返回 $`\hat f(x)=g(x)/\hat c`$；排序场景可直接返回 $`g(x)`$。
4. 以目标真实概率阈值 $`\tau`$ 决策时，比较 $`g(x)`$ 与 $`\hat c\tau`$。

### 5.2 加权重训（论文实验使用）

1. 按 5.1 得到校准的 $`g`$ 和 $`\hat c`$。
2. 对每个 $`x\in U`$ 计算 $`w(x)`$。
3. 为每个 $`x\in U`$ 创建正、负两个带权副本；将 $`P`$ 保持为权重 1 的正例。
4. 用支持逐样本 `sample_weight` 的二分类学习器训练最终模型。

**实现要点**:
- 步骤 4 创建 `base_estimator` 的**新实例**（通过 `sklearn.base.clone`），在增强的加权数据集上训练。
- 新模型**替换** $`g`$ 成为最终模型；后续 `predict`/`predict_proba`/`decision_function` 均使用此新模型。
- 加权重训后 `predict_proba` 直接返回最终模型的概率输出（已经是 $`f(x)`$ 近似，无需再除以 $`\hat c`$）。
- `_decision_function` 使用最终模型的 `decision_function` 或 `predict_proba[:, 1]`。

### 5.3 实现保护（非论文规定）

- 在计算 $`w(x)`$ 前将用于分母的 $`g(x)`$ 限制在 $`[\epsilon,1-\epsilon]`$；记录发生裁剪的数量。
- 当 $`\hat c\le\epsilon`$、$`\hat c>1`$、或大量 $`w(x)\notin[0,1]`$ 时抛出明确错误/警告，要求检查校准、数据划分和 SCAR。
- 验证集很小时，$`\hat c`$ 方差会很大；应报告验证集中标注正例数量。

---

## 6. API 接口

### 6.1 构造函数

```python
class ElkanNotoClassifier(BasePUClassifier):
    def __init__(
        self,
        base_estimator=None,             # 默认 sklearn.linear_model.LogisticRegression()
        calibration_method="sigmoid",    # "sigmoid" (Platt) | "isotonic"
        n_cv_folds=3,                    # 分层 K-fold out-of-fold 预测估计 ĉ
        eps=1e-12,                       # g(x) 数值裁剪阈值
        mode="probability_correction",   # "probability_correction" | "weighted_retraining"
        random_state=None,
    ):
```

### 6.2 方法映射（对应 BasePUClassifier 契约）

| 方法 | 约定 |
|---|---|
| `fit(X, y_pu, *, class_prior=None, sample_weight=None)` | 签名匹配基类；`class_prior` 被接受但忽略（Elkan-Noto 自行估计）；内部自动 K-fold 分层划分验证集并估计 ĉ |
| `_predict(X)` | 返回 `(f(x) >= 0.5).astype(int)`，shape `(n_samples,)` dtype int；阈值 τ=0.5 作用在 $`f(x)`$ 上（等价于 $`g(x) \ge 0.5\hat c`$） |
| `_decision_function(X)` | 返回 $`f(x)=g(x)/\hat c`$，shape `(n_samples,)`；概率校正模式下由 $`g`$ 计算，加权重训模式下由最终模型输出 |
| `predict_proba(X)` | 返回 `np.column_stack([1 - f_hat, f_hat])`，shape `(n_samples, 2)`；col0 = $`P(y=0\mid x)`$，col1 = $`P(y=1\mid x)`$。$`f(x)/\hat c`$ 可能超过 1（$`c<1`$ 时），属算法设计预期行为 |
| `predict_label_proba(X)` | 返回 $`g(x)=p(s=1\mid x)`$，shape `(n_samples,)`；仅概率校正模式下可用，加权重训返回 `None` |
| `score_samples(X)` | 复用 `_decision_function`，无需覆盖 |
| `get_pu_metadata()` | 返回 `family/assumption/scenario/propensity_/class_prior_` 等元数据 |
| `get_params()` / `set_params()` | 由 sklearn `BaseEstimator` 提供，兼容 Pipeline / GridSearchCV |

### 6.3 拟合属性（`fit` 后设置）

| 属性 | 类型 | 含义 |
|---|---|---|
| `self.propensity_` | `float` | 估计的标注倾向 $`\hat c = p(s=1\mid y=1)`$ |
| `self.class_prior_` | `float` 或 `None` | 类别先验 $`\widehat{p(y=1)}`$；仅 `scenario=SINGLE_TRAINING_SET` 时计算；case-control 不可识别 |
| `self._is_fitted` | `bool` | 基类管理，`fit()` 结束时设为 `True` |
| `self._X_shape_` | `tuple[int, int]` | 训练数据形状 |
| `self.classes_` | `np.ndarray` | `np.array([0, 1])` |

### 6.4 额外方法的处理

| 原方法卡提议 | 实现处理 |
|---|---|
| `estimate_propensity()` | → 私有 `_estimate_propensity()`，在 `fit()` 内部调用，结果存 `self.propensity_` |
| `fit_weighted()` | → 不作为公共方法；由 `mode="weighted_retraining"` + `fit()` 内部触发 |
| `estimate_class_prior()` | → 结果存 `self.class_prior_`；仅 `scenario=SINGLE_TRAINING_SET` 时可用，case-control 时标记为不可识别 |

---

## 7. Toolbox 集成映射

### 文件与注册

| 项目 | 内容 |
|---|---|
| 目标文件 | `pu_toolbox/estimators/classic/elkan_noto.py` |
| 类名 | `ElkanNotoClassifier(BasePUClassifier)` |
| 注册名称 | `"elkan_noto"`（`implementation_status=NATIVE`） |
| 别名 | `["en", "elkan-noto", "elkan_noto_calibration"]` |
| 导出 | `estimators/classic/__init__.py` 添加 `from .elkan_noto import ElkanNotoClassifier` |

### 注册表更新

> ✅ 已完成。`elkan_noto` 条目 `implementation_status=NATIVE`，已绑定 `ElkanNotoClassifier`。

### 类级元数据

```python
family = AlgorithmFamily.CLASSIC_CALIBRATION
assumption = (Assumption.SCAR,)
scenario = (Scenario.SINGLE_TRAINING_SET,)
requires_class_prior = False
implementation_status = ImplementationStatus.NATIVE
source_status = SourceStatus.THIRD_PARTY_ONLY
backend = Backend.SKLEARN
maturity = Maturity.STABLE
```

### 参考实现

| 优先级 | 代码库 | 用途 |
|---|---|---|
| **主参考** | [`pulearn/pulearn`](https://github.com/pulearn/pulearn) | 生产级 sklearn 兼容实现（`ElkanotoPuClassifier` + `WeightedElkanotoPuClassifier`），BSD-3-Clause。用作算法验证基准和 API 设计参考 |
| 历史参考 | [`aldro61/pu-learning`](https://github.com/aldro61/pu-learning) | 原始实现（Python 2，2013 年废弃），BSD-3-Clause。仅历史价值 |

---

## 8. 测试参考

### 8.1 合成 SCAR 校正测试

1. 生成有完整 $`y`$ 的二分类数据；仅对 $`y=1`$ 按固定概率 $`c`$ 随机保留为 $`s=1`$。
2. 训练 $`g`$，以独立标注正例估计 $`\hat c`$。
3. 验证 $`\hat c`$ 接近生成的 $`c`$，且 $`g(x)/\hat c`$ 相比 $`g(x)`$ 更接近真实 $`p(y=1\mid x)`$（例如 Brier score / log loss）。

### 8.2 排序不变性测试

在 $`\hat c>0`$ 下，验证 $`g(x)`$ 与 $`g(x)/\hat c`$ 的排序完全一致；对应 ROC-AUC 相同。

### 8.3 加权数据构造测试

对每个无标签样本验证会产生两条记录，权重分别为 $`w(x)`$ 与 $`1-w(x)`$，且权重和为 1；标注正例权重为 1。

### 8.4 先验可识别性开关测试

- `single_training_set`：验证 $`[n+\sum_Uw(x)]/m`$ 可返回。
- `case_control`：调用类别先验估计应失败并说明“本文条件下不可识别”。

### 8.5 校准与异常输入测试

覆盖：无验证标注正例、$`\hat c\le0`$、预测值接近 1、未校准基础模型、以及明显违反 SCAR 的模拟数据；后者至少应产生诊断警告，而不是承诺无偏结果。

---

## 9. 论文实验参考

| 项目 | 论文设置 |
|---|---|
| 合成示例 | 500 个正例、1000 个负例，二维 Gaussian；20% 正例随机标注 |
| 真实数据 | TCDB 的 2453 个已知正例；SwissProt 中 4906 个无标签记录 |
| 基础模型 | 线性核 SVM；概率输出用 Platt scaling 校准 |
| 评估 | 10-fold CV；accuracy、F1、ROC-AUC、固定假阳性率下的召回 |
| 结论 | 概率校正与加权重训在该任务上优于论文比较的 biased SVM；不应泛化为所有数据集的性能保证 |

---

## 10. 源码状态

| 字段 | 内容 |
|---|---|
| Source status | `third_party_only`（无官方代码，有第三方实现） |
| 主参考实现 | [`pulearn/pulearn`](https://github.com/pulearn/pulearn) — sklearn 兼容、活跃维护（v0.2.0, 2026-03）、BSD-3-Clause、含概率校正与加权重训两种方法 |
| 历史参考 | [`aldro61/pu-learning`](https://github.com/aldro61/pu-learning) — Python 2、2013 年废弃、仅有概率校正、不兼容 sklearn、不可直接复用 |
| 实现策略 | **Native clean-room**（`docs/resources_optimized.md` 已裁决）：以论文 §2–3 公式为权威依据，以 pulearn 为算法验证参考，API 严格对齐本项目 `BasePUClassifier` 契约 |
| License | BSD-3-Clause（第三方代码）；本项目实现为独立 Native 代码 |
