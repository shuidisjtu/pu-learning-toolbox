# Method Card: Elkan–Noto PU Learning

> 参数契约（签名、参数表、返回结构）以 [API 参考](../../user/reference/api.md) 为权威；本文档只记论文研究内容、实现边界与复现状态。

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Learning Classifiers from Only Positive and Unlabeled Data |
| Authors | Charles Elkan, Keith Noto |
| Venue | KDD |
| Year | 2008 |
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

## 2. 问题设定与符号

单一训练集设定：样本先从总体 $`p(x,y,s)`$ 抽取，再只记录 $`(x,s)`$；标注正例构成集合 $`P`$，其余为无标签集合 $`U`$。目标是在 SCAR 下由 $`p(s=1\mid x)`$ 恢复真实正类概率。

| 论文符号 | 含义 |
|---|---|
| $`x`$ | 特征样本 |
| $`y\in\{0,1\}`$ | 真实正/负标签 |
| $`s\in\{0,1\}`$ | 是否被标注 |
| $`P`$ | 验证集或训练集中的 $`s=1`$ 样本 |
| $`U`$ | $`s=0`$ 的无标签样本 |
| $`g(x)`$ | 非传统分类器，$`p(s=1\mid x)`$ |
| $`f(x)`$ | 传统分类器，$`p(y=1\mid x)`$ |
| $`c`$ | 正例被标注的常数概率 $`p(s=1\mid y=1)`$ |
| $`w(x)`$ | 无标签样本为真实正例的后验概率 |
| $`m`$ | 单一训练集总样本数 |
| $`n`$ | 已标注正例数 |

---

## 3. 核心公式

### 3.1 中心引理：从 $`g`$ 到真实正类概率

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

### 3.2 概率校正与分类阈值

对测试样本：

```math
\hat f(x)=\frac{g(x)}{\hat c}.
```

若以真实正类概率阈值 $`\tau`$ 做二分类，则等价地在 $`g`$ 空间使用阈值 $`\hat c\tau`$；论文使用自然阈值 $`\tau=0.5`$，即 $`g(x)\ge 0.5\hat c`$。

### 3.3 无标签样本的软标签权重

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

### 3.4 类别先验（仅单一训练集）

```math
\widehat{p(y=1)}=\frac{1}{m}\left[n+\sum_{x\in U}w(x)\right].
```

等价的另一估计形式为：

```math
\widehat{p(y=1)}=\frac{n/m}{\hat c}.
```

---

## 4. 算法概要

### 4.1 概率校正（推荐用于概率输出）

1. 将标注正例设为 $`s=1`$、无标签样本设为 $`s=0`$，训练并校准 $`g(x)\approx p(s=1\mid x)`$。
2. 用独立验证集的标注正例预测值均值计算 $`\hat c`$。
3. 返回 $`\hat f(x)=g(x)/\hat c`$；排序场景可直接返回 $`g(x)`$。
4. 以目标真实概率阈值 $`\tau`$ 决策时，比较 $`g(x)`$ 与 $`\hat c\tau`$。

### 4.2 加权重训（论文实验使用）

1. 按 4.1 得到校准的 $`g`$ 和 $`\hat c`$。
2. 对每个 $`x\in U`$ 计算 $`w(x)`$。
3. 为每个 $`x\in U`$ 创建正、负两个带权副本；将 $`P`$ 保持为权重 1 的正例。
4. 用支持逐样本 `sample_weight` 的二分类学习器训练最终模型。

---

## 5. 论文边界

- 本文假设 **SCAR**：在真实正类内，被标注的概率与特征 $`x`$ 无关。若标注机制依赖 $`x`$（SAR），$`f(x)=g(x)/c`$ 和后续权重均会有系统偏差。
- 论文的类别先验估计只适用于**单一训练集**：样本先从总体 $`p(x,y,s)`$ 抽取，再只记录 $`(x,s)`$。若 $`P`$ 与 $`U`$ 是独立收集的 case-control 数据，$`p(y=1)`$ 不可由本文识别——§3.4 的两个估计式在 case-control 数据中不得用于报告可识别的类别先验。
- 仅做排序时无需估计 $`c`$：$`f(x)`$ 是 $`g(x)`$ 的正比例变换，二者排序相同。
- 概率校正依赖 $`g(x)\approx p(s=1\mid x)`$，不是普通分类分数；AUC 高不代表概率可用于本方法。
- 原论文未给出置信区间、SCAR 检验、SAR 修正或现代深度模型的校准 protocol；这些不能作为本文保证的一部分。

---

## 6. 实现注记

> 见 ADR-0014 #2（OOF 每折正例占比偏低导致 `c_hat` 系统性低估约 25%，折内 `sqrt(n_U/n_P_fold)` 加权校正；最终全量 `g` 拟合保持规范不加权）

**加权重训实现要点**（对应 §4.2 步骤 4）：

- 创建 `base_estimator` 的**新实例**（通过 `sklearn.base.clone`），在增强的加权数据集上训练。
- 新模型**替换** $`g`$ 成为最终模型；后续 `predict`/`predict_proba`/`decision_function` 均使用此新模型。
- 加权重训后 `predict_proba` 直接返回最终模型的概率输出（已经是 $`f(x)`$ 近似，无需再除以 $`\hat c`$）。
- 概率校正模式下 `decision_function` 由 $`g`$ 计算，加权重训模式下由最终模型输出。

**实现保护**（非论文规定）：

- 在计算 $`w(x)`$ 前将用于分母的 $`g(x)`$ 限制在 $`[\epsilon,1-\epsilon]`$；记录发生裁剪的数量。
- 当 $`\hat c\le\epsilon`$、$`\hat c>1`$、或大量 $`w(x)\notin[0,1]`$ 时抛出明确错误/警告，要求检查校准、数据划分和 SCAR。
- 验证集很小时，$`\hat c`$ 方差会很大；应报告验证集中标注正例数量。

---

## 7. 论文实验参考

| 项目 | 论文设置 |
|---|---|
| 合成示例 | 500 个正例、1000 个负例，二维 Gaussian；20% 正例随机标注 |
| 真实数据 | TCDB 的 2453 个已知正例；SwissProt 中 4906 个无标签记录 |
| 基础模型 | 线性核 SVM；概率输出用 Platt scaling 校准 |
| 评估 | 10-fold CV；accuracy、F1、ROC-AUC、固定假阳性率下的召回 |
| 结论 | 概率校正与加权重训在该任务上优于论文比较的 biased SVM；不应泛化为所有数据集的性能保证 |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `third_party_only`（无官方代码，有第三方实现） |
| 主参考实现 | [`pulearn/pulearn`](https://github.com/pulearn/pulearn) — sklearn 兼容、活跃维护（v0.2.0, 2026-03）、BSD-3-Clause、含概率校正与加权重训两种方法 |
| 历史参考 | [`aldro61/pu-learning`](https://github.com/aldro61/pu-learning) — Python 2、2013 年废弃、仅有概率校正、不兼容 sklearn、不可直接复用 |
| 实现策略 | Native clean-room：以论文 §2–3 公式为权威依据，以 pulearn 为算法验证参考 |
| License | BSD-3-Clause（第三方代码）；本项目实现为独立 Native 代码 |
