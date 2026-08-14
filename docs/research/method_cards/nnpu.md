# Method Card: Non-Negative PU Learning (nnPU)

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Positive-Unlabeled Learning with Non-Negative Risk Estimator |
| Authors | Ryuichi Kiryo, Gang Niu, Marthinus C. du Plessis, Masashi Sugiyama |
| Venue | NIPS |
| Year | 2017 |
| Setting | 两样本 PU 学习：可靠正例 P + 边缘分布无标签 U |
| Requires class prior | `True`（必填；本文不负责估计类别先验） |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| GPU required | `False`（深度模型训练建议使用） |

### Assumptions

令 $`X\in\mathbb{R}^d`$，$`Y\in\{+1,-1\}`$：

```math
p_p(x)=p(x\mid Y=+1),\qquad
p_n(x)=p(x\mid Y=-1),\qquad
p(x)=\pi_p p_p(x)+\pi_n p_n(x),
```

其中：

```math
\pi_p=p(Y=+1),\qquad \pi_n=1-\pi_p.
```

训练数据为：

```math
X_p=\{x_i^p\}_{i=1}^{n_p}\sim p_p(x),\qquad
X_u=\{x_i^u\}_{i=1}^{n_u}\sim p(x).
```

实现前提：

- $`0<\pi_p<1`$ 且在训练期间固定；
- P 样本均为真实正类；
- U 样本是正负类混合，不要求提供真实标签；
- 理论结论还依赖有界性、Lipschitz、函数类复杂度等附加条件，见第 5 节。

---

## 2. 问题设定与符号

nnPU 解决柔性模型下 uPU 的负经验风险问题：当模型过于灵活时，无偏经验风险估计 $`r(g)`$ 可能显著为负，使训练目标无下界并诱发过拟合。论文在 uPU 风险基础上对负类风险项施加非负约束（训练时用分支梯度规则），保留无偏性优势的同时抑制过拟合。

| 论文符号 | 含义 |
|---|---|
| $`g(x)`$ | 实值判别函数 |
| $`\ell(t,y)`$ | 预测分数 $`t`$ 对标签 $`y`$ 的损失 |
| $`\pi_p`$ | 正类先验 |
| $`\widehat R_p^+(g)`$ | P 样本按正类计算的经验风险 |
| $`\widehat R_p^-(g)`$ | P 样本按负类计算的经验风险 |
| $`\widehat R_u^-(g)`$ | U 样本按负类计算的经验风险 |
| $`r(g)`$ | 估计的负类风险项 $`\widehat R_u^- - \pi_p\widehat R_p^-`$ |
| $`\widehat R_{\mathrm{pu}}(g)`$ | 无偏 PU 风险（uPU） |
| $`\widetilde R_{\mathrm{pu}}(g)`$ | 非负 PU 风险（nnPU） |
| $`\beta`$ | mini-batch 负风险容忍阈值 |
| $`\gamma`$ | 校正分支步长折扣 |

---

## 3. 核心公式

### 3.1 三个部分风险

```math
\widehat R_p^+(g)
=
\frac{1}{n_p}\sum_{i=1}^{n_p}\ell(g(x_i^p),+1),
```

```math
\widehat R_p^-(g)
=
\frac{1}{n_p}\sum_{i=1}^{n_p}\ell(g(x_i^p),-1),
```

```math
\widehat R_u^-(g)
=
\frac{1}{n_u}\sum_{i=1}^{n_u}\ell(g(x_i^u),-1).
```

如支持 `sample_weight`，三个经验均值必须在 P、U 组内分别归一化：

```math
\widehat R
=
\frac{\sum_i w_i\ell_i}{\sum_i w_i},
```

不得除以拼接后的总样本权重。

### 3.2 uPU 风险

由：

```math
\pi_n R_n^-(g)=R_u^-(g)-\pi_pR_p^-(g),
```

得到无偏经验风险：

```math
\boxed{
\widehat R_{\mathrm{pu}}(g)
=
\pi_p\widehat R_p^+(g)
-\pi_p\widehat R_p^-(g)
+\widehat R_u^-(g)
}
```

定义负类风险项：

```math
r(g)=\widehat R_u^-(g)-\pi_p\widehat R_p^-(g),
```

则：

```math
\widehat R_{\mathrm{pu}}(g)=\pi_p\widehat R_p^+(g)+r(g).
```

当模型过于柔性时，$`r(g)`$ 可能显著小于 0，使总经验风险无下界并诱发过拟合。

### 3.3 nnPU 风险

论文公式 (6)：

```math
\boxed{
\widetilde R_{\mathrm{pu}}(g)
=
\pi_p\widehat R_p^+(g)
+
\max\left\{
0,\,
\widehat R_u^-(g)-\pi_p\widehat R_p^-(g)
\right\}
}
```

即：

```math
\widetilde R_{\mathrm{pu}}(g)
=
\pi_p\widehat R_p^+(g)+\max\{0,r(g)\}.
```

该值用于：

- PU 验证集上的风险评估；
- 训练过程中的风险监控；
- 早停或模型选择。

### 3.4 训练分支：Algorithm 1

对每个成对 mini-batch 计算：

```math
r_i
=
\widehat R_u^-(g;X_u^i)
-
\pi_p\widehat R_p^-(g;X_p^i).
```

优化量为：

```math
L_{\mathrm{opt}}=
\begin{cases}
\pi_p\widehat R_p^+(g;X_p^i)+r_i,
& r_i\ge -\beta,\\[4pt]
-\gamma r_i,
& r_i<-\beta.
\end{cases}
```

对应行为：

- `r_i >= -beta`：按 uPU 风险正常下降；
- `r_i < -beta`：停止优化正类风险，反向推动 $`r_i`$ 增大，避免负类风险继续向负方向发散；
- `beta=0`：无负风险容忍，论文默认 nnPU；
- `gamma=1`：完整校正步长；
- `gamma=0`：校正分支不更新（除非优化器另含正则项）。

> 在 `r_i < -beta` 时，Algorithm 1 的梯度来自 $`-r_i`$，不是来自 $`\pi_p\widehat R_p^+ + \max(0,r_i)`$；直接对 `max` 反向传播不等价于论文算法——见 ADR-0014 #9。

### 3.5 $`\beta`$ 的范围

论文给出：

```math
0\le\beta\le
\pi_p\sup_t\max_y\ell(t,y).
```

对 sigmoid loss，$`\sup\ell=1`$，因此：

```math
0\le\beta\le\pi_p.
```

对无界 loss，无法依据论文给出有限上界（实现建议见第 6 节）。

---

## 4. 算法概要

### 4.1 训练流程

1. 校验 `class_prior`、PU 标签及 P/U 样本数量；拆分为 $`X_p`$ 与 $`X_u`$。
2. 分别打乱并生成 P、U mini-batch；每一步配对一个 P 批次和一个 U 批次。
3. 前向计算三个部分风险及 $`r_i`$；按 `r_i >= -beta` 选择正常分支或校正分支。
4. 用外部 SGD 类优化器更新参数；每轮在完整训练集或独立验证集上计算并记录 uPU/nnPU 风险。
5. 依据验证 nnPU 风险或监督验证指标早停；推理时返回原始分数，并以 0 为默认分类阈值。

### 4.2 loss 选择

论文推荐 sigmoid loss：

```math
\ell_{\mathrm{sig}}(t,y)=\sigma(-ty).
```

原因：

- 有界；
- Lipschitz；
- 满足对称条件；
- 梯度在有限输入处非零；
- 可用常规梯度优化器训练柔性模型。

### 4.3 风险评估

独立 PU 验证集可直接计算公式 (6)：

- 使用 zero-one loss：用于分类风险评估；
- 使用训练 surrogate loss：用于与训练目标一致的早停。

该风险有正偏，但在论文条件下保持一致；不要把它解释成 accuracy 或概率校准误差。

---

## 5. 论文边界

- nnPU 解决的是柔性模型下 uPU 的**负经验风险与过拟合**问题；它不是概率校准方法。
- `decision_function` 输出的是判别分数 $`g(x)`$。论文不保证该分数等于 $`p(y=1\mid x)`$，因此不要把它直接包装成 `predict_proba`。
- nnPU 风险估计有正偏；其优势是偏差随样本量指数衰减，并在给定条件下保持一致性和最优收敛阶。
- 类别先验设错会直接改变风险分解。论文实验显示低估 $`\pi_p`$ 往往比轻度高估伤害更大，但这不是调高先验的理论依据。
- 论文形式化为 P、U 两样本问题：P 来自 $`p(x\mid Y=+1)`$，U 来自边缘分布 $`p(x)`$。P 集必须是可靠正例，本文不处理正例标签噪声。
- 论文实验允许 P 与 U 依赖；但每个集合内部的经验均值仍应分别计算，不得把 P、U 拼接后以统一分母求平均。
- 仅当模型足够柔性、负类风险项出现负值时，nnPU 才与 uPU 明显不同；线性模型上二者可能近似或完全一致。
- 早停不得依据持续下降的 uPU 训练风险；优先使用独立 PU 验证集上的 nnPU 风险，或有真实标签时使用监督验证指标。

### 理论保证

| 结论 | 核心内容 | 主要附加条件 |
|---|---|---|
| 偏差 | nnPU 风险一般高于 uPU 风险，因而有正偏 | 固定 $`g`$ |
| 偏差衰减 | 偏差随 $`n_p,n_u`$ 增长指数衰减 | loss/模型输出有界，$`R_n^-(g)\ge\alpha>0`$ |
| 一致性 | 固定 $`g`$ 时，$`\widetilde R_{\mathrm{pu}}(g)\to R(g)`$ | 同上 |
| 收敛阶 | $`O_p(\pi_p/\sqrt{n_p}+1/\sqrt{n_u})`$ | 同上；该阶对固定 $`g`$ 最优 |
| MSE 改善 | 在给定条件下，nnPU 的 MSE 小于 uPU | 对称 loss、$`n_u\gg n_p`$、负风险事件概率非零等 |
| 经验风险最小化一致性 | nnPU 学到的模型趋近函数类中的真实风险最优模型 | loss Lipschitz、函数类闭合于取负、复杂度受控等 |
| 与 uPU 同阶 | nnPU 的估计误差界与 uPU 同阶 | 常数项不同，不代表 nnPU 内在更差 |

以下不属于论文保证：

- 类别先验估计；
- 概率校准；
- 多分类；
- 正例标签噪声；
- 特征依赖的标注机制修正；
- 任意错误 $`\pi_p`$ 下的鲁棒性；
- 所有数据集上优于 PN/uPU。

---

## 6. 实现注记

> 见 ADR-0014 #9（Algorithm 1 校正分支梯度来自 $`-r_i`$，对 `max(0,·)` 反向传播不等价；含 `r_i < -beta` 分支行为上下文）

- **[状态]** 已实现为 native 分类器（torch backend，2026-07-16），接口按 `BasePUClassifier` 契约对齐。
- **`beta` 范围（实现建议）**：对无界 loss，论文无法给出有限上界；实现只要求 `beta >= 0`，并默认 `beta=0`。
- **mini-batch 组织**：当两个 loader 长度不同，可将一个 epoch 定义为遍历较长 loader，并循环较短 loader。无论采用何种策略，都应：记录实际更新步数；确保每个更新步同时含 P 和 U；分别对 P、U 批次求均值；不依赖两类样本数相同。
- **loss 扩展（MVP 边界）**：MVP 只内置 sigmoid loss。若项目已有统一 loss protocol，可允许自定义 differentiable loss，但必须声明：是否有界及上界、是否支持 `target ∈ {-1,+1}`、是否按样本返回 loss、是否可用于概率评估（通常不可）。
- **[项目现状]** Toolbox 已实现 Convex PU 相关部分；nnPU 应优先复用已有的标签规范、类别先验校验、P/U 风险分解与 loss 抽象（具体复用点需结合仓库代码确认）。

---

## 7. 论文实验参考

| 项目 | 论文设置 |
|---|---|
| 数据集 | MNIST、epsilon、20News、CIFAR-10 |
| 重点 | 使用 MLP/CNN 等柔性模型验证 uPU 负风险过拟合 |
| P 样本数 | 主要实验中 $`n_p=1000`$ |
| U 样本 | 使用全部训练数据构造 U |
| Loss | sigmoid loss |
| 正则化 | $`\ell_2`$ regularization |
| 优化器 | MNIST/epsilon/CIFAR-10 用 Adam；20News 用 AdaGrad |
| nnPU 参数 | $`\beta=0,\ \gamma=1`$ |
| 训练轮数 | 图中展示 200 epochs |
| 主要结论 | uPU 在四个数据集均出现过拟合；nnPU 修复该问题 |
| 与 PN 比较 | nnPU 在 MNIST、epsilon、CIFAR-10 优于有限 N 数据的 PN，在 20News 上相当 |
| 先验敏感性 | 测试 $`0.8\pi_p`$ 至 $`1.2\pi_p`$；低估通常伤害更大，轻度高估有时因额外偏差而更稳 |

图表对应信息：

- 第 4 页 Figure 1：线性模型上 nnPU 与 uPU 可相同；MLP 上 uPU 训练风险变负、测试风险上升，nnPU 保持稳定。
- 第 8 页 Figure 2：四个深度模型实验中，uPU 均表现出负风险过拟合，nnPU 显著抑制。
- 第 9 页 Figure 3：展示类别先验误设的敏感性；不得据此自动修正或放大用户提供的先验。

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact` |
| Official code | [`kiryor/nnPUlearning`](https://github.com/kiryor/nnPUlearning)（官方 PyTorch 实现，MIT） |
| 实现状态 | native 实现（backend TORCH，支持 GPU）；接口按 `BasePUClassifier` 契约 |
| 复现风险 | 论文实验为深度模型协议（200 epochs、Adam/AdaGrad、$`\ell_2`$ 正则）；先验必须外部提供，设错直接改变风险分解；训练与评估须区分报告值（公式 (6)）与优化量（Algorithm 1 分支规则），`max` 误实现会静默偏离论文算法 |
