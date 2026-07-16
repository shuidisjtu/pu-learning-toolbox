# Method Card: Non-Negative PU Learning (nnPU)

## 1. 待办与注意

### 1.1 待办

- 实现并严格区分两个量：
  1. **风险报告值**：论文公式 (6) 的非负风险估计量；
  2. **训练优化量**：论文 Algorithm 1 的分支梯度规则。  
  二者在负类风险项小于阈值时不相同，不得用一次 `max(0, ·)` 反向传播替代 Algorithm 1。
- `class_prior`（$`\pi_p`$）为必填参数；本文不负责估计类别先验。
- 初始版本内置 sigmoid loss：
```math
  \ell_{\mathrm{sig}}(t,y)=\frac{1}{1+\exp(ty)}.
```
  使用数值稳定的 `sigmoid(-y * score)` 实现，不直接计算指数。
- 训练时分别构造 P、U mini-batch；每个更新步必须同时具备正例批次和无标签批次。
- 默认使用论文实验设置 `beta=0.0`、`gamma=1.0`；记录每轮进入校正分支的比例。
- 训练历史至少记录 `positive_risk`、`negative_risk`、`upu_risk`、`nnpu_risk`、`optimization_loss` 和 `correction_fraction`。
- 增加 uPU 负经验风险回归测试：柔性模型或人工 logits 下，uPU 风险可继续下降至负值，而 nnPU 风险不得随之变负。
- **[项目现状，来自阅读批注]** Toolbox 已实现 Convex PU 相关部分。nnPU 应优先复用已有的标签规范、类别先验校验、P/U 风险分解与 loss 抽象；具体复用点需结合仓库代码确认。

### 1.2 注意

- nnPU 解决的是柔性模型下 uPU 的**负经验风险与过拟合**问题；它不是概率校准方法。
- `decision_function` 输出的是判别分数 $`g(x)`$。论文不保证该分数等于 $`p(y=1\mid x)`$，因此不要把它直接包装成 `predict_proba`。
- nnPU 风险估计有正偏；其优势是偏差随样本量指数衰减，并在给定条件下保持一致性和最优收敛阶。
- 类别先验设错会直接改变风险分解。论文实验显示低估 $`\pi_p`$ 往往比轻度高估伤害更大，但这不是调高先验的理论依据。
- 论文形式化为 P、U 两样本问题：P 来自 $`p(x\mid Y=+1)`$，U 来自边缘分布 $`p(x)`$。P 集必须是可靠正例，本文不处理正例标签噪声。
- 论文实验允许 P 与 U 依赖；但每个集合内部的经验均值仍应分别计算。实现中不得把 P、U 拼接后以统一分母求平均。
- 仅当模型足够柔性、负类风险项出现负值时，nnPU 才与 uPU 明显不同；线性模型上二者可能近似或完全一致。
- 早停不得依据持续下降的 uPU 训练风险；优先使用独立 PU 验证集上的 nnPU 风险，或有真实标签时使用监督验证指标。

---

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Positive-Unlabeled Learning with Non-Negative Risk Estimator |
| Authors | Ryuichi Kiryo, Gang Niu, Marthinus C. du Plessis, Masashi Sugiyama |
| Venue | NIPS |
| Year | 2017 |
| Family | `pu_risk_estimation` |
| Setting | 两样本 PU 学习：可靠正例 P + 边缘分布无标签 U |
| Requires class prior | `True` |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| Produces calibrated probability | `False` |
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
- 理论结论还依赖有界性、Lipschitz、函数类复杂度等附加条件，见第 6 节。

---

## 3. 符号与记号

| 论文符号 | 含义 | 开发侧对应（建议） |
|---|---|---|
| $`g(x)`$ | 实值判别函数 | `model(X)` / raw score |
| $`\ell(t,y)`$ | 预测分数 $`t`$ 对标签 $`y`$ 的损失 | `loss_fn(score, target)` |
| $`\pi_p`$ | 正类先验 | `class_prior_` |
| $`\widehat R_p^+(g)`$ | P 样本按正类计算的经验风险 | `positive_risk` |
| $`\widehat R_p^-(g)`$ | P 样本按负类计算的经验风险 | `positive_as_negative_risk` |
| $`\widehat R_u^-(g)`$ | U 样本按负类计算的经验风险 | `unlabeled_as_negative_risk` |
| $`r(g)`$ | 估计的负类风险项 $`\widehat R_u^- - \pi_p\widehat R_p^-`$ | `negative_risk` |
| $`\widehat R_{\mathrm{pu}}(g)`$ | 无偏 PU 风险（uPU） | `upu_risk` |
| $`\widetilde R_{\mathrm{pu}}(g)`$ | 非负 PU 风险（nnPU） | `nnpu_risk` |
| $`\beta`$ | mini-batch 负风险容忍阈值 | `beta` |
| $`\gamma`$ | 校正分支步长折扣 | `gamma` |

统一标签建议：

- 外部 PU 标签：`1 = labeled positive`，`0 = unlabeled`；
- loss 内部目标：正类 `+1`，负类 `-1`；
- 预测：`g(x) >= 0` 判为正类。

---

## 4. 核心公式

### 4.1 三个部分风险

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

### 4.2 uPU 风险

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

### 4.3 nnPU 风险

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

### 4.4 训练分支：Algorithm 1

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

> **关键实现约束**：在 `r_i < -beta` 时，Algorithm 1 的梯度来自 $`-r_i`$，不是来自  
> $`\pi_p\widehat R_p^+ + \max(0,r_i)`$。直接对 `max` 反向传播会保留正类风险梯度且不给 $`r_i`$ 校正梯度，不等价于论文算法。

### 4.5 $`\beta`$ 的范围

论文给出：

```math
0\le\beta\le
\pi_p\sup_t\max_y\ell(t,y).
```

对 sigmoid loss，$`\sup\ell=1`$，因此：

```math
0\le\beta\le\pi_p.
```

对无界 loss，无法依据论文给出有限上界；初始实现建议只要求 `beta >= 0`，并默认 `beta=0`。

---

## 5. 算法概要

### 5.1 训练流程

1. 校验 `class_prior`、PU 标签及 P/U 样本数量。
2. 将训练数据拆分为 $`X_p`$ 与 $`X_u`$。
3. 分别打乱并生成 P、U mini-batch；每一步配对一个 P 批次和一个 U 批次。
4. 前向计算三个部分风险及 $`r_i`$。
5. 按 `r_i >= -beta` 选择正常分支或校正分支。
6. 用外部 SGD 类优化器更新参数。
7. 每轮在完整训练集或独立验证集上计算并记录 uPU/nnPU 风险。
8. 依据验证 nnPU 风险或监督验证指标早停。
9. 推理时返回原始分数，并以 0 为默认分类阈值。

### 5.2 mini-batch 组织

推荐使用两个独立 sampler/data loader：

- `positive_loader`：只产生 P 样本；
- `unlabeled_loader`：只产生 U 样本。

**[项目适配建议]** 当两个 loader 长度不同，可将一个 epoch 定义为遍历较长 loader，并循环较短 loader。无论采用何种策略，都应：

- 记录实际更新步数；
- 确保每个更新步同时含 P 和 U；
- 分别对 P、U 批次求均值；
- 不依赖两类样本数相同。

### 5.3 loss 选择

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

**[项目适配建议]** MVP 只内置 sigmoid loss。若项目已有统一 loss protocol，可允许自定义 differentiable loss，但必须声明：

- 是否有界及上界；
- 是否支持 `target ∈ {-1,+1}`；
- 是否按样本返回 loss；
- 是否可用于概率评估（通常不可）。

### 5.4 风险评估

独立 PU 验证集可直接计算公式 (6)：

- 使用 zero-one loss：用于分类风险评估；
- 使用训练 surrogate loss：用于与训练目标一致的早停。

该风险有正偏，但在论文条件下保持一致；不要把它解释成 accuracy 或概率校准误差。

---

## 6. 理论保证与边界

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

## 7. API 接口

> **[项目适配，待仓库核验]** 以下接口按参考资料卡中的 `BasePUClassifier` 契约设计；类名、参数位置和 backend 需与项目现有抽象对齐。

### 7.1 构造函数

```python
class NonNegativePUClassifier(BasePUClassifier):
    def __init__(
        self,
        model=None,
        loss="sigmoid",
        beta=0.0,
        gamma=1.0,
        optimizer=None,
        batch_size=256,
        max_epochs=200,
        patience=20,
        random_state=None,
    ):
```

参数约束：

| 参数 | 约束 |
|---|---|
| `model` | 输出 shape `(n_samples,)` 或 `(n_samples, 1)` 的实值分数 |
| `loss` | 初始版本仅 `"sigmoid"`；自定义 loss 需满足第 5.3 节协议 |
| `beta` | `>= 0`；sigmoid 下建议同时校验 `beta <= class_prior` |
| `gamma` | `[0,1]` |
| `optimizer` | SGD-like；论文使用 Adam/AdaGrad |
| `batch_size` | 应能为 P、U 分别产生非空批次 |
| `patience` | 早停耐心轮数（正整数）；仅传入 `validation_data` 时生效 |
| `max_epochs` | 正整数 |

### 7.2 方法映射

| 方法 | 约定 |
|---|---|
| `fit(X, y_pu, *, class_prior, sample_weight=None, validation_data=None)` | `class_prior` 必填；拆分 P/U 后执行 Algorithm 1；`validation_data` 为 `(X_val, y_pu_val)` 时启用早停 |
| `_decision_function(X)` | 返回原始 $`g(x)`$，shape `(n_samples,)` |
| `_predict(X)` | 返回 `(g(x) >= 0).astype(int)` |
| `predict_proba(X)` | 不实现或明确抛出 `NotImplementedError`；论文不提供后验概率 |
| `evaluate_pu_risk(X, y_pu, *, class_prior=None, loss=None, non_negative=True)` | 显式计算完整数据上的 uPU/nnPU 风险 |
| `get_training_history()` | 返回每轮风险与校正分支统计 |
| `score_samples(X)` | 复用 `_decision_function` |
| `get_pu_metadata()` | 返回 family、class prior、loss、beta、gamma 等元数据 |

不要将 `score()` 重载为 PU 风险；标准分类 `score(X, y_true)` 与 `evaluate_pu_risk()` 应保持语义分离。

### 7.3 拟合属性

| 属性 | 类型 | 含义 |
|---|---|---|
| `model_` | estimator/module | 已拟合模型 |
| `class_prior_` | `float` | 固定的 $`\pi_p`$ |
| `n_positive_` | `int` | P 样本数 |
| `n_unlabeled_` | `int` | U 样本数 |
| `history_` | mapping/list | 训练历史 |
| `classes_` | `np.ndarray` | `np.array([0, 1])` |
| `_is_fitted` | `bool` | 拟合状态 |

`history_` 最低字段：

```text
epoch
positive_risk
negative_risk
upu_risk
nnpu_risk
optimization_loss
correction_fraction
```

---

## 8. Toolbox 集成映射

> **[项目适配，待仓库核验]** 本节依据参考资料卡结构及阅读批注推定，不是论文规定。

### 文件与注册

| 项目 | 建议 |
|---|---|
| 类名 | `NonNegativePUClassifier` |
| 注册名称 | `"nnpu"` |
| 别名 | `["non_negative_pu", "non-negative-pu", "kiryo_nnpu"]` |
| Family | 使用项目已有的 risk-estimation / deep-PU 类别；不要新建重复枚举 |
| 依赖 | 优先复用 Convex PU 的风险分解、先验校验和标签转换 |
| Backend | 以项目当前可训练柔性模型的 backend 为准 |

### 模块拆分建议

- `NonNegativePURisk`：只负责部分风险、uPU/nnPU 报告值和 Algorithm 1 优化量；
- `NonNegativePUClassifier`：负责数据拆分、batch、优化器、预测和元数据；
- 公共测试夹具：与 Convex PU 共用 P/U 数据构造和手工风险计算。

### 集成边界

- 不在 nnPU 类中实现类别先验估计；
- 不自动将任意基础分类器的 `predict_proba` 当作 $`g(x)`$；
- 不把已有 Convex PU 替换为 nnPU：二者应作为独立方法注册；
- 可共享 uPU 风险实现，但 nnPU 必须额外实现校正分支与诊断历史。

---

## 9. 测试参考

### 9.1 部分风险与公式一致性

用固定 logits 手工计算：

```math
\widehat R_p^+,\quad
\widehat R_p^-,\quad
\widehat R_u^-,\quad
r,\quad
\widehat R_{\mathrm{pu}},\quad
\widetilde R_{\mathrm{pu}}.
```

逐项与实现比较，覆盖 `r > 0`、`r = 0`、`r < 0`。

### 9.2 负经验风险回归测试

构造使 P 样本被强烈预测为正、U 样本被强烈预测为负的 logits：

- 验证 $`r<0`$；
- 验证 uPU 风险可小于 0；
- 验证 nnPU 报告风险始终 $`\ge 0`$；
- 验证该现象被记录为校正分支，而不是数值错误。

### 9.3 分支梯度测试

- `r >= -beta`：梯度等于 uPU 风险梯度；
- `r < -beta`：梯度等于 $`-\gamma r`$ 的梯度；
- 校正分支不得包含 $`\pi_p\widehat R_p^+`$ 的梯度；
- `gamma=0` 时校正分支参数不更新（忽略独立正则化）。

### 9.4 `max` 误实现防护测试

构造 `r < 0`：

- 对 `pi_p * positive_risk + max(0, r)` 反向传播；
- 对 Algorithm 1 校正分支反向传播；
- 断言两者梯度不同，防止未来重构误将二者合并。

### 9.5 $`\beta`$ 边界测试

sigmoid loss 下覆盖：

- `beta < 0`：拒绝；
- `beta = 0`：标准 nnPU；
- `beta = class_prior`：所有可达 mini-batch 通常进入 uPU 分支；
- `beta > class_prior`：按严格模式拒绝或明确警告。

### 9.6 类别先验测试

覆盖：

- `class_prior is None`；
- `class_prior <= 0`；
- `class_prior >= 1`；
- 合法先验；
- 先验轻度低估/高估时训练仍可运行，并在元数据中保留实际使用值。

### 9.7 P/U batch 测试

覆盖：

- 无 P 样本；
- 无 U 样本；
- P 数量小于 batch size；
- P/U loader 长度差异大；
- 每个更新步均包含两个非空批次；
- `sample_weight` 在两组内独立归一化。

### 9.8 输出语义测试

- `decision_function` 返回有限实值；
- `predict` 仅按 0 阈值输出 `{0,1}`；
- `predict_proba` 不得伪造概率；
- `evaluate_pu_risk(non_negative=True)` 与公式 (6) 一致。

### 9.9 过拟合行为测试

使用小规模可重复的柔性模型：

- uPU 训练风险继续下降并出现负值；
- uPU 验证风险随后恶化；
- nnPU 的负类风险被拉回阈值附近；
- nnPU 验证风险不随 uPU 的负风险发散。

该测试可作为慢速/集成测试，不要求在单元测试中复现论文精度。

---

## 10. 论文实验参考

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

## 11. 实现验收清单

- [ ] `class_prior` 强制校验并持久化；
- [ ] P/U 风险分别求均值；
- [ ] uPU、nnPU 报告值与优化量分离；
- [ ] 校正分支实现为 $`-\gamma r`$；
- [ ] 默认 `beta=0`、`gamma=1`；
- [ ] sigmoid loss 数值稳定；
- [ ] 不伪造 `predict_proba`；
- [ ] 训练历史包含校正分支比例；
- [ ] 覆盖负经验风险回归测试；
- [ ] 与现有 Convex PU 共享公共组件但保持独立注册；
- [ ] 所有项目路径、枚举和 backend 适配项已结合仓库实际代码复核。
