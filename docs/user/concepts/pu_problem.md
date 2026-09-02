# PU 问题设定

## 1. 问题定义

PU（Positive-Unlabeled）学习只有两类监督信号：**已标记正样本**（labeled positive）与**未标记样本**（unlabeled）。未标记集合同时包含真实正类和真实负类——这是与监督学习（有完整正负标签）的本质差异，也是本工具所有方法要解决的困难所在。

> **数据从哪里来**：本工具接收的 `X`, `y_pu` 是用户整理好的 PU 数据——抽样（选择
> 哪些样本）与标签标记（哪些正例被标注）由研究者在工具箱之外完成，工具箱将
> 其视为已给输入。数据模拟器（`make_sar_*`/`make_scar_labels`/`make-demo-data`）
> 只用于合成研究，不替代真实采样流程。

只允许正类被标记（单边标记）：

```math
S=1 \Longrightarrow Y=1,
\qquad
P(S=1\mid Y=0,X)=0.
```

其中 $`S`$ 表示样本是否被标记为正类，$`Y`$ 表示真实类别。

## 2. 符号表

| 符号 | 含义 |
|---|---|
| $`x`$ | 样本特征 |
| $`y \in \{0,1\}`$ | 真实类别标签（训练时不可完全观测） |
| $`s \in \{0,1\}`$ | 是否被标记为正类 |
| $`\pi = P(y=1)`$ | 类先验 |
| $`c = P(s=1 \mid y=1)`$ | SCAR 下正类被标记的常数概率 |
| $`c(x) = P(s=1 \mid y=1, x)`$ | SAR 下实例相关的标记倾向 |

本项目标签规范：PU 标签为 $`\{1, 0\}`$（1 = 已标记正类，0 = 未标记）；PNU 扩展为 $`\{+1, -1, 0\}`$。

## 3. 观测恒等式：$`q = \pi\bar c`$

设 $`q = P(S=1)`$ 为可观测的 labeled-positive rate，则：

```math
q = P(Y=1)P(S=1\mid Y=1) = \pi\bar c,
```

其中 SCAR 下 $`\bar c = c`$，SAR 下 $`\bar c = E[c(X)\mid Y=1]`$（正类平均标记倾向）。由此：

- 给定 $`\pi`$，可计算 implied mean propensity $`\bar c = q/\pi`$；
- 给定 $`\bar c`$，可计算 implied class prior $`\pi = q/\bar c`$；
- **仅凭 $`q`$ 不能分别识别 $`\pi`$ 与 $`\bar c`$**；
- $`\pi < q`$ 或 $`\bar c < q`$ 会推出大于 1 的概率，与观测数据矛盾（工具在敏感性分析中标记 `is_consistent=False`）。

## 4. π 的角色

类先验是 PU 方法族的分水岭：

- **需要 π 的方法**：风险估计类（uPU、nnPU、PNU、LLSVM、Dist-PU）与深度方法（Self-PU、WConPU、DGPU）用 π 构造无偏风险或加权目标；π 错误会显著影响结果（uPU 的 π 低估可导致召回率崩溃）。
- **不需要 π 的方法**：Elkan-Noto（估计标记概率 c）、PUSB / LBE（SAR 假设下建模标记倾向）、InfoMax PU（自动估计 π）、类先验估计器本身（penL1 / KM1/KM2 / ReCPE）。

π 的来源优先级与自动估计流程见 [howto/pipeline.md](../howto/pipeline.md) 的「类先验解析优先级」；π 假设错误的影响范围见 [howto/sensitivity_analysis.md](../howto/sensitivity_analysis.md)。

## 5. 与监督学习的差异

| 维度 | 监督学习 | PU 学习 |
|---|---|---|
| 负类标签 | 显式 | 不存在（未标记 = 混合） |
| 评估指标 | accuracy 等直接计算 | 依赖 π（`pu_zero_one_risk`）或需要 oracle 真值（AUC） |
| 标记机制 | 无关 | SCAR / SAR 假设影响方法选择 |

标记机制（SCAR/SAR）的定义与识别边界见 [scar_sar.md](scar_sar.md)；数据画像工具如何给出假设证据见 [howto/data_profiling.md](../howto/data_profiling.md)。
