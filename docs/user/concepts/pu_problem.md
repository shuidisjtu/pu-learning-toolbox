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
| $`\pi_{population} = P(Y=1)`$ | 目标总体的正类先验；下文 π 未加下标时指此值 |
| $`c = P(s=1 \mid y=1)`$ | SCAR 下正类被标记的常数概率 |
| $`e(x) = P(S=1 \mid Y=1, X=x)`$ | 个体标记倾向，旧文档也记为 c(x) |
| $`\bar c = E[e(X)\mid Y=1]`$ | 正类平均标记倾向 |
| $`q=P(S=1)`$ | OS 总体中的已标记比例 |
| $`\pi_U`$ | 原生未标记池中的正类比例 |

本项目标签规范：PU 标签为 $`\{1, 0\}`$（1 = 已标记正类，0 = 未标记）；PNU 扩展为 $`\{+1, -1, 0\}`$。

## 3. 观测恒等式：$`q = \pi\bar c`$

设 $`q = P(S=1)`$ 为可观测的 labeled-positive rate，则：

```math
q = P(Y=1)P(S=1\mid Y=1) = \pi\bar c,
```

其中 SCAR 下 $`\bar c = c`$，SAR 下 $`\bar c = E[e(X)\mid Y=1]`$（正类平均标记倾向）。
这里的 q、π 和标记事件来自同一 OS 总体；TS 拼接比例 r 的定义见[先验语义](class_prior_semantics.md)。由此：

- 给定 $`\pi`$，可计算 implied mean propensity $`\bar c = q/\pi`$；
- 给定 $`\bar c`$，可计算 implied class prior $`\pi = q/\bar c`$；
- $`q`$ 约束 $`\pi`$ 与 $`\bar c`$ 的乘积；分别估计两者需要附加信息；
- $`\pi < q`$ 或 $`\bar c < q`$ 会推出大于 1 的概率，与观测数据矛盾（工具在敏感性分析中标记 `is_consistent=False`）。

## 4. π 的角色

类先验是 PU 方法族的分水岭：

- **需要解析 π 的方法**：风险估计类（uPU、nnPU、PNU、LLSVM、Dist-PU）与深度方法（Self-PU、WConPU、DGPU）用 π 构造风险或加权目标；π 错误会显著影响结果（uPU 的 π 低估可导致召回率崩溃）。
- **不强制外部 π 的方法**：Elkan-Noto 估计标记概率 c；LBE 可用可选 π 初始化后验；InfoMax PU 可内部估计 π。当前 PUSB baseline 不消费先验；类先验估计器本身则负责估计。

`requires_class_prior=False` 表示调用者可省略外部先验。LDCE/KLDCE 从标记比例与
flip_probability 派生先验；InfoMax 可内部估计先验。逐方法事实见
[语义基线](../../dev/pu_research_semantics.md#4-历史-os_or_ts-的逐实现清单)。

π 的来源优先级与自动估计流程见 [howto/pipeline.md](../howto/pipeline.md) 的「类先验解析优先级」；π 假设错误的影响范围见 [howto/sensitivity_analysis.md](../howto/sensitivity_analysis.md)。

## 5. 与监督学习的差异

| 维度 | 监督学习 | PU 学习 |
|---|---|---|
| 负类标签 | 显式 | 不存在（未标记 = 混合） |
| 评估指标 | accuracy 等直接计算 | 依赖 π（`pu_zero_one_risk`）或需要 oracle 真值（AUC） |
| 标记机制 | 无关 | SCAR / SAR 假设影响方法选择 |

标记机制（SCAR/SAR）的定义与识别边界见 [scar_sar.md](scar_sar.md)；数据画像工具如何给出假设证据见 [howto/data_profiling.md](../howto/data_profiling.md)。

## 6. 用独立维度描述研究设置

| 维度 | 需要回答的问题 | 阅读入口 |
|---|---|---|
| 输入采样 | P/U 如何取得？U 来自总体还是移除 P 后的剩余池？ | [OS/TS 与风险视图](pu_sampling.md) |
| 标记机制与抽样政策 | SCAR/SAR？Bernoulli、定额还是排序选择？ | [SCAR/SAR](scar_sar.md) |
| 训练风险池 | 哪些样本进入正例项和混合项？ | [风险分解](pu_sampling.md#3-风险分解决定实际损失池) |
| 先验目标与来源 | 总体、U 还是拼接表？已知还是估计？ | [先验语义](class_prior_semantics.md) |
| 验证监督与准则 | PA/OA 使用哪些标签，怎样选择 checkpoint 和阈值？ | [PU 选模](pu_model_selection.md) |
| 表征与训练 | 二维特征、可训练 CNN 还是冻结 encoder？ | [表征](representations.md) |
| 评测与比较 | test 代表哪个总体，split/seed/预算是否可比？ | [实验操作](../howto/custom_experiments.md) |

例如“OS 输入 + 固定数 LBE-A 权重抽样 + TS-compatible 风险池 + 已知总体先验 + OA +
可训练 ResNet-18”描述了多个独立选择。SCAR/SAR 描述标记机制，OS/TS 描述输入采样，
历史 `os_or_ts` 描述训练风险视图。

已有真实 PU 数据可按“采样 → 标记假设 → 先验 → Pipeline”阅读；
合成/对照研究可按“采样 → 标记政策 → 风险视图 → 选模 → 表征 → DIY”阅读。
具体来源与公式条件见各概念页末尾；当前实现与后续修复边界见
[语义基线](../../dev/pu_research_semantics.md)。
