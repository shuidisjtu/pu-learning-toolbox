# 选择 PU 方法

> 前置条件：先完成 [快速开始](../quickstart.md)。
> 概念：PU 问题与 π 的角色见 [pu_problem.md](pu_problem.md)，SCAR/SAR 见 [scar_sar.md](scar_sar.md)。

## 1. 推荐的决策路径：让推荐器选

本工具内置自动推荐器 `recommend_methods`——数据画像（规模/稀疏性/PU 比例）叠加
用户输入（scenario / assumption / π），对注册表全部方法做七维评分，返回 top-k 与
每个候选的理由和风险警告。**除非你想理解选型原理或对推荐结果有疑问，不需要手工查表。**

```python
from pu_toolbox import recommend_methods

result = recommend_methods(
    X, y_pu,
    class_prior=0.3,      # 已知 π 传入；None 会排除需要 π 的方法
    has_gpu=True,
    top_k=5,
)

for c in result.candidates:
    print(f"{c.rank}. {c.name} (score={c.score:.1f})")
    for r in c.reasons:
        print(f"    - {r}")
```

七维评分：assumption 匹配 + maturity + source 可信度 + 数据规模 + 训练成本 + GPU +
标记充足度（各维权重可经 `ScoringConfig` 调整）。`class_prior` 是**硬过滤**：为 None
时直接排除需要 π 的方法。完整签名与返回结构见 [API 参考](../reference/api.md)。

## 2. π 在 PU 中的角色（决策的轴心）

- **需要 π 的方法**：风险估计类（uPU、nnPU、PNU、LLSVM、Dist-PU）与部分深度方法
  （Self-PU、WConPU、DGPU）用 π 构造无偏风险或加权目标。π 错误的影响是实打实的——
  例如 uPU 把 π 低估一半，召回率可以崩到 0。
- **不需要 π 的方法**：Elkan-Noto（估计标记概率 c）、PUSB / LBE（SAR 假设下建模
  标记倾向）、InfoMax PU（自动估计 π）。
- **π 未知时**：先用类先验估计器（recpe / pen_l1 / km1 / km2，`PUPipeline` 会自动做）
  或选不依赖 π 的方法。

先验解析优先级与自动估计流程见 [howto/pipeline.md](../howto/pipeline.md)；π 假设错误的
影响范围与敏感性分析见 [howto/sensitivity_analysis.md](../howto/sensitivity_analysis.md)。

## 3. 按数据条件决策

从**你的数据**出发，而不是从算法清单出发：

| 数据条件 | 推荐 | 理由 |
|---|---|---|
| π 已知 | 风险估计类（uPU / nnPU / PNU / LLSVM / Dist-PU） | 用 π 构造无偏风险，理论保证最强 |
| π 未知 | 先估计 π（recpe / pen_l1 / km1 / km2），或选不依赖 π 的方法（Elkan-Noto / InfoMax PU） | 先验估计器是上游步骤，`PUPipeline` 自动串联 |
| 有部分负样本可用 | PNU | 组合 PN/PU/NU 三种风险，信息最全 |
| 怀疑标记有偏（SAR） | PUSB / LBE / DGPU | 显式建模 selection bias；SCAR 方法在此假设下有偏 |
| 小数据（< 1000 样本） | Elkan-Noto、非深度方法 | 深度方法需要大量数据；LLSVM 固定 epoch 训练成本占比高 |
| 大数据 + GPU | nnPU / Dist-PU / Self-PU / WConPU / InfoMax PU | 深度方法在数据充足时收益最大 |
| labeled positive 极少 | 简单 baseline 优先（Elkan-Noto / PUSB） | 深度方法在标记不足时不一定优于简单方法 |
| 需要审计证据 | 提供 `y_true` 走审计画像与 oracle 指标 | 可识别假设证据 + supervised 指标 |

推荐器会把这些条件编码进七维评分并给出逐条理由；上表用于理解推荐器为什么这样排，
以及手动复核其建议。

## 4. 算法族的设计思想

各算法族解决不同的问题，理解族的设计思想比记方法清单更有用：

- **Class Prior Estimation**（penL1 / KM1/KM2 / ReCPE）：上游估计 π，本身不做分类。
  penL1 凸优化、Kernel Mean 用核均值匹配、ReCPE 基于判别模型。
- **Risk Estimation**（uPU / nnPU / PNU / Dist-PU / LDCE / KLDCE / LLSVM）：用 π 构造
  无偏的风险估计。uPU 的负风险问题 → nnPU 非负修正；PNU 扩展到有部分负样本；
  Dist-PU 从标签分布视角建模。
- **Classic & Calibration**（Elkan-Noto）：经典校准路线，估计标记概率并校正输出，
  实现简单、SCAR 下的可靠基线。
- **Bias-Aware / SAR**（PUSB / LBE / DGPU）：显式建模标记倾向。PUSB 面向 selection
  bias、LBE 显式估计 labeling bias、DGPU 判别-生成联合建模（DGPU 同时见于 Deep PU
  条目，族归属以 registry `Fam.DEEP_PU` 为准）。
- **Deep PU**（Self-PU / InfoMax PU / WConPU / DGPU）：表征学习 + PU 风险修正。
  InfoMax PU 信息论表征 + 自动估计 π；WConPU 加权对比学习 + 难负样本挖掘；
  Self-PU 自训练 + meta reweight + 蒸馏（需要 clean validation）。

每篇论文的完整方法卡见 [method_cards/](../../research/method_cards/)（公式、复现状态、
实现边界）。

## 5. 风险提示

无论自动推荐还是手工选型，以下风险必须显式考虑：

1. SCAR 不成立时，Elkan-Noto、uPU、nnPU 等 SCAR 假设方法的结论可能有偏。
2. π 估计错误会显著影响风险估计方法——用敏感性分析界定影响范围。
3. labeled positive 过少时，深度方法不一定优于简单 baseline。
4. selection bias 明显时，优先尝试 PUSB / LBE / DGPU。
5. 推荐器对候选给出 warnings（实验性实现、需 π、SCAR 风险）——阅读理由与警告，
   不要只看分数。

## 6. 领域全景（不在本工具内）

TIcE、AlphaMax（类先验估计）、PU Bagging、Biased SVM、Weighted LR（分类器）是 PU
学习领域的经典方法，本工具**不提供实现**，此处列出供领域参照，未来版本可能纳入。

## 下一步

- 用推荐结果跑完整实验：[howto/pipeline.md](../howto/pipeline.md)
- 精确参数契约：[API 参考](../reference/api.md)
