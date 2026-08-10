# SCAR/SAR 机制与识别边界

## 1. 两种标记机制

设真实类别为 $`Y`$，是否观察到正标签为 $`S`$。两种常用标记机制：

```math
\text{SCAR}: P(S=1\mid Y=1,X)=c,
\qquad
\text{SAR}: P(S=1\mid Y=1,X)=c(X).
```

- **SCAR**（Selected Completely At Random）：正类被标记的概率是常数 $`c`$，与特征无关。
- **SAR**（Selected At Random）：标记倾向 $`c(x)`$ 随特征变化，即实例相关的选择偏差（selection bias）。

## 2. 数据生成机制

模拟器支持三种机制（`make_sar_propensity` / `make_sar_dataset` 的 `mechanism` 参数）：

| 机制 | 定义 | 用途 |
|---|---|---|
| `scar` | 正类 propensity 为常数 | SCAR 控制组 |
| `linear` | 标准化特征投影经过 sigmoid | 单调 instance-dependent bias |
| `nonlinear` | 在线性投影上加入中心化二次项 | 非线性 selection bias |

`feature_weights` 控制投影方向，默认所有特征等权并进行 L2 归一化。`strength=0` 时，线性和非线性机制退化为常数 propensity，可作为实现正确性的边界检查。生成方法见 [howto/sar_simulation.md](../howto/sar_simulation.md)。

## 3. 识别边界：为什么 SCAR/SAR 通常无法从数据中识别

仅观察到特征 $`X`$ 和 PU 标记 $`S`$ 时，通常无法识别 SCAR 与 SAR。未标记集合同时包含真实正类和真实负类，因此即使分类器能很好地区分"已标正例"和"未标记样本"，也可能只是因为 $`X`$ 能预测真实类别 $`Y`$——不能据此断言标记策略依赖 $`X`$。

工具明确区分两种证据：

| `evidence` | 使用的数据 | `is_identifying` | 可以怎样解释 |
|---|---|---:|---|
| `observed_mixture` | 全部已标正例 vs 未标记混合样本 | `False` | 仅作为筛查信号，不能证明 SAR |
| `audited_positives` | 已知真实正例内部的已标 vs 未标 | `True` | 可直接检查 $`S`$ 是否依赖正例特征 |
| `not_evaluated` | 特征含非有限值（数据质量不满足要求；样本量不足时为 `inconclusive`，不改变 evidence） | `False` | 先修复报告中的问题 |

即便审计模式返回 `plausible`，含义也只是"没有检测到强特征依赖"，而不是证明 SCAR 成立。有限样本、模型表达能力和未观测特征仍可能造成漏检。

## 4. 假设选择的影响

- SCAR 假设下的方法（Elkan-Noto、uPU、nnPU 等）在标记确实依赖特征时结果可能有偏。
- SAR-aware 方法（PUSB、LBE、DGPU）显式建模标记倾向，SCAR 数据下仍可用（退化为常数 propensity 的特例）。
- 画像报告给出 `sar_signal` 警告时，应优先评估 PUSB/LBE 并做敏感性分析。

画像工具如何使用这些证据：见 [howto/data_profiling.md](../howto/data_profiling.md)；假设错配的风险与选型建议：见 [method_selection.md](method_selection.md)。
