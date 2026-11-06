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
| `scar` | $`P(S=1\mid Y=1,X)=c`$（直接赋予常数 `label_frequency`） | SCAR 控制组 |
| `linear` | $`P(S=1\mid Y=1,X)=\sigma(\gamma\,w^\top\tilde{x}+b)`$（标准化特征投影经过 sigmoid，$`\gamma`$ 为 `strength`、$`b`$ 为校准截距） | 单调 instance-dependent bias |
| `nonlinear` | 在线性投影上加入中心化二次项：$`\sigma(\gamma\,(w^\top\tilde{x}+\tfrac12\,q)+b)`$，$`q`$ 为 $`w^\top\tilde{x}`$ 在真实正例上的中心化二次项 | 非线性 selection bias |

`feature_weights` 控制投影方向，默认所有特征等权并进行 L2 归一化。`strength=0` 时，线性和非线性机制退化为常数 propensity，可作为实现正确性的边界检查。生成方法见 [howto/sar_simulation.md](../howto/sar_simulation.md)。

> **默认机制是 SAR**：`make_sar_dataset` 不传 `mechanism` 时解析为 `"linear"` 并发出 `UserWarning`。
> 类先验估计器与多数 PU 分类器假设 SCAR；用该函数生成的数据检验 SCAR 方法前，请显式
> 确认机制（SCAR 场景传 `mechanism="scar"`）。

## 3. 观测筛查与正例审计

观测模式比较已标记正例与未标记混合样本，提供分布差异筛查信号。
审计模式在真实正例内部比较已标与未标样本，检验标记选择对特征的依赖。

工具明确区分两种证据：

| `evidence` | 使用的数据 | `is_identifying` | 可以怎样解释 |
|---|---|---:|---|
| `observed_mixture` | 全部已标正例 vs 未标记混合样本 | `False` | 已标正例与 U 的分布差异筛查 |
| `audited_positives` | 已知真实正例内部的已标 vs 未标 | `True` | 可直接检查 $`S`$ 是否依赖正例特征 |
| `not_evaluated` | 特征含非有限值（数据质量不满足要求；样本量不足时为 `inconclusive`，不改变 evidence） | `False` | 先修复报告中的问题 |

`plausible` 表示当前检验未检测到强特征依赖；检出率受样本量、模型表达能力与可观测特征影响。

## 4. 假设选择的影响

- SCAR 假设下的方法（Elkan-Noto、uPU、nnPU 等）在标记确实依赖特征时结果可能有偏。
- SAR 方法依赖各自的附加结构。例如 LBE 联合拟合类别与标记倾向；DGPU 依赖条件生成组件。
  当前 `pusb` 是 observed-P-vs-U logistic baseline，`pusb_kernel` 是另一个核方法入口，
  方法身份和适用条件见[选型原理](method_selection.md)。
- 画像报告给出 `sar_signal` 警告时，先调查标签取得方式，再按方法假设选择对照并做敏感性分析。

画像工具如何使用这些证据：见 [howto/data_profiling.md](../howto/data_profiling.md)；假设错配的风险与选型建议：见 [method_selection.md](method_selection.md)。

## 5. 生成概率、抽样权重与实际计数

机制描述选择偏好，抽样政策决定怎样落实为标签。两者应分别记录。

| 当前入口 | 数值的含义 | 抽样政策与计数 |
|---|---|---|
| `make_sar_propensity` | 正例个体的 Bernoulli 标记概率；负类为 0 | 正例均值校准至 `label_frequency` |
| `make_sar_labels` | 使用上述概率 | 独立 Bernoulli，实际数量随机；默认允许补标 |
| `make_scar_labels` / `SCARGenerator` | c 是请求的正例标记比例 | 真正例池中均匀无放回，数量为 `min(n_pos, max(1, round(c*n_pos)))`；无正例返回全 U |
| `SARLBEAGenerator` / `SARLBEBGenerator` | 正例间的相对抽样权重 | 固定配额、加权无放回；权重决定相对选择偏好 |

`make_sar_labels(ensure_labeled=True)` 在 Bernoulli 未选中任何正例时补标 propensity
最高的真正例；相同最大值按原始顺序取第一例。返回 propensity 描述 Bernoulli 阶段的概率，
补标由 `ensure_labeled` 政策单独记录。研究纯 Bernoulli 机制时显式设为 `False`。
固定数的最小一例和取整也会使实际标记率偏离请求值。

当前 LBE 生成器在调用所在 clean 分区上用真实标签拟合辅助 LogisticRegression，
再仅在真阳性中选择 P。训练与 pu_val 分区分别生成，辅助评分不跨角色拟合。
这是模拟数据构造所用的真值；传给算法或 PA 的 PU 视图不包含这些真值。

## 6. 论文定义与具名实现

设辅助正类概率为 $`\hat p(x)`$。下表分列 R4 Appendix C.3 与已核对代码，
避免同名机制掩盖公式差异。

| 机制 | R4 论文 | 工具箱当前实现或参考库差异 |
|---|---|---|
| LBE-A | $`\hat p^{10}`$ 权重，定额无放回 | 工具箱默认 `k=10`；归一化后再混入 10% 均匀权重，与参考库一致 |
| LBE-B | $`(1-\hat p)^{10}`$ 权重，定额无放回 | 工具箱为 `clip(1.5+shrink_coef-p, 0)^k`，默认 `shrink_coef=1, k=10`，即参考库的 `(2.5-p)^10` |
| PUSB 选择预设 | $`\hat p^{20}`$ 排序取 top-N | 参考库采用接受/拒绝后截断，可少于请求数；工具箱尚未提供该具名生成预设 |

R4 Appendix C.2 配额用 `floor(c*n_pos)`；工具箱当前使用上节的 round / 最小一例政策。
论文公式、参考代码和工具箱实现分别记录公式与版本。
工具箱的 `pusb` 分类器与此表的 PUSB **数据选择机制**是不同入口。

模拟记录至少包括机制公式/参数、辅助评分拟合范围、请求/实际数量与标记率、
抽样政策、seed、取整/补标规则和来源版本。LBE 当前元数据已记录计数、参数及辅助真值来源，
调用者补充 `sampling_policy`；通用 Bernoulli 入口可返回 propensity。
操作入口见 [SAR 模拟教程](../howto/sar_simulation.md)，其余契约见
[API 参考](../reference/api.md)。

## 7. 来源

- Bekker & Davis (2020), Definition 1/2：
  [PU survey](https://doi.org/10.1007/s10994-020-05877-5)，SCAR/SAR 定义及假设。
- Chen et al. (2026), Appendix C.2/C.3：
  [PU-Bench](https://openreview.net/forum?id=tb8DabMbMq)，定额机制与论文公式。
- [固定版本参考代码](https://github.com/XiXiphus/PU-Bench/blob/2d95a19eefd72e66ff30128ec2f65e1d1d4cc077/data/data_utils.py)：
  LBE-A/B 平滑与权重见行 443–472，PUSB 接受/拒绝见行 86–122。
- 工具箱实现：[通用 propensity/Bernoulli](../../../pu_toolbox/preprocessing/selection_bias.py)、
  [固定数标记](../../../pu_toolbox/preprocessing/pu_labeling.py)、
  [实验生成器](../../../pu_toolbox/experiment/strategies.py)。
