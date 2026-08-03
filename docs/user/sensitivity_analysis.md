# PU 类先验与标记倾向敏感性分析

`analyze_pu_sensitivity` 用于回答一个有限但重要的问题：当同一组模型输出面对不同的
类先验或平均标记倾向假设时，依赖这些假设的 PU 指标会如何变化，以及这些假设是否与
观测标记率相容。它不会训练模型，也不会声称从 PU 数据中同时识别类先验和 propensity。

## 1. 统计边界

令 $`S=1`$ 表示样本被标记为正例，$`Y=1`$ 表示真实正例。只允许正例被标记时：

```math
q = P(S=1) = P(Y=1)P(S=1\mid Y=1) = \pi\bar c.
```

其中 $`q`$ 是可观测的 labeled-positive rate，$`\pi`$ 是类先验；SCAR 下
$`\bar c=c`$，SAR 下 $`\bar c=E[c(X)\mid Y=1]`$。因此：

- 给定 $`\pi`$，可计算 implied mean propensity $`\bar c=q/\pi`$；
- 给定 $`\bar c`$，可计算 implied class prior $`\pi=q/\bar c`$；
- 仅凭 $`q`$ 不能分别识别 $`\pi`$ 和 $`\bar c`$；
- $`\pi<q`$ 或 $`\bar c<q`$ 会推出大于 1 的概率，与观测数据矛盾。

本接口把最后一类点保留在结果中并设置 `is_consistent=False`，方便审计配置错误。

## 2. 基本使用

```python
from pu_toolbox.diagnostics import analyze_pu_sensitivity

analysis = analyze_pu_sensitivity(
    y_valid,
    fitted_classifier.predict(X_valid),
    scores=fitted_classifier.decision_function(X_valid),
    class_priors=[0.20, 0.25, 0.30, 0.35, 0.40],
    label_propensities=[0.20, 0.30, 0.40, 0.50, 0.80],
)

print(analysis.to_frame(axis="class_prior"))
print(analysis.metric_ranges)
```

`y_pu` 必须使用可规范化为 `{1, 0}` 的 PU 标签，`y_pred` 必须为 `{1, 0}`。
两者都必须同时包含 labeled-positive 和 unlabeled 组。至少提供一个网格：

| 参数 | 合法范围 | 含义 |
|---|---|---|
| `class_priors` | 每项在 `(0, 1)` | 假定的 $`\pi`$ |
| `label_propensities` | 每项在 `(0, 1]` | 假定的正例平均标记概率 $`\bar c`$ |
| `scores` | 有限一维数组，可选 | 用 0 为阈值计算 PU zero-one risk |

未提供 `scores` 时，risk 使用 `y_pred` 映射出的 `+1/-1` 分数。precision 始终使用
`y_pred`，所以显式分数只影响 risk，不影响 estimated precision。

## 3. 输出含义

每个 `SensitivityPoint` 包含：

| 字段 | 含义 |
|---|---|
| `axis`, `value` | 当前扫动的参数及其假定值 |
| `class_prior` | 假定或由 $`q/\bar c`$ 推出的类先验 |
| `label_propensity` | 假定或由 $`q/\pi`$ 推出的平均 propensity |
| `is_consistent` | 是否满足概率边界和观测恒等式 |
| `consistency_reason` | 不一致时的直接原因 |
| `pu_estimated_precision` | $`\min(\pi\widehat{Recall}_P/\widehat P(\hat Y=1),1)`$ |
| `pu_zero_one_risk` | $`2\pi\widehat{FNR}_P+\widehat{FPR}_U-\pi`$ |

`metric_ranges` 仅汇总 `is_consistent=True` 且指标可用的点，报告 minimum、maximum、
span 和样本点数。边界 $`\pi=1`$ 虽代数上可相容，但现有 PU 指标要求 `(0,1)`，因此
该点指标为 `None`。

## 4. 保存与实验记录

```python
analysis.save("artifacts/sensitivity.json")
analysis.save("artifacts/sensitivity.md")
analysis.save("artifacts/sensitivity.csv")
```

JSON 使用严格模式，不写入 `NaN` 或 `Infinity`。CSV 适合绘图；JSON 保留 feasible
region、指标区间与 provenance；Markdown 适合附在实验报告中。

## 5. 正确解读

该接口是 **fixed-output assumption sensitivity**：模型预测保持不变，只审计评价指标和
`q=\pi\bar c` 的代数一致性。它适合训练后报告、类先验估计器对比和配置排错，但不能
支持“训练算法对先验误设稳健”的结论。

若要做后者，必须固定 split 和 seed 集合，在每个 $`\pi`$ 或 propensity 假设下重新拟合
模型，保存逐 seed 结果并报告均值、标准差和失败率。那属于 paper-like benchmark，不能
与本接口生成的固定输出曲线混写。

完整可运行示例：

```bash
python examples/minimal/09_sensitivity_analysis.py
```
