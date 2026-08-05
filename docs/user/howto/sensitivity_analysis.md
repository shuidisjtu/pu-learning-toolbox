# 类先验与标记倾向敏感性分析

> 前置条件：先完成 [快速开始](../quickstart.md) 与 [PUPipeline 端到端训练评估](pipeline.md)。
> 概念：观测恒等式 $`q=\pi\bar c`$ 与不可识别性见 [concepts/pu_problem.md](../concepts/pu_problem.md)。

`analyze_pu_sensitivity` 用于回答一个有限但重要的问题：当同一组模型输出面对不同的
类先验或平均标记倾向假设时，依赖这些假设的 PU 指标会如何变化，以及这些假设是否与
观测标记率相容。它不会训练模型，也不会声称从 PU 数据中同时识别类先验和 propensity。

## 1. 基本使用

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
两者都必须同时包含 labeled-positive 和 unlabeled 组。`class_priors` 与
`label_propensities` 至少提供一个；参数合法范围与每个 `SensitivityPoint` 的字段含义
见 [API 参考](../reference/api.md)。要点：

- 未提供 `scores` 时，risk 使用 `y_pred` 映射出的 `+1/-1` 分数；precision 始终使用
  `y_pred`，所以显式分数只影响 risk，不影响 estimated precision。
- `metric_ranges` 仅汇总 `is_consistent=True` 且指标可用的点。边界 $`\pi=1`$ 虽代数上
  可相容，但现有 PU 指标要求 `(0,1)`，因此该点指标为 `None`。

## 2. 保存与实验记录

```python
analysis.save("artifacts/sensitivity.json")
analysis.save("artifacts/sensitivity.md")
analysis.save("artifacts/sensitivity.csv")
```

JSON 使用严格模式，不写入 `NaN` 或 `Infinity`。CSV 适合绘图；JSON 保留 feasible
region、指标区间与 provenance；Markdown 适合附在实验报告中。

## 3. 正确解读

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

## 下一步

- 回到 [用户文档首页](../README.md) 选择下一个任务
- 精确参数契约：[API 参考](../reference/api.md)
