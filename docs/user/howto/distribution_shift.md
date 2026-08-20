# 审计并处理 PU 分布漂移

> 前置条件：准备来自历史/训练环境的源域 `(X_source, y_source_pu)`，以及来自当前
> 部署环境的目标域 `X_target`。要启动加权适配，还需要一小批目标域 PU 标签
> `y_target_pu`。

## 先理解输出边界

本功能先判断源域和目标域是否可区分，再估计边际密度比
`p_target(x) / p_source(x)`。它提供的是**协变量漂移加权基线**：只有在
`p_source(y|x) = p_target(y|x)` 时，权重才有相应的适配解释。

- 域分类 OOF AUC 高：存在可观测差异，但不能单独判断是协变量、类先验还是概念漂移；
- 域分类 OOF AUC 低：当前审计器没有发现强差异，不等于证明两个分布相同；
- ESS 低、权重集中在边界或归一化前相对质量很低：源域覆盖不足，不应自动适配；
- 源域 CV 指标不是目标域性能。目标域真值只用于单独标记的 oracle 指标。

## CLI：先做漂移审计

源域与目标域 CSV 必须具有非数字表头和相同的特征列数：

```bash
pu-toolbox shift-audit \
  --source-data source/X.csv \
  --source-labels source/y_pu.csv \
  --target-data target/X.csv \
  --target-labels target/y_pu.csv \
  --out-dir shift_results/
```

目标域标签还未收集时可省略 `--target-labels`；审计仍会执行，但报告中的
`adaptation_ready` 固定为 `false`。

输出目录包含：

| 文件 | 内容 |
|---|---|
| `shift_report.json` | 严格 JSON：域 AUC、严重度、ESS、权重分位数、问题与实现边界 |
| `shift_report.md` | 人可读解释和下一步建议 |
| `source_importance_weights.csv` | `source_row` 与均值归一化的源域权重 |

常用参数：`--alpha 0.1` 控制相对密度比上界变换，`--cv 5` 控制域分类 OOF
折数，`--probability-clip 1e-6` 防止域概率的 odds 数值发散。

## Python：仅审计

```python
from pu_toolbox.diagnostics import analyze_pu_shift

shift = analyze_pu_shift(
    X_source,
    y_source_pu,
    X_target,
    y_target_pu=y_target_pu,  # 可省略；省略时不能进入适配
    alpha=0.1,
    cv=5,
    random_state=42,
)

print(shift.domain_auc, shift.severity)
print(shift.weight_summary["effective_sample_fraction"])
shift.save("shift_report.json")
shift.save("source_importance_weights.csv")
```

权重 CSV 与源域行严格一一对应。JSON 只存权重摘要和产物名，不内嵌整列权重，避免报告
随样本量无限膨胀。

## Python：显式启动协变量适配

```python
from pu_toolbox.workflows import ShiftAwarePUPipeline

workflow = ShiftAwarePUPipeline(
    classifier="elkan_noto",  # 默认；必须真正支持 sample_weight
    cv=5,
    shift_cv=5,
    random_state=42,
)
report = workflow.fit_evaluate(
    X_source,
    y_source_pu,
    X_target,
    y_target_pu=y_target_pu,
    y_true_target=y_true_target,        # 可选，仅 oracle 评估
    target_class_prior=0.30,            # 可选，启用目标域先验依赖指标
)

print(report.shift.to_markdown())
print(report.target_metrics["pu_auc_roc"].mean)
```

工作流会把源域权重按 CV 训练折切片，并传给最终全量重训。`upu`、`pnu` 等声明
`sample_weight_support="ignored"` 的分类器会立即失败；不会出现命令成功但权重未生效的
情况。当前支持权重的完整列表见 [API 参考](../reference/api.md#sample_weight-三档语义)。

若只需要“审计 + 未加权源域基线”，调用：

```python
report = workflow.fit_evaluate(
    X_source,
    y_source_pu,
    X_target,
    adapt=False,
)
```

覆盖检查失败时，默认抛出 `PipelineError`。`allow_unstable=True` 可用于受控研究实验，但
警告仍保留在报告中；这不是生产环境的推荐设置。

## 怎样行动

| 结果 | 建议 |
|---|---|
| AUC 低、ESS 高 | 保留监控；仍需目标域验证，不要写成“无漂移” |
| AUC 中等/高、ESS 高、覆盖正常 | 检查协变量漂移假设后比较未加权与加权模型 |
| ESS 低或相对质量很低 | 优先补采目标域/相似源域数据，不依赖极端权重 |
| 两域已标记正例率变化明显 | 分别审计类先验与标记倾向；已标记率不是类先验 π |
| 怀疑 `p(y|x)` 已变化 | 使用目标域 PU 数据做专门适配/重训；当前边际权重没有联合漂移保证 |

## 研究级联合漂移

AISTATS 2025 的联合漂移方法估计 `p_target(x,y) / p_source(x,y)`，并交替训练共享
特征、分类头和类别条件权重模型。当前版本没有把边际权重包装成该论文的复现。论文级扩展
只有在公式金标准、退化情形和 paper-like benchmark 完成后才会升级声明；开发清单见
[分布漂移感知 PU 补充清单](../../dev/distribution_shift_aware_pu_checklist.md)。

## 下一步

- 精确对象、参数和报告字段：[API 参考](../reference/api.md)
- 类先验/标记倾向不确定性：[敏感性分析](sensitivity_analysis.md)
- 完整设计与理论边界：[开发者设计说明](../../dev/distribution_shift_aware_pu.md)
