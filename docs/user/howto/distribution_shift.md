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

## CLI：配对比较后再决定

```bash
pu-toolbox shift-run \
  --source-data source/X.csv \
  --source-labels source/y_pu.csv \
  --target-data target/X.csv \
  --target-labels target/y_pu.csv \
  --target-class-prior 0.30 \
  --classifier elkan_noto \
  --cv 5 \
  --out-dir shift_comparison/
```

命令在同一目标集上配对运行未加权和加权两臂，写出
`shift_comparison.json/.md`、`shift_report.json`、`source_importance_weights.csv` 和
`target_predictions.csv`。若提供 `--target-true-labels`，默认以 oracle AUC 决策；否则可用
目标类先验依赖的 PU risk。两者都没有时仍给出对照数值，但结论固定为 `audit_only`。
覆盖不稳定且未设置 `--allow-unstable` 时不执行加权臂，结论为 `collect_target_data`。

## 连续窗口监控

```python
from pu_toolbox.diagnostics import PUShiftMonitor

monitor = PUShiftMonitor(X_reference, y_reference_pu, cv=5)
window, audit = monitor.update(
    X_august,
    y_window_pu=y_august_pu,
    window_id="2026-08",
    timestamp="2026-08-01T00:00:00+08:00",
)
print(window.alert_level, window.alert_codes)
monitor.save_history("shift_history.json")
```

告警覆盖高域漂移、不可适配、相邻窗口 AUC 突跳和标记率突跳。历史只保存窗口摘要，不保存
特征与标签；用同配置的新监控器调用 `load_history(...)` 后可继续追加。

## 区分类先验变化与标记机制变化

```python
from pu_toolbox.diagnostics import analyze_domain_assumptions

assumptions = analyze_domain_assumptions(
    X_source,
    y_source_pu,
    X_target,
    y_target_pu,
    source_class_prior=0.25,   # 省略则两个域分别用 pen_l1 估计
    target_class_prior=0.35,
)
print(assumptions.conclusion)
print(assumptions.differences)
```

报告利用 `P(S=1)=π·c̄` 分开类先验与平均正例标记倾向，并提供局部 3×3 先验敏感性表。
`c̄` 只是聚合量：即便它在两域相同，也不能证明 SCAR 或排除特征依赖 SAR。

## 不确定样本拒绝与主动复核

```python
from pu_toolbox.diagnostics import analyze_pu_uncertainty

review = analyze_pu_uncertainty(
    fitted_model,
    X_target,
    y_pu=y_target_pu,
    min_confidence=0.60,
    query_budget=20,
    query_strategy="shift_weighted",
    importance_weight=target_relevance_weight,
)
review.save("uncertainty_rows.csv")
```

`selective_prediction=-1` 表示交给人工/下游规则，不是负类。三种查询策略分别是纯不确定性、
漂移权重乘不确定性、以及在不确定候选中的聚类多样性；已有标记正例默认不进入查询池。
这里的概率边际是模型不确定性启发式量，不是经覆盖校准的置信区间。

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
特征、分类头和类别条件权重模型。`pu_toolbox.estimators.research.JointShiftPUClassifier`
提供一个明确标为 research 的 sklearn 近似：软类别条件域比、两域先验比、有界相对权重
以及交替 PU 更新。它不是论文共享神经特征和精确 PU 风险目标的复现，也未进入注册表/`auto`。
只有在公式金标准、退化情形和 paper-like benchmark 完成后才会升级声明；开发清单见
[分布漂移感知 PU 补充清单](../../dev/distribution_shift_aware_pu_checklist.md)。

## 下一步

- 精确对象、参数和报告字段：[API 参考](../reference/api.md)
- 类先验/标记倾向不确定性：[敏感性分析](sensitivity_analysis.md)
- 完整设计与理论边界：[开发者设计说明](../../dev/distribution_shift_aware_pu.md)
