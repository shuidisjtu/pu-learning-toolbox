# PU 诊断报告

`build_diagnostic_report` 将数据画像、模型输出、PU 指标、可选监督指标和行动建议组合为一个可保存、可审计的报告。报告生成过程不会训练或修改 estimator，也不会自动修复数据。

## 1. API 与三种模式

### 1.1 纯数据报告

```python
from pu_toolbox.diagnostics import build_diagnostic_report

report = build_diagnostic_report(
    X,
    y_pu,
    class_prior=0.30,
    random_state=42,
)
```

这种模式只运行 `profile_pu_data`。所有依赖预测、分数或真实标签的指标仍保留在报告中，但其 `basis` 为 `unavailable`，同时给出原因。

### 1.2 已拟合 estimator

```python
classifier.fit(X_train, y_train, class_prior=0.30)

report = build_diagnostic_report(
    X_valid,
    y_valid,
    estimator=classifier,
    class_prior=0.30,
)
```

报告调用 `predict`，并优先读取 `decision_function`；若不存在，则读取二分类 `predict_proba(X)[:, 1]`。若 estimator 提供 `get_pu_metadata()`，其算法假设、实现状态和训练诊断也会进入 `model.metadata`。

生成器不会调用 `fit`。未拟合的 sklearn estimator 会得到明确的 `ValueError`，避免把底层 `NotFittedError` 混入批量报告流程。

### 1.3 显式输出

```python
report = build_diagnostic_report(
    X_valid,
    y_valid,
    y_pred=predictions,
    scores=decision_scores,
    class_prior=0.30,
)
```

该模式适合外部框架或无法 pickle 的模型。`estimator` 与 `y_pred/scores` 互斥，防止报告混用来自不同模型的结果。

## 2. 可选审计真值

仅当拥有人工复核、延迟结果或合成实验真值时传入 `y_true`：

```python
report = build_diagnostic_report(
    X_test,
    y_test_pu,
    estimator=classifier,
    y_true=y_test_true,
    class_prior=0.30,
)
```

`y_true` 有两个用途：

1. 在真实正例内部运行可识别的 selection-dependence 检查。
2. 计算 accuracy、F1 和 ROC AUC，并标为 `supervised_oracle`。

它不会传给 estimator，也不会参与模型训练。正式复现实验必须区分 validation 与 test：测试集真值只应用于最终评价，不能据此选择超参数、分类阈值或类先验。

## 3. 指标证据级别

每个 `DiagnosticMetric` 都包含 `value`、`basis`、`available` 和 `reason`：

| `basis` | 指标来源 | 是否需要额外假设 |
|---|---|---|
| `pu_observed` | PU 标签和模型预测 | 不需要真实标签 |
| `class_prior_dependent` | PU 标签、预测和类先验 | 数值随类先验改变 |
| `supervised_oracle` | 真实标签和预测/分数 | 仅用于有真值评价 |
| `unavailable` | 输入不足或数值无效 | `reason` 解释缺失原因 |

固定指标如下：

| 指标 | 证据级别 | 含义 |
|---|---|---|
| `labeled_positive_recall` | `pu_observed` | 已标正例中被预测为正的比例 |
| `unlabeled_negative_rate` | `pu_observed` | 未标记样本中被预测为负的比例 |
| `predicted_positive_rate` | `pu_observed` | 全体样本的预测正类比例 |
| `pu_estimated_precision` | `class_prior_dependent` | 使用类先验校正的精确率估计 |
| `pu_zero_one_risk` | `class_prior_dependent` | 基于离散预测的 PU 零一风险 |
| `accuracy` / `f1` / `roc_auc` | `supervised_oracle` | 仅在提供 `y_true` 时计算 |

PU-only 不等于无假设。例如，在 SAR 下，已标正例可能并不代表全部真实正例，因此 `labeled_positive_recall` 只能描述观测标记子集。

## 4. 报告结构

`PUDiagnosticReport` 包含：

| 字段 | 内容 |
|---|---|
| `data_profile` | 标签规模、特征质量和 SCAR/SAR 证据 |
| `model` | 输入模式、estimator 类型、分数来源和 metadata |
| `metrics` | 固定指标及各自的证据级别 |
| `prediction_statistics` | 预测正类数/比例、有限分数数量和分数范围 |
| `issues` | 数据与模型输出问题、严重级别及行动建议 |
| `provenance` | 样本量、随机种子、真值/类先验是否提供及 profiler 参数 |

`schema_version` 当前为 `1.0`，位于 `report.to_dict()` 的顶层。消费报告的程序应根据 schema version 解析，而不是依赖 Markdown 文本。

## 5. 输出与保存

```python
payload = report.to_dict()
json_text = report.to_json()
markdown_text = report.to_markdown()

report.save("artifacts/diagnostic.json")
report.save("artifacts/diagnostic.md")
```

JSON 使用严格编码：未定义的 AUC、无穷 PU 比例等值会转成 `null`，不会输出非标准的 `NaN` 或 `Infinity`。`save` 根据 `.json`、`.md` 或 `.markdown` 后缀推断格式，也可以显式传入 `format="json"` 或 `format="markdown"`。

## 6. 问题代码

报告继承 Data Profiler 的全部问题，并增加模型输出检查：

| Code | 级别 | 说明 |
|---|---|---|
| `constant_predictions` | warning | 全部样本被预测为同一类；复核阈值、先验和收敛状态 |
| `nonfinite_scores` | error | 分数包含 `NaN/inf`；监督 AUC 被标为不可用 |
| `constant_scores` | warning | 模型没有提供排序信息；检查训练和特征变化 |

缺少预测、类先验或真实标签属于指标不可用状态，而不是数据错误，所以记录在对应 metric 的 `reason` 中，不会制造大量重复 issue。

## 7. 正确工作流

1. 划分训练、验证和测试数据，并保持 PU 标记比例。
2. 所有插补、标准化、特征筛选都只在训练折拟合。
3. 训练 estimator；报告生成器只读取已拟合模型。
4. 在验证集生成不含 `y_true` 的 PU 报告，用预先声明的规则决策。
5. 锁定配置后，在测试集加入 `y_true` 生成最终 oracle 报告。
6. 保存 JSON 作为机器可读产物，Markdown 用于沟通和审阅。

诊断报告不会替代统计显著性检验、类先验敏感性分析或算法推荐。后两项仍是独立工作包。

## 8. 示例

```bash
python examples/minimal/08_diagnostic_report.py
```

示例在 SAR 合成数据上划分训练/测试集，拟合 PUSB，并生成同时包含 PU 指标、审计指标和 selection evidence 的 Markdown 报告。
