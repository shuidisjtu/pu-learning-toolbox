# 端到端 PU 工作流（PUPipeline）

`PUPipeline` 把 **数据画像 → 类先验估计 → 模型训练 → PU 分层交叉验证 → 评估诊断**
封装为一次调用：非专家用户只需要 `(X, y_pu)` 和一个方法名（或 `"auto"`），
即可得到完整、可审计的训练评估报告。

## 快速上手

```python
from pu_toolbox import PUPipeline

pipe = PUPipeline()                      # classifier="auto"，自动选算法
report = pipe.fit_evaluate(X, y_pu)      # X: (n, d)，y_pu: {1, 0} PU 标签

print(report.summary())                  # 人类可读摘要
report.to_markdown()                     # 完整 Markdown 报告
report.to_json()                         # 严格 JSON（无 NaN）
report.save("results/pipeline.json")
```

`report.final_model` 是在全量数据上重新拟合的最终模型，可直接 `predict` / `decision_function`。

## 五步流程

1. **校验**：`validate_pu_X_y`（标签规范化、形状、最少正样本、正样本数 ≥ CV 折数）。
2. **先验解析**（见下节优先级）：需要类先验的方法缺先验时自动估计一次（全量数据，**绝不使用 `y_true`**）。
3. **数据画像**：`profile_pu_data` 一次，产出 SCAR/SAR 证据与数据质量检查。
4. **交叉验证**：默认 `PUStratifiedKFold(5, shuffle=True, random_state=...)`，
   每折独立 fit/predict，逐折计算指标并聚合（mean ± std）。
5. **最终模型 + 诊断**：全量 refit 后调用 `build_diagnostic_report`，指标与
   画像问题汇总进 `PipelineReport`。

## 类先验解析优先级

| 优先级 | 来源 | 报告中的 source |
|---|---|---|
| 1 | `fit_evaluate(class_prior=...)` 显式传入 | `user` |
| 2 | 分类器实例构造参数 `class_prior=...` | `constructor` |
| 3 | `prior_estimator` 自动估计（默认 `"recpe"`） | `estimated` |
| 4 | 方法不需要先验 | `none` |

- `prior_estimator="recpe"`（默认）/ `"pen_l1"` / `"km1"` / `"km2"`（后两者映射到
  `KernelMeanPriorEstimator`）或传入估计器实例。
- `prior_estimator=None` 且方法需要先验且未提供 → 抛出 `PipelineError`
  （消息给出三条出路，并注明 `y_true` 从不用于先验估计）。
- 画像模块会审计"先验 < 标注正样本比例"的不一致并给出警告
  （`inconsistent_class_prior`）。

## 分类器选择

| `classifier=` | 行为 |
|---|---|
| `"auto"`（默认） | 先估先验 → `recommend_methods` 推荐 → 按 rank 扫描选中第一个**可自动实例化**的候选 |
| `"nnpu"` / `"upu"` 等注册名 | 直接使用（大小写不敏感，支持别名），构造时自动注入 `class_prior` 与 `random_state` |
| `UPUClassifier(...)` 实例 | 原样使用（`clone` 到每折），不注入任何参数 |

注意：构造器有必填非 `class_prior` 参数的方法（如 `"ldce"` 需要
`flip_probability`、`"dgpu"` 需要生成器）**不能从名字自动实例化**——`"auto"` 会
跳过它们并在 `report.provenance["skipped_candidates"]` 记录原因；显式指定名字则
在构造时即报错，请改传实例。

## 指标与可用性

默认指标 `DEFAULT_METRICS = ("pu_zero_one_risk", "pu_recall", "pu_estimated_precision", "pu_auc_roc")`。
别名：`pu_risk`、`auc`/`roc_auc`、`recall`、`precision`、`accuracy`、`f1`、`negative_rate`。

| 指标 | 需要 | basis |
|---|---|---|
| `pu_recall` / `pu_negative_rate` | 仅 `y_pu` + 预测 | `pu_observed` |
| `pu_zero_one_risk` / `pu_estimated_precision` | 预测 + 类先验 | `class_prior_dependent` |
| `pu_auc_roc` / `pu_accuracy` / `pu_f1` | `y_true` | `supervised_oracle` |

缺失输入（无 `y_true`、无 `decision_function`、无先验）时对应指标**跳过并记录原因**
（`CVMetric.available=False`、`reason` 说明），不中断流程。单折内异常（如 AUC 折内
单类）只跳过该折，`mean`/`std` 在已计算折上聚合。

## 错误场景

| 场景 | 异常 |
|---|---|
| 无效 classifier / prior 名 | 构造时 `PipelineError`（fail-fast） |
| `"ldce"` 等不可自动实例化 | 构造时 `PipelineError`（提示传实例） |
| 无正样本 / 正样本 < `MIN_POSITIVE_SAMPLES` | `ValidationError` |
| 正样本 < CV 折数 | `ValidationError`（提示减折数） |
| 需要先验且最终缺失 | `PipelineError` |
| 先验估计值 ∉ (0, 1) | `PipelineError` |
| 未知指标名 / 非法 CV | 构造时 `ValueError` / `TypeError` |

## 与手动流程对比

手动流程（`examples/minimal/05_recpe_pipeline.py`）需要 ~30 行样板：
画像 → 先验估计 → 网格调参 → CV → 评估。`PUPipeline` 一行等价，且报告
`provenance` 完整记录每步决策（classifier 解析、先验来源、跳过候选、随机种子），
满足可审计要求。
