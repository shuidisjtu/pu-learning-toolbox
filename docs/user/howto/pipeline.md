# PUPipeline 端到端训练评估

> 前置条件：先完成 [快速开始](../quickstart.md)。
> 概念：PU 问题设定与 π 的角色见 [concepts/pu_problem.md](../concepts/pu_problem.md)，
> SCAR/SAR 见 [concepts/scar_sar.md](../concepts/scar_sar.md)。

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

## 类先验解析

先验按「显式传入 → 构造参数 → 自动估计 → 不需要」四级解析（完整优先级表见
[API 参考](../reference/api.md)）。要点：

- `prior_estimator="recpe"`（默认）/ `"pen_l1"` / `"km1"` / `"km2"`（后两者映射到
  `KernelMeanPriorEstimator`）或传入估计器实例。
- 显式指定 classifier 时，`prior_estimator=None` 且方法需要先验且未提供 →
  抛出 `PipelineError`（消息给出三条出路，并注明 `y_true` 从不用于先验估计）；
  auto 模式下同样输入降级为无先验推荐（见下）。
- 画像模块会审计"先验 < 标注正样本比例"的不一致并给出警告
  （`inconsistent_class_prior`）。

## 分类器选择

| `classifier=` | 行为 |
|---|---|
| `"auto"`（默认） | 先估先验 → `recommend_from_profile` 推荐 → 按 rank 扫描选中第一个**可自动实例化**的候选 |
| `"nnpu"` / `"upu"` 等注册名 | 直接使用（大小写不敏感，支持别名），构造时自动注入 `class_prior` 与 `random_state` |
| `UPUClassifier(...)` 实例 | 原样使用（`clone` 到每折），不注入任何参数 |

注意：构造器有必填非 `class_prior` 参数的方法（如 `"ldce"` 需要
`flip_probability`、`"dgpu"` 需要生成器）**不能从名字自动实例化**——`"auto"` 会
跳过它们并在 `report.provenance["skipped_candidates"]` 记录原因；显式指定名字则
在构造时即报错，请改传实例。

**降级语义**：auto 模式下先验估计失败（估计器异常或估计值越界）不中断流程——
降级为无先验推荐（需先验方法被排除），`report.prior.degraded` 与 issues 中的
`prior_estimation_failed` 记录原因。显式指定 classifier 时估计失败仍报错。
估计先验被画像审计警告（`inconsistent_class_prior`）时，
`report.provenance["prior_audit_flagged"]=True`，提示自动选出的需先验方法需谨慎。

### 深度算法与架构选择

`PUPipeline` 已为深度算法显式接入架构选择（`wconpu` / `infomax_pu` 支持
`encoder` 骨架注入；`self_pu` 亦可按名实例化但尚未适配 cnn），需先安装可选
依赖 `pip install pu-toolbox[torch]`。深度算法须**显式指定**——`auto` 模式下
深度方法虽在推荐器候选内，但因 GPU/数据规模/训练成本评分低，实际不会被选中。

| 参数 | 默认 | 说明 |
|---|---|---|
| `architecture` | `"mlp"` | `"mlp"`（表格数据）或 `"cnn"`（4-D NCHW 图像） |
| `backbone` | `"cnn13"` | CNN 骨架：`"cnn13"` / `"resnet18"` / `"resnet50"`（仅 `architecture="cnn"` 时有效） |
| `device` | `"cpu"` | 传给深度分类器的 torch 设备（如 `"cuda"`） |

- 显式指定深度分类器且其构造签名声明 `encoder` 参数时放行（当前为 `wconpu` /
  `infomax_pu`），`class_prior` 仍按「显式 > 估计」顺序注入；
  `architecture="cnn"` 时 pipeline 用 `build_encoder` 构建并注入 CNN 编码器；
  未声明 `encoder` 的深度分类器（如 `self_pu`）配 cnn 在构造期报 `PipelineError`
- `architecture="cnn"` 要求 4-D NCHW 图像输入（`.npy` 数组）；2-D 表格配
  `cnn` 或 4-D 图像配 `mlp` 都会报错
- 深度训练较慢（WConPU 默认 800 epoch），`cv>1` 时 pipeline 会打印训练成本
  提示，可减少折数（`cv` 最小为 2）

## 指标与可用性

默认指标 `DEFAULT_METRICS = ("pu_zero_one_risk", "pu_recall", "pu_estimated_precision", "pu_auc_roc")`。
各指标的依赖与证据级别（`pu_observed` / `class_prior_dependent` / `supervised_oracle`）见
[API 参考](../reference/api.md)。

缺失输入（无 `y_true`、无 `decision_function`、无先验）时对应指标**跳过**并记录
（`CVMetric.available=False`；仅当全部折都被跳过时才由 `reason` 说明原因），
不中断流程。单折内异常（如 AUC 折内单类）只跳过该折，`mean`/`std` 在已计算折上聚合。

## 错误场景

设计原则是 **fail-fast**：无效 classifier / prior 名、不可自动实例化方法在构造时即抛
`PipelineError`；数据问题（无正样本、正样本 < CV 折数）抛 `ValidationError`。完整异常
表与降级语义见 [API 参考](../reference/api.md)。

## 与手动流程对比

手动流程（`examples/minimal/05_recpe_pipeline.py`）需要 ~30 行样板：
画像 → 先验估计 → 网格调参 → CV → 评估。`PUPipeline` 一行等价，且报告
`provenance` 完整记录每步决策（classifier 解析、先验来源、跳过候选、随机种子），
满足可审计要求。

## 下一步

- 生成诊断报告：[diagnostic_reports.md](diagnostic_reports.md)
- 类先验与标记倾向敏感性分析：[sensitivity_analysis.md](sensitivity_analysis.md)
- 精确参数契约：[API 参考](../reference/api.md)
