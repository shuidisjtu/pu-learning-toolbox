# API 参考

> 这不是入门读物：先读 [快速开始](../quickstart.md) 或对应的操作指南。
> 本文档只给精确契约；参数含义与使用示例在对应 howto 中。

## PUPipeline

用法见 [howto/pipeline.md](../howto/pipeline.md)。

```python
pipe = PUPipeline(
    classifier="auto",          # 注册方法名 / "auto"（推荐器选算法）/ 分类器实例
    prior_estimator="recpe",    # "recpe"/"pen_l1"/"km1"/"km2"（后两者映射到
                                # KernelMeanPriorEstimator）/ 估计器实例 / None
    cv=5,                       # PU 分层 CV 折数，或自定义 splitter
                                # （默认 cv=None → 解析为 5 折 PUStratifiedKFold）
    metrics=DEFAULT_METRICS,    # 指标名元组，见下方指标表
                                # （默认 metrics=None → DEFAULT_METRICS）
    random_state=42,
    architecture="mlp",         # 深度算法架构："mlp"（表格）/ "cnn"（4-D NCHW 图像，需显式 wconpu/infomax_pu）
    backbone="cnn13",           # CNN 骨架：cnn13/resnet18/resnet50（仅 cnn 有效）
    device="cpu",               # 深度分类器 torch 设备（如 "cuda"）
)
report = pipe.fit_evaluate(X, y_pu, y_true=None, class_prior=None)
```

### 类先验解析优先级

| 优先级 | 来源 | 报告中的 source |
|---|---|---|
| 1 | `fit_evaluate(class_prior=...)` 显式传入 | `user` |
| 2 | 分类器实例构造参数 `class_prior=...` | `constructor` |
| 3 | `prior_estimator` 自动估计（默认 `"recpe"`） | `estimated` |
| 4 | 方法不需要先验 | `none` |

- `prior_estimator=None` 且方法需要先验且未提供 → 抛出 `PipelineError`。
- 画像模块审计"先验 < 标注正样本比例"的不一致（`inconsistent_class_prior`）。
- auto 模式先验估计失败不中断流程：降级为无先验推荐（`report.prior.degraded`）。

### 指标

`DEFAULT_METRICS = ("pu_zero_one_risk", "pu_recall", "pu_estimated_precision", "pu_auc_roc")`。
别名：`pu_risk`、`auc`/`roc_auc`、`recall`、`precision`、`accuracy`、`f1`、`negative_rate`。

| 指标 | 需要 | basis |
|---|---|---|
| `pu_recall` / `pu_negative_rate` | 仅 `y_pu` + 预测 | `pu_observed` |
| `pu_zero_one_risk` / `pu_estimated_precision` | 预测 + 类先验 | `class_prior_dependent` |
| `pu_auc_roc` / `pu_accuracy` / `pu_f1` | `y_true` | `supervised_oracle` |

缺失输入（无 `y_true`、无 `decision_function`、无先验）时对应指标跳过并记录原因
（`CVMetric.available=False`），不中断流程。

### classifier 选择语义

| `classifier=` | 行为 |
|---|---|
| `"auto"`（默认） | 先估先验 → `recommend_from_profile` 推荐 → 按 rank 扫描选中第一个**可自动实例化**的候选 |
| `"nnpu"` / `"upu"` 等注册名 | 直接使用（大小写不敏感，支持别名），构造时自动注入 `class_prior` 与 `random_state` |
| `UPUClassifier(...)` 实例 | 原样使用（`clone` 到每折），不注入任何参数 |

构造器有必填非 `class_prior` 参数的方法（如 `"ldce"` 需要 `flip_probability`）
不能从名字自动实例化——`"auto"` 会跳过它们并在 `report.provenance["skipped_candidates"]`
记录原因；显式指定名字则构造时即报错，请改传实例。

### 深度算法与架构选择

`architecture` / `backbone` / `device` 参数契约：

| 参数 | 默认 | 取值 | 语义 |
|---|---|---|---|
| `architecture` | `"mlp"` | `"mlp"` / `"cnn"` | `"cnn"` 需显式深度分类器且其构造签名声明 `encoder` 参数（当前 `wconpu` / `infomax_pu`）；未声明（如 `self_pu`）、`auto` 或非深度方法配 cnn 抛 `PipelineError` |
| `backbone` | `"cnn13"` | `"cnn13"` / `"resnet18"` / `"resnet50"` | 仅 `architecture="cnn"` 有效；非法值抛 `ValueError` |
| `device` | `"cpu"` | torch 设备字符串 | 透传给深度分类器（`_fresh_estimator` 按签名注入） |

- 深度算法接入契约：要获得 `architecture="cnn"` 支持，分类器构造签名必须声明
  `encoder` 参数（特征提取器形态，pipeline 注入 `build_encoder` 产物，
  `_fresh_estimator` 按签名守卫注入）；未声明时配 cnn 在构造期即被拒绝
- 显式 `wconpu` / `infomax_pu`：放行必填参数检查，`class_prior` 按「显式 >
  估计」顺序注入；`architecture="cnn"` 时 encoder 由 pipeline 在 `fit_evaluate`
  内懒构建（`build_encoder("cnn", backbone=..., in_channels=...)`）并注入
- 输入维度：4-D NCHW + 显式深度分类器 + cnn → 正常（prior 估计与数据画像在
  展平视图上进行，CV splitter 按索引切分）；4-D + mlp 或非深度分类器 →
  `PipelineError`；2-D + cnn → `PipelineError`
- deep + `cv>1` 时打印训练成本警告（n_splits+1 次训练），建议减少折数（`cv` 最小为 2）
- `auto` 行为不变：深度方法虽在推荐器候选内，但因 GPU/数据规模/训练成本评分低，实际不会被选中

### 错误场景

| 场景 | 异常 |
|---|---|
| 无效 classifier / prior 名 | 构造时 `PipelineError`（fail-fast） |
| `"ldce"` 等不可自动实例化 | 构造时 `PipelineError`（提示传实例） |
| 无正样本 / 正样本 < `MIN_POSITIVE_SAMPLES` | `ValidationError` |
| 正样本 < CV 折数 | `ValidationError`（提示减折数） |
| 需要先验且最终缺失 | `PipelineError` |
| 先验估计值 ∉ (0, 1) | auto：降级为无先验推荐；显式：`PipelineError` |
| 先验估计器异常 | auto：降级（`prior.degraded` 记录）；显式：`PipelineError` |
| 未知指标名 / 非法 CV | 构造时 `ValueError` / `TypeError` |

## build_encoder

深度分类器的统一编码器构建入口（`pu_toolbox/estimators/deep/vision.py`），
PUPipeline 在 `architecture="cnn"` 时内部调用；也可手动传给分类器。

```python
encoder = build_encoder(
    architecture,             # "mlp" | "cnn"
    *,
    backbone="cnn13",         # "cnn13" / "resnet18" / "resnet50"（仅 cnn）
    in_channels,              # 图像通道数（如 RGB=3）
    normalization_mean=None,  # 每通道均值；默认 0.5
    normalization_std=None,   # 每通道标准差；默认 0.5
)
```

- `"mlp"` → 返回 `None`（分类器内置 MLP 路径，表格数据）
- `"cnn"` → 返回 `build_wconpu_backbone(...)` 图像骨干（4-D NCHW 输入，
  内嵌通道标准化）
- 非法 `architecture` → `ValueError`

## InfoMaxPUClassifier

深度 PU 分类器（PURL 表示学习 → 类先验估计 → nnPU 分类），构造参数多
（`representation_*` / `classifier_*` 系列）；需要细粒度控制时直接传实例给
`PUPipeline`。`encoder` 参数：

```python
clf = InfoMaxPUClassifier(
    encoder=None,     # 外置编码器（如 build_encoder("cnn", ...)）；None → 内置 MLP
    device="cpu",     # torch 设备
    ...
)
```

- `encoder=None`（默认）：内置 MLP 编码器，向后兼容（表格数据）
- 传入外置编码器：替代内部 `nn.Sequential(Linear...)` 编码部分，`ratio_head_`
  接在编码器特征之后；`fit` 放行 4-D NCHW 图像输入

## profile_pu_data

用法见 [howto/data_profiling.md](../howto/data_profiling.md)。

```python
report = profile_pu_data(
    X,
    y_pu,
    y_true=None,               # 提供后启用可识别的审计模式
    class_prior=None,
    min_labeled_positives=30,
    max_unlabeled_to_positive=100.0,
    low_variance_threshold=1e-12,
    scar_auc_threshold=0.65,
    cv=5,
    random_state=42,
)
```

所有 `y_pu == 1` 的样本必须在 `y_true` 中也是正类，否则接口拒绝输入。

### 返回 PUDataProfile

| 字段 | 内容 |
|---|---|
| `summary` | 样本数、特征数、已标正例数、未标记数、PU 比例、稀疏性、类先验及隐含标记频率 |
| `feature_statistics` | 缺失/无穷值数量、常数列和低方差列索引 |
| `selection_diagnostic` | AUC、状态、证据来源、可识别性、实际折数和评估样本数 |
| `issues` | 带稳定 `code`、严重级别、解释和行动建议的问题列表 |
| `assumption_hints` | 对 SCAR/SAR、类先验依赖和结果边界的说明 |

辅助接口：`report.has_errors` / `report.has_warnings` / `report.format_text()` / `report.to_dict()`。

`selection_diagnostic["status"]` 取值：`plausible`（CV AUC 不高于阈值）、`at_risk`
（AUC 高于阈值）、`inconclusive`（某一组少于两个样本，或存在非有限特征值）。

### 问题代码

| Code | 级别 | 建议 |
|---|---|---|
| `no_labeled_positives` | error | 检查标签编码或收集可信正例 |
| `no_unlabeled_samples` | error | 改用监督学习流程或补充未标记总体 |
| `missing_features` / `infinite_features` | error | 在仅由训练折拟合的 pipeline 中清理或插补 |
| `few_labeled_positives` | warning | 使用重复验证、置信区间并尽量补充正例 |
| `extreme_pu_imbalance` | warning | 使用 PU 分层切分和抗不平衡指标 |
| `constant_features` | warning | 在训练 pipeline 内删除常数列 |
| `high_dimensional_data` | warning | 使用正则化模型，避免在全数据上预处理 |
| `inconsistent_class_prior` | warning | 复核类先验、采样总体和标签定义 |
| `sar_signal` | warning | 审计正例支持 SAR；优先评估 PUSB/LBE 并做敏感性分析 |
| `observed_selection_signal` | info | 只有非识别性信号；补充审计或标记策略信息 |
| `low_variance_features` | info | 复核特征缩放；低方差列可能不携带信号 |
| `selection_diagnostic_inconclusive` | info | 评估组样本不足或特征非有限；补充审计或可信正例标签 |

## recommend_methods / recommend_from_profile

用法与决策原理见 [concepts/method_selection.md](../concepts/method_selection.md)。

```python
result = recommend_methods(
    X, y_pu,
    scenario=None,       # Scenario 或字符串，如 "case_control"
    assumption=None,     # Assumption 或字符串，如 "scar"
    class_prior=None,    # 已知 P(Y=1)；None 时排除需先验的方法
    has_gpu=False,
    top_k=5,
    random_state=42,
    config=None,         # ScoringConfig；默认 DEFAULT_CONFIG
)

result = recommend_from_profile(
    profile,             # 已有 PUDataProfile，跳过重复 profiling
    scenario=None, assumption=None, class_prior=None,
    class_prior_source=None,  # 先验来源说明（如 "user"/"estimated"），写入 provenance
    has_gpu=False, top_k=5, config=None,
)
```

返回 `RecommendationResult`：`candidates`（`MethodCandidate`：name/score/rank/reasons/warnings/metadata）、
`filters_applied`、`global_warnings`、`provenance`。导出：`result.to_json()` / `to_markdown()` / `save(path)`。

## 数据生成

用法见 [howto/sar_simulation.md](../howto/sar_simulation.md)。

```python
propensity = make_sar_propensity(X, y_true, mechanism="linear",
                                 label_frequency=0.4, strength=1.5)
y_pu, propensity = make_sar_labels(X, y_true, mechanism="nonlinear",
                                   label_frequency=0.4, random_state=42,
                                   return_propensity=True)
X, y_pu, y_true, propensity = make_sar_dataset(
    n_samples=1000, n_features=8, class_prior=0.3, separation=2.0,
    mechanism="linear", label_frequency=0.4, strength=1.5, random_state=42)
```

- `mechanism`：`"scar"` / `"linear"` / `"nonlinear"`（定义见 [concepts/scar_sar.md](../concepts/scar_sar.md)）。
- `label_frequency`：正类 propensity 的目标均值（校准），不是抽样后的精确比例。
- `make_sar_labels` 默认 `ensure_labeled=True`：小样本抽样未选中任何正类时选择
  propensity 最高的真实正类，保证下游可训练。
- 返回值 `propensity` 表示 `P(S=1|Y,X)`，真实负类位置固定为零。
- CLI 演示数据 `make-demo-data`（`--n` 每类样本数、`--c` 标注概率）内部使用
  `make_scar_dataset`（`make_scar_dataset(n, c, n_features=5, separation=4.0,
  random_state=None)` → `(X, y_pu, class_prior)`，机制固定为 SCAR）。

## analyze_pu_sensitivity

用法见 [howto/sensitivity_analysis.md](../howto/sensitivity_analysis.md)。

```python
analysis = analyze_pu_sensitivity(
    y_valid,
    y_pred,                              # 必须 {0,1}
    scores=None,                         # 有限一维数组；0 为阈值计算 PU zero-one risk
    class_priors=None,                   # 每项在 (0, 1)
    label_propensities=None,             # 每项在 (0, 1]
)
```

`class_priors` 与 `label_propensities` 至少提供一个；`y_pu` 必须同时包含
labeled-positive 和 unlabeled 组。返回 `PUSensitivityAnalysis`，每点含：

| 字段 | 含义 |
|---|---|
| `axis`, `value` | 当前扫动的参数及其假定值 |
| `class_prior` | 假定或由 `q/c̄` 推出的类先验 |
| `label_propensity` | 假定或由 `q/π` 推出的平均 propensity |
| `is_consistent` | 是否满足概率边界和观测恒等式 |
| `consistency_reason` | 不一致时的直接原因 |
| `pu_estimated_precision` | `min(π·Recall̂_P/P̂(Ŷ=1), 1)` |
| `pu_zero_one_risk` | `2π·FNR̂_P + FPR̂_U − π` |

`metric_ranges` 仅汇总 `is_consistent=True` 且指标可用的点。导出：
`analysis.to_frame(axis=...)` / `analysis.save(path)`（JSON 严格模式，无 NaN/Infinity）。

## build_diagnostic_report

用法见 [howto/diagnostic_reports.md](../howto/diagnostic_reports.md)。三种模式：

```python
# 纯数据（只运行 profile_pu_data；预测类指标 basis=unavailable）
report = build_diagnostic_report(X, y_pu, class_prior=0.30, random_state=42)

# 已拟合 estimator（调用 predict，优先 decision_function，回退 predict_proba[:, 1]；
# 不调用 fit；未拟合 estimator 抛 ValueError）
report = build_diagnostic_report(X_valid, y_valid, estimator=classifier,
                                 class_prior=0.30)

# 显式输出（estimator 与 y_pred/scores 互斥）
report = build_diagnostic_report(X_valid, y_valid, y_pred=predictions,
                                 scores=decision_scores, class_prior=0.30)
```

`y_true`（可选）：启用可识别的 selection-dependence 检查并计算
`accuracy`/`f1`/`roc_auc`（标为 `supervised_oracle`）；不会传给 estimator。

### 指标证据级别

| `basis` | 指标来源 | 是否需要额外假设 |
|---|---|---|
| `pu_observed` | PU 标签和模型预测 | 不需要真实标签 |
| `class_prior_dependent` | PU 标签、预测和类先验 | 数值随类先验改变 |
| `supervised_oracle` | 真实标签和预测/分数 | 仅用于有真值评价 |
| `unavailable` | 输入不足或数值无效 | `reason` 解释缺失原因 |

固定指标：`labeled_positive_recall` / `unlabeled_negative_rate` / `predicted_positive_rate`
（`pu_observed`）；`pu_estimated_precision` / `pu_zero_one_risk`
（`class_prior_dependent`）；`accuracy` / `f1` / `roc_auc`（`supervised_oracle`）。

### 报告问题代码（模型输出检查）

| Code | 级别 | 说明 |
|---|---|---|
| `constant_predictions` | warning | 全部样本被预测为同一类；复核阈值、先验和收敛状态 |
| `nonfinite_scores` | error | 分数包含 `NaN/inf`；监督 AUC 被标为不可用 |
| `constant_scores` | warning | 模型没有提供排序信息；检查训练和特征变化 |

`PUDiagnosticReport` 顶层 `schema_version` 当前为 `1.0`；`save()` 按 `.json`/`.md` 后缀推断格式，
JSON 严格编码（未定义值转 `null`）。
