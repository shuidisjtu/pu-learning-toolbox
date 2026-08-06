# 结果解读指南（中文）

面向最终用户的中文解读模板。agent 用用户的语言写总结，本指南给出
字段含义与建议模板（英文 SKILL.md 的 Step 4 引用本文件）。

## profile.json 关键字段

- `summary`：样本数、特征数、标注正样本数、观测正样本率 P(S=1) 等。
- `selection_diagnostic.status`：
  - `plausible` — 数据与 SCAR 假设无明显冲突，可以按 SCAR 方法继续；
  - `inconclusive` — 证据不足，无法判定（常见于正样本过少）；
  - `at_risk` — 存在 selection-bias 迹象，SCAR 方法可能偏置；
  - `error` — 数据本身无法支撑可靠建模。
- `issues[]`：每条含 `code` / `severity` / `message` / `action`。
  向用户转述 `message` 并给出 `action` 建议。

## recommendation.json 关键字段

- `candidates[]`：按 `rank` 排序；`score` 越高越好；`reasons[]` 是
  入选理由（逐条转述），`warnings[]` 是该方法的风险提示。
- `global_warnings[]`：全局性注意事项（如先验来源、稀疏数据）。
- `filters_applied`：被过滤掉的候选原因（如需要 class_prior 但未提供）。

## report.json 关键字段

- `prior`：`value` + `source`（user / estimated / constructor / none）。
  提醒用户先验对 PU 结果影响大，估计值应结合领域知识审视。
- `cv_metrics`：每个指标的 `available` / `mean` / `std` /
  `n_computed`。`available=false` 时转述 `reason`（如缺 y_true）。
- `provenance`：`classifier`、`classifier_mode`（auto 时说明是推荐器
  选中的）、`prior_estimator`。
- `issues[]`：`has_errors` 为 true 时先处理错误再下结论。

## sensitivity.json 关键字段

- `observed_label_rate`：观测正样本率，是先验的可识别性下限。
- `points[]`：每点的 `class_prior` / `label_propensity` / `is_consistent`
  / `estimated_precision` / `pu_zero_one_risk`。
- `has_inconsistent_assumptions`：存在与观测不符的假设组合——转述
  具体是哪几组，说明结果对这些假设的敏感性。

## 行动建议模板（结论结构）

1. **假设判定**：先讲 SCAR/SAR 结论与对所选方法的影响。
2. **先验**：数值、来源，提醒敏感性。
3. **方法**：选中的算法 + 入选理由 + 风险提示。
4. **指标**：只引用 `available=true` 的指标，写 `mean ± std`。
5. **下一步**：最多 3 条具体建议（修数据 / 换先验 / 对比方法）。
