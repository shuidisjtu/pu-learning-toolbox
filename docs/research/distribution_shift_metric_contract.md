# PU 分布漂移与部署评价契约

```yaml
schema_version: 1
status: stable_api_contract
scope:
  - shift_audit
  - shift_run
  - shift_aware_pipeline
  - shift_monitor
  - uncertainty_review
purpose: cross_domain_and_deployment_evaluation
```

## 1. 目的与边界

本契约规定源域到目标域的 PU 漂移审计、协变量加权适配、连续监控和选择性复核的
评价口径。它不把域差异检测结果冒充分类性能，也不声称从观测数据唯一识别漂移类型。

`JointShiftPUClassifier` 和 `DynamicJointShiftPUClassifier` 属于 research 路径，按
[JointShift 研究评估协议](joint_shift_research_protocol.md) 单独评价。

## 2. 指标分层

### 2.1 审计指标

| 字段 | 含义 | 解释边界 |
|---|---|---|
| `domain_auc` | 源/目标域 OOF 域分类 ROC-AUC，方向对称到 `[0.5, 1]` | 只表示可观测域可分性，不识别漂移类型 |
| `severity` | 基于 domain AUC 阈值的 `low`/`moderate`/`high` | 工程告警级别，不是统计显著性 |
| `effective_sample_size` | 源域重要性权重的 ESS | 衡量加权后的有效信息量 |
| `effective_sample_fraction` | ESS / 源域样本数 | 覆盖稳定性门槛 |
| `weight_summary` | 权重范围、分位数、原始均值 | 排查权重集中和支持集不足 |
| `probability_clip_fraction` | 域概率触及裁剪边界的比例 | 反映密度比数值风险 |
| `relative_boundary_fraction` | 相对权重触及上界的比例 | 反映权重饱和 |

域 AUC 低不证明无漂移，域 AUC 高也不等价于概念漂移、类先验漂移或联合漂移。ESS
过低、权重边界率过高或原始相对权重质量过低时，报告必须提示覆盖风险。

### 2.2 目标域评估指标

目标域指标必须与源域指标分开保存于 `target_metrics`，不得用源域交叉验证结果替代。

- 提供 `target_true_labels` 时，允许报告目标域 oracle 指标，例如 ROC-AUC、AP、F1 和
  balanced accuracy；
- 没有目标真值但提供可靠 `target_class_prior` 时，允许报告目标域先验依赖的 PU risk
  和 estimated precision；
- 两者都没有时，目标域只能做审计，不能声称适配改善。

加权与未加权两臂必须在同一目标集、同一切分、同一指标口径下配对比较。源域 CV 只能
评价训练稳定性，不得作为目标域性能代理。

### 2.3 部署监控与复核字段

窗口级监控字段包括：

- `domain_auc`、相邻窗口 AUC 变化；
- `effective_sample_fraction` 与覆盖门禁；
- 标记率及其相邻窗口变化；
- `adaptation_ready`、`alert_level`、`alert_codes`；
- 选择性预测的 `coverage`、拒绝率和置信度摘要；
- `n_queries`、复核策略和人工复核队列。

这些字段用于运行时风险控制和数据采集决策，不参与普通分类器超参数优化。
主动复核只产生候选行，不自动创造标签。

## 3. 结论状态

`adaptation_ready` 是布尔字段，表示目标 PU 标签已提供且 ESS、权重边界和相对质量
通过当前覆盖门槛。

`shift-run` 的推荐结论使用以下状态：

| 状态 | 触发条件 | 禁止的结论 |
|---|---|---|
| `adaptation_ready` | 适配所需目标 PU 与覆盖门禁均满足 | 仍不能声称已修复概念/联合漂移 |
| `audit_only` | 缺少目标真值和可靠目标先验，或仅执行审计 | 不得声称目标性能改善 |
| `collect_target_data` | 覆盖门禁失败且未显式允许不稳定适配 | 不得自动执行加权适配 |

`allow_unstable` 只允许受控研究实验；报告必须保留覆盖警告，不能改变结论边界。

## 4. 产物与 provenance

稳定工作流至少保存：

```text
shift_report.json
shift_report.md
source_importance_weights.csv
shift_comparison.json/.md
target_predictions.csv
shift_history.json
```

JSON/Markdown 必须分开表达审计指标、目标指标、结论状态和实现边界。manifest 或报告
provenance 记录数据域、配置、折数、随机种子、阈值、代码 commit 和依赖版本。

监控历史只保存窗口摘要，不保存原始样本；恢复历史时必须校验监控配置一致性。

## 5. 验收规则

1. 相同分布数据的 domain AUC 接近随机，权重均值和 ESS 计算符合数学金标准；
2. 已知均值漂移数据的 domain AUC、severity 和权重诊断按预期升高；
3. 高 AUC 与低 ESS 时触发覆盖告警，不自动宣称适配成功；
4. 目标真值缺失时，目标 oracle 指标明确不可用；
5. 目标真值和目标先验均缺失时，结论固定为 `audit_only`；
6. 覆盖门禁失败且未 override 时，结论为 `collect_target_data`；
7. 加权/未加权比较使用同一目标集并输出配对差异；
8. shift-monitor、review CLI/UI 与结构化产物字段保持一致。

本契约只描述稳定分布漂移与部署路径。改变字段语义、结论状态、覆盖门槛或目标指标
可用条件时，必须提升 `schema_version`。
