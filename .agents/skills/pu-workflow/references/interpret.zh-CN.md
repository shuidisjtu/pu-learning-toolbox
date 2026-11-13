# 结果解读指南（中文）

读取实际产物，以“设置 → 结果 → 适用条件 → 下一步”组织解释。直接陈述分析支持的结论，
数值来自产物。按任务选择下列字段，集中说明与当前结果有关的条件。

写作时先确定每项计算直接回答的问题，再陈述该问题的结果。例如，写
“P/U 可分性 AUC 为 0.80，证据类型为 observed_mixture；方法选择结合已记录的标记政策”，
或“本次固定预测扫描中，PU 经验风险从 0.1 变化到 −0.1”。把需要进一步研究的问题写成
具体动作，如“补充独立真值验证集后计算监督指标”。每段围绕当前证据及其结论展开。

## 数据与假设证据

`profile.json` 或 Pipeline 报告的 `profile` 包含：

- `summary`：样本数、特征数、标记正例数、标记比例及输入采样设置。
- `selection_diagnostic.status`：`plausible` 表示诊断支持按 SCAR 路线分析；
  `at_risk` 表示诊断发现选择偏差迹象；`inconclusive` 表示当前诊断信息量有限，
  结合具体样本数和 `issues` 中的建议说明下一步。
- `issues[]`：读取 `code/severity/message/action`，说明具体观测和对应动作。

先说明 OS、独立 TS 或具名有限池设计；模拟研究另写来源数据、抽样政策与标记机制。
按用户授权和可执行路线继续分析，将诊断结果用于方法选择和结果解释。

## 推荐与先验链

`recommendation.json` 中的 `candidates[]` 按 `rank` 排序，读取 `score`、`reasons`、
`warnings`、`global_warnings` 和 `filters_applied`，说明当前数据与能力条件下的推荐依据。

Pipeline 报告 schema `1.1` 的 `prior` 按以下顺序解释：

| 阶段 | 字段 | 解释 |
|---|---|---|
| 来源 | `source/estimator/auto_selected` | 用户值、构造值或估计方法，以及自动选择情况 |
| 原始输入 | `input_value/input_target/estimation_pool` | 输入或估计得到的数值、目标分布与估计池 |
| 总体解析 | `value/target/input_sampling/conversion` | 总体先验和 OS/TS 下使用的换算 |
| 估计细节 | `estimation_details` | 方法提供的辅助目标、近似或其他元数据 |
| 训练消费 | `training_consumption` | 每折和最终 refit 实际传给方法的先验与目标 |

例如，产物记录 OS 标记比例 0.2、原生 U 先验 0.25、总体先验 0.4 时，可写：
“本次估计的原生 U 正类比例为 0.25，按 OS 质量关系换算的总体先验为 0.4。”
随后按实际 `training_consumption` 说明所用方法消费的目标。

报告 `degraded`、`prior_audit_flagged` 或相关 issue 有值时，说明实际采用的路线和原因。

## 训练、调参与方法比较

`report.json` 的 `cv_metrics` 按指标读取 `available/mean/std/n_computed/reason`。
对可用指标写明均值、标准差和有效折数，并解释其方向、是否使用真实标签及评估总体。
对待补充的信息直接给出 `reason` 对应的动作。

从 `provenance` 与 `cv_provenance` 读取方法、seed、架构和运行设置；按 `issues` 说明具体发现。
`tuning.json` 或 `comparison.json` 另记录候选范围、评分、选择结果和执行预算。
将评分准则下选出的候选与其报告相连，明确结果对应的搜索范围。

## 研究实验与表征

读取 manifest 中的设置、候选轨迹、各协议选择和最终评测，按实际字段组织：

- 来源级 split、标记政策、输入采样、风险视图、先验输入与训练消费。
- PA/OA 各自的候选、checkpoint、阈值和验证指标，以及对应 test 结果。
- clean support 的来源、规模和使用阶段。
- 二维特征、原生 CNN 或冻结特征路径；encoder/backbone、初始化或预训练来源、
  拟合范围和表示身份。

PA 分数按当前 OS 计算式和所用总体先验解释；OA 是 clean 验证真值上的准确率。
比较时说明共享的 split、seed、候选预算和表征条件。总体结论引用实际 test 或多 seed 汇总。

## 敏感性与扩展结果

- 固定预测敏感性：读取 `sensitivity.json` 的 `observed_label_rate` 与 `points`，
  记录模型、先验/标记率网格、`is_consistent` 和指标变化，说明合法假设区间内的结果。
- 重新训练比较：记录各次训练所用先验、方法、seed 和评估结果。
- 漂移：报告 domain AUC、权重/重叠证据及实际执行的 baseline/weighted 指标差。
- 监控：报告窗口 ID、告警代码、变化值和阈值。
- 人工复核：报告策略、预算、覆盖率和可追踪的查询行。
- 基准审计：报告检查结果、覆盖规模及需要修复的具体产物。

## 交付内容

根据任务形成连贯的中文说明：使用了什么数据和设置、执行了哪些计算、得到哪些结论、
相关条件是什么、接下来采取什么具体动作。附报告或模型路径，并将建议集中在最有用的事项上。
