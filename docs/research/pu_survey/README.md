# PU Survey 文档索引

本目录只保留长期有效的协议、任务、交付、准入和复核材料；阶段性排序已合并到
[执行计划](survey_execution_plan.md)，不再单独维护后续优先级文档。

## 权威源

| 文档 / 数据 | 定位 |
|---|---|
| [`pu_survey_protocol.md`](pu_survey_protocol.md) | Survey 协议要求：数据集、四角色输入、OS/TS 视图、PA/OA 选模、PN oracle 与主榜门禁 |
| [`survey_execution_plan.md`](survey_execution_plan.md) | 唯一任务分工真相源：已完成交付、未完成任务、主责/复核、依赖、风险、留痕与决策摘要 |
| `pu_toolbox/experiment/method_ledger.json` | 方法台账真相源：方法身份、来源、视图、校准、适配和待复核状态 |
| `pu_toolbox/experiment/survey_protocol_v1.json` | 机器可读执行协议，当前 `survey-v1.2` |
| `pu_toolbox/experiment/survey_comparison_v3.json` | 机器可读对照矩阵，当前 `survey-comparison-v3` |

> 代码级事实以源码、测试和提交历史为准；本目录文档负责记录协议边界、交付证据和人工复核状态，
> 不重复抄录提交过程或参考文献全文。

## 文档分区

### `architecture/`：长期有效的制品与 checkpoint 设计

- [`artifact_storage_architecture.md`](architecture/artifact_storage_architecture.md)：制品分级、规模推演与迁移路径。
- [`epoch_checkpoint_delivery.md`](architecture/epoch_checkpoint_delivery.md)：逐 epoch 保存、PA/OA 独立恢复与回收语义。
- [`epoch_checkpoint_reclaim_plan.md`](architecture/epoch_checkpoint_reclaim_plan.md)：D25 回收方案及其实施边界。

### `delivery/`：阶段交付、批次快照与验收材料

- P1.4：[`p1_4_review.md`](delivery/p1_4_review.md)。
- P2.0：[`p2_0a_delivery.md`](delivery/p2_0a_delivery.md)、[`p2_0a_review.md`](delivery/p2_0a_review.md)、
  [`p2_0b_delivery.md`](delivery/p2_0b_delivery.md)、[`p2_0c_delivery.md`](delivery/p2_0c_delivery.md)、
  [`p2_0c_review.md`](delivery/p2_0c_review.md)、[`p2_0e_delivery.md`](delivery/p2_0e_delivery.md)。
- P2.1：[`p2_1_b1_snapshot.md`](delivery/p2_1_b1_snapshot.md)、[`p2_1_b2_snapshot.md`](delivery/p2_1_b2_snapshot.md)、
  [`p2_1_b3a_snapshot.md`](delivery/p2_1_b3a_snapshot.md)、[`p2_1_b3ab_snapshot.md`](delivery/p2_1_b3ab_snapshot.md)、
  [`p2_1_b4_snapshot.md`](delivery/p2_1_b4_snapshot.md)、[`p2_1_b5_review_checklist.md`](delivery/p2_1_b5_review_checklist.md)、
  [`p2_1_handoff_checklist.md`](delivery/p2_1_handoff_checklist.md)。
- P2.2：[`p2_2_delivery.md`](delivery/p2_2_delivery.md)、[`p2_2_review.md`](delivery/p2_2_review.md)、
  [`p21_workbook_review_20261004.md`](delivery/p21_workbook_review_20261004.md)。

### `admission/`：候选方法、CNN、资源和准入证据

- [`p3_preintegration_handoff.md`](admission/p3_preintegration_handoff.md)：P3 预集成交接及正式准入边界。
- [`p3_pulda_puet_admission_spec_draft.md`](admission/p3_pulda_puet_admission_spec_draft.md)：PULDA/PUET 准入规格草案。
- [`pn_oracle_integration.md`](admission/pn_oracle_integration.md)：PN oracle 接入边界；端到端 CNN baseline 的实现与正式纳入分开记录。
- [`vpu_sampling_audit.md`](admission/vpu_sampling_audit.md)：VPU 采样假设、视图和 PA 选模边界。
- [`gradpu_source_review_20261005.md`](admission/gradpu_source_review_20261005.md)：GradPU 来源与协议复核准备。
- [`candidate_cnn_storage_review_20261005.md`](admission/candidate_cnn_storage_review_20261005.md)：CNN 存储工程证据；不等于正式预算或 P3.3 验收。

### `reviews/`：独立复核和阶段性工程记录

- [`pilot_independent_review_20261004.md`](reviews/pilot_independent_review_20261004.md)：Pilot 的代理技术复核，不代替签署。
- [`heng958_independent_review.md`](reviews/heng958_independent_review.md)：HENG958 负责范围的历史复核记录。
- [`collaborator_followup_20261004.md`](reviews/collaborator_followup_20261004.md)：数据收集、CNN oracle、校准和交付边界的补全记录。
- [`independent_progress_20261005.md`](reviews/independent_progress_20261005.md)：独立工程预集成和合成数据证据；不改变冻结矩阵或正式准入。

### `data/` 与 `assets/`

`data/` 保存机器可读索引、摘要、准入草稿和资源探针；`assets/` 保存协议配图。
原始数据、645 行交付表和执行机分析目录不入库，按交付文档中的制品索引和访问约定处理。

## 当前阅读入口

1. 先读 [`survey_execution_plan.md`](survey_execution_plan.md) 了解任务主责和未完成项。
2. 需要协议口径时读 [`pu_survey_protocol.md`](pu_survey_protocol.md)。
3. 需要交付证据、候选准入或独立复核时进入对应子目录，不在本索引重复复制结论。
## 新增工程证据（2026-10-07 合并）

- [五方法准入证据](p3_admission_evidence_20261005.md)、[公开结果对照准备](p3_public_comparison_20261005.md)。
- [阶段选模与预算草案](p3_stage_budget_review_20261007.md)。
- Holistic-PU：[来源与 CNN](holistic_source_cnn_review_20261007.md)、[后段初始化](holistic_stage_initialization_20261007.md)、[LZO 选择](holistic_lzo_selection_20261007.md)。
- [PULNS 来源与 CNN](pulns_source_cnn_review_20261007.md)、[GEN-PU 来源与 CNN](genpu_source_cnn_review_20261007.md)。
- [扩展 CNN 资源探针说明](extended_cnn_resource_review_20261007.md)：工具已交付，21 次实测尚未交付有效记录。

以上均为工程准备；正式资源预算、配方、数值验收和负责人签署仍按执行计划推进。
