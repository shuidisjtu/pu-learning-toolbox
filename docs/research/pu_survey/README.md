# PU Survey 文档索引

本目录只保留长期有效的协议、任务、交付、准入和复核材料；阶段性排序已合并到
[执行计划](survey_execution_plan.md)，不再单独维护后续优先级文档。

## 权威源

| 文档 / 数据 | 定位 |
|---|---|
| [`pu_survey_protocol.md`](pu_survey_protocol.md) | Survey 协议要求：数据集、四角色输入、OS/TS 视图、PA/OA 选模、PN oracle 与主榜门禁 |
| [`survey_execution_plan.md`](survey_execution_plan.md) | 能力与缺口真相源：现在能跑什么、缺什么、预算/资源口径、执行边界与决策摘要 |
| `pu_toolbox/experiment/method_ledger.json` | 方法台账真相源：方法身份、来源、视图、校准、适配和待复核状态 |
| `pu_toolbox/experiment/survey_protocol_v2.json` | 机器可读执行协议，当前 `survey-v2.0`；`review_items` 登记本次修订的接受/待复核状态 |
| `pu_toolbox/experiment/survey_protocol_v1.json` | 冻结归档的 `survey-v1.2` 矩阵，供带日期的 P3 证据与历史对照矩阵绑定 |
| `pu_toolbox/experiment/survey_comparison_v4.json` | 机器可读对照矩阵，当前 `survey-comparison-v4`（`v1`–`v3` 保留为历史绑定） |

> 代码级事实以源码、测试和提交历史为准；本目录文档负责记录协议边界、交付证据和人工复核状态，
> 不重复抄录提交过程或参考文献全文。

## 文档分区

### `architecture/`：长期有效的制品与 checkpoint 设计

- [`artifact_storage_architecture.md`](architecture/artifact_storage_architecture.md)：制品分级、规模推演与迁移路径。
- [`epoch_checkpoint_delivery.md`](architecture/epoch_checkpoint_delivery.md)：逐 epoch 保存、PA/OA 独立恢复与回收语义。
- [`epoch_checkpoint_reclaim_plan.md`](architecture/epoch_checkpoint_reclaim_plan.md)：D25 回收方案及其实施边界。

### `delivery/`：阶段交付、批次快照与验收材料

- P4.1-A/P2.2 最新推进：[`p41_p22_progress_20261016.md`](reviews/p41_p22_progress_20261016.md)，
  正式 CI 条件接线、原始制品分批接收、来源候选补记与 RV3–RV7 边界；不代替配方冻结/独立重放/本人签署。
- 正式 registry CI：[`p4_1_production_ci_20261016.md`](delivery/p4_1_production_ci_20261016.md)，
  当前文件缺失时 skipped，真实冻结时须改为无条件必需存在。
- P2.2 AutoDL 制品核对：[`p2_2_autodl_check_20261016.md`](reviews/p2_2_autodl_check_20261016.md)，归档摘要、PA/OA 选中权重与逐 epoch 记录；B2 退出码仍未找到。
- P2.2 C2/C3 裁决：[`p2_2_c2_c3_ruling_20261016.md`](reviews/p2_2_c2_c3_ruling_20261016.md)，PA 准则锁定、54 条阻断映射解除（矩阵 v3 暂不更新）。
- P2.1/P2.2新修正包接收：[`p21_correction_receiving_20261015.md`](reviews/p21_correction_receiving_20261015.md)，三数据页保持、独立文件身份与说明审计；原训练制品/签署仍待。
- Issue #41阶段B：[`label_semantics_stage_b_20261015.md`](delivery/label_semantics_stage_b_20261015.md)，公共入口门禁/展示/ADR；2026-10-16 用户确认 RV2 完成，Issue #41 可关闭。
- P1.4：[`p1_4_review.md`](delivery/p1_4_review.md)。
- P1.4-R当前接收验收：[`p1_4r_receiving_acceptance_20261015.md`](reviews/p1_4r_receiving_acceptance_20261015.md)，五数据集25split/125文件/5原归档技术通过；人工复核与已知限制保留。
- P1.4-R 独立重建诊断：[`split_rebuild_diagnostic_20261015.md`](delivery/split_rebuild_diagnostic_20261015.md)，平台字节归因不替代原收据/归档验收。
- P1.4-R 完整独立重建（接收前历史记录）：[`p1_4r_full_rebuild_20261015.md`](reviews/p1_4r_full_rebuild_20261015.md)，原交付现已接收，当前状态以上方接收验收为准。
- P1.4-R 早期部分复核（历史截点）：[`p1_4r_independent_rebuild_20261015.md`](reviews/p1_4r_independent_rebuild_20261015.md)。
- P2.0：[`p2_0a_delivery.md`](delivery/p2_0a_delivery.md)、[`p2_0a_review.md`](delivery/p2_0a_review.md)、
  [`p2_0b_delivery.md`](delivery/p2_0b_delivery.md)、[`p2_0c_delivery.md`](delivery/p2_0c_delivery.md)、
  [`p2_0c_review.md`](delivery/p2_0c_review.md)、[`p2_0e_delivery.md`](delivery/p2_0e_delivery.md)。
- P2.1：[`p2_1_b1_snapshot.md`](delivery/p2_1_b1_snapshot.md)、[`p2_1_b2_snapshot.md`](delivery/p2_1_b2_snapshot.md)、
  [`p2_1_b3a_snapshot.md`](delivery/p2_1_b3a_snapshot.md)、[`p2_1_b3ab_snapshot.md`](delivery/p2_1_b3ab_snapshot.md)、
  [`p2_1_b4_snapshot.md`](delivery/p2_1_b4_snapshot.md)、[`p2_1_b5_review_checklist.md`](delivery/p2_1_b5_review_checklist.md)、
  [`p2_1_handoff_checklist.md`](delivery/p2_1_handoff_checklist.md)。
- P2.2：[`p2_2_delivery.md`](delivery/p2_2_delivery.md)、[`p2_2_review.md`](delivery/p2_2_review.md)、
  [`p21_workbook_review_20261004.md`](delivery/p21_workbook_review_20261004.md)。
- P4.1-A：[`p4_1_runtime_binding_20261014.md`](delivery/p4_1_runtime_binding_20261014.md)：训练前绑定、pilot 摘要固定/续跑与 draft CI；不是正式配方冻结。
  [`p4_1_pool_binding_20261014.md`](delivery/p4_1_pool_binding_20261014.md)：版本化候选池、PA/OA 独立候选与 trajectory 身份；不批准新参数搜索。
- LBE 独立神经变体：[`lbe_neural_integration_20261014.md`](delivery/lbe_neural_integration_20261014.md)，不替换冻结 linear `lbe`。
- 独立 clean-support 制品与 CPU 入口：[`clean_support_workflow_20261017.md`](delivery/clean_support_workflow_20261017.md)：五数据集 train 内分拆、五角色收据、剩余 train-only 预处理与 Self-PU/LaGAM/PULNS 单候选 OA 回放；不批准正式预算。
- [Clean-support checkpoint 双轨](delivery/clean_support_checkpoints_20261022.md)：Self-PU/LaGAM 的 meta-active 与 PULNS 的 episode 快照、显式 PA+clean-support/OA、容量预检和选后回收；默认终点 OA 保留，非正式准入。
- [五数据集真实 support 制品](delivery/clean_support_materialization_20261022.md)：本机 25 份五角色制品、逐 raw 映射/剩余 train 预处理核验、三方法真实 Spambase CPU 工程链；Git 仅登记清单，不含 NPZ/原始数据，不等于正式预算或 GPU 结果。
- [GEN-PU 单候选工程 recipe](delivery/genpu_recipe_20261023.md)：#126 四项工程规格、实际 PA/OA manifest、真实 CIFAR 预固定小子集与全宽 ResNet-18 的 batch8/256 GPU 峰值/回放；正式 v3/预算及全量验收仍待，不自动关 issue。

### `admission/`：候选方法、CNN、资源和准入证据

- [线性/神经 LBE 临时成对实验](admission/lbe_comparison_exploration_20261016.md)：
  已有四路二维分区、同标记、终点或显式 EM-round PA/OA 选择，独立探索性结果，不改旧 Pilot。
- [`vpu_schedule_review_20261015.md`](admission/vpu_schedule_review_20261015.md)：VPU 可选源 Adam 重置调度与固定轮内训练步数，不改变默认或 D22 准则。
- [`vpu_encoder_stream_20261015.md`](delivery/vpu_encoder_stream_20261015.md)：VPU 注入 CNN 与 CPU 数据分批工程路径，非作者图像复现/正式准入。
- [`cvir_source_cnn_review_20261015.md`](admission/cvir_source_cnn_review_20261015.md)：CVIR 历史来源/图像复核。2026-10-22 已补[独立校准与先验接线](delivery/cvir_calibration_20261022.md)：全局训练 U∪P 先排序后截断、按实际视图换算先验与 PA/OA 工程选模；正式 v3/配方预算、真实数据/GPU 仍待。
- [`lagam_source_cnn_review_20261015.md`](admission/lagam_source_cnn_review_20261015.md)：LaGAM 虚拟范围/CNN/分批与阶段成本；2026-10-17 #115 已将 registry/台账采样定位纠正为 OS，support 标签预算与 PA 限制仍待审。
- [LaGAM 全文与实现交叉复核](admission/lagam_paper_review_20261020.md)：补齐原文获取；登记零梯度、β/簇数来源分歧与 support 子类配额，绑定数值行为测试；保留默认实现及未签署/未准入状态。

- [`p3_preintegration_handoff.md`](admission/p3_preintegration_handoff.md)：P3 预集成交接及正式准入边界。
- [`p3_pulda_puet_admission_spec_draft.md`](admission/p3_pulda_puet_admission_spec_draft.md)：PULDA/PUET 准入规格草案。
- [`pn_oracle_integration.md`](admission/pn_oracle_integration.md)：PN oracle 接入边界；端到端 CNN baseline 的实现与正式纳入分开记录。
- [`vpu_sampling_audit.md`](admission/vpu_sampling_audit.md)：VPU 采样假设、视图和 PA 选模边界。
- [`vpu_d21_d22_review_20261018.md`](admission/vpu_d21_d22_review_20261018.md)：VPU D21/D22 只读复核；D21 成立，D22 裁决不变；“与上游一致”的验证 P 项采样表述已更正为已登记差异；不改代码、不新增准入。
- [`gradpu_source_review_20261005.md`](admission/gradpu_source_review_20261005.md)：GradPU 来源与协议复核准备。
- [`kldce_source_behavior_review_20261019.md`](admission/kldce_source_behavior_review_20261019.md)：KLDCE 生产 SMO 与论文附录差异、作者代码来源状态核查；不代表方法负责人签署。
- [Self-PU 生命周期、参数与评分补核](admission/selfpu_lifecycle_20261020.md)：失败重训/概率拟合检查、有限参数/整数预算早失败、逐子模块模式恢复及 OS/TS × 一致性范围 × 选择路径回归。
- [Dist-PU 显式 encoder 与 CPU-stream](admission/distpu_encoder_stream_20261020.md)：staged/histogram opt-in 图像训练、分批界限、CV/模板隔离、旧 MLP 与冻结 feature-adapter 兼容；正式规格和资源仍待。
- [`candidate_cnn_storage_review_20261005.md`](admission/candidate_cnn_storage_review_20261005.md)：CNN 存储工程证据；不等于正式预算或 P3.3 验收。

### `reviews/`：独立复核和阶段性工程记录

- [20261021 任务交接复核回复](reviews/heng958_task_review_20261022.md)：v2_p4/v2_p10、#122、RV5/RV6 技术意见、issue 收尾边界与 PN-CNN 5 seed 暂缓；不代替本人签署或修改冻结协议。
- [`pilot_independent_review_20261004.md`](reviews/pilot_independent_review_20261004.md)：Pilot 的代理技术复核，不代替签署。
- [`heng958_independent_review.md`](reviews/heng958_independent_review.md)：HENG958 负责范围的历史复核记录。
- [`collaborator_followup_20261004.md`](reviews/collaborator_followup_20261004.md)：数据收集、CNN oracle、校准和交付边界的补全记录。
- [`independent_progress_20261005.md`](reviews/independent_progress_20261005.md)：独立工程预集成和合成数据证据；不改变冻结矩阵或正式准入。
- [`p3_1_acceptance_review_20261016.md`](reviews/p3_1_acceptance_review_20261016.md)：P3.1 逐方法技术验收复核（2026-10-18 汇总、2026-10-19 P3MIX 全文接入补记）；#114–#117 与新增方法结论。
- [`p3_1_v2_delta_20261021.md`](reviews/p3_1_v2_delta_20261021.md)：13 方法在 survey-v2.0 下的 delta 表（旧结论 → v2 影响 → 新处置）与 #119–#126 阶段 C 处置记录；技术复核建议，不代替 RV8 签署。
- [`p3mix_paper_review_20261019.md`](admission/p3mix_paper_review_20261019.md)：P3MIX 全文逐项核对、实现映射及未解除的正式准入前置。
- [`rv6_ts_calibration_technical_review_20261019.md`](reviews/rv6_ts_calibration_technical_review_20261019.md)：五方法 TS 校准作用范围与不变量核验；工程结论不代替 HENG958 方法学签署。
- [`rv8_method_admission_20261021.md`](reviews/rv8_method_admission_20261021.md)：P3.1 RV8 14 条新增方法准入证据复核；shuidisjtu 签署，14 条均为有条件接受，不构成正式准入。
- [`additional_ts_calibration_review_20261019.md`](reviews/additional_ts_calibration_review_20261019.md)：PULDA、PUET、GradPU 的 TS 风险角色和回归证据补核；不构成来源重放或负责人签署。
- [`additional_ts_calibration_review_20261019b.md`](reviews/additional_ts_calibration_review_20261019b.md)：Robust-PU、Split-PU、GEN-PU 的阶段/角色级 TS 校准范围；保留各自来源与协议待审项。
- [`infomax_source_ambiguity_review_20261019.md`](admission/infomax_source_ambiguity_review_20261019.md)：InfoMax 类别分组、batch size 与 KM 变体的论文来源核对；临时配置值不冒充论文事实。
- [`selfpu_consistency_scope_review_20261019.md`](admission/selfpu_consistency_scope_review_20261019.md)：Self-PU 论文式 (13)/(16) 定义域核对及显式 full-meta consistency 适配；默认冻结消融路径不变。
- [`selfpu_source_pin_review_20261019.md`](admission/selfpu_source_pin_review_20261019.md)：官方仓库/MIT 核实、来源 commit pin 与 clean-room 实现边界。
- [`selfpu_source_behavior_review_20261019.md`](admission/selfpu_source_behavior_review_20261019.md)：固定官方 commit 的 meta/test 数据流核查；区分上游源码行为、论文定义与工具箱隔离协议。
- [`algorithm_issues_114_117_20261017.md`](reviews/algorithm_issues_114_117_20261017.md)：#114–#117 工程整改、验证证据与独立实验调用边界。
- [`algorithm_completion_20261017.md`](reviews/algorithm_completion_20261017.md)：新增算法收尾状态、fail-closed 预算与正式准入边界。
- [`puet_adaptation_completion_20261017.md`](reviews/puet_adaptation_completion_20261017.md)：PUET 风险质量证据、适配回归与 `max_features` 越界登记；不改变训练代码或正式准入。

### `data/` 与 `assets/`

`data/` 保存机器可读索引、摘要、准入草稿和资源探针；`assets/` 保存协议配图。
原始数据、645 行交付表和执行机分析目录不入库，按交付文档中的制品索引和访问约定处理。

## 当前阅读入口

当前交接：[20261021 任务复核回复](reviews/heng958_task_review_20261022.md)；
先处理五项 review 与 #126/#88/#122 收尾，再推进 P3.2，具体排序只看执行计划。
PN-CNN 的 5 seed 实验按用户指令暂缓至协议定稿，不取消其他任务的五 seed 要求。

[PULDA / GradPU #119/#120 配方决策](admission/candidate_image_owner_decisions_20261018.md)及
[GradPU #121 调度/batch 声明](admission/gradpu_training_recipe_review_20261018.md)保留作证据，
不再因历史“待回复”文字重开已收尾项；
[Split-PU #122 JSD 分母与 mini-batch 反例](admission/splitpu_122_characterization_20261018.md)仍有待决事项。

1. 先读 [`survey_execution_plan.md`](survey_execution_plan.md) 了解现在能跑什么、缺什么。
2. 需要协议口径时读 [`pu_survey_protocol.md`](pu_survey_protocol.md)。
3. 需要交付证据、候选准入或独立复核时进入对应子目录，不在本索引重复复制结论。
## 新增工程证据（2026-10-07 合并）

- [五方法准入证据](p3_admission_evidence_20261005.md)、[公开结果对照准备](p3_public_comparison_20261005.md)。
- [阶段选模与预算草案](p3_stage_budget_review_20261007.md)。
- Holistic-PU：[来源与 CNN](holistic_source_cnn_review_20261007.md)、[后段初始化](holistic_stage_initialization_20261007.md)、[LZO 选择](holistic_lzo_selection_20261007.md)。
- [PULNS 来源与 CNN](pulns_source_cnn_review_20261007.md)、[GEN-PU 来源与 CNN](genpu_source_cnn_review_20261007.md)。
- [扩展 CNN 资源探针说明](extended_cnn_resource_review_20261007.md)：工具已交付，21 次实测尚未交付有效记录。
- 2026-10-17 独立 BBE/TEDⁿ：[BBE](../method_cards/BBE.md)、[TEDn](../method_cards/TEDn.md)；native TS、显式 α_U，不接 CVIR/OS Survey。
- 2026-10-18 [五候选方法失败重训状态整改](reviews/candidate_fit_failure_20261018.md)：异常/取消后禁止旧或部分模型公开推理，成功训练与校准口径不变。
- 2026-10-18 [PAN 来源与推理模式补证](admission/pan_source_inference_review_20261018.md)：保持论文目标/OS，补九文件与实际交替梯度，仅修推理的逐层模式恢复。
- 2026-10-18 [RP 来源与实际 OOF 补证](admission/rp_source_review_20261018.md)：四文件双摘要、剪枝/重权/实际拟合隔离与成本；修常数概率的均值舍入 NaN，不改数学算法/默认或正式准入。

以上均为工程准备；正式资源预算、配方、数值验收和负责人签署仍按执行计划推进。
