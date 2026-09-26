# 进度清单

> 总体策略：**framework-first**——先完成稳定框架与 API 契约，用 mock estimator 跑通链路，再逐个集成论文算法。当前 21 个注册方法均已完成 clean-room 核心实现（NATIVE）；另有一个隔离的联合漂移 research 求解器。新接入的 GradPU、PUET、VPU、PULDA 仍属实验性子集，未完成 Survey P3.1/P3.2 正式验收。
> **Method Card 为可选文档**，新算法接入不要求必写。

## 阶段历史（已闭环）

Phase 0-9 已闭环（框架 → 核心风险估计 → 机制 → 推荐诊断 → SAR → 深度 → 分布漂移 → 联合漂移）；逐条明细与批次历史见 git log。

## 未完成项

- [ ] Phase 3 官方数据/历史环境全量运行（依赖外部官方数据与历史环境，非工具箱缺口）
- [ ] Phase 6 WConPU 官方视觉 + DGPU EDM paper-like 全量（依赖 CUDA/授权数据）
- [ ] InfoMax 未公开类别分组、batch size 与 KM 变体核对
- ⚠️ v1 范围外：Phase 2 三经典包装器 + TIcE/AlphaMax 类先验估计

## 发布状态

### 未发布（Survey P2 前置，不放行正式 P2.1）

- **P2.0a 工程交付**（2026-09-17 已签署验收）：版本化执行矩阵、runner 强制消费与参数锁、CIFAR adapter/native 接线、逐 epoch checkpoint 保存与 PA/OA 独立恢复。见 [交付记录](../research/pu_survey/p2_0a_delivery.md)、[复核包](../research/pu_survey/p2_0a_review.md)。
- **P1.2/P1.4 数据制品**（2026-09-19 重建）：split manifest 新增 `provenance` 块（来源/版本/引用/许可逐条按官方原文）；三数据集 × 5 seed 统一重建，60 npz 逐字节不变；切分流水线效率修复与跨 seed 文本嵌入一致性（PR #63/#64）。索引见 [split_artifacts_index](../research/pu_survey/data/split_artifacts_index.json)。
- **split 制品跨机传输**：`survey_splits_archive.py` pack/verify 双向校验（逐文件摘要 + 从 `.npz` 重算索引摘要），载体定为网盘带外传（2026-09-20 已发送，索引先于发送入库）。
- **全 pilot 跑批驱动与磁盘预算**：`run_survey_pilot.py` 编排 645 次运行，按 manifest 判定已完成、半完成单元按 seed 拆分；**磁盘容量以代码输出为准**——累计 `348,443,368,000 B`（≈324.5 GiB）、跑前门禁 `18,874,368,000 B`（≈17.58 GiB），旧值与换算过程见 [p2_0a_delivery §6](../research/pu_survey/p2_0a_delivery.md) 与执行计划决策 D15。
- **PA 正式选模准则（R9）**（2026-09-20）：proxy accuracy（Wang et al. 2026 Def. 1 OS 分支）落地，π 必传，阈值网格与 OA 同构。见 [p2_0a_review R9 补记](../research/pu_survey/p2_0a_review.md)。
- **P2.0b 标签语义门禁 + P2.0c 对照预注册**：工程完成、合作者签署待办。见 [p2_0b 交付](../research/pu_survey/p2_0b_delivery.md)、[p2_0c 交付](../research/pu_survey/p2_0c_delivery.md)。
- **TS-OS 校准接入训练执行链（P2.0e，2026-09-23）**：训练视图默认由方法台账 `native_sampling_assumption` 推导、`--os-or-ts` 可覆盖；`nnpu` 已接线（逐 mini-batch `D_U^k ← D_U^k ∪ D_P^k`）、`upu` 已接线（无标签风险项的集合与 RBF 中心候选池跟随视图，中心数量不随视图变化），逐 run 实际视图入 manifest 并成为公平性分组维度；剩余 `pusb_kernel`、`dist_pu`、`self_pu` 三个适用 Pilot 方法待各自独立设计；线性 `pusb` 待适用性裁决（其训练信号直接把 U 当作负类，不存在同形的无标签损失输入，且不在 Pilot 矩阵内，见 D16）。见 [执行计划 P2.0e](../research/pu_survey/survey_execution_plan.md)、[实验层关键设计机制](experiment_layer.md)。

### 未发布（随下一版本发布）

- **PN oracle 接入**：`CleanLabelGenerator` + 视图/声明双向 fail-loud + `--oracle` 入口。见 [pn_oracle_integration](../research/pu_survey/pn_oracle_integration.md)。
- **双架构阶段 0-2**：Registry 4 能力字段、build_encoder 导出、nnpu encoder 试点、CNN 提示文案回归、契约路线 B 收口。见 [dual_architecture_plan §5](dual_architecture_plan.md)。
- **Survey 语义统一（issue #42）**：PUSB 拆为 `pusb`/`pusb_kernel`，先验门禁改由 registry 驱动，台账↔registry 一致性契约。见 [survey_execution_plan](../research/pu_survey/survey_execution_plan.md)。
- **Self-PU 声明收口（issue #38/#45）**：`input_ndims` 恢复 `{2,4}`，非原生 CNN 由 `native_architectures` 承载。
- **SAR-OA 执行路径（issue #43）**：`--labeling-mechanism` 与 `--method` 正交，SAR 强制 OA-only。见 [协议 §2.3](../research/pu_survey/pu_survey_protocol.md)、[D8](../research/pu_survey/survey_execution_plan.md)。

### 已发布版本

- 1.11.0（2026-08-29）：pu-workflow skill 扩展场景 + NaN/Inf 拒绝 + 最低版本要求升至 1.10.0
- 1.10.0（2026-08-29）：传统 PU 调优收尾（KLDCE b₀ 修复、契约 v2、六轮写回，ADR-0016 闭环）
- 1.9.0（2026-08-27）：七方法传统 PU benchmark + AP/balanced-accuracy/Brier/ECE 指标 + KLDCE 原生 SMO
- 1.8.0（2026-08-21）：联合漂移研究求解器 + shift-monitor/review CLI + UI 部署面板
- 1.7.0（2026-08-21）：配对漂移适配 + 窗口告警 + 不确定性/主动复核
- 1.6.0（2026-08-21）：分布漂移审计 + 协变量加权 + shift-audit CLI
- 1.5.1（2026-08-16）：CNN 序列化、PUTuner 坏参数隔离、UI 历史持久化
- 1.5.0（2026-08-15）：classifier_params + PUTuner + Streamlit UI

### 收尾统计

- **算法**：21 个已注册方法，全部 native 实现
- **质量门禁**：8 道（test_quality / doc_links / project_metadata / math_rendering / api_docs / skill_sync / baseline_configs / format）
- **v1 范围外**：Phase 2 三个经典包装器 + TIcE/AlphaMax 类先验估计
- **依赖外部**：Phase 3 官方历史环境、WConPU CUDA/授权数据、DGPU EDM/CelebA 全量运行

历史执行记录见 git log；关键决策见 [`docs/adr/`](../adr/)。
