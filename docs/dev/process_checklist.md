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
- **跑批子集选择（P2.0f，2026-09-27）**：`run_survey_pilot.py` 新增 `--datasets`（逗号分隔白名单，**默认仍为全量**，只收窄不放宽）与 `--plan-json`（须配 `--dry-run`，把展开后的计划落成 JSON）。过滤施加在**矩阵**而非计划上，且在任何读取之前生效——协议按 `--protocol` 加载并先收窄，splits 的摘要与先验读取也只覆盖本分片的数据集（按数据集目录遍历，未纳入的目录根本不会被打开），故未选数据集内自相矛盾的先验不会阻断本分片（两个分片各自准备、各自持有，谁都不该为不跑的数据停摆）——于是计划、容量估算、续跑判定与批次一起收窄；未知数据集名直接报错并列出矩阵实际规划的集合（打错字母若静默产出空计划并退出 0，另一台主机会被误认为已覆盖）；过滤生效时收尾汇总自报范围。用途不限于跨主机拆分——批次按 `(dataset, …)` 字典序执行、**cifar10 排在最前**，故不经此机制无法「先跑最便宜的数据集把链路验通再上大的」。口径取自 PUBench `sweep.py` 与 PU-Bench `run_train.py`（两者均为 argv 多值白名单，且均无分片机制）。（PR #79）
- **PA 正式选模准则（R9）**（2026-09-20）：proxy accuracy（Wang et al. 2026 Def. 1 OS 分支）落地，π 必传，阈值网格与 OA 同构。见 [p2_0a_review R9 补记](../research/pu_survey/p2_0a_review.md)。
- **P2.0b 标签语义门禁 + P2.0c 对照预注册**：工程完成、合作者签署待办。见 [p2_0b 交付](../research/pu_survey/p2_0b_delivery.md)、[p2_0c 交付](../research/pu_survey/p2_0c_delivery.md)。
- **TS-OS 校准接入训练执行链（P2.0e，2026-09-23）**：训练视图默认由方法台账 `native_sampling_assumption` 推导、`--os-or-ts` 可覆盖；`nnpu` 已接线（逐 mini-batch `D_U^k ← D_U^k ∪ D_P^k`）、`upu` 已接线（无标签风险项的集合与 RBF 中心候选池跟随视图，中心数量不随视图变化）、`pusb_kernel` 已接线（只校准内部 CV 训练折与终拟合；**验证折固定保持 OS**，属协议驱动的跨目标选参，见 D17；中心池与阈值池本就不随视图变化）、`dist_pu` 已接线（只把分布对齐与熵的角色集合换成 `X_U ∪ X_P`；**对齐目标仍是总体 π**，正例项、Mixup 物理池与 RNG 序列不变，见 D18）、`self_pu` 已接线（只把负 PU 项的**角色集合**由「未信任 U 行」换成「未信任 U 行 ∪ 正例批」并按行数质量混合；**α_P 不是 π**，trusted 人口与 pace、伪标签、meta 矩阵、consistency、验证/选模与 checkpoint 一概不随视图变化，见 D19），逐 run 实际视图入 manifest 并成为公平性分组维度；五个适用方法**无剩余待接线项**；线性 `pusb` 经 2026-09-27 裁决为**不适用校准**（其训练信号直接把 U 当作负类，不存在同形的无标签损失输入，且不在 Pilot 矩阵内，见 D16 ①，该完成口径措辞未随之调整）。见 [执行计划 P2.0e](../research/pu_survey/survey_execution_plan.md)、[实验层关键设计机制](experiment_layer.md)。

### 未发布（随下一版本发布）

- **PN oracle 接入**：`CleanLabelGenerator` + 视图/声明双向 fail-loud + `--oracle` 入口。见 [pn_oracle_integration](../research/pu_survey/pn_oracle_integration.md)。
- **双架构阶段 0-2**：Registry 4 能力字段、build_encoder 导出、nnpu encoder 试点、CNN 提示文案回归、契约路线 B 收口。见 [dual_architecture_plan §5](dual_architecture_plan.md)。
- **Survey 语义统一（issue #42）**：PUSB 拆为 `pusb`/`pusb_kernel`，先验门禁改由 registry 驱动，台账↔registry 一致性契约。见 [survey_execution_plan](../research/pu_survey/survey_execution_plan.md)。
- **Self-PU 声明收口（issue #38/#45）**：`input_ndims` 恢复 `{2,4}`，非原生 CNN 由 `native_architectures` 承载。
- **训练视图分层（P2.0e 后续）**：角色构造下沉到中立核心层 `pu_toolbox/core/training_views.py`（`TrainingView` + `build_training_view`：P / 原始 U / 损失 U 三角色与来源索引，不认识台账·方法名·manifest），实验层只留政策与历史——台账门禁、路由裁决、旧 `run_view` 词表在出口现算、manifest 构造。依赖方向固定 `estimators → core`、禁止 `estimators → experiment`（实测反向导入会拉起 28 个 estimator 模块），由子进程导入边界测试守住。所有权契约：三组 positions 与 `source_indices` 自有并冻结，`source_features`/`source_labels` 借用不复制且只读。**不迁移任何生产 estimator**——本轮实测各方法角色构造重复量仅一行到数行，生产迁移推迟到首个真实待接入算法。见 [P2.0e 交付 §10](../research/pu_survey/p2_0e_delivery.md)、[架构 §2.1](architecture.md)。
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
