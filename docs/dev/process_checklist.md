# 进度清单

> 总体策略：**framework-first**——先完成稳定框架与 API 契约，用 mock estimator 跑通链路，再逐个集成论文算法。当前全部注册方法均有 clean-room 核心实现（NATIVE）；另有一个隔离的联合漂移 research 求解器。实现可运行与论文数值复现、研究结果审核分别判断；Survey 的当前能力与实验状态见 [Survey README](../research/pu_survey/README.md)。
> **Method Card 为可选文档**，新算法接入不要求必写。

## 阶段历史（已闭环）

Phase 0-9 已闭环（框架 → 核心风险估计 → 机制 → 推荐诊断 → SAR → 深度 → 分布漂移 → 联合漂移）；逐条明细与批次历史见 git log。

## 未完成项

- [ ] Phase 3 官方数据/历史环境全量运行（依赖外部官方数据与历史环境，非工具箱缺口）
- [ ] Phase 6 WConPU 官方视觉 + DGPU EDM paper-like 全量（依赖 CUDA/授权数据）
- [x] InfoMax 类别分组、batch size 与 KM 变体来源核对：原文仅说明 MNIST/Fashion-MNIST 各分成两组、表示学习按 4:1 mini-batch 交替更新，并采用 KM；未给类别 ID、数值 batch size 或 KM1/KM2。现配置 `[0,1,2,3,4]`、256、KM1 均保留为 provisional 工程选择，结果 `paper_claim=false`；详见[来源审计](../research/method_cards/InfoMax-PU.md)。
- ⚠️ v1 范围外：Phase 2 三经典包装器 + TIcE/AlphaMax 类先验估计

## 发布状态

### 未发布（PU 研究能力整合）

- [x] P0 语义基线：固定四篇论文/两个参考库来源；核对采样、先验估计池与逐实现风险消费；
  实际执行 S1–S3 最小复现，形成 [ADR-0021](../adr/0021-pu-research-semantics.md)。
- [x] P1 公共知识：采样、先验、PA/OA、表征四篇概念页，SCAR/SAR 政策与来源对照，
  用户/开发导航；新增的 7 个 Python 文档片段实际执行通过。
- [x] P2-S1：同池定额入口 `make_fixed_count_labels`，旧名字兼容包装与 2.0 迁移计划；
  独立 TS 高斯生成器记录来源、总体参数和经验先验。
- [x] P2-S2：固定 Gaussian 旧场景，混合 U 教学路径使用新生成器；
  [示例 15](../../examples/minimal/15_pu_sampling.py)、[采样教程](../user/howto/pu_sampling.md)与 API 同步。
- [x] P2-S3：先验目标声明、ReCPE 复制流程、Pipeline 的解析与消费记录，连同 OS/TS 画像及指标回归。
  每折按实际标记比例换算训练参数；PUSBKernel 分别解析训练表与风险池先验；报告 schema 为 1.1。
- [x] P3：固定正例配额与均匀抽样单源；Pipeline/CVIR 共享先验换算；registry 消费目标声明。
  跨入口标签、全选后的随机流、元数据及层间依赖回归通过。
- [x] P4：示例 14–18 覆盖采样、SAR 政策、先验传递、独立 PA/OA、encoder 与完整 DIY；
  新增 encoder 教程，补齐可组合研究路径，CPU 实际运行和 31 项教学/快照回归通过。
- [ ] P5：协议与报告设置摘要收口。

S1/S2 已按失败回归驱动完成实现；旧同池入口保留输出与随机轨迹。
文档按有证据支持的结论和适用条件直接陈述。专项与适用静态门禁结果见
[语义基线的验证记录](pu_research_semantics.md#7-本批验证记录2026-11-06)。

### 未发布（实验层收敛）

实验层保留公共四角色 API、独立 PA/OA 选模、当前 Survey 配方执行和结果报告生成能力。
训练产物及结果审核由研究团队维护；历史试点协议、队列、交接和阶段审批记录退出工具箱。
当前能力与实验状态见 [Survey README](../research/pu_survey/README.md)，
设计与扩展方式见[实验层设计](experiment_layer.md)，操作链见
[运行指南](../research/pu_survey/runbook.md)。

发布前检查：

- 公共 API 不读取 Survey 配置或项目历史文档；自定义策略仍可注入。
- 数据角色隔离、视图/先验换算、独立 support 预算与 PA/OA 选模约束仍生效。
- 结果读取拒绝损坏选择制品，并保留失败、缺 seed 和缺指标；均值/样本标准差及成本分账正确。
- 工作簿、诊断卡与结论可由现行入口生成；文献事实与实验观测分别记录。
- 运行适用的测试和质量门禁；不把未执行的 GPU、外部数据或结果审核报告为完成。

### 未发布（随下一版本发布）

- **PN oracle 接入**：`CleanLabelGenerator` + 视图/声明双向 fail-loud + `--oracle` 入口。见[实验层设计 §2](experiment_layer.md#2-当前能力与边界)。
- **双架构阶段 0-2**：Registry 4 能力字段、build_encoder 导出、nnpu encoder 试点、CNN 提示文案回归、契约路线 B 收口。见 [dual_architecture_plan §5](dual_architecture_plan.md)。
- **Survey 语义统一（issue #42）**：PUSB 拆为 `pusb`/`pusb_kernel`，先验门禁改由 registry 驱动，台账↔registry 一致性契约。采样视图见[实验协议 §2.3](../research/pu_survey/pu_survey_protocol.md#23-pu-数据集生成)，先验语义见[实验协议 §3](../research/pu_survey/pu_survey_protocol.md#3-class-prior-以及相关参数设置)。
- **Self-PU 声明收口（issue #38/#45）**：`input_ndims` 恢复 `{2,4}`，非原生 CNN 由 `native_architectures` 承载。
- **训练视图分层**：`core/training_views.py` 承载中立角色构造，实验层负责视图路由、先验语义与 manifest；estimator 不依赖 experiment。核心视图拥有并冻结角色位置与来源索引，借用的特征和标签只读。依赖方向见[架构 §2.1](architecture.md)。
- **SAR-OA 执行路径（issue #43）**：`--labeling-mechanism` 与 `--method` 正交，SAR 强制 OA-only。见[实验协议](../research/pu_survey/pu_survey_protocol.md)。

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

- **算法**：全部注册方法为 native 实现（当前集合见 registry / `pu-toolbox list-methods`）
- **质量门禁**：清单见 [README](../../README.md) 的 Quality gates 与 CI 的 quality job
- **v1 范围外**：Phase 2 三个经典包装器 + TIcE/AlphaMax 类先验估计
- **依赖外部**：Phase 3 官方历史环境、WConPU CUDA/授权数据、DGPU EDM/CelebA 全量运行

历史执行记录见 git log；关键决策见 [`docs/adr/`](../adr/)。
