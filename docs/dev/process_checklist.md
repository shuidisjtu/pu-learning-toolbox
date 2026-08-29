# 进度清单

> 总体策略：**framework-first**——先完成稳定框架与 API 契约，用 mock estimator 跑通链路，再逐个集成论文算法。当前 17 个注册方法均已完成 clean-room 核心实现（NATIVE）；另有一个隔离的联合漂移 research 求解器，后续重点是官方数据、历史环境和 paper-like benchmark。
> 实际执行顺序与原始路线图有调整：优先实现 PU 特有的风险估计方法（工具箱核心差异化能力），经典分类器包装器后移。
> **Method Card 为可选文档**，新算法接入不要求必写。

## 阶段历史（已闭环）

- Phase 0 ✅ 项目骨架：pyproject + Core 基类 + Registry（初始 15 个 api_only 占位，现已按实现状态升级）+ 测试框架
- Phase 1 ✅ 核心 PU 风险估计：Elkan-Noto / uPU / nnPU / ReCPE / PNU / PU splitters / metrics / minimal examples
- Phase 2 ✅ 部分：penL1 类先验与算法推荐器完成；三经典包装器与 TIcE/AlphaMax 列 v1 范围外
- Phase 3 ✅ 机制就绪：PUSB benchmark 全链路（来源锁/manifest/shard 聚合/断点续跑/审计器）；官方数据全量运行依赖外部，见下
- Phase 4 ✅ 推荐与诊断：Data Profiler / SCAR-SAR 提示 / 推荐器 / 诊断报告 / 敏感性分析
- Phase 5 ✅ SAR：数据模拟器、PUSB/LBE/Centroid/LLSVM 接口与 SCAR vs SAR 对比 benchmark
- Phase 6 ✅ 深度 PU 大部分：Self-PU / Dist-PU / InfoMax PU / WConPU / DGPU 全链路（clean-room 多 seed、Fashion-MNIST 3-seed smoke、InfoMax 暂定协议 20 seeds）；剩余见下
- Phase 7 ✅ 分布漂移感知 PU 第一版：OOF 域审计、有界边际相对权重、ESS/覆盖门禁、
  `ShiftAwarePUPipeline`、`shift-audit` CLI 与三类产物；联合漂移动态训练列后续研究
- Phase 8 ✅ 分布漂移决策扩展：配对加权对照与 `shift-run`、窗口监控、双域先验/标记
  机制分解、不确定性拒绝/主动复核；新增研究级类别条件联合漂移近似求解器
- Phase 9 ✅ 联合漂移研究执行层：论文式动态共享特征目标、四类对照与五类消融、公开表格
  数据多 seed/CI 协议、双域先验 bootstrap，以及部署监控 CLI/UI

> 逐条明细与批次历史见 git log(压缩前旧路径:git log --all -- docs/project_management/process_checklist.md)。

## 未完成项

- [ ] Phase 3 官方数据/历史环境全量运行（依赖外部官方数据与历史环境提供，非工具箱缺口）
- [ ] Phase 6 WConPU 官方视觉 + DGPU EDM paper-like 全量（依赖 CUDA/授权数据）
- [ ] InfoMax 未公开类别分组、batch size 与 KM 变体核对
- ⚠️ v1 范围外：Phase 2 三经典包装器 + TIcE/AlphaMax 类先验估计

## 发布状态 (v1.11.0)

- **版本**: `1.11.0`（2026-08-29：pu-workflow skill 更新——新增可选扩展场景
  （漂移迁移 `shift-audit`/`shift-run`、部署监控 `shift-monitor`/`review`、基准审计
  `audit-benchmark`，各带强制检查点）、输入契约补充 NaN/Inf 拒绝、技能最低版本
  要求升至 `pu-toolbox >= 1.10.0`）
- **版本**: `1.10.0`（2026-08-29：传统 PU 第一次调优收尾——KLDCE b₀ 类对称修复
  （低先验全负根因）、契约 v2 基线重跑与 KLDCE 调优轮重跑（r3，默认参数即有效
  工作点）、Elkan-Noto 调优轮重跑（r2，`mode=weighted_retraining` 12/12 全单元
  confirmed）；第 6 步写回三轮全部落地——LDCE 组合默认（v4）、uPU squared（v5）、
  elkan_noto weighted_retraining（v6），每轮重锁基线 + 确认种子重跑 +
  companion 逐单元审计，当前对齐基线 baseline_v6，ADR-0016 闭环）
- **版本**: `1.9.0`（2026-08-27：新增七方法传统 PU 可复现 benchmark、锁定基线、
  数据泄露预检、断点续跑、配对统计比较与七轮调优证据（ADR-0016 verdict 留档）；
  新增 AP、balanced accuracy、Brier score、ECE 指标及概率可用性契约；KLDCE
  改为原生 SMO 内层求解并修复收敛诊断，LDCE 默认迭代上限提升；修正 Elkan–Noto
  等非零原生阈值模型的 PU 零一风险语义）
- **版本**: `1.8.0`（2026-08-21：新增 AISTATS 2025 联合漂移 PU clean-room 动态目标、
  对照/消融与公开数据 benchmark、双域 bootstrap 区间，以及 `shift-monitor`/`review` CLI
  和 UI 部署面板）
- **版本**: `1.7.0`（2026-08-21：新增配对漂移适配比较、窗口告警历史、双域 PU
  假设分析、不确定性/主动复核，以及明确标为 research 的联合漂移近似求解器）
- **版本**: `1.6.0`（2026-08-21：新增分布漂移审计、协变量加权 PU 工作流、
  `PUPipeline.sample_weight` 严格传递契约和 `shift-audit` CLI）
- **版本**: `1.5.1`（2026-08-16：验收修复——CNN 模型序列化、PUTuner 坏参数隔离、
  UI 运行历史持久化）
- **版本**: `1.5.0`（2026-08-15：新增 `classifier_params` 与 CLI
  `--classifier-param`，支持按注册名调整模型；新增 PU-aware `PUTuner`，搜索阶段仅做
  CV 并只重训最佳候选；新增 Streamlit 图形界面，支持数据上传、模型配置、参数搜索、
  指标与诊断展示，以及报告、预测和模型下载）
- **算法**: 17 个已注册方法，全部 native 实现
- **质量门禁**: 7 道（test_quality / doc_links / project_metadata / math_rendering / skill_sync / baseline_configs / format）
- **v1 范围外**: Phase 2 三个经典包装器与 TIcE/AlphaMax 类先验估计
- **依赖外部**: Phase 3 官方历史环境，以及 WConPU CUDA/授权数据和 DGPU EDM/CelebA
  全量运行；InfoMax 暂定 Fashion-MNIST 20-seed 协议已执行

历史执行记录见 git log；关键决策见 [`docs/adr/`](../adr/)。
