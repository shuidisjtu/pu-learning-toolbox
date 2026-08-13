# 进度清单

> 实际执行顺序与原始路线图有调整：优先实现 PU 特有的风险估计方法（工具箱核心差异化能力），经典分类器包装器后移。
> 阶段定义以本文档为准，`docs/dev/roadmap.md` 为高层路线图。
> **Method Card 为可选文档**，新算法接入不要求必写。

## 阶段历史（已闭环）

- Phase 0 ✅ 项目骨架：pyproject + Core 基类 + Registry（初始 15 个 api_only 占位，现已按实现状态升级）+ 测试框架
- Phase 1 ✅ 核心 PU 风险估计：Elkan-Noto / uPU / nnPU / ReCPE / PNU / PU splitters / metrics / minimal examples
- Phase 2 ✅ 部分：penL1 类先验与算法推荐器完成；三经典包装器与 TIcE/AlphaMax 列 v1 范围外
- Phase 3 ✅ 机制就绪：PUSB benchmark 全链路（来源锁/manifest/shard 聚合/断点续跑/审计器）；官方数据全量运行依赖外部，见下
- Phase 4 ✅ 推荐与诊断：Data Profiler / SCAR-SAR 提示 / 推荐器 / 诊断报告 / 敏感性分析
- Phase 5 ✅ SAR：数据模拟器、PUSB/LBE/Centroid/LLSVM 接口与 SCAR vs SAR 对比 benchmark
- Phase 6 ✅ 深度 PU 大部分：Self-PU / Dist-PU / InfoMax PU / WConPU / DGPU 全链路（clean-room 多 seed、Fashion-MNIST 3-seed smoke、InfoMax 暂定协议 20 seeds）；剩余见下

> 逐条明细与批次历史见 git log(压缩前旧路径:git log --all -- docs/project_management/process_checklist.md)。

## 未完成项

- [ ] Phase 3 官方数据/历史环境全量运行（依赖外部官方数据与历史环境提供，非工具箱缺口）
- [ ] Phase 6 WConPU 官方视觉 + DGPU EDM paper-like 全量（依赖 CUDA/授权数据）
- [ ] InfoMax 未公开类别分组、batch size 与 KM 变体核对
- ⚠️ v1 范围外：Phase 2 三经典包装器 + TIcE/AlphaMax 类先验估计

## 发布状态 (v1.3.0)

- **版本**: `1.3.0`（2026-08-10：类先验估计用户角度体验修复——`make_sar_dataset` 默认 SAR 警示、CLI 报告估计可靠性上下文（估计器名/边界注/Assumption Notes）、demo 分离度 1.0、list-priors 别名分组、`--prior-param` 非法值前置拦截、NaN 友好报错、推荐器不再对非识别 at_risk 信号提升 SAR 方法、`is_scar_plausible` 改名 `is_observed_dependence_absent`）
- **算法**: 17 个已注册方法，全部 native 实现
- **质量门禁**: 6 道（test_quality / doc_links / project_metadata / math_rendering / skill_sync / format）
- **v1 范围外**: Phase 2 三个经典包装器与 TIcE/AlphaMax 类先验估计
- **依赖外部**: Phase 3 官方历史环境，以及 WConPU CUDA/授权数据和 DGPU EDM/CelebA
  全量运行；InfoMax 暂定 Fashion-MNIST 20-seed 协议已执行

历史执行记录见 git log；关键决策见 [`docs/adr/`](../adr/)。
