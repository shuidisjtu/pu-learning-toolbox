# 路线图

## 总体策略

**framework-first**：先完成稳定框架与 API 契约，用 mock estimator 跑通链路，再逐个集成论文算法。当前 16 个方法均已完成 clean-room 核心实现（NATIVE），后续重点是官方数据、历史环境和 paper-like benchmark。

```
框架与 API 契约 → Registry + metadata → 核心 PU 风险估计
    → 经典包装器 + 补充估计 → benchmark
    → SAR / Selection-Biased PU → 深度 PU
```

## 版本路线

```
0.1.0  核心 PU 风险估计（Elkan-Noto, uPU, nnPU, ReCPE, PNU, LDCE, penL1）
0.2.0  经典包装器 + 补充类先验估计（KM1/KM2）
0.3.0  Benchmark + LLSVM + PUSB + LBE + Dist-PU
0.4.0  推荐器 + 诊断报告 + 敏感性分析
0.5.0  SAR / selection-biased PU
0.6.0  Self-PU, Dist-PU, InfoMax PU, WConPU, DGPU
1.0.0  API 稳定
```

## 阶段叙事

**Phase 0 — 项目骨架 ✅**：pyproject、包骨架、Core 基类、labels/validation、Registry 系统与 api_only 占位跑通链路。

**Phase 1 — 核心 PU 风险估计 (v0.1) ✅**：Elkan-Noto、uPU、nnPU、ReCPE、PNU 与 LDCE 完成，构成工具箱核心差异化能力。

**Phase 2 — 经典包装器与补充估计 (v0.2) ✅**：penL1/KM1/KM2 类先验估计器、算法推荐器、class-prior 敏感性分析完成；PU Bagging、Biased SVM、Weighted LR、TIcE、AlphaMax 明确划为 v1 范围外。

**Phase 3 — Benchmark + 集成 (v0.3) ✅**：PNU、PUSB、LBE、LLSVM native 实现完成；paper-like benchmark 配置建立。

**Phase 4 — 推荐与诊断 (v0.4) ✅**：Data Profiler、SCAR/SAR 假设提示、算法推荐器、诊断报告、敏感性分析完成。

**Phase 5 — SAR / Selection-Biased PU (v0.5) ✅**：SAR/selection-bias 数据模拟器、PUSB、LBE、LLSVM 与 SCAR vs SAR 对比 benchmark 完成。

**Phase 6 — 深度 PU (v0.6) 进行中**：Self-PU 核心、Dist-PU、InfoMax PU、WConPU、DGPU 核心接口与统一 runner 完成（含 clean-room 多 seed、Fashion-MNIST official-data smoke）。剩余：未公开视觉/validation 指标细节、授权数据与 DGPU EDM paper-like 全量运行。

## 进度明细

任务粒度的完成状态以 [process_checklist.md](../project_management/process_checklist.md) 为权威来源，本文档只保留阶段叙事与版本路线，不再逐条重复。
