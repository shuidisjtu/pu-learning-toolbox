# Development Roadmap

## 总体策略

**framework-first**：先完成稳定框架与 API 契约，用 mock estimator 跑通链路，再逐个集成论文算法。当前 15 个方法均已完成 clean-room 核心实现（NATIVE），后续重点是官方数据、历史环境和 paper-like benchmark。

```
框架与 API 契约 → Registry + metadata → 核心 PU 风险估计
    → 经典包装器 + 补充估计 → benchmark
    → SAR / Selection-Biased PU → 深度 PU
```

## Phase 0 — 项目骨架 (已完成)

pyproject.toml、包骨架、Core 基类、labels/validation、Registry 系统、15 api_only 占位、67 条测试（Phase 0 完成时）。

## Phase 1 — 核心 PU 风险估计 (v0.1) ✅

> 实际优先实现的模块，构成工具箱核心差异化能力。

- ~~Elkan-Noto~~ ✅ — 经典 PU 校准基线
- ~~uPU / Convex PU~~ ✅ — 无偏 PU 风险估计
- ~~nnPU~~ ✅ — 非负 PU 风险估计
- ~~ReCPE~~ ✅ — 类先验估计
- ~~PNU~~ ✅ — 半监督 PU 扩展
- PU splitters、基础 metrics ✅
- minimal examples ✅

## Phase 2 — 经典包装器与补充估计 (v0.2)

> 原 Phase 1 剩余部分：经典分类器的 PU 包装 + 额外类先验估计器。

- PU Bagging、Biased SVM、Weighted LR（v1 范围外）
- ~~penL1~~ ✅、TIcE / AlphaMax 类先验估计（v1 范围外）
- ~~算法推荐器~~ ✅
- ~~class-prior sensitivity analysis~~ ✅

## Phase 3 — Benchmark + 集成 (v0.3)

- ~~PNU~~ ✅ (native)、~~PUSB~~ ✅ (native)、~~LBE~~ ✅ (native)、~~LLSVM~~ ✅ (native)
- paper-like benchmark 配置

## Phase 4 — 推荐与诊断 (v0.4)

- Data Profiler、SCAR/SAR 假设提示 ✅
- ~~算法推荐器~~ ✅
- 诊断报告 ✅
- 类先验与标记倾向敏感性分析 ✅

## Phase 5 — SAR / Selection-Biased PU (v0.5)

- SAR / selection bias 数据模拟器 ✅
- ~~PUSB~~ ✅、~~LBE~~ ✅、~~LLSVM~~ ✅
- SCAR vs SAR 对比 benchmark ✅
- [x] Centroid Estimation (LDCE) — 提前完成，已移入 Phase 1

## Phase 6 — 深度 PU (v0.6)

- ~~Self-PU Method Card、接口与核心实现~~ ✅（paper-like benchmark 待完成）
- ~~Dist-PU~~ ✅
- ~~InfoMax PU、WConPU、DGPU 核心接口、统一 runner、配置锁和 clean-room 多 seed~~ ✅
- 公开官方数据执行层、可重复 PU split、resume/provenance、Fashion-MNIST smoke 与资源审计 ✅
- InfoMax PU 论文网络、优化器与 20-seed paper-protocol 配置 ✅
- InfoMax PU 独立 validation split、KM1/KM2 类先验接口与 runner 集成 ✅
- 精确视觉 backbone、历史 split、授权数据与 DGPU EDM paper-like 全量运行待完成

## 工作包分解

| WP | 内容 | 优先级 | 依赖 |
|---|---|---|---|
| WP0 | 项目骨架、CI、文档 | P0 | — |
| WP1 | Base API、标签规范、数据校验 | P0 | WP0 |
| WP2 | 算法注册表、metadata | P0 | WP1 |
| WP3 | LLSVM 实现 ✅ | P1 | WP2 |
| WP4 | PU 数据切分与 CV ✅ | P0 | WP1 |
| WP5 | Elkan-Noto ✅、uPU ✅、nnPU ✅、ReCPE ✅ | P0 | WP1 |
| WP6 | PNU ✅、LDCE ✅ | P0 | WP5 |
| WP7 | PU Bagging、Biased SVM、Weighted LR（v1 范围外） | P3 | WP1 |
| WP8 | penL1 ✅ / KM1/KM2 ✅ / TIcE / AlphaMax 类先验估计（TIcE/AlphaMax v1 范围外） | P1 | WP1 |
| WP9 | Metrics 与 diagnostics | P1 | WP1, WP8 |
| WP10 | 算法推荐器 ✅ | P1 | WP2, WP9 |
| WP11 | paper-like benchmark | P1 | WP6 |
| WP12 | SAR 抽象、selection bias 模拟器 | P2 | WP1, WP8 |
| WP13 | PUSB ✅ / LBE ✅ / LLSVM ✅ | P2 | WP11, WP12 |
| WP14 | Self-PU / Dist-PU ✅ | P2 | WP6, WP11 |
| WP15 | InfoMax / WConPU / DGPU 核心接口、runner、配置锁、clean-room 与 Fashion-MNIST official-data smoke ✅；精确视觉/EDM 运行待完成 | P3 | WP14 |

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
