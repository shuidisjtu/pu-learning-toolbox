# Development Roadmap

## 总体策略

**framework-first + source-aware integration**：先完成稳定框架与 API 契约，用 mock estimator 跑通链路，再逐个集成论文算法。允许先不实现全部方法，但不允许没有清晰接口和元数据占位。

```
框架与 API 契约 → Registry + metadata → 经典 PU MVP
    → 类先验估计 + 风险估计 → 源码 adapter + benchmark
    → SAR / Selection-Biased PU → 深度 PU
```

## Phase 0 — 项目骨架 (已完成)

pyproject.toml、包骨架、Core 基类、labels/validation、Registry 系统、15 api_only 占位、67 tests。

## Phase 1 — 经典 PU MVP (v0.1)

- penL1 / L1 类先验估计
- ~~Elkan-Noto~~ ✅、PU Bagging、Biased SVM、Weighted LR
- TIcE / AlphaMax
- PU splitters、基础 metrics
- minimal examples、Advisor 规则版

## Phase 2 — 类先验与风险估计 (v0.2)

- ReCPE wrapper
- uPU ✅ / nnPU / PNU loss
- class-prior sensitivity analysis

## Phase 3 — 源码 Adapter (v0.3)

- nnPU、PNU、PUSB、LBE、LLSVM adapter
- adapter smoke test、paper-like benchmark 配置

## Phase 4 — 推荐器与诊断 (v0.4)

- Data Profiler、SCAR/SAR 假设提示
- 算法推荐器、诊断报告
- 类先验与标记倾向敏感性分析

## Phase 5 — SAR / Selection-Biased PU (v0.5)

- SAR / selection bias 数据模拟器
- PUSB、LBE、Centroid Estimation、LLSVM
- SCAR vs SAR 对比 benchmark

## Phase 6 — 深度 PU (v0.6)

- Self-PU、Dist-PU
- InfoMax PU、Contrastive PU、DGPU（research extension）

## 工作包分解

| WP | 内容 | 优先级 | 依赖 |
|---|---|---|---|
| WP0 | 项目骨架、CI、文档 | P0 | — |
| WP1 | Base API、标签规范、数据校验 | P0 | WP0 |
| WP2 | 算法注册表、metadata | P0 | WP1 |
| WP3 | Source adapter 接口 | P0 | WP2 |
| WP4 | PU 数据切分与 CV | P0 | WP1 |
| WP5 | Elkan-Noto ✅、PU Bagging、Weighted LR、Biased SVM | P0 | WP1 |
| WP6 | 类先验估计 penL1/TIcE/AlphaMax/ReCPE | P0 | WP1 |
| WP7 | Metrics 与 diagnostics | P1 | WP1, WP6 |
| WP8 | Advisor 推荐器 | P1 | WP2, WP7 |
| WP9 | uPU / nnPU / PNU | P1 | WP6 |
| WP10 | 官方源码 adapter + benchmark | P1 | WP3, WP9 |
| WP11 | SAR 抽象、selection bias 模拟器 | P2 | WP1, WP6 |
| WP12 | PUSB / LBE / Centroid / LLSVM | P2 | WP10, WP11 |
| WP13 | Self-PU / Dist-PU | P2 | WP9, WP10 |
| WP14 | InfoMax / Contrastive / DGPU | P3 | WP13 |

## 版本路线

```
0.1.0  经典 PU + 类先验 + registry 占位
0.2.0  uPU / nnPU / PNU 风险模块
0.3.0  官方源码 adapter + paper-like benchmark
0.4.0  推荐器 + 诊断报告
0.5.0  SAR / selection-biased PU
0.6.0  Self-PU, Dist-PU
1.0.0  API 稳定
```
