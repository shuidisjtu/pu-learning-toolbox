# 进度清单

> 实际执行顺序与原始路线图有调整：优先实现 PU 特有的风险估计方法（工具箱核心差异化能力），经典分类器包装器后移。
> 阶段定义以本文档为准，`development_roadmap.md` 为高层路线图。
> **Method Card 为可选文档**，新算法接入不要求必写。

## Phase 0 — 项目骨架 ✅

- [x] pyproject.toml + 包骨架 + 分层依赖
- [x] Core 基类
- [x] labels.py, validation.py, exceptions.py, config.py, tags.py, random.py
- [x] Registry 系统（Phase 0 初始建立 15 个 api_only 占位，现已按实现状态升级）
- [x] 测试框架（测试套件通过）

## Phase 1 — 核心 PU 风险估计 (v0.1)

> 实际优先实现的模块：经典校准 → 无偏风险 → 非负风险 → 类先验估计。
> 这些是工具箱的核心差异化能力，也是后续深度方法（Dist-PU, Self-PU 等）的前置依赖。

- [x] **Elkan-Noto** — 经典 PU 校准基线 (Elkan & Noto, 2008)
- [x] **uPU / Convex PU** — 无偏 PU 风险估计 (du Plessis et al., 2015)
- [x] **nnPU** — 非负 PU 风险估计 (Kiryo et al., 2017)
- [x] **ReCPE** — 类先验估计 (Yao et al., 2022)
- [x] **PNU** — 半监督 PU 扩展 (Sakai et al., 2017)
- [x] PU splitters（`PUStratifiedKFold` 等）
- [x] 基础 metrics（AUC, F1, PU estimated risk）
- [x] minimal examples（`examples/minimal/`）

## Phase 2 — 经典包装器与补充估计 (v0.2)

> 原 Phase 1 剩余部分。经典分类器的 PU 包装 (Bagging, SVM, LR) 和额外的类先验估计器。

- [ ] PU Bagging 分类器 ⚠️ v1 范围外
- [ ] Biased SVM 分类器 ⚠️ v1 范围外
- [ ] Weighted Logistic Regression 分类器 ⚠️ v1 范围外
- [x] penL1 类先验估计
- [ ] TIcE / AlphaMax 类先验估计 ⚠️ v1 范围外
- [ ] 算法推荐器（规划中，非当前范围） ⚠️ v1 范围外

## Phase 3 — Benchmark + 集成 (v0.3)

- [x] PUSB、LBE、Dist-PU、LLSVM native interfaces
- [x] 前五篇负责论文的 paper-like 复现实验规格
- [x] 前五篇统一 benchmark runner、官方来源/配置锁和 clean-room 多 seed 结果
- [ ] 官方数据、历史环境和 paper-like 全量运行

## Phase 4 — 推荐与诊断 (v0.4)

- [x] Data Profiler、SCAR/SAR 假设提示
- [ ] 算法推荐器（规划中）
- [x] 诊断报告
- [x] 类先验与标记倾向敏感性分析

## Phase 5 — SAR / Selection-Biased PU (v0.5)

- [x] SAR / selection bias 数据模拟器
- [x] PUSB、LBE、Centroid Estimation 和 LLSVM 接口/实现
- [x] SCAR vs SAR 对比 benchmark（PUSB/LBE，3 mechanisms × 10 seeds）

## Phase 6 — 深度 PU (v0.6)

- [x] Self-PU Method Card
- [x] Self-PU 接口与实现
- [x] Dist-PU
- [x] InfoMax PU、WConPU、DGPU Method Card、核心接口与 registry
- [x] InfoMax PU、WConPU、DGPU 统一 runner、论文配置锁及 clean-room 多 seed benchmark
- [x] 公开官方数据加载、确定性 PU split、断点续跑、provenance 与 Fashion-MNIST 3-seed smoke
- [x] 完整论文配置运行前审计（GPU、EDM、授权数据、实现差距分别报告）
- [x] InfoMax PU 论文网络协议（mini-batch PURL、BN/ReLU、gradient noise、300×3 nnPU、Adam/AdaGrad）
- [x] InfoMax PU 独立 validation split、KM1/KM2 类先验估计与结果误差记录
- [ ] InfoMax PU、WConPU、DGPU 官方视觉/文本与 EDM paper-like 全量运行

## 最近完成记录

| 日期 | 方法 | 状态 | 代码与文档 | 验证 |
|---|---|---|---|---|
| 2026-07-16 | 测试架构重构 | 分层架构 + 权威级别 | `tests/contract/`；`tests/unit/losses/`；pytest markers | `241 passed`；ruff 检查通过 |
| 2026-07-16 | nnPU / Non-Negative PU | 已完成 native 实现 | `pu_toolbox/losses/nnpu.py`；`pu_toolbox/estimators/risk/nnpu.py`；`docs/research/method_cards/nnpu.md` | `149 passed`；ruff 检查通过 |
| 2026-07-16 | ReCPE / Regrouping CPE | 已完成 native 实现 | `pu_toolbox/prior/recpe.py`；`docs/research/method_cards/ReCPE.md` | `133 passed`；ruff 检查通过 |
| 2026-07-13 | uPU / Convex PU | 已完成 native 实现 | `pu_toolbox/losses/upu.py`；`pu_toolbox/estimators/risk/upu.py`；`docs/research/method_cards/Convex_Formulation_for_PU_DATA_Learning.md` | `130 passed`；ruff 检查通过 |
| 2026-07-18 | PNU | 已完成 native 实现 | `pu_toolbox/losses/pnu.py`；`pu_toolbox/estimators/risk/pnu.py`；`pu_toolbox/utils/basis.py`；`docs/research/method_cards/PNU.md` | `270 passed`；ruff 检查通过 |
| 2026-07-10 | Elkan-Noto | 已完成 native 实现 | `pu_toolbox/estimators/classic/elkan_noto.py`；`docs/research/method_cards/Elkan_Noto.md` | `102 passed`；ruff 检查通过 |
| 2026-07-21 | LDCE | 已完成 native 实现 | `pu_toolbox/estimators/risk/ldce.py`；`docs/research/method_cards/LDCE.md` | `311 passed`；ruff + 质量门禁通过 |
| 2026-07-21 | KLDCE | 已完成 native 实现（QP oracle 版，RBF kernel） | `pu_toolbox/estimators/risk/kldce.py`；`pu_toolbox/utils/centroid.py`；`docs/research/method_cards/KLDCE.md` | `357 passed`；ruff + 质量门禁通过 |
| 2026-07-21 | penL1 / Dist-PU / PUSB / LBE | 已完成统一接口与核心实现 | `pu_toolbox/prior/pen_l1.py`；`pu_toolbox/estimators/risk/dist_pu.py`；`pu_toolbox/estimators/bias_aware/{pusb,lbe}.py`；对应 Method Cards | 新增方法测试与 registry/contract 测试通过；ruff 新增文件通过 |
| 2026-07-21 | 前五篇 Method Card 深化 | 按 KLDCE/PNU/nnPU 结构补齐论文信息、假设、符号、公式、算法、API、测试与复现风险 | `class_prior_estimation.md`、`ReCPE.md`、`Dist-PU.md`、`PUSB.md`、`LBE.md` | 文档结构和代码落点已核对；PUSB/LBE/Dist-PU 的完整 paper-like benchmark 仍待完成 |
| 2026-07-23 | LLSVM | 已完成 native 实现 | `pu_toolbox/losses/llsvm.py`；`pu_toolbox/estimators/classic/llsvm.py`；`docs/research/method_cards/LLSVM.md` | 新增 loss + estimator 测试通过；ruff + 质量门禁通过 |
| 2026-07-27 | Self-PU Method Card | 已完成严谨 Method Card，接口与实现待开始 | `docs/research/method_cards/Self-PU.md` | 已记录论文公式、三阶段流程、API 规格、测试验收标准与复现风险 |
| 2026-07-27 | InfoMax PU / WConPU / DGPU | 已完成严谨 Method Card、clean-room 核心接口与 registry 集成 | `estimators/deep/`；对应 Method Cards | `501 passed`；新增文件 ruff 与格式检查通过；DGPU paper-like 运行需 EDM backend |
| 2026-07-27 | 前五篇复现实验规格 | 已补充 CPE、ReCPE、Dist-PU、PUSB、LBE 的数据、调参、对照、统计和验收协议 | 五份对应 Method Card | 文档规格完成；benchmark runner、官方配置锁定和实际多 seed 实验仍待完成 |
| 2026-07-27 | 前五篇 benchmark 落地 | 统一 JSON runner、5 份官方配置、4 个不可变来源锁；完成 5 方法 × 5 seeds | `benchmarks/assigned_methods/` | 25 trials；新增 runner tests 通过；完整官方数据/GPU/历史环境运行仍待完成 |
| 2026-07-30 | SAR 模拟与对比 | 公共 propensity/labels/dataset API；SCAR、线性 SAR、非线性 SAR 配对 benchmark | `preprocessing/selection_bias.py`；`benchmarks/assigned_methods/` | 22 个 simulator cases；60 个实际 trials；`536 passed`；ruff、文档链接与测试质量检查通过 |
| 2026-07-31 | Data Profiler | 结构化质量报告、问题级别/行动建议、SCAR/SAR 可识别性提示与审计模式 | `preprocessing/data_profiler.py`；`docs/user/data_profiling.md` | 12 个 profiler cases；`548 passed`；ruff、文档与测试质量门禁通过 |
| 2026-08-01 | 诊断报告 | 数据/模型/指标组合报告；证据级别；严格 JSON/Markdown 输出 | `diagnostics/report.py`；`docs/user/diagnostic_reports.md` | 13 个 report cases；`561 passed`；ruff、文档与测试质量门禁通过 |
| 2026-08-01 | 兼容性与项目治理 | 真实 Python matrix、全目录质量门禁、Hatchling 边界、wheel 安装冒烟、贡献/PR 规范与状态对齐 | `pyproject.toml`；`.github/`；`CONTRIBUTING.md`；`docs/development_compatibility.md` | `562 passed`；104 个 Python 文件 lint/format 通过；wheel/sdist 构建及隔离安装通过 |
| 2026-08-03 | 类先验与标记倾向敏感性 | 固定输出假设扫描、观测恒等式相容性、指标区间及 JSON/Markdown/CSV 导出 | `diagnostics/sensitivity.py`；`docs/user/sensitivity_analysis.md` | 13 个 sensitivity cases；测试、ruff、文档与测试质量门禁通过 |
| 2026-08-03 | Self-PU 核心实现 | 动态 trusted set、meta reweight、双 student/EMA teacher 蒸馏、checkpoint 与 registry | `estimators/deep/self_pu.py`；`docs/user/self_pu.md`；Self-PU Method Card | 13 个专项 cases + 统一 contract；完整 paper-like benchmark 待完成 |
| 2026-08-03 | 深度 PU benchmark | InfoMax PU/WConPU/DGPU 统一 runner、三份论文配置锁、Gaussian generator 和实际多 seed 结果 | `benchmarks/deep_pu/` | 3 methods × 3 seeds = 9 trials；13 个 runner cases；官方视觉/EDM 运行待完成 |
| 2026-08-04 | 深度 PU 官方数据执行层 | 公开数据加载、确定性 case-control split、resume 防混写、原始数据/split/配置/代码哈希与完整配置 preflight | `benchmarks/deep_pu/official_data.py`；`preflight_paper.py`；official-data 配置与结果 | Fashion-MNIST 3 seeds 已执行；ROC-AUC `0.4420 ± 0.0874`；`623 passed`；完整视觉/EDM 仍按 blocker 清单推进 |
| 2026-08-04 | InfoMax PU 论文网络与先验链路 | PURL mini-batch、BN/ReLU、gradient noise；300×3 nnPU；独立 validation；KM1/KM2 与先验误差审计 | `estimators/deep/infomax_pu.py`；`prior/kernel_mean.py`；InfoMax official-data protocol 配置 | 定向 `29 passed`；20-seed 配置待 preflight；未公开 class split、batch size 和 KM 变体仍待核对 |
