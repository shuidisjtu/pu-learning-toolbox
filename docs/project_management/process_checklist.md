# 进度清单

> 实际执行顺序与原始路线图有调整：优先实现 PU 特有的风险估计方法（工具箱核心差异化能力），经典分类器包装器后移。
> 阶段定义以本文档为准，`docs/dev/roadmap.md` 为高层路线图。
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
- [x] 算法推荐器（recommend_methods / recommend_from_profile）

## Phase 3 — Benchmark + 集成 (v0.3)

- [x] PUSB、LBE、Dist-PU、LLSVM native interfaces
- [x] 前五篇负责论文的 paper-like 复现实验规格
- [x] 前五篇统一 benchmark runner、官方来源/配置锁和 clean-room 多 seed 结果
- [x] 前五篇 official preflight（源码、数据、历史环境、CUDA/MATLAB 与 toolbox 实现差距分轴审计）
- [x] PUSB official-aligned RBF 适配器、IJCNN1 数据校验与缩小网格 smoke
- [x] PUSB IJCNN1 仓库扩展：`pi=0.2` 完整网格 3 seeds × 3 U、uLSIF 与断点续跑
- [x] PUSB 论文 Table 2 六数据集来源/hash/loader 锁定与官方采样可行性审计
- [x] PUSB 严格完整单元/官方兼容策略、精确计划、checkpoint/resume 与 provenance runner
- [x] PUSB strict plan：45 单元/4500 trials、45 shards、精确聚合与配对 95% CI 报告
- [ ] 官方数据、历史环境和 paper-like 全量运行（工具箱侧机制已就绪：preflight 审计、执行层、数据/源码锁定与 blocker 诊断；全量运行依赖外部官方数据与历史环境（Dist-PU 需 Py3.7/numpy1.19/MATLAB 等），官方数据不内置工具箱，由执行方提供 —— 非工具箱缺口）

## Phase 4 — 推荐与诊断 (v0.4)

- [x] Data Profiler、SCAR/SAR 假设提示
- [x] 算法推荐器（recommend_methods / recommend_from_profile）
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
- [x] WConPU NCHW、13-layer CNN/ResNet、SimAugment/RandAugment 与 cosine scheduler 接入
- [x] WConPU clean 10% validation、二维 loss-weight grid、候选级 resume 与最优参数 refit
- [ ] InfoMax PU、WConPU、DGPU 官方视觉/文本与 EDM paper-like 全量运行

## 发布状态 (v1.0.0)

- **版本**: `1.0.0`（首个正式版，未发布过 dev 版）
- **算法**: 17 个已注册方法，全部 native 实现
- **测试**: 811 passed（unit + integration = PR 快层；e2e + slow = nightly 顶层）
- **质量门禁**: 6 道（test_quality / doc_links / project_metadata / math_rendering / skill_sync / format）
- **v1 范围外**: Phase 2 三个经典包装器与 TIcE/AlphaMax 类先验估计
- **依赖外部**: Phase 3/6 官方数据与历史环境全量运行（非工具箱缺口，由执行方提供）

历史执行记录见 git log；关键决策见 `docs/project_management/decision_log.md`。
