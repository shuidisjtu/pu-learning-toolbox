# 进度清单

> 实际执行顺序与原始路线图有调整：优先实现 PU 特有的风险估计方法（工具箱核心差异化能力），经典分类器包装器后移。
> 阶段定义以本文档为准，`development_roadmap.md` 为高层路线图。
> **Method Card 为可选文档**，新算法接入不要求必写。

## Phase 0 — 项目骨架 ✅

- [x] pyproject.toml + 包骨架 + 分层依赖
- [x] Core 基类
- [x] labels.py, validation.py, exceptions.py, config.py, tags.py, random.py
- [x] Registry 系统（15 api_only 占位）
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
- [ ] paper-like benchmark 配置

## Phase 4 — 推荐与诊断 (v0.4)

- [ ] Data Profiler、SCAR/SAR 假设提示
- [ ] 算法推荐器（规划中）
- [ ] 诊断报告
- [ ] 类先验与标记倾向敏感性分析

## Phase 5 — SAR / Selection-Biased PU (v0.5)

- [ ] SAR / selection bias 数据模拟器
- [x] PUSB、LBE；Centroid Estimation 已完成；LLSVM 已完成
- [ ] SCAR vs SAR 对比 benchmark

## Phase 6 — 深度 PU (v0.6)

- [ ] Self-PU；~~Dist-PU~~ ✅
- [ ] InfoMax PU、Contrastive PU、DGPU（research extension）

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
