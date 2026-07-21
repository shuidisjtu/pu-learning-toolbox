# 进度清单

> 实际执行顺序与原始路线图有调整：优先实现 PU 特有的风险估计方法（工具箱核心差异化能力），经典分类器包装器后移。
> 阶段定义以本文档为准，`development_roadmap.md` 为高层路线图。

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

- [ ] PU Bagging 分类器
- [ ] Biased SVM 分类器
- [ ] Weighted Logistic Regression 分类器
- [ ] penL1 类先验估计
- [ ] TIcE / AlphaMax 类先验估计
- [ ] Advisor 规则版

## Phase 3 — 源码 Adapter (v0.3)

- [ ] PNU、PUSB、LBE、LLSVM adapter
- [ ] adapter smoke test
- [ ] paper-like benchmark 配置

## Phase 4 — 推荐器与诊断 (v0.4)

- [ ] Data Profiler、SCAR/SAR 假设提示
- [ ] 算法推荐器
- [ ] 诊断报告
- [ ] 类先验与标记倾向敏感性分析

## Phase 5 — SAR / Selection-Biased PU (v0.5)

- [ ] SAR / selection bias 数据模拟器
- [ ] PUSB、LBE、Centroid Estimation、LLSVM
- [ ] SCAR vs SAR 对比 benchmark

## Phase 6 — 深度 PU (v0.6)

- [ ] Self-PU、Dist-PU
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
| 2026-07-21 | LDCE | 已完成 native 实现 | `pu_toolbox/estimators/risk/ldce.py`；`docs/research/method_cards/LDCE_KLDCE.md` | `311 passed`；ruff + 质量门禁通过 |
