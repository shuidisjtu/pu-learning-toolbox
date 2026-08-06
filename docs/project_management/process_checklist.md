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
- [ ] 官方数据、历史环境和 paper-like 全量运行

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

## 最近完成记录

| 日期 | 方法 | 状态 | 代码与文档 | 验证 |
|---|---|---|---|---|
| 2026-08-06 | Deep PU 接入 Pipeline/CLI | MLP/CNN 架构选择（两级 architecture+backbone）、.npy 4D NCHW 图像输入、WConPU/InfoMax 显式接入（prior 注入、维度/架构校验、训练成本提示）、InfoMax encoder 插拔、build_encoder 统一入口 | `pu_toolbox/workflows/pipeline.py`；`pu_toolbox/cli/run.py`；`pu_toolbox/estimators/deep/{infomax_pu,vision}.py`；`docs/user/{howto/cli,howto/pipeline,reference/api}.md` | `630 passed`；4 项质量门禁 + ruff（check+format）全通过 |
| 2026-08-06 | CI 稳定性修复 | B905：test_pipeline_deep.py 两处 `zip` 补 `strict=False`（本地 lint 此前只查 `pu_toolbox/` 漏 `tests/`）；encoder determinism 测试改构建前 seed + 随机输入 + eval 模式 + 显式容差（消除 BN 对常数 batch 的 subnormal 平台噪声） | `tests/unit/workflows/test_pipeline_deep.py`；`tests/unit/estimators/test_vision.py` | CI 13 job 全绿（3 平台 × 3 Python + quality + wheel）；本地 `630 passed` |
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
| 2026-07-31 | Data Profiler | 结构化质量报告、问题级别/行动建议、SCAR/SAR 可识别性提示与审计模式 | `preprocessing/data_profiler.py`；`docs/user/howto/data_profiling.md` | 12 个 profiler cases；`548 passed`；ruff、文档与测试质量门禁通过 |
| 2026-08-01 | 诊断报告 | 数据/模型/指标组合报告；证据级别；严格 JSON/Markdown 输出 | `diagnostics/report.py`；`docs/user/howto/diagnostic_reports.md` | 13 个 report cases；`561 passed`；ruff、文档与测试质量门禁通过 |
| 2026-08-01 | 兼容性与项目治理 | 真实 Python matrix、全目录质量门禁、Hatchling 边界、wheel 安装冒烟、贡献/PR 规范与状态对齐 | `pyproject.toml`；`.github/`；`CONTRIBUTING.md`；`docs/dev/compatibility.md` | `562 passed`；104 个 Python 文件 lint/format 通过；wheel/sdist 构建及隔离安装通过 |
| 2026-08-03 | 类先验与标记倾向敏感性 | 固定输出假设扫描、观测恒等式相容性、指标区间及 JSON/Markdown/CSV 导出 | `diagnostics/sensitivity.py`；`docs/user/howto/sensitivity_analysis.md` | 13 个 sensitivity cases；测试、ruff、文档与测试质量门禁通过 |
| 2026-08-03 | Self-PU 核心实现 | 动态 trusted set、meta reweight、双 student/EMA teacher 蒸馏、checkpoint 与 registry | `estimators/deep/self_pu.py`；`docs/user/howto/self_pu.md`；Self-PU Method Card | 13 个专项 cases + 统一 contract；完整 paper-like benchmark 待完成 |
| 2026-08-03 | 深度 PU benchmark | InfoMax PU/WConPU/DGPU 统一 runner、三份论文配置锁、Gaussian generator 和实际多 seed 结果 | `benchmarks/deep_pu/` | 3 methods × 3 seeds = 9 trials；13 个 runner cases；官方视觉/EDM 运行待完成 |
| 2026-08-04 | 深度 PU 官方数据执行层 | 公开数据加载、确定性 case-control split、resume 防混写、原始数据/split/配置/代码哈希与完整配置 preflight | `benchmarks/deep_pu/official_data.py`；`preflight_paper.py`；official-data 配置与结果 | Fashion-MNIST 3 seeds 已执行；ROC-AUC `0.4420 ± 0.0874`；`623 passed`；完整视觉/EDM 仍按 blocker 清单推进 |
| 2026-08-04 | InfoMax PU 论文网络与先验链路 | PURL mini-batch、BN/ReLU、gradient noise；300×3 nnPU；独立 validation；KM1/KM2 与先验误差审计 | `estimators/deep/infomax_pu.py`；`prior/kernel_mean.py`；InfoMax official-data protocol 配置 | 定向 `29 passed`；20-seed 配置待 preflight；未公开 class split、batch size 和 KM 变体仍待核对 |
| 2026-08-04 | WConPU 视觉执行链路 | NCHW、clean-room CNN13、torchvision ResNet-18/50、SimAugment/RandAugment、cosine scheduler、向量化 contrastive loss | `estimators/deep/vision.py`；WConPU estimator 与 CIFAR-10 protocol 配置 | 视觉/runner 定向 `29 passed`；5-seed × 800 epoch CUDA 待执行 |
| 2026-08-05 | WConPU clean validation 选参 | 10% 真值验证集隔离、二维 loss-weight grid、候选级落盘/resume、稳定并列规则与最优参数 refit | `benchmarks/deep_pu/official_data.py`；WConPU CIFAR-10 protocol；Method Card | 定向 `25 passed`；验证集不进入训练；完整 CUDA 实验待执行 |
| 2026-08-05 | 方法卡 MathJax 渲染修复 | `\operatorname`→`\mathrm`；修复 LLSVM/LDCE 缺失上下标参数与 `$` 配对；新增 MathJax 渲染质量门禁 | `docs/research/method_cards/`；`scripts/check_math_rendering.py` | `534 passed`；4 项质量门禁 + ruff 全通过 |
| 2026-08-05 | CI 平台矩阵扩展 | tests job 覆盖 Ubuntu/Windows/macOS 三平台 × 3 Python（3×3），统一 bash shell；quality/package 保持 Linux（静态检查平台无关） | `.github/workflows/tests.yml`；`docs/dev/compatibility.md` | 4 项质量门禁 + ruff 通过；本地 Windows `534 passed` |
| 2026-08-05 | CI 跨平台首跑修复 | actions 升级 Node 24（checkout v5.1.0 / setup-python v6.3.0 / setup-uv v9.0.0，消除 Node 20 弃用警告）；nnPU 早停测试改为强可分数据 + Adam lr=5e-3（3-5 epoch 触发，对 BLAS 归约非确定性留 20-30 倍余量） | `.github/workflows/tests.yml`；`tests/unit/estimators/test_nnpu.py` | CI 11 job 全绿（三平台 × 3 Python + quality + wheel）；本地 Windows `534 passed` |
| 2026-08-05 | PUPipeline 端到端工作流 | 画像→先验→训练→PU 分层 CV→评估 一键封装；先验解析优先级（显式>构造>估计，绝不使用 y_true）；auto 模式经推荐器选算法并跳过不可实例化候选；指标可用性语义（缺 y_true/scores/先验→跳过并记录）；同时修复 LLSVM predict 契约违约（{1,-1}→{0,1}）与测试质量门禁类级 marker 继承 bug | `pu_toolbox/workflows/`；`tests/unit/workflows/test_pipeline.py`；`docs/user/howto/pipeline.md` | `549 passed`；4 项质量门禁 + ruff（check+format）全通过 |
| 2026-08-05 | pu-toolbox CLI | argparse 薄封装：`run` 一键式全流程（双 CSV 输入、目录三件套输出、退出码 0/1/2）、`list-methods`/`list-priors`（registry 实时读取 + 复用 `_missing_required_params` 判定自动实例化）、`make-demo-data`（SCAR 演示数据，自洽闭环）、`[project.scripts]` 入口 | `pu_toolbox/cli/`；`docs/user/howto/cli.md` | 新增 20 cases（4 文件）；门禁与 ruff 全通过；手动验证 make-demo-data → run 闭环 |
| 2026-08-05 | CLI 代码评审修复 | max-effort 评审 15 项修复：CI wheel 计数 15→16 + console script 冒烟；headerless CSV 检测报错（防静默丢首样本）；metrics 剥离空白；多列 labels 报错；`--prior-estimator none` + auto 降级为无先验推荐（pipeline 深度修复）；make-demo-data 统一错误映射 + `--n>=3` 校验；list-priors 补 registry 别名；km1/km2 移除误导别名（kldce 同类缺陷）；CLI 测试套件 141s→3.6s（`--classifier upu`）；文档 15→16 计数 + KLDCE.md 守卫声明如实化 | `pu_toolbox/cli/`；`pu_toolbox/workflows/pipeline.py`；`pu_toolbox/registry/builtin_methods.py`；`.github/workflows/tests.yml`；`tests/unit/cli/` | `583 passed`；4 项质量门禁 + ruff（check+format）全通过；手动验证闭环与错误路径 |
| 2026-08-05 | 默认 run 提速（A+B） | LLSVM 收敛早停（trailing-window 相对损失 + min_epochs 下限，默认开，tol=5e-4 校准）；推荐器新增第 7 维训练成本（TrainingCost 元数据 + cost_max=10，llsvm 唯一 HIGH）；auto 小数据改选快方法 | `estimators/classic/llsvm.py`；`advisor/rules.py`；`core/tags.py`；`registry/{metadata,builtin_methods}.py`；`tests/unit/{estimators/test_llsvm,advisor/}` | `597 passed`；4 项门禁 + ruff 全通过；实测 auto 30s→2.05s（y_true 场景选 UPU 95.5，无 y_true 选 PUSB）、显式 llsvm 30s→7.1s |
| 2026-08-05 | 文档体系重构 | docs/ 按受众分层（user/ 旅程化：快速开始→概念→操作→API 参考；dev/ 开发者文档；project_management/ 原位）；重写 method_selection/根 README 双语/roadmap/architecture；check_doc_links 硬编码路径同步；删除一次性任务分工 division.txt | `docs/user/`、`docs/dev/`、`docs/README.md`、根 README 双语；`scripts/check_doc_links.py` | 4 项质量门禁全绿（含 math 渲染）；无代码改动 |
