# Development Roadmap

## 1. 总体开发策略

本 Toolbox 推荐采用 **framework-first + source-aware integration** 的开发方式：先完成稳定框架、API 契约、registry、metadata、测试体系和示例，再逐个集成论文算法。这样即使短期内没有实现全部论文方法，也可以先让项目具备可安装、可扩展、可测试、可演示的工程基础。

推荐路线：

```text
项目骨架与 API 契约
        ↓
算法注册表、metadata、source policy、mock estimator
        ↓
经典 PU Learning MVP
        ↓
类先验估计、指标、PU CV、诊断系统
        ↓
官方源码 adapter 与 paper-like benchmark
        ↓
PyTorch 风险估计方法：uPU / nnPU / PNU（v0.2）
        ↓
深度分布对齐：Dist-PU（v0.6）
        ↓
SAR / Instance-Dependent / Selection-Biased PU
        ↓
深度表征、对比学习、生成式 PU 等研究扩展
```

核心判断：**允许先不实现所有论文算法，但不允许没有清晰接口、元数据、测试占位和集成规范。**

> 阶段对应关系（以 [`A.md`](A.md) 阶段 1–4 为统一框架）：
> - Phase 0 → A.md 阶段 1（项目骨架与 API 契约）
> - Phase 1–2 → A.md 阶段 2（v0.1 Baseline + 类先验接口）
> - Phase 3–4 → A.md 阶段 3–4（adapter 在阶段 3，推荐器/诊断增强在阶段 4）
> - Phase 5–6 → A.md 阶段 4 延伸（SAR / Deep PU 接口预留与调研，完整实现超出 1 个月范围，为 post-MVP 工作）

---

## 2. Phase 0：项目骨架与规范

### 目标

建立可维护的工程基础，为后续逐个集成论文算法做准备。

### 任务

1. 创建项目目录；
2. 配置 `pyproject.toml`；
3. 配置测试框架；
4. 配置 lint / format；
5. 配置 CI；
6. 建立基础文档；
7. 定义 Base 类；
8. 定义标签规范；
9. 定义算法注册表；
10. 定义 source metadata；
11. 定义官方源码 adapter 接口；
12. 定义 `implementation_status` 和 `source_status` 枚举。

### 交付物

```text
BasePUClassifier
BasePriorEstimator
BasePropensityEstimator
BasePULoss
BaseSourceAdapter
normalize_pu_labels
validate_pu_X_y
get_algorithm_registry
AlgorithmMetadata
SourceMetadata
```

### 验收标准

- 项目可安装；
- 核心包可导入；
- 单元测试可运行；
- Base API 清晰；
- 所有 estimator 支持 `get_params` / `set_params`；
- 随机种子处理统一；
- registry 能注册 `api_only` 算法；
- advisor 能识别算法是否已实现、是否有官方源码、是否需要外部 adapter。

---

## 3. Phase 1：经典 PU Learning MVP

### 目标

形成第一个可用版本，重点解决“用户能不能快速跑通”的问题。

### 优先实现

1. Elkan-Noto；
2. PU Bagging；
3. Biased SVM；
4. Weighted Logistic Regression；
5. TIcE 或 AlphaMax；
6. PU splitters；
7. 基础 metrics；
8. minimal examples；
9. method advisor 的规则版；
10. 15 篇论文方法的 registry metadata 占位。

### 交付物

```text
pu_toolbox v0.1
```

### 能力

1. 用户可输入 PU 数据；
2. 可训练经典模型；
3. 可估计类先验；
4. 可做 PU 专用切分；
5. 可输出基础评估结果；
6. 可运行 minimal example；
7. 可看到未实现论文算法的状态说明；
8. 可根据 `resources_optimized.md` 查询哪些算法有官方源码可优先集成。

---

## 4. Phase 2：类先验估计与风险估计基础

### 目标

支持表格中 class-prior estimation 与 risk-consistent loss function 方向的核心方法。

### 优先实现

1. Class-Prior Estimation 基础接口；
2. ReCPE wrapper；
3. Convex PU / uPU loss；
4. nnPU loss；
5. PNU loss；
6. class prior sensitivity analysis；
7. toy data 数值测试。

### 交付物

```text
pu_toolbox v0.2
```

### 验收标准

- 可以在 case-control 设置下训练 uPU / nnPU；
- 所有风险估计方法都显式接收或估计 `class_prior`；
- 当 `class_prior` 不合理时给出 warning；
- uPU / nnPU loss 在 toy data 上数值稳定；
- nnPU 优先与作者源码或官方相关实现做结果对齐。

---

## 5. Phase 3：官方源码 adapter 与复现体系

### 目标

把 `resources_optimized.md` 中标记为 `official_exact` 或 `official_related` 的方法纳入可管理的 adapter 体系，而不是散乱复制代码。

### 优先适配

1. nnPU；
2. PNU；
3. PUSB；
4. LBE；
5. LLSVM；
6. Centroid 相关代码包。

> Self-PU、Dist-PU 的 adapter 工作在 Phase 6（深度 PU）中随完整实现一并推进，不在本阶段单独处理。

### 任务

源码 adapter 的设计规范、依赖检查、输入输出转换、许可证管理等详见 [`architecture.md`](architecture.md) §10。本阶段的具体任务聚焦于：

1. 为 `resources_optimized.md` 中标记为 `official_exact` 的方法逐个创建 adapter；
2. 写 adapter smoke test 和 paper-like benchmark 配置；
3. 在 registry 中标记 `official_adapter` 或 `official_aligned_native`。

### 交付物

```text
pu_toolbox[source] v0.3
```

### 验收标准

- 不破坏 core 包轻量性；
- 未安装外部源码时 core 仍可正常使用；
- adapter 不改变统一 estimator API；
- 每个 adapter 至少有一个 smoke test；
- 每个 adapter 都有许可证备注和引用提示；
- 无法直接集成的源码保留 reproduction guide。

---

## 6. Phase 4：模型推荐器与诊断系统

### 目标

提高非专家用户可用性，让用户知道“当前数据该试哪些方法、为什么、有哪些风险”。

### 任务

1. 数据画像模块；
2. SCAR/SAR 假设提示；
3. 算法推荐器；
4. 类先验敏感性分析；
5. 标记倾向敏感性分析；
6. 概率校准；
7. 诊断报告生成器；
8. source-aware recommendation：有官方源码的方法优先提示。

### 交付物

```text
pu_toolbox v0.4
```

### 推荐器输出

至少返回 3 个候选算法，并说明：

1. 适用假设；
2. 是否需要类先验；
3. 是否支持稀疏数据；
4. 是否需要 GPU；
5. 训练复杂度；
6. 推荐原因；
7. 潜在风险；
8. 实现状态；
9. 官方源码状态；
10. 如果当前算法只是 `api_only`，应提示用户它尚不能训练。

---

## 7. Phase 5：SAR / Instance-Dependent / Selection-Biased PU

### 目标

形成项目差异化能力，处理 SCAR 不成立时的 PU Learning。

### 优先实现

1. SCAR 数据模拟器（`scar_simulator.py`，已在早期阶段作为基础工具完成）；
2. SAR 数据模拟器（`sar_simulator.py`）；
3. selection bias 数据模拟器（`selection_bias_simulator.py`）；
4. PropensityEstimator；
5. PUSB；
6. LBE；
7. Centroid Estimation；
8. LLSVM；
9. SAR 专用文档与示例。

### 交付物

```text
pu_toolbox[sar] v0.5
```

### 验收标准

- 能生成已知 `c(x)` 的 SAR 合成数据；
- 能估计实例相关标记倾向；
- 能比较 SCAR 模型与 SAR 模型表现；
- 有清晰的失败模式说明；
- 有 SAR example 和 benchmark；
- LBE、LLSVM 等有作者源码的算法优先提供 adapter 或 official-aligned native 实现。

---

## 8. Phase 6：深度表征与研究扩展

### 目标

覆盖更复杂研究方向，但不进入 MVP。

### 候选方向

> Self-PU 和 Dist-PU 均存在官方源码（`official_exact`），其 adapter 和完整实现在本阶段一并推进，不拆分到 Phase 3。

1. Self-PU；
2. Dist-PU；
3. Information-Theoretic PU；
4. Weighted Contrastive PU；
5. Discriminative-Generative PU；
6. Online PU；
7. Graph PU；
8. Anomaly Detection PU；
9. Federated PU。

这些方向应在核心稳定后逐步加入，因为它们会显著提高依赖复杂度、benchmark 成本、文档复杂度和用户使用门槛。

---

## 9. 开发任务分解表

| 工作包 | 内容 | 优先级 | 依赖 |
|---|---|---|---|
| WP0 | 项目骨架、CI、文档结构 | P0 | 无 |
| WP1 | Base API、标签规范、数据校验 | P0 | WP0 |
| WP2 | 算法注册表、metadata、aliases | P0 | WP1 |
| WP3 | source metadata 与 adapter 接口 | P0 | WP2 |
| WP4 | PU 数据切分与 CV | P0 | WP1 |
| WP5 | Elkan-Noto、PU Bagging、Weighted LR、Biased SVM | P0 | WP1 |
| WP6 | 类先验估计 TIcE / AlphaMax / ReCPE | P0 | WP1 |
| WP7 | metrics 与 diagnostics | P1 | WP1, WP6 |
| WP8 | advisor 推荐器 | P1 | WP2, WP7 |
| WP9 | uPU / nnPU / PNU Torch 后端 | P1 | WP6 |
| WP10 | 官方源码 adapter 与 paper-like benchmark | P1 | WP3, WP9 |
| WP11 | SAR 抽象、selection bias 模拟器 | P2 | WP1, WP6 |
| WP12 | PUSB / LBE / Centroid / LLSVM | P2 | WP10, WP11 |
| WP13 | Self-PU / Dist-PU | P2 | WP9, WP10 |
| WP14 | InfoMax / Contrastive / DGPU | P3 | WP13 |
| WP15 | Agent skill | P2 | WP8 |

---

## 10. 论文算法实施优先级

| 优先级 | 方法 | 原因 |
|---|---|---|
| P0 | Elkan-Noto | 经典 baseline，易实现，适合 core |
| P0 | TIcE / AlphaMax / ReCPE 接口 | 支撑风险估计方法 |
| P1 | uPU / nnPU | 现代风险估计核心，且有官方相关实现 |
| P1 | PNU | 与 PU / NU / PN 风险组合有关，适合扩展 risk module |
| P1 | PUSB / LBE | 处理 selection bias / labeling bias，形成差异化 |
| P2 | LLSVM / Centroid | 重要 SAR / bias-aware 方向，但实现复杂度较高 |
| P2 | Dist-PU / Self-PU | 深度方法代表，有官方源码可适配 |
| P3 | InfoMax PU / Weighted Contrastive PU | 表征学习方向，适合 research extension |
| P3 | DGPU | 生成式混合方法，训练和依赖成本高，最后实现 |

---

## 11. 测试体系

### 11.1 核心模块

目标覆盖率：95%–100%。包括标签规范化、数据校验、基类接口、类先验估计器、splitters、metrics、registry、经典算法、metadata schema 和 source policy。

### 11.2 深度算法

需要 smoke test、小规模 toy data 收敛测试、CPU/GPU 兼容测试、随机种子复现测试、loss 数值稳定性测试和 class prior 边界测试。

### 11.3 Source Adapter

每个 adapter 必须测试依赖缺失提示、输入输出格式转换、随机种子传递、统一 API 输出、benchmark 配置运行和许可证信息查询。

### 11.4 Benchmark Regression

每个重要版本发布前，固定数据集和配置，比较 AUC、F1、average precision、训练时间、内存占用、随机种子稳定性，以及对 `π` 和 `c(x)` 的敏感性。

---

## 12. 示例脚本设计

```bash
python examples/minimal/run_elkan_noto.py
python examples/minimal/run_pu_bagging.py
python examples/minimal/run_prior_estimation.py

python examples/paper_like/reproduce_nnpu_mnist.py
python examples/paper_like/reproduce_pnu.py
python examples/paper_like/reproduce_lbe.py
python examples/paper_like/reproduce_dist_pu.py

python examples/sar_cases/run_labeling_bias_demo.py
python examples/sar_cases/compare_scar_vs_sar.py
python examples/sar_cases/run_selection_bias_demo.py

python examples/source_adapter_demo/check_official_nnpu.py
python examples/source_adapter_demo/check_lbe_package.py
```

---

## 13. 版本策略

```text
0.1.0  MVP，可训练经典 PU 模型，registry 中包含论文方法占位
0.2.0  支持类先验估计与 uPU / nnPU / PNU 风险模块
0.3.0  支持官方源码 adapter 与 paper-like benchmark
0.4.0  支持推荐器与诊断报告
0.5.0  支持 SAR / selection-biased PU 核心算法
0.6.0  支持 Self-PU、Dist-PU 等深度代表方法
1.0.0  API 稳定版本
```

废弃策略：

1. 先给出 `DeprecationWarning`；
2. 至少保留一个 minor version 过渡期；
3. 移除时在 changelog 中标记 `[breaking]`；
4. 重命名或行为变化标记 `[deprecate]`。

---

## 14. MVP 推荐实现清单

### v0.1 必做

```text
core/
  BasePUClassifier
  BasePriorEstimator
  BasePropensityEstimator
  BasePULoss
  normalize_pu_labels
  validate_pu_X_y

registry/
  get_algorithm
  get_algorithm_registry
  AlgorithmMetadata
  SourceMetadata
  SourcePolicy

source_adapters/
  BaseSourceAdapter

estimators/
  ElkanNotoPUClassifier
  BaggingPUClassifier
  WeightedLogisticPUClassifier
  BiasedSVMClassifier

prior/
  TIcEEstimator
  AlphaMaxEstimator

model_selection/
  PUStratifiedKFold
  PUStratifiedShuffleSplit

metrics/
  pu_label_frequency
  lee_liu_score
  check_degenerate_prediction

advisor/
  rule-based Top-3 recommendation

examples/
  minimal examples
```

### v0.1 可以只做 API 占位

```text
ReCPE
uPU
nnPU
PNU
PUSB
LBE
Centroid Estimation
LLSVM
Self-PU
Dist-PU
InfoMax PU
Weighted Contrastive PU
DGPU
```

这些占位类必须遵守 [`architecture.md`](architecture.md) §8 中 `api_only` 状态的约束：仅提供 API 契约，不得包含训练逻辑。
