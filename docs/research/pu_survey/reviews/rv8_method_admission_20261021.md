# P3.1 RV8 新增方法准入证据复核（2026-10-21）

> 状态：**已完成并签署**。签署人 shuidisjtu，2026-10-21。
> 范围：`method_ledger.json` 的 `pending_owner_review_methods` 共 14 条：P3MIX、PUET、
> PULDA、Grad-PU、Robust-PU、Split-PU、PAN、RP、PULNS、GEN-PU、Holistic-PU、CVIR、
> LaGAM、TEDn。前 13 条即 P3.1 宽口径集合；TEDn 只做 RV8，不计入 P3.1 的 13 方法。
> 口径：协议 [`survey-v2.0`](../../../../pu_toolbox/experiment/survey_protocol_v2.json)
> （canonical sha256 `0033d8105c98a4b053381a708d9dcbc46cc00c1f4e16cf21655a475c553a8956`）、
> [13 方法 v2 delta 表](p3_1_v2_delta_20261021.md)与 #119–#126 阶段 C 处置。
> 边界：RV8 复核的是**准入证据是否成立**，不是正式准入。本记录不修改
> `admitted=false`、冻结协议、比较矩阵或台账历史签署。

## 1. 判定规则

- 四档与 delta 表一致：**接受 / 有条件接受 / 退回 / 延后**。
- 本批 14 条均为**有条件接受**：来源、公式与行为证据成立，但正式配方、预算与资源
  仍受 P3.2/P3.3/P4.1-B 约束；个别方法另有保留项（见 §4）。
- 校准按 v2 §2.3 A2：原生 TS 且存在可替换 U 风险项的方法必须校准；原生 OS 或
  无同形 U 风险项的实现登记“不适用”。
- 架构按 v2 §2.5 A4/Q1：只有论文或方法机制必需才登记例外；作者实现选择不算。
- 证据由 HENG958 提供实现/来源材料；HENG958 不自审其新增算法，判定与签署由
  shuidisjtu 完成。

## 2. 汇总

| # | 方法 | 来源身份 | 采样 / 先验 | 校准 | 结论 |
|---|---|---|---|---|---|
| 1 | P3MIX | paper-only | OS / 无外部先验 | 不适用 | 有条件接受 |
| 2 | PUET | official_related | TS / population π | 已校准 | 有条件接受 |
| 3 | PULDA | official_related | TS / population π | 已校准 | 有条件接受 |
| 4 | Grad-PU | paper-only | TS / 无先验 | 已校准 | 有条件接受 |
| 5 | Robust-PU | official_exact | TS / population π | 已校准 | 有条件接受 |
| 6 | Split-PU | official_exact | TS / population π | 已校准 | 有条件接受 |
| 7 | PAN | official_related | OS / 无先验 | 不适用 | 有条件接受 |
| 8 | RP | official_related | OS / 内部估 π | 不适用 | 有条件接受 |
| 9 | PULNS | paper-only | 原生 TS + clean support / 无先验 | 不适用 | 有条件接受（OA-only） |
| 10 | GEN-PU | official_related | TS / population π | 已校准 | 有条件接受 |
| 11 | Holistic-PU | official_related | OS / 无外部先验 | 不适用 | 有条件接受 |
| 12 | CVIR | official_exact | TS 原生 / α_U | 未接 | 有条件接受（OS 路径） |
| 13 | LaGAM | official_exact | OS + clean support / 无先验 | 不适用 | 有条件接受（OA-only） |
| 14 | TEDn | official_exact | 原生 TS / 内部 BBE | 不适用 | 有条件接受 |

“条件”指 §4 的 P3.2/P3.3/P4.1-B 前置与个别保留项；不表示证据不足或需要退回。

## 3. 逐条记录

### 1. P3MIX（`p3mix`）

- **身份**：paper-only（已核对 14 页 ICLR 2022 全文，未发现作者代码地址）；NATIVE，
  含 P3Mix / P3Mix-E / P3Mix-C。
- **采样/先验**：OS；`class_prior` 仅保留兼容并校验，不参与训练。
- **校准**：不适用（原生 OS，无可替换的 U 风险项）。
- **架构**：论文 MLP/CNN 是实验选择，不构成共享架构例外；共享 MLP 或注入可训练 CNN。
- **判定**：有条件接受。原文交叉表、实现映射与专项测试成立；正式 recipe、预算与
  资源转 P3.2/P3.3。
- **证据**：[原文交叉表](../admission/p3mix_paper_review_20261019.md)、
  [方法卡](../../method_cards/P3MIX.md)。
- **签署**：shuidisjtu，2026-10-21。

### 2. PUET（`puet`）

- **身份**：official_related（仅 clean-room nnPU 二次分支）；NATIVE，随机森林。
- **采样/先验**：TS；population π，按节点正类风险质量 `π·n_P,node/n_P` 分配。
- **校准**：已校准。v2 A2 明确把“逐树根复制 P 为未标记风险角色”纳入等价实现。
- **架构**：非神经共享 backbone；CPU 树模型按 CPU 条款，不占 GPU 预算。
- **判定**：有条件接受。#114 修复已验证；候选/选模与 CPU 树预算转 P3.3。
- **证据**：[PUET 适配补项](puet_adaptation_completion_20261017.md)、
  [方法卡](../../method_cards/PUET.md)。
- **签署**：shuidisjtu，2026-10-21。

### 3. PULDA（`pulda`）

- **身份**：official_related（锁定 commit `7b3dcad`）；known-prior PULDA，非 PULDA\*。
- **采样/先验**：TS；population π。
- **校准**：已校准（逐 batch U∪P 风险角色）。
- **架构**：共享 ResNet-18 端到端，属 benchmark-adapted 主路径，**不登记架构例外**；
  MLP64 与分类头维持 method-private，不与共享 MLP 混排。
- **判定**：有条件接受。#119 决策已定稿（训练 BN、共享增强 none、LDA warmup）；
  runner 分层与 120 快照转 P3.2/P3.3。
- **证据**：[#119 决策清单](../admission/candidate_image_owner_decisions_20261018.md)、
  [TS 校准补核](additional_ts_calibration_review_20261019.md)、
  [方法卡](../../method_cards/PULDA.md)。
- **签署**：shuidisjtu，2026-10-21。

### 4. Grad-PU（`gradpu`）

- **身份**：paper-only（未确认作者仓库）；NATIVE，Eq. (5)–(7) 独立实现。
- **采样/先验**：TS；目标不使用 `class_prior`，fit 仅接口兼容。
- **校准**：已校准（U 风险与原始图像梯度插值同时替换）。
- **架构**：`cnn13_no_bn` 登记为 **method-private 架构例外**：论文明确移除 BN（与
  梯度正则不兼容），按 Q1 单列并保留 SHOULD 级共享架构对照；不进入标准
  ResNet-18 主榜。
- **判定**：有条件接受。BN fail-closed、原图梯度、二阶反传、checkpoint 与 fold
  隔离测试成立；200 快照转 P3.3，配方/预算（#121）转 P3.2/P4.1-B。
- **证据**：[配方复核](../admission/gradpu_training_recipe_review_20261018.md)、
  [来源复核](../admission/gradpu_source_review_20261005.md)、
  [方法卡](../../method_cards/GradPU.md)。
- **签署**：shuidisjtu，2026-10-21。

### 5. Robust-PU（`robust_pu`）

- **身份**：official_exact（论文直接链接锁定仓库，表示来源可得，不是源码重放）。
- **采样/先验**：TS；case-control 的 U 为总体边缘分布，`π_U=π`，仅用于 nnPU 预训练。
- **校准**：已校准；与 `runtime_prior_contract` 一致。
- **架构**：共享 MLP 或注入可训练 CNN，无架构例外。
- **判定**：有条件接受。默认 SPL 与官方实现一致；预训练 batching 与预算按适配登记。
- **证据**：[来源证据](../data/robustpu_source_review_20261018.json)、
  [TS 校准补核 b](additional_ts_calibration_review_20261019b.md)、
  [方法卡](../../method_cards/Robust-PU.md)。
- **签署**：shuidisjtu，2026-10-21。

### 6. Split-PU（`split_pu`）

- **身份**：official_exact（锁定 commit `82fb973`）。
- **采样/先验**：TS；`class_prior` 为 population π，仅用于 nnPU teacher。
- **校准**：已校准（TS 只作用 teacher）。
- **架构**：共享 MLP 或注入可训练 CNN，无架构例外。
- **判定**：有条件接受，但带 RV8 保留项：JSD 权重方向仍待方法负责人裁决
  （#122 保留），当前实现跟随源码 teacher=0.7；TS 均衡 batch 与总体 π 的比例
  差异登记为近似，是否加权重校准待决。保留项闭合前不得声称论文式复现。
- **证据**：[#122 表征](../admission/splitpu_122_characterization_20261018.md)、
  [TS 校准补核 b](additional_ts_calibration_review_20261019b.md)、
  [方法卡](../../method_cards/Split-PU.md)。
- **签署**：shuidisjtu，2026-10-21。

### 7. PAN（`pan`）

- **身份**：official_related（README 称论文代码，正文未直链，无 LICENSE）。
- **采样/先验**：OS；目标不需要外部先验。
- **校准**：不适用（原生 OS）。
- **架构**：保留论文 Eq. (7) 的 paper-faithful 实现，C/D 双编码器属方法机制，不登记
  为共享 backbone 例外；正式采样跟随论文 single-training-set。
- **判定**：有条件接受。#123 决定已入卡、台账与扩展草稿；对抗训练预算、C-only
  快照与 D/optimizer 续训转 P3.2/P3.3。
- **证据**：[来源/推理复核](../admission/pan_source_inference_review_20261018.md)、
  [来源 JSON](../data/pan_source_review_20261018.json)、
  [方法卡](../../method_cards/PAN.md)。
- **签署**：shuidisjtu，2026-10-21。

### 8. RP（`rp`）

- **身份**：official_related（PU clean-positive 特例的独立适配，非通用双向噪声 PN）。
- **采样/先验**：OS；无外部先验，`ρ1` 由 OOF confident counts 估计、π 由观测比例推导。
- **校准**：不适用（原生 OS，无同形 U 风险项）。
- **架构**：经典 sklearn 基学习器，无神经 backbone。
- **判定**：有条件接受。PU 特例与作者 `fit(pulearning=True)` 路径一致；退化先验
  测试已补（`pi≥1`、`pi0` 夹断、`min_retained`）；OOF+refit 预算转 P3.2。
- **证据**：[来源复核](../admission/rp_source_review_20261018.md)、
  [方法卡](../../method_cards/RP.md)。
- **签署**：shuidisjtu，2026-10-21。

### 9. PULNS（`pulns`）

- **身份**：paper-only（未确认作者仓库）；NATIVE，RL selector + 终端分类器。
- **采样/先验**：论文为原生 TS（P/U 分别抽样，`γ` 为 U 内正类比例，不是标记率 `c`）；
  工程按 clean-support 路径登记为 OS-compatible，无外部先验。
- **校准**：不适用（原生 TS 但无可替换的同形 U 风险项；训练依赖 clean support）。
- **架构**：共享 MLP 或注入可训练 CNN；clean support 是额外训练监督，不构成
  架构例外。
- **判定**：有条件接受（OA-only）。标准 PU-only PA 不适用；`PA+clean-support` 与
  support 规模/预算按 v2 A3/A7/§3.2，`v2_p4` 仍 pending；正式 PA 路径与 support
  制品转 P3.2。
- **证据**：[clean support 工作流](../delivery/clean_support_workflow_20261017.md)、
  [数据集适配](../data/pulns_dataset_adaptation_20261017.json)、
  [来源复核](../pulns_source_cnn_review_20261007.md)、
  [方法卡](../../method_cards/PULNS.md)。
- **签署**：shuidisjtu，2026-10-21。

### 10. GEN-PU（`genpu`）

- **身份**：official_related（作者主页链接归档，未发现 LICENSE，`license=unknown`）。
- **采样/先验**：TS；population π，生成分布与 Du 假样本按 π/1−π 混合并加权。
- **校准**：已校准。
- **架构**：GAN 生成器/判别器是方法机制必需组件，按方法组件登记，不是共享 backbone
  例外；D/PN 的共享表征仍适用。
- **判定**：有条件接受（#126 保留）。论文 Eq. (3)–(11) 目标已定，作者 demo 仅差异
  对照；四个 recipe/预算 blocker 转 P3.2/P4.1-B。
- **证据**：[来源复核](../genpu_source_cnn_review_20261007.md)、
  [方法卡](../../method_cards/GEN-PU.md)。
- **签署**：shuidisjtu，2026-10-21。

### 11. Holistic-PU（`holistic_pu`）

- **身份**：official_related（作者仓库锁定，未发现 LICENSE，`license=unknown`）。
- **采样/先验**：OS；无外部先验，趋势伪标签的 U 内正类比例仅作诊断。
- **校准**：不适用（原生 OS）。
- **架构**：LZO/初始化是 recipe 选择，不构成架构例外；共享 encoder 适用。
- **判定**：有条件接受。论文 pairwise/variance 路径、默认 fixed warmup + continue
  已记录；LZO/初始化 recipe 与存储 profile 转 P3.2/P3.3。
- **证据**：[LZO 选择](../holistic_lzo_selection_20261007.md)、
  [阶段初始化](../holistic_stage_initialization_20261007.md)、
  [方法卡](../../method_cards/Holistic-PU.md)。
- **签署**：shuidisjtu，2026-10-21。

### 12. CVIR（`cvir`）

- **身份**：official_exact（作者代码已锁定；表示来源可得，不是重放）。
- **采样/先验**：论文为原生 TS，`α_U` 是 U 内正类比例，不是自动获得的 population π。
- **校准**：未接。`calibration_hooked=false`，TS 主表门未闭合。
- **架构**：共享 MLP 或注入可训练 CNN，无架构例外。
- **判定**：有条件接受，仅限 **OS known-prior 路径**。TS 主表必须在“接入
  TS-compatible 校准 / 论证不适用 / 明确延后”中给出结论（P3.2）；在此之前不得以
  `ts-compatible` 呈现。
- **证据**：[来源与 CNN](../admission/cvir_source_cnn_review_20261015.md)、
  [RV6 TS 校准复核](rv6_ts_calibration_technical_review_20261019.md)、
  [方法卡](../../method_cards/CVIR.md)。
- **签署**：shuidisjtu，2026-10-21。

### 13. LaGAM（`lagam`）

- **身份**：official_exact（作者源码锁定；source_scope/EMA/损失等差异按 review 登记）。
- **采样/先验**：OS；`class_prior` 不进入目标。
- **校准**：不适用（原生 OS，无可直接替换的同形 U 风险角色）。
- **架构**：显式 clean support + head-only virtual meta update；clean support 是额外
  训练监督，不是架构例外。
- **判定**：有条件接受（OA-only）。`v2_p4` 仍 pending；support 制品与
  `PA+clean-support` 路径转 P3.2；许可与公开数值仍待。
- **证据**：[全文复核](../admission/lagam_paper_review_20261020.md)、
  [来源与 CNN](../admission/lagam_source_cnn_review_20261015.md)、
  [方法卡](../../method_cards/LaGAM.md)。
- **签署**：shuidisjtu，2026-10-21。

### 14. TEDn（`tedn`）

- **身份**：official_exact（锁定 receipt 与 license 记录在来源 review）。
- **采样/先验**：原生 TS，由调用方显式声明 `input_sampling="ts"`；内部 BBE 估计
  `α_U`，不接受 `class_prior`，也不按 `α_U` 加权风险。
- **校准**：不适用（原生 TS；TEDn 本身不执行 TS-compatible 校准）。
- **架构**：共享 MLP 或注入可训练 encoder，无架构例外。
- **判定**：有条件接受（只走 RV8，不计入 P3.1 的 13 方法）。`input_sampling` 声明尚未
  自动验证；正式准入与公开数值另行验收。
- **证据**：[评分模式复核](../admission/tedn_scoring_modes_20261019.md)、
  [来源 JSON](../data/tedn_source_review_20261017.json)、
  [方法卡](../../method_cards/TEDn.md)。
- **签署**：shuidisjtu，2026-10-21。

## 4. 保留项与移交

以下保留项不改变“有条件接受”，但必须在其 RV8 记录或结果说明中写明。共通前置：
正式资源 profile（P3.3）、正式预算/配方冻结（P4.1-B）、全部 22 方法门禁（P4.2）。

| 方法 | 保留项 | 去向 / owner |
|---|---|---|
| Split-PU | JSD 权重方向（#122 保留）、TS batch 比例、台账失效指针 | #122 / P3.2，shuidisjtu |
| CVIR | TS-compatible 校准或“不适用”论证 | P3.2，shuidisjtu |
| PULNS | `v2_p4`、support 制品物化、`PA+clean-support` 路径 | P3.2，shuidisjtu |
| LaGAM | `v2_p4`、support 路径、许可与公开数值 | P3.2，shuidisjtu |
| GEN-PU | 四个 recipe/预算 blocker（#126 保留） | P3.2/P4.1-B，shuidisjtu |
| Holistic-PU | LZO/初始化 recipe、存储 profile | P3.2/P3.3，shuidisjtu |
| PAN | 对抗训练预算、C-only 快照、D/optimizer 续训 | P3.2/P3.3，shuidisjtu |
| RP | OOF+refit 预算/选模规格 | P3.2，shuidisjtu |
| PULDA | runner 分层、120 快照 profile | P3.2/P3.3，shuidisjtu |
| Grad-PU | 200 快照 profile、默认 LR/batch/候选池 | P3.3/P3.2，shuidisjtu |
| PUET | CPU 树预算、候选/选模 | P3.3，shuidisjtu |
| P3MIX | 正式 recipe/预算/资源 | P3.2/P3.3，shuidisjtu |
| TEDn | `input_sampling` 自动验证、正式准入 | P3.2 / 另行验收，shuidisjtu |
| Robust-PU | 无额外保留项 | 共通前置 |

## 5. 签署

- **签署人**：shuidisjtu（GitHub @shuidisjtu）
- **日期**：2026-10-21
- **范围**：上列 14 条 RV8 准入证据复核
- **结论**：14 条均为“有条件接受”，无“退回”；条件与保留项见 §4。
- **声明**：本人已逐条核对来源身份、采样/先验口径、校准适用性、架构归属与证据链。
  HENG958 提供实现与来源证据，未自审其新增算法。本签署不构成正式准入；正式准入
  仍待 P3.3 资源、P4.1-B 预算/配方冻结与 P4.2 全部 22 方法门禁。
- **电子签名**：shuidisjtu

记录版本：`rv8-20261021-1`。
