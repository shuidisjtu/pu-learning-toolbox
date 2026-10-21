# PU 调研实验执行计划（pilot → 主榜）

> 本文件只维护 Survey 的任务分工、验收边界、风险与留痕约定，以及仍影响执行的决策摘要。
> 实验协议和比较矩阵分别以 [`pu_survey_protocol.md`](pu_survey_protocol.md)、
> `pu_toolbox/experiment/survey_protocol_v2.json` 与 `survey_comparison_v4.json` 为准；
> `survey_protocol_v1.json`/`survey_comparison_v1..v3.json` 保留为历史冻结绑定。
> 状态核对日期：**2026-10-21**；RV1/RV2 验收基线为 PR #110 的 `8b4e751`。
> 复核责任明确归属 **shuidisjtu** 或 **HENG958**；
> RV1/RV2 与 P1.4-R 已由用户接受技术复核证据并确认完成，见对应验收记录。
> P2.2 的 C1–C6 经用户 2026-10-16 决定调整范围：仅执行 C2/C3，其余按通过处理（未逐项核查），见 P2.2 与 RV3/RV4/RV7。

## 1. 任务分工与验收

**HENG958** 负责新增算法、CNN 适配与训练、资源测量；**shuidisjtu** 负责数据、实验协议、
注册表与结果审计，并复核 HENG958 的新增实现。HENG958 复核 shuidisjtu 交付的既有 Pilot 材料。
每项任务只有一名主责；工程 smoke、单方放行或代码存在不等于正式准入。

### 1.1 已完成的工程交付

2026-10-17 算法独立扩展：[BBE](../method_cards/BBE.md) 已补论文统计量与
二维 PvU 内部留出估计入口；目标显式为 α_U，不接 OS 总体先验或现有 CVIR，
不改冻结矩阵。另补独立 [TEDⁿ](../method_cards/TEDn.md) 完整交替训练工程路径；
截至 2026-10-19，TEDn 已实验级注册至 API/registry/CLI/UI，要求调用方显式声明原生
`input_sampling="ts"`，但不执行 TS-compatible 校准（要求调用方声明原生 TS 数据），也不自动验证采样设计。真实数据数值验收、
正式预算和准入仍未完成。

| 编号 | 任务 | 证据 |
|---|---|---|
| P1.1 | 环境与 GPU 工程验证 | 已有目标设备 smoke；正式运行仍须 frozen-lock 环境 |
| P1.2 | Pilot 三数据集获取与版本审计 | 来源、版本、标签映射与许可已登记；剩余数据见 P1.2-R |
| P1.2-R | 剩余五数据集获取与许可登记 | **2026-10-14 完成（shuidisjtu 确认）**。MNIST、F-MNIST、20News、Connect-4 由 HENG958 搜集、shuidisjtu 核对，原始包校验通过且执行端可取用，见[收据](data/p1_2_r_raw_receipt_20261014.json)。ADNI 槽位使用非 ADNI 官方数据的 Kaggle 替代集，shuidisjtu 接受当前交付，见[替代数据记录](data/adni_substitute_receipt_20261012.json)；不据此代填 HENG958 的明确复核。许可核查结果保留，当前继续研究实验；未声明/Unknown 不代表无使用限制，公开发表或再分发前重新确认使用条件，不阻塞当前实验。 |
| P1.3a | 方法台账 | 冻结 Pilot 为 7 个 PU 方法 + PN oracle；新增方法状态见 P3.1/RV8 |
| P1.3b | Survey 脚本 | 四路输入、PA/OA、归档与 oracle 入口已有脚本级证据 |
| P1.4 | Pilot 数据产物 | 15 split / 75 文件 / 3 归档技术核验通过；[证据](delivery/p1_4_review.md)，HENG958 复核见 RV1 |
| P1.4-R | 8 数据集预处理与正式 split | **2026-10-16 完成（用户确认）**。实验所需 8 个数据集均已完成预处理与划分，由 shuidisjtu 与 HENG958 分别负责部分数据集的收集并互相验收对方产物；`data/` 下产物满足实验协议，不再另要求 HENG958 签署。五数据集原交付 10-15 接收验收通过（25 split/125 文件/5 归档，与原仓库索引一致；源标签/行/特征、四角色隔离及 train-only 预处理已核），见[接收验收](reviews/p1_4r_receiving_acceptance_20261015.md)。已知限制：ADNI 槽位为替代集，subject 级隔离不可判、许可限制保留；前三集 manifest 仅换行不同；20News 原缓存身份已验证，与重建数值差异仅记录不判等价；不混用原接收与重建身份，不覆盖旧 Pilot 或冻结协议；新主榜协议/配方/资源仍另行批准 |
| P2.0a | Pilot 共享规格与 oracle 对齐 | shuidisjtu 已签署；[交付](delivery/p2_0a_delivery.md)、[复核](delivery/p2_0a_review.md) |
| P2.0b | 标签语义门禁 | 单方技术放行；[交付](delivery/p2_0b_delivery.md)，HENG958 复核见 RV2 |
| P2.0c | 对照预注册 | 锚点与规则已冻结；[复核记录](delivery/p2_0c_review.md)，C2 裁决见 RV3，其余 C 项见 P2.2 范围调整 |
| P2.0d | SAR-OA 路径 | HENG958 已复核（2026-09-16） |
| P2.0e | 五方法 TS-OS 校准接入 | 工程已接线；[交付](delivery/p2_0e_delivery.md)，HENG958 方法学复核见 RV6 |
| P2.1 | Pilot 跑批 | 645/645 完成；[交接](delivery/p2_1_handoff_checklist.md)，结果复核由 P2.2 负责人收尾覆盖，RV7 已关闭 |
| P2.2 | Pilot 交付与审计收尾 | **2026-10-16 收尾完成（shuidisjtu）**。三个分析入口在干净的 `7ee6a30` 检出上重跑，645 份 manifest 树摘要与 10 个产物同基准一致（3 个 JSON 仅 `generated_at` 不同）；AutoDL 上五批归档摘要、PA/OA 选中权重（存在、sha256 一致、CPU 可加载）与逐 epoch 选择记录核对通过，B4 非选中 epoch 权重已回收但选中权重完好，见[核对记录](reviews/p2_2_autodl_check_20261016.md)；C2/C3 已裁决，见 RV3/RV4。范围按用户 10-16 决定：C1/C4/C5/C6 按通过处理（未逐项核查，无 HENG958 签署）。已知限制：B2 退出码仍未找到（日志显示 215/215 完成），保留“未核对”；B1 回收披露与本次所见未调和；负责人侧复验不写成合作者独立复核；原索引不改，新表不冒充旧发布身份，31 条低 AUC 不事后翻转或设新阈值 |
| P3.1 | P3MIX 完整集成与新增方法准入证据 | **2026-10-21 收口（shuidisjtu）**：`survey-v2.0` 落库；13 方法 [v2 delta](reviews/p3_1_v2_delta_20261021.md) 与 #119–#126 处置完成；[RV8](reviews/rv8_method_admission_20261021.md) 14 条签署。Split-PU JSD（#122）与 GEN-PU recipe（#126）作保留项；正式准入待 P3.3/P4.1-B |
| RV1 | Pilot 数据制品复核 | **2026-10-16 完成（用户确认）**；接受数据身份、四角色隔离及预处理技术证据，见[验收记录](delivery/p1_4_review.md)；不再另要求 HENG958 签署 |
| RV2 | 标签语义门禁复核 | **2026-10-16 完成（用户确认）**；170 项相关测试及 PNU 三值直调通过，见[验收记录](delivery/label_semantics_stage_b_20261015.md)；Issue #41 可关闭 |
| RV3 | PA 阻断映射 eligibility；C2 | **2026-10-16 完成（shuidisjtu 裁决）**；54 条 `blocked_pending_pa_criterion` 解除，归类与其 OA 对应映射一致，不新增数值裁决，见[裁决记录](reviews/p2_2_c2_c3_ruling_20261016.md)。**机器矩阵 v3 暂不更新**，以裁决记录为准，正式使用对照前须另发 v4；36 个锚点、7 条映射、4 条 contradictions 保持 `pending_review`，对照陈述继续视为未复核 |
| RV4 | PA 准则裁决；C3 | **2026-10-16 完成（shuidisjtu 裁决）**；锁定 `pu_val` proxy accuracy 准则（Def. 1 OS 分支），依据为试点 54 个配对单元的 PA/OA 差异，见[裁决记录](reviews/p2_2_c2_c3_ruling_20261016.md)；锁定不等于 PA 单元获得数值裁决 |
| RV7 | Pilot 深度结果与 C5/C6 复核 | **2026-10-16 按范围调整关闭（用户决定）**；C5/C6 按通过处理，**未做独立重放或逐项核查**，也无 HENG958 签署，见 P2.2 与[复核包](delivery/p2_2_review.md) §2.3。深度结果仍以 P2.2 的三个分析入口重跑为证据 |
| RV8 | 新增算法正式准入复核 | **2026-10-21 完成（shuidisjtu 签署）**：14 条（13 方法 + TEDn）均为有条件接受，无退回；共通前置为 P3.3/P4.1-B，个例保留项见[记录](reviews/rv8_method_admission_20261021.md). |

### 1.2 未完成任务

「当前状态」列先写已有基础，再以 **待办：** 显式列出尚需完成的事项；「完成条件」只写验收口径与边界。

| 编号 | 任务 | 主责 / 复核 | 前置 | 当前状态 | 完成条件 |
|---|---|---|---|---|---|
| P3.2 | CNN/共享特征适配与算法收尾 | **HENG958**；shuidisjtu 复核 | P3.1 收口 | 多方法已有 CNN/特征工程路径；Holistic-PU、PULNS、GEN-PU 新路径及[阶段预算证据](p3_stage_budget_review_20261007.md)已补；2026-10-19 复跑 19 个 CNN/encoder、图像协议、候选训练收据及分批预测测试文件，退出码 0；9 项 CUDA-only 测试因本机无 CUDA 跳过，不能据此声称 GPU 验收。另将非正式 oracle checkpoint smoke 的“大小不可估计”告警显式断言为预期行为；配套门禁确认 versioned pilot 在缺少协议绑定容量估算时 fail-closed。各方法的适配例外见 §1.3.4。<br>**待办：** 确定正式共享规格；获批图像/特征路径、分批训练预测、阶段预算及 checkpoint 的跨方法统一验收；说明增强/调度与原论文的适配差异；完成执行证据与合作者复核；补齐批准协议的公开数值对照与其余已接入方法逐项复核 | 交付可追溯工程证据；资源验收见 P3.3-A/B |
| PN-CNN | 端到端 CNN PN 全监督 oracle baseline | **HENG958** 实现/训练；shuidisjtu 登记协议并复核 | 新协议与存储预算登记 | `PilotOracleCNN` 已有代码和合成验证；旧协议 native CNN oracle 仍不可运行；真实 CIFAR 五 seed 未交付。<br>**待办：** ① shuidisjtu 完成新协议/存储预算登记；② HENG958 完成真实 CIFAR GPU smoke；③ 交付五 seed 权重、指标和 manifest | 用真实 train PN 标签端到端训练 ResNet-18 与分类头，与对应 PU 共享 split/表征/预算；仅用 clean_val accuracy 作 OA 选模；每 dataset/seed 一次、五 seed |
| P3.3-A | 批准结构的资源 profile | **HENG958**；shuidisjtu 复核预算依据 | P3.2 的批准结构 | [扩展资源 receipt](data/candidate_extended_cnn_storage_probe_20261007.json)已完成 7 recipe×3 seed 的独立 CPU 技术测量并通过 checker；仍是合成输入、非批准结构/正式上界，GPU 未测，不能外推正式数据集。<br>**待办：** 测真实批准结构的序列化、内存/磁盘及保守配额 | 测量覆盖批准结构；PUET 按 CPU 条款处理，不计入 GPU 覆盖 |
| P3.3-B | 正式资源与持久化验收 | **HENG958**；shuidisjtu 复核制品 | 对应候选 P4.1-B 冻结绑定；正式协议批准；frozen-lock 环境与资源可用 | 未开始（前置未满足）。<br>**待办：** 多 seed CPU/GPU 运行；OOM/重试、保存加载、selected checkpoint 对账；逐 run 环境/成本记录 | 仅验收批准范围 |
| P4.1-A | 注册表绑定工程 | **shuidisjtu**；HENG958 复核接口与方法参数映射 | 无（与 RV5/RV6 并行） | runner/CLI、pilot、外部白名单/审计/汇总、draft CI 及[版本化候选池身份](delivery/p4_1_pool_binding_20261014.md)已接通，含全字段删除拒收、PA/OA 独立候选与 trajectory 对账。正式 registry 的条件 CI 已接入：[draft schema 无条件检查；正式 v1 文件出现时执行 `--require-present --require-locked`](delivery/p4_1_production_ci_20261016.md)。冻结前该文件不存在，故正式门禁尚未实际验证，配方也未冻结。<br>**待办：** ① 冻结正式 registry 时将 CI 条件改为无条件 fail-closed，并核对真实 CI 结果；② HENG958 复核接口与方法参数映射 | 正式使用须采用外部 v2 白名单，不把 draft 当 locked 或代签；新增参数搜索仍需协议/registry 语义与预算批准 |
| P4.1-B | 配方审阅与冻结 | **shuidisjtu**；HENG958 核对配置与实现一致性 | P4.1-A 就绪；候选/选模/标签预算与 P3.3-A 资源依据齐备 | 未开始（前置未满足）。<br>**待办：** 双方审阅配置；登记实际批准范围、正式 registry/resolved snapshot 与版本摘要 | 此为候选/预算批准，不代替 RV8 最终方法复核，不回填旧 manifest |
| P4.2 | 主榜运行结果聚合与分析 | **shuidisjtu**；HENG958 复核深度结论 | 拟纳入数据、PN-CNN、方法准入、P3.3-B/P4.1-B 与正式运行制品均具备 | 未开始（前置未满足）。<br>**待办：** 聚合并分析主榜运行结果 | 22 项通过门禁；按机制、PA/OA、预算族及训练路径分层；区分文献事实/观测/推断，不生成跨数据集总排名 |
| RV5 | Pilot Self-PU 元重加权/消融口径复核 | **HENG958**；shuidisjtu 提供既有 Pilot 协议证据 | 既有 Pilot 协议证据 | Pilot 实际为 ablation，不是完整版。<br>**待办：** 确认 clean 标签监督、消融范围与 PA 资格 | 记录 HENG958 方法学结论；不将消融说成完整论文复现 |
| RV6 | 五方法 TS 接线复核 | **HENG958**；shuidisjtu 提供既有接线证据 | 既有接线证据 | `nnpu`/`upu`/`pusb_kernel`/`dist_pu`/`self_pu` 工程接线已有；2026-10-19 已补[逐方法作用范围/不变量技术核验](reviews/rv6_ts_calibration_technical_review_20261019.md)，五方法及公共路由专项测试通过。**待办：** HENG958 确认 mini-batch 校准、PUSBKernel 的 TS 训练/OS 验证选择口径，以及分布损失/加权负风险的适配结论 | 记录 HENG958 逐方法结论；工程技术建议和路由回归均不代替方法学复核 |

### 1.3 执行边界

本节只列仍约束执行的边界与例外；逐方法来源、行为与配方细节见方法卡与 `admission/`，
处置见 [13 方法 v2 delta](reviews/p3_1_v2_delta_20261021.md) 与
[RV8 记录](reviews/rv8_method_admission_20261021.md)。2026-10-14–20 的逐日工程流水
已归档在 git 历史与上述记录，不再逐条重复。

#### 1.3.1 冻结与版本

- 协议以 `survey_protocol_v2.json`（`survey-v2.0`）为准；`survey_protocol_v1.json` 与
  `survey_comparison_v1..v3.json` 仅作历史绑定，对照矩阵以 `survey_comparison_v4.json` 为准。
- 冻结文件只在版本化提案中修改；新数据、方法或配方登记新协议，不覆盖旧 645-run 结果、
  旧 manifest 或历史签署。
- **P3.1 完成 ≠ 正式准入**；正式准入仍待 P3.3 资源、P4.1-B 预算/配方冻结与 P4.2 门禁。

#### 1.3.2 数据、角色与标签预算

- train / pu_val / clean_val / test 四角色严格隔离；test 不参与先验、候选、阈值或
  checkpoint 选择。
- clean support 从 train 切出并移除，与其余四路不相交；规模 Spambase 100、其余 500，
  正负标签计入训练期标签预算；覆盖 Spambase、Connect-4、MNIST、F-MNIST、CIFAR-10。
- PULNS、LaGAM 与 Self-PU 完整分支需要该 support；无 support 的 Self-PU 路径只作消融。
- ADNI 槽位为替代数据，subject 级隔离与许可限制保留；未声明/未知许可不阻塞研究实验，
  公开发表或再分发前重新确认。

#### 1.3.3 选模与 PA/OA

- PA 只在 `pu_val` 上用 proxy accuracy 选配置；OA 只在 `clean_val` 上选；两者不混用。
- `PA+clean-support` 与纯 PU PA 同表另列，不参与纯 PU 优势比较；PULNS/LaGAM 只报 OA
  或 `PA+clean-support`。
- checkpoint 完成后离线独立选择；推理快照不用于续训 optimizer。

#### 1.3.4 来源与复现声明

- `official_exact` 只表示作者来源已锁定，不表示逐行一致或数值复现；`official_related`
  表示相关来源；无来源保持 paper-only。所有适配统一标 `benchmark-adapted`。
- 不照搬作者 test accuracy 选模。Self-PU 作者代码用 test split 做 meta 更新并按测试
  真值选 checkpoint 的事实已登记；工具箱 clean-support 路径不复制该行为。
- 仍影响执行的例外：

| 方法 | 已登记差异 | 处置 |
|---|---|---|
| Dist-PU | clean-room 全量目标与作者两阶段、scheduler、直方图目标不同 | 另有显式 `author_staged` 路径；默认 clean-room 不变 |
| Split-PU | 论文 Eq. (6) 与源码 JSD 权重方向相反；TS 均衡 batch ≠ 总体 π | #122 待方法负责人裁决 |
| PAN | 论文 Eq. (7) 与源码目标、采样不一致 | 保留 paper-faithful；正式采样跟随论文 OS |
| CVIR | `α_U` 为 U 内比例；原生 TS 未接校准 | TS 主表门未闭合，P3.2 决定 |
| Grad-PU | 论文要求移除 BN | `cnn13_no_bn` 单列 method-private，附 SHOULD 共享对照 |
| PULNS / LaGAM | clean support 是额外监督；PA 不适用 | 只报 OA 或 `PA+clean-support`；P3.2 |
| GEN-PU / Holistic-PU | 作者归档未发现 LICENSE | `license=unknown`；不复制、执行或再分发 |
| KLDCE / PUSB 线性 | 求解器或来源身份与论文/官方路径不同 | 方法卡与台账已披露，不改求解策略 |

#### 1.3.5 资源与预算

- 正式资源验收使用批准结构、frozen-lock 环境与逐 run 记录；合成输入、缩短 smoke 或旧
  backbone 探针不能替代 P3.3-B。
- PULDA 120、GradPU 200 快照与正式候选的存储/峰值尚未实测；正式候选池、epoch/batch
  与资源配额在 P3.3/P4.1-B 批准前保持 `null`/`false`。

#### 1.3.6 推进顺序（2026-10-21）

1. **P3.2**：共享规格、`PA+clean-support` 路径、CVIR 校准/`α_U` 结论、support 制品。
2. **P3.3-A**：批准结构的资源 profile。
3. **P4.1-B**：候选/预算与配方冻结。
4. **P3.3-B**：正式多 seed 资源与持久化验收。
5. **P4.2**：主榜聚合；22 项门禁全部通过后才发布。

#### 1.3.7 留痕

- 每项任务只有一名主责；证据、结论、身份与日期由主责记录，代理检查不代替签署。
- 退回项另记整改；Issue/PR 记录作为处置留痕，不把工程通过写成方法准入。

## 2. 风险与留痕约定

### 2.1 风险

1. **GPU 与正式环境**：正式资源验收须使用批准配置、frozen-lock 环境、兼容的驱动/依赖和逐 run 资源记录；合成 CPU/CUDA 探针不能代替 P3.3-B。
2. **数据与标签**：公开原始数据不等于许可、正式 split 或四角色验收；新数据和版本变化必须留来源、映射、授权范围与摘要，ADNI 官方数据未获批不得纳入；`adni` 槽位使用的是替代数据，须按 D1 的修订标注。
3. **方法准入**：代码级接入、方法卡、smoke、资源 profile 或阶段批准均不等于正式准入；P3.1 收口与 RV8 签署仍不等于正式准入，PULNS/LaGAM 等额外标签预算单独裁决。
4. **存储峰值**：D25 回收发生在选模和 test 评测之后，只降低最终留存，不降低训练期间峰值；新候选按全部 epoch × component × candidate × retry 估算，不外推缩短探针。

### 2.2 留痕约定

- Artifact 固化代码/依赖身份、Python/PyTorch/CUDA/GPU、方法/数据配置、注册表版本、seed、split/label manifest、选择 artifact 与结果 schema；缺必要身份的结果不得进入主表。
- 历史 Pilot 的代码 commit 和依赖锁身份由批次计划/证据包承载，不要求逐 run manifest 重复记录；P4.1-A 未完成前不把 draft 当 locked，不回填历史 manifest。
- 工程资源探针记录实际源码摘要；旧读数对应测量时源码，后续修复不自动构成重测证据。
- 2026-10-07 扩展 CNN receipt 已绑定 `4bfa59b` 基线及逐文件 SHA256；其 21 条 profile 仅证明当前合成 CPU 路径可复核，不解除 P3.3-A/B 的批准结构、GPU、OOM/重试或正式制品前置。
- 复核结论记录主责人、复核人、代码/制品身份、证据、日期和整改状态；`接受`、`有条件接受`、`退回`不得混写。
- 新阶段进展只更新本节任务表；新增或修订决策追加附录 A，不重写历史提交或冻结协议。

## 附录 A. 决策账本（D1–D26）

> 本附录只保留仍影响执行的决策摘要；当前任务与责任以 §1 为准，实验规则与比较矩阵以
> `pu_survey_protocol.md` 和机器可读 JSON 为准。实现细节以对应交付文档、代码与提交历史为准。

| # | 决策 | 内容 | 日期 |
|---|---|---|---|
| D1 | **ADNI 数据获取** | ADNI 官方数据仅在获批持有人提供授权材料后纳入，未取得不记作官方 ADNI 获取完成。**2026-10-12 修订**：经与学长、导师讨论，为避免流程过于耗时，`adni` 槽位改用 Kaggle 4 类 Alzheimer MRI 的 GitHub 转存（论文 2 的 Table B.1 称其 Alzheimer MRI，附录称 ADNI），固定在转存仓库 commit 与逐文件摘要；它不是 ADNI 数据，所有登记、报告与快照须写明「替代数据」；Kaggle 页面许可为 Unknown（2026-10-12 核对），同一受试者切片是否跨集未核实，见[替代数据记录](data/adni_substitute_receipt_20261012.json)。 | 2026-09-08（修订 2026-10-12） |
| D2 | 协议分工执行口径 | 研究团队准备四角色数据，工具箱 `fit` 消费已划分输入；工具箱不替代数据划分。 | 2026-09-08 |
| D3 | 依赖锁与 CUDA 环境落地 | `uv.lock` 入库，PR 使用提交的锁，nightly 用 `--upgrade` 重新解析；正式 GPU 运行另核驱动与依赖身份，详见 [ADR-0012](../../adr/0012-dependency-release-policy.md)。 | 2026-10-05 |
| D4 | issue #41 阶段 A/B 拆分 | 标签语义阶段 A 服务 Pilot 门禁，阶段 B 覆盖 pipeline/CLI/UI 收口；实现细节见[P2.0b 交付](delivery/p2_0b_delivery.md)。 | 2026-09-14 |
| D5 | Self-PU `input_ndims` 恢复 `{2,4}` | Self-PU 支持四维输入不代表原生 CNN；展平/adapter 能力与端到端图像训练须分开报告。 | 2026-09-14 |
| D6 | PU-Bench PN 行降级为背景参考 | PU-Bench PN 用 PU-only proxy 选模，本项目 PN 用 clean validation accuracy；其 PN 数字仅作背景参考，不参与数值裁决。 | 2026-09-14 |
| D7 | SAR 标记频率口径修正 | SAR 的 c 网格为 `{0.05,0.5}`，SCAR 为 `{0.1,0.3,0.5}`。 | 2026-09-15 |
| D8 | SAR 执行路径设计（issue #43） | 机制与方法正交；SAR 仅 OA，c 使用规范 token，目录及 manifest 保留请求值，不接受同值异拼写。 | 2026-09-15 |
| D9 | SCAR 标记数取整口径 | 标记数采用协议的 `round(c·n₊)`，记录实际 c；PU-Bench 的向下取整属于协议差异。 | 2026-09-06 |
| D10 | **对照矩阵修订（v1 → v2）** | 修正 15 个 PU-Bench 映射中错误的预处理差异登记，并补入 case-control 与我方 `ts` 等价的训练视图差异，发布 `survey_comparison_v2.json`。仅改 `protocol_differences`，锚点、判定规则与资格不变；v1 留档。IMDB 差异由 D12②继续更正。。 | 2026-09-23 |
| D11 | **split 分母口径与 c 取整的补登记** | ① split 先从全量池取 20% test，再按剩余池 90/5/5 分 train/PA/OA；因此全量比例为 72/4/4/20，“5%”的分母是去掉 test 后的池。② 标注数按 `round(c·n₊)`，PU-Bench 代码向下取整；这是需登记的协议差异，不改变既有 split。。 | 2026-09-23 |
| D12 | **P2.0b/P2.0c 单方放行 + 对照矩阵 v3（F8）** | ① P2.0b/P2.0c 经单方技术验收，移除对应正式阻断；对照矩阵的 HENG958 复核状态与阻断不变，不等于签署。② 更正 IMDB L2 预处理差异，发布 v3，仅改差异登记，v1/v2 留档。③ 协议摘要 `c15b0c9e…` → `c019a87d…`，版本仍为 `survey-v1.2`，三版矩阵同步重绑。。 | 2026-09-25 |
| D13 | **F10 修复：聚合按实际运行视图分区** | 聚合层按 `(comparability_group, run_view)` 分区，下层公平性校验保持 fail-closed。`run_view` 是实际视图，不是原生假设；其与校准标记不一致则拒收。Oracle 仅属 OS，不广播至 TS。报告 schema 升至 1.1；mechanism 分区由 D26 补充，覆盖率格网遗留见 D26。。 | 2026-09-26 |
| D14 | **F11 修复：续跑状态按实际训练视图限定** | 续跑键改为 `(manifest_identity, run_view)`，按逐单元预期视图判断完成。Pilot 与单元入口共用视图解析和校验；聚合器遇非法制品终止，续跑扫描视为未完成。显式 `ts` 须提供 estimator class 并在启动任何 batch 前校验，不允许静默降级；作用域内含非法方法或 oracle 时拒绝。。 | 2026-09-26 |
| D15 | **F12 修复：checkpoint 体积按训练路径与架构三元组分派** | Checkpoint 体积按 `(training_path, backbone, model_family)` 登记，区分 CNN 全网与 adapter head；签署的 ResNet18 45 MiB 常量不变。未知有 epoch 组合 fail-loud，`None` 仅表示不产出逐 epoch 权重；新增 profile 须附序列化证据，Pilot 在启动子进程前预检。不改保存频率和训练预算；旧估值勘误见[P2.0a 交付](delivery/p2_0a_delivery.md) §6。。 | 2026-09-26 |
| D16 | **P2.0e 逐方法接线口径** | ① 区分适用方法接线与线性 `pusb` 的适用性裁决，后者处置由 D20②落地。② 各方法须独立推导，不推广 UPU 规则。③ UPU 的无标签风险及 RBF 中心候选池用 `U ∪ P`，中心数仍受校准前 `n_U` 上限约束，避免引入容量差异。④ 复核未获、阻断移除的后续口径见 D20①。。 | 2026-09-26 |
| D17 | **P2.0e：`pusb_kernel` 接线口径** | 仅适用于 `pusb_kernel`。① CV 训练折与最终 refit 的无标签风险用 `U ∪ P`，分母为 `n_U+n_P`，正例项/先验/正则不变。② 验证折保持 OS，选参目标不随视图变化，TS 不是“TS 最优调参”基线。③ RBF 候选池本已为完整 X，不套用 UPU 规则。⑤ 属协议选择而非论文事实；复核状态见 D20。细节见[PUSB 方法卡](../method_cards/PUSB.md)。。 | 2026-09-27 |
| D18 | **P2.0e：`dist_pu` 接线口径** | 仅适用于 `dist_pu`。① alignment/entropy 角色改用 `U ∪ P`；正例 BCE、总体 π、Mixup 池不变。② alignment 有 population identity 依据，③ entropy 扩展是协议选择，不能等同数学必然。⑤ 仅锁 RNG 调用与抽样结构，不锁 loss/梯度；⑥ `sample_weight` 为 `IGNORED`，不套用其他方法策略。观测范围、其余不变量见[方法卡](../method_cards/Dist-PU.md)，复核状态见 D20。。 | 2026-09-27 |
| D19 | **P2.0e：`self_pu` 接线口径** | 仅适用于 `self_pu`。① 负 PU 角色用“未信任 U ∪ P”，不改身份或增加前向；② 行数质量混合中 α_P≠π。③ 均匀 U 权重时等于并集均值；clean-meta 是质量守恒扩展，非论文 TS→OS 公式。④ trusted/pace/伪标签/meta/consistency 等不随视图；⑥ 验证选模保持 OS。⑧ Pilot 恒为 ablation，不代表含 meta 的完整方法。⑤ 既有一致性项收窄及其余不变量见[方法卡](../method_cards/Self-PU.md)；⑨ 放行口径由 D20 取代。。 | 2026-09-27 |
| D20 | **P2.0e 单方放行（承接 D12 的做法）** | ① 五个适用方法的 TS 接线经单方技术验收，移除 `ts_view_collaborator_review`；方法学复核仍未获，历史制品不改写。证据见[P2.0e 交付](delivery/p2_0e_delivery.md)。② 线性 `pusb` 不适用校准，以本记录完成明确处置，不另产裁决文件。③ P2.0e 工程完成，不等于 HENG958 对既有接线的签署或新方法准入。。 | 2026-09-27 |
| D21 | **P3.1 局部：VPU 采样假设裁决与训练视图接入口径** | ① VPU 原生视图裁决为 `ts`，边缘池用训练分区 `P ∪ U`；② OS 是协议消融，③ 不登记为 `both`，此处 TS 是视图语义，不要求独立两样本生成。④ **验证口径由 D22 更正**，不再采用“与上游分歧”的旧判断。⑤ 既有适配缺口见[VPU 审计](admission/vpu_sampling_audit.md)。⑥ 接线及显式 OS 路由随后落地（D23），不代表新方法准入或 §1 列示复核完成。 | 2026-09-28 |
| D22 | **VPU 验证变分风险的定位：source diagnostic，不作选模准则** | ① 更正 D21④：VPU 验证风险保持上游 `U_val ∪ P_val` 定义。② 协议的 OS 验证约束用于选参/裁决，source diagnostic 不受该约束。③ 允许记录和视图对照，但禁止作 PA/OA 选模；未来进入矩阵且需 PA 选模时须另行裁决。④ D17②/D19⑥不能推广到 VPU。⑤ 仅补协议文档，不改代码或协议 JSON；⑥ 不构成 P3.1 完成。⑦ 2026-10-18 复核补注：上游 `val_p_loader` 由全量 clean 类标签抽样，并非 `val_x_loader` 的正例子集；D22 的 source diagnostic 裁决不变，采样差异见[复核](admission/vpu_d21_d22_review_20261018.md)。 | 2026-09-28 |
| D23 | **`route_training_view` 的非对称转发规则（协议级）** | ① `ts` 请求对未声明目标仍 fail-loud；显式 `os` 仅向 `fit` 具名声明 `os_or_ts` 的目标转发，`None` 不转发，`**kwargs` 不算 OS 声明。② 防止默认 TS 的 VPU 丢失显式 OS 请求；③ 非对称是刻意设计。⑤ 替代方案及来源见原文和[P2.0e 交付](delivery/p2_0e_delivery.md) §11。⑥ 不改方法默认视图或冻结矩阵，不构成 P3.1 完成。 | 2026-09-28 |
| D24 | **放行三个 manifest 正式资格阻断位（承接 D12/D20 的做法）** | ①② 移除旧 Linux 环境偏差、Self-PU clean-meta OA 集成与 PA 待签署三个静态正式阻断；保留逐 epoch 独立选模的动态阻断、`protocol_deviation` 和 Self-PU 消融标记，不代替 HENG958 对 PA/Self-PU 协议的复核。③ 摘要 `c019a87d…` → `287c2f45…`，版本仍为 `survey-v1.2`，矩阵与 README 同步重绑。④(a) 54 条 PA 映射仍不能数值裁决；(b) 历史制品不改写；(c) 同一 `(dataset, seed, c)` 单元整组重跑，不混新旧摘要；(d) 不放行方法学复核。 | 2026-09-28 |
| D25 | **逐 epoch checkpoint 回收：保留选中、回收其余（容量治理）** | ② PA/OA 选模与 test 评测后、manifest 前，仅逐文件回收未选中权重；保留选中权重与全部 epoch 元数据，原 `path`/`sha256` 加 `reclaimed`，不改成 `path=null`。③ 复现范围为“选中权重＋全部选择记录”，不关闭 capture，不降低训练期峰值。④ 选模、阈值与 test 指标不变，B3a 无需重跑；**B3a/B3b 全量留存、B4 起启用回收**，见[交付记录](architecture/epoch_checkpoint_delivery.md) §4。⑤ 不动动态阻断，默认关闭，不放行方法学复核。 | 2026-09-30 |
| D26 | **F13 修复：聚合分组补上 mechanism 维度** | 聚合键由 D13 扩展为 `(comparability_group, run_view, mechanism)`，避免 SCAR/SAR 在相同 c 下混组。只改聚合分区，下层公平性与可比性校验不变；实际视图/机制作为分组维度，不改协议 group 值。实现与验收见[P2.2 交付](delivery/p2_2_delivery.md)；D13 的覆盖率格网问题未由本修复解决。。 | 2026-09-30 |
