# PU 调研实验执行计划（pilot → 主榜）

> 本文件维护 Survey 的实验能力、缺口、预算/资源口径与执行边界，以及仍影响执行的决策摘要。
> 实验协议和比较矩阵分别以 [`pu_survey_protocol.md`](pu_survey_protocol.md)、
> `pu_toolbox/experiment/survey_protocol_v3.json`（执行协议）与 `survey_comparison_v4.json` 为准；
> `survey_protocol_v2.json` 是 645-run pilot 的历史绑定（比较矩阵仍绑它），
> `survey_protocol_v1.json`/`survey_comparison_v1..v3.json` 仅作历史留档。
> 状态核对日期：**2026-10-24**（v3 协议已冻结、runnable 40/64；25 份 clean-support 制品本机重建；GEN-PU 配方与 PULDA/GradPU 图像 runner 已补；PN-CNN 5 seed 继续暂缓）

## 1. 实验能力与缺口

> 本节按“PU Survey 实验设施现在能做什么”组织；任务编号只作关联。状态分三层，避免把工程接线
> 当结果：**runnable**（`survey_protocol_v3.json` 的 `execution_units` 标 `runnable=true`，
> 有预算/runner，能产出合规 manifest，当前 **40 / 64**）、**formal**（清掉 `formal_blockers`、
> 冻结 P4.1-B 预算并绑进比较矩阵，当前 **0**）、**evidence**（已有真实数据读数，目前只有
> 单 seed 工程 smoke，没有正式多 seed/GPU 批次）。判定标准：命令可执行并产出合规 manifest
> 才算“可用”，只有代码存在或测试通过不算。状态核对日期：2026-10-24。

### 1.1 能力矩阵

| 能力 | 现在能用？ | 缺什么 | 补完后怎么算可用 |
|---|---|---|---|
| 8 数据集四路 split | **是** | — | `run_survey_pilot.py --dry-run` 能列出单元并读取 split |
| clean-support 制品（5 数据集 × seeds 0–4） | **是（本机，2026-10-24 重建）** | 无；收据属新批次，跨机传递仍须按收据核验 | 25 份逐份 `load_support_dataset` 通过并登记[机器清单](data/clean_support_materialization_20261024.json) |
| 协议内单跑（v3 40 个单元） | **是** | 覆盖面只有 spambase/imdb/cifar10 | `--protocol survey_protocol_v3.json` 跑通并产出 manifest |
| 5 方法阶段预算（GEN-PU/PULDA/Robust/Split/Holistic） | **是（工程）** | 正式 P4.1-B 冻结 | 阶段快照/成本收据与运行 manifest 对账 |
| 图像三条路径 | **部分**：GEN-PU/PULDA 原生 CNN 与 Robust/Split/Holistic 的 `cnn_feature_adapter` 已标 runnable；GradPU 私有 `cnn13_no_bn` 装配已通、行仍 blocked | GradPU 首次真实运行的 checkpoint 契约；全量预算 | 行翻 runnable 并出真实 CIFAR manifest |
| clean-support 主链（PULNS/LaGAM/Self-PU full-meta） | **是**（spambase seed0 三方法已跑通） | 5 数据集正式 units/预算 | 正式 support 批次 manifest 与制品对账 |
| PA/OA 选模（含 `PA+clean-support` 双轨） | **是** | PA 准则正式放行（`per_epoch_independent_PA_OA_checkpoint_selection`） | P2.0c 规则落地后解除阻断 |
| 分组跑 + 汇总 | **是** | — | `--methods/--datasets/--training-paths` 分批 + 汇总入口 |
| v3 ↔ comparison/registry 绑定 | **否** | 运行逐条记 `missing_versioned_profile_mapping`；`survey_comparison_v4.json` 仍绑 `survey-v2.0` | 版本化 profile 映射落地，结果进比较矩阵 |
| 22 方法主榜 | **否** | 24 个 blocked 单元 + 只有 3 个数据集 + 上一条绑定 | P4.2 门禁 |

### 1.2 缺口任务

只列仍未关闭的缺口；已完成项并入 §1.1，不再留在这里。

| 缺口 | 影响 | 任务 | 负责人 | 完成判据 |
|---|---|---|---|---|
| 24 个 blocked 单元 | CVIR/KLDCE/P3MIX/PAN/PUET/RP/VPU 全行、GradPU 全行、LaGAM/PULNS 的 cifar10 行不能跑 | 逐方法补预算与 runner，按各自 `non_runnable_reason` 一条条翻 runnable | HENG958 | 单元 runnable 且产出 manifest |
| v3 ↔ comparison/registry 绑定 | 结果进不了比较矩阵 | 版本化 profile 映射，重绑 comparison | shuidisjtu | 运行不再记 `missing_versioned_profile_mapping` |
| P4.1-B 正式预算冻结 | 没有正式准入 | 5 方法阶段预算转正；其余方法族补冻结预算 | shuidisjtu 起草，HENG958 核对 | P4.1-B 记录 |
| recipe registry 仍停在 v0.1 draft | 只覆盖 upu/nnpu/kldce/dist_pu/pusb_kernel/lbe/self_pu/pn_oracle 且绑 `survey-v2.0`；v3 中需要 recipe 的 5 个方法（GEN-PU、GradPU、PAN、VPU、Holistic-PU）无登记 | 把 recipe 登记扩到 v3 并纳入这 5 个方法 | shuidisjtu 打包，HENG958 出配方数值 | registry 绑 v3，每方法有 recipe 摘要与证据 |
| support 正式 units（5 数据集） | v3 只有 spambase 的 LaGAM/PULNS/Self-PU full-meta 三行 | 为三方法补 connect_4/mnist/fashion_mnist/cifar10 单元 | HENG958 | 5 数据集 support manifest 与制品对账 |
| 数据集覆盖（v3 仅 3 个） | 主榜缺 connect_4/mnist/fashion_mnist/adni/twenty_newsgroups | 补这些数据集的 units 与预算 | shuidisjtu | 每数据集至少一行 runnable |
| GPU 机时实测 | 无法排 AutoDL 正式批次 | GradPU 首跑、三个 adapter 行真实 CIFAR 单 seed、PN-CNN 单 seed 计时与资源峰值 | shuidisjtu | 实测表反推机时 |
| PA 准则正式放行 | PA 行仍带 `per_epoch_independent_PA_OA_checkpoint_selection` | P2.0c 规则落地 | HENG958 | 阻断位移除 |
| #122 Split-PU 裁决 | Split-PU 口径未定 | 负责人裁决 JSD 方向/分母与 TS 比例口径 | HENG958 | 裁决记录（测试全绿不等于关闭） |
| PN-CNN 5 seed | oracle 行 | 协议定稿后再安排（D30） | — | 另行安排，不标完成 |

### 1.3 预算与资源

**预算（P4.1-B 前仍是工程口径）**

- 已有工程冻结（5 方法阶段预算）：GEN-PU 100、PULDA 120、Robust-PU 30、Split-PU 40、
  Holistic-PU 130 份快照，逐组件 8–64 MiB；
- 已有显式 budget 令牌：PULNS、LaGAM、Self-PU full-meta（spambase 行）；
- 仍是 draft：CVIR、GradPU、KLDCE、P3MIX、PAN、PUET、RP、VPU 只有 `draft_*` 令牌，未冻结；
- `seeds`：0–4；`c_tokens`：SCAR 0.1/0.3/0.5，SAR 0.05/0.5；
- 标签预算：|P|、clean support 大小（Spambase 100、其余 500）。

这些参数保持统一与不变是为了维护实验的公平性。冻结后不按运行批次临时改；
P4.1-B 完成前不把 draft 预算当 locked 预算。

**资源**

- CPU/GPU 时间默认不限，不作门禁；
- 单 run 峰值磁盘/内存按方法实测或按参数×快照估算；P2.1 pilot 的 B4 任务参考为 8.34 GiB → 选后
  42.5 MiB；
- 只保留选中 checkpoint，其余按 D25 回收；按组清理，不保留全过程权重；
- P3.3 因此只做“峰值估算 + 保留规则 + 一次 smoke”，不再作为“资源批准”门。

**分批执行**

- 按 method、dataset、training-path（含 backbone）或 labeling mechanism 分组跑，
  `run_survey_pilot.py --methods/--datasets/--training-paths` 原生支持，v3 单元用
  `run_survey_experiment.py --protocol ...` 逐单元执行；
- 每组 manifest 绑定同一协议/比较摘要，汇总时按
  `(comparability_group, run_view, mechanism)` 分区；
- 不要求同批跑完，也不要求一次性占用全部磁盘；汇总只在最后统一做。

### 1.4 执行边界

#### 1.4.1 冻结与版本

- 执行协议为 `survey_protocol_v3.json`（`survey-v3.0`，`accepted`；runnable 40/64）。
  `survey_protocol_v2.json`（`survey-v2.0`）继续是 645-run pilot 的历史绑定，不被覆盖；
  `survey_protocol_v1.json` 与 `survey_comparison_v1..v3.json` 仅作历史绑定。
- 比较矩阵 `survey_comparison_v4.json` 仍绑 `survey-v2.0`：v3 结果在版本化 profile 映射落地前
  不进比较矩阵（运行逐条记 `missing_versioned_profile_mapping`）。
- `runnable` 只表示实现能力，不是正式准入。v3 的 `formal_blockers` 未清、P4.1-B 预算未冻结。
  v3 中 `formal_blockers` 与 `draft_scope.limitations` 的措辞仍早于最后两笔接线（图像行、
  clean-support 行），与 `execution_units` 已不一致；在真实运行对齐之前按“未验证”处理，
  不据此改准入。
- 冻结文件只在版本化提案中修改；新数据、方法或配方登记新协议，不覆盖旧 645-run 结果、
  旧 manifest 或历史签署。
- **P3.1 完成 ≠ 正式准入**；正式准入仍待预算冻结与主榜门禁。

#### 1.4.2 数据、角色与标签预算

- train / pu_val / clean_val / test 四角色严格隔离；test 不参与先验、候选、阈值或
  checkpoint 选择。
- clean support 从 train 切出并移除，与其余四路不相交；规模 Spambase 100、其余 500，
  正负标签计入训练期标签预算；覆盖 Spambase、Connect-4、MNIST、F-MNIST、CIFAR-10。
  25 份制品已于 2026-10-24 在本机重建（`data/clean-support-20261024-u5axe92t`，
  [清单](data/clean_support_materialization_20261024.json)），v3 目前只有 spambase 上的
  LaGAM/PULNS/Self-PU full-meta 三个 support 单元。
- PULNS、LaGAM 与 Self-PU 完整分支需要该 support；无 support 的 Self-PU 路径只作消融。
- ADNI 槽位为替代数据，subject 级隔离与许可限制保留；未声明/未知许可不阻塞研究实验，
  公开发表或再分发前重新确认。

#### 1.4.3 选模与 PA/OA

- PA 只在 `pu_val` 上用 proxy accuracy 选配置；OA 只在 `clean_val` 上选；两者不混用。
- `PA+clean-support` 与纯 PU PA 同表另列，不参与纯 PU 优势比较；PULNS/LaGAM 只报 OA
  或 `PA+clean-support`。
- clean-support 主 CLI 已接通：制品目录同时作 `data_dir`，`--split-ref` 指向制品自身
  `support_manifest.json`；用 `data/splits/*` 配对会因 train 已移除 support 被拒。
- checkpoint 完成后离线独立选择；推理快照不用于续训 optimizer。

#### 1.4.4 来源与复现声明

- `official_exact` 只表示作者来源已锁定，不表示逐行一致或数值复现；`official_related`
  表示相关来源；无来源保持 paper-only。所有适配统一标 `benchmark-adapted`。
- 不照搬作者 test accuracy 选模。Self-PU 作者代码用 test split 做 meta 更新并按测试
  真值选 checkpoint 的事实已登记；工具箱 clean-support 路径不复制该行为。
- 仍影响执行的例外：

| 方法 | 已登记差异 | 处置 |
|---|---|---|
| Dist-PU | clean-room 全量目标与作者两阶段、scheduler、直方图目标不同 | 另有显式 `author_staged` 路径；默认 clean-room 不变 |
| Split-PU | 论文 Eq. (6) 与源码 JSD 权重方向相反；TS 均衡 batch ≠ 总体 π | #122 待HENG958裁决 |
| PAN | 论文 Eq. (7) 与源码目标、采样不一致 | 保留 paper-faithful；正式采样跟随论文 OS |
| CVIR | 原始 U 的 α_U 与总体 π 不同；已补全局风险池校准与视图先验换算的独立工程入口 | v3 已声明 3 个单元但全 blocked；预算/runner 与真实数据/GPU 仍待，不把工程对照当论文数值复现 |
| Grad-PU | 论文要求移除 BN | `cnn13_no_bn` 单列 method-private，附 SHOULD 共享对照 |
| PULNS / LaGAM | clean support 是额外监督；PA 不适用 | 只报 OA 或 `PA+clean-support`；主 CLI 已接通，正式批次待 P4.1-B |
| GEN-PU / Holistic-PU | 作者归档未发现 LICENSE | `license=unknown`；不复制、执行或再分发 |
| KLDCE / PUSB 线性 | 求解器或来源身份与论文/官方路径不同 | 方法卡与台账已披露，不改求解策略 |

### 1.5 推进顺序与推进记录

**当前推进顺序**（2026-10-24，与 §1.2 的缺口一一对应）：

1. **24 个 blocked 单元**（HENG958）：CVIR/KLDCE/P3MIX/PAN/PUET/RP/VPU/GradPU 全行与
   LaGAM/PULNS 的 cifar10 行，逐条补预算/runner 翻 runnable。
2. **v3 ↔ comparison/registry 绑定**（shuidisjtu）：版本化 profile 映射，重绑比较矩阵，
   否则结果只能留在 pilot 口径。
3. **P4.1-B**（shuidisjtu 起草，HENG958 核对）：5 方法阶段预算转正，其余方法族补冻结预算；
   随后 P3.3 核对单 run 峰值、选后保留与 smoke。
4. **GPU 机时实测**（shuidisjtu）：先测后开机，只跑算得清的 AutoDL 批次。
5. **分组跑批 + 汇总、P4.2 主榜门禁**：只运行满足协议/预算/配方前置的单元。

**PN-CNN 的 5 seed 实验单独暂缓**：按用户 2026-10-22 指令，协议定稿后再安排；
不启动、不补造结果、不标记完成。该指令不取消其他方法或 support 制品的 seeds 0–4，
也不删除 PN-CNN 已有实现、历史证据或协议中的 oracle 角色要求。

以下按日期记录推进过程（最新在下）；逐项技术意见与未解除的边界见各日复核文档，
2026-10-21 交接的技术复核回复见[heng958 任务复核](reviews/heng958_task_review_20261022.md)。

2026-10-22 P3.2 工程补记：[clean-support checkpoint 双轨](delivery/clean_support_checkpoints_20261022.md)。
Self-PU/LaGAM 的显式 SCAR OS 路径已接通真实 meta、完成 meta epoch/teacher 快照、
`PA+clean-support`/OA 独立选择、容量估算和选后回收；旧终点 OA/默认消融不变。
CLI 可显式执行单候选 CPU 工程链，不代表正式单位、真实数据/GPU/预算验收；
同日第二批补齐 PULNS episode 回调和同一双轨 CLI；只保存完成 episode 的实际 classifier，
不把 reward probe/内部训练当作额外选择 epoch。v2 两个 pending review 项事后在 v3 中置 accepted。

同日后续：[CVIR 校准与先验](delivery/cvir_calibration_20261022.md)。新增显式训练
U∪P 全局排序/截断和按 PU 计数换算先验，保留默认 OS 与旧显式 α_U adapter。
独立 SCAR 工程入口可记录 PA/OA checkpoint 选择，仍拒绝 versioned Survey/SAR；
v3 的 CVIR 单元至今全 blocked。

同日数据补齐：[五数据集 clean-support 制品](delivery/clean_support_materialization_20261022.md)。
Spambase/Connect-4/MNIST/F-MNIST/CIFAR-10 的 seeds 0–4 共 25 份已在本机生成并按 raw
逐行/五角色摘要/剩余 train 预处理验收（100/500，support_seed=split_seed）。
Self-PU/LaGAM/PULNS 的真实 Spambase 单 seed 短 CPU 双轨 CLI 也已跑通并回放，
不等于正式配方、多 seed/GPU 或论文效果验收。制品不入 Git，只登记
[摘要清单](data/clean_support_materialization_20261022.json)，跨机器传递仍须核验；
该批制品当时不在本机，2026-10-24 已按同一口径重建（见下）。

2026-10-23：[GEN-PU recipe 与真实图像 GPU 交付](delivery/genpu_recipe_20261023.md)。
四项 blocker 已逐项落实工程配方/阶段与成本守卫；实际单候选命令产出 PA/OA manifest，
全宽 ResNet-18、真实 CIFAR 预固定小子集，batch8/256 短 GPU 验证均0告警/0重试并回放。
仅记录本机工程规格，不改 frozen JSON/台账或签署，不代替 v3/P4.1-B 正式预算与全量验收。
#126 未关闭；本日后续已补[PULDA/GradPU 图像 runner 与120/200峰值](delivery/candidate_image_runner_20261023.md)：
标准宽度 ResNet-18/CNN13-no-BN、固定 CIFAR 小子集各1 seed，0告警/0重试，
双轨选择回放、选后2份快照；完整快照数量不冒充全量数据/默认batch/正式预算验收。
同日后续已补[Self-PU 显式 TS/full-meta 工程双轨](delivery/selfpu_ts_support_20261023.md)，
真实 Spambase 同制品/标记/参数的短 OS/TS 运行均0告警/0重试并回放；
不修改原生风险、不宣称校准效果改善。下一步正式 full-meta/support units 与13方法 profiles/units
继续同 v3 对接，冻结正式预算/阶段资格；PN-CNN 5 seed 暂缓不变。

本日另补[v3 profile/unit 接入请求与校验](delivery/survey_v3_proposal_20261023.md)：
范围按 RV8 对齐（13方法含P3MIX、不含VPU；VPU另补，TEDn不计13），22目标算法、
Self-PU 完整版单列后23 profile、8数据集184个阻断请求。该 planning schema 不可执行，
不改 v2 基线/默认，未填正式预算或签署；下一步仍是 profile identity、正式五角色/配方、
多阶段预算校验和新版本 registry/comparison 的实际接线，不能把生成草案标成13方法可跑。

2026-10-24 已补[profile 与算法身份分离](delivery/profile_identity_20261024.md)：
显式 draft schema 下的模型、先验/视图、CLI、批次规划及完整来源摘要续跑已接通；
同算法多 profile 不污染 registry、不覆盖输出。旧 v2 保持，新扩展强制工程验证，
不挪用旧 comparison/registry；full-meta/support 与 CVIR 在缺正式适配时仍提前拒绝。
下一步转入方法族预算/阶段资格、五角色适配和新版本 registry/comparison；
未把这些工程测试或184个请求标成正式准入，PN-CNN 5 seed 仍暂缓。

同日补[GEN-PU/PULDA 显式阶段预算](delivery/staged_budget_20261024.md)：
二维 draft 的预算参数、原生架构身份、实际快照容量和阶段/优化器成本收据已接通；
GEN-PU 不再把 GAN 轮数当 PN 快照数，PULDA 计入两段但不把 TS 并集误计为额外 batch。
其他方法族、native CNN、正式预算/阶段资格及五角色/registry/comparison 仍未完成；
不据此把 P4.1-B 或13方法正式可跑标成完成。

同日后续补[Robust-PU/Split-PU/Holistic-PU 阶段预算](delivery/episodic_budget_20261024.md)：
schema 2 已接通三方法 native_2d draft 的实际 CLI/pilot 与成本收据，
Robust episode 内多轮不多造快照；Split 预留 cap、核验原生早停和逐 round 覆盖并可续跑；
Holistic LZO 回选状态仍完整计入已执行 warmup 和前向，OS/TS 边界保持。
阶段预算汇总不再投影成占位1；同快照数也须核对内层/GAN等全部显式参数。
连同前批共5方法具有阶段预算工程接线，不等于5份正式配方批准；
五角色、其他方法族、图像正式绑定、新 registry/comparison 与数值验收仍待。

同日已冻结 [v3 协议与 clean-support 主链](delivery/v3_freeze_20261024.md)：
`survey_protocol_v3.json`（`survey-v3.0`，accepted）runnable 40/64，并补 25 份
[clean-support 制品本机重建](delivery/clean_support_rebuild_20261024.md)。
重建用同 raw 字节、同准备源码，成员 25/25 与 10-22 记录一致，但收据是新的（本机 numpy
重压缩与 provenance 时间戳），不能冒充旧批次的跨机副本。主 CLI 消费时把制品目录同时作为
`data_dir`、`--split-ref` 指向制品自身 manifest；用 `data/splits/*` 配对会因 train 已移除
support 而被拒。三个 clean-support 方法已在重建制品上跑通单 seed CPU 链路，仍非正式批次。

## 2. 风险与留痕约定

### 2.1 风险

1. **运行环境**：正式运行仍须 frozen-lock 环境、兼容的驱动/依赖和逐 run 身份记录；合成探针不能代替真实 smoke。
2. **数据与标签**：公开原始数据不等于许可、正式 split 或四角色验收；新数据和版本变化必须留来源、映射、授权范围与摘要，ADNI 官方数据未获批不得纳入；`adni` 槽位使用的是替代数据，须按 D1 的修订标注。
3. **方法准入**：代码级接入、方法卡、smoke、资源 profile 或阶段批准均不等于正式准入；P3.1 收口与 RV8 签署仍不等于正式准入；support 标签预算按 v3 中已 accepted 的 `v2_p4_clean_support_policy` 执行，PULNS/LaGAM 的正式批次仍需 P4.1-B 预算冻结。
4. **存储峰值**：按 D29 只保留选中 checkpoint 并按组清理；D25 回收不降低训练期峰值，单 run 峰值仍按全部 epoch × component × candidate × retry 估算，不外推缩短探针。

### 2.2 留痕约定

- Artifact 固化代码/依赖身份、Python/PyTorch/CUDA/GPU、方法/数据配置、注册表版本、seed、split/label manifest、选择 artifact 与结果 schema；缺必要身份的结果不得进入主表。
- 历史 Pilot 的代码 commit 和依赖锁身份由批次计划/证据包承载，不要求逐 run manifest 重复记录；P4.1-A 未完成前不把 draft 当 locked，不回填历史 manifest。
- 工程资源探针记录实际源码摘要；旧读数对应测量时源码，后续修复不自动构成重测证据。
- 2026-10-07 扩展 CNN receipt 已绑定 `4bfa59b` 基线及逐文件 SHA256；其 21 条 profile 仅证明当前合成 CPU 路径可复核，不解除 P3.3 的峰值估算与保留规则验证。
- 复核结论记录主责人、复核人、代码/制品身份、证据、日期和整改状态；`接受`、`有条件接受`、`退回`不得混写。
- 新阶段进展更新 §1 能力矩阵与缺口任务；新增或修订决策追加附录 A，不重写历史提交或冻结协议。

## 附录 A. 决策账本（D1–D32）

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
| D27 | **分批执行与统一汇总** | 最终实验按 method、dataset、training-path 或 labeling mechanism 分组跑，不要求同批或一次性跑完；每组 manifest 绑定同一协议/比较摘要，汇总按 `(comparability_group, run_view, mechanism)` 分区。分组不改变可比性。 | 2026-10-21 |
| D28 | **计算资源不设上限** | CPU/GPU 时间默认不限，不作为跑批门禁；正式运行仍须 frozen-lock 环境与逐 run 身份记录。 | 2026-10-21 |
| D29 | **存储与 checkpoint 保留** | 只保留选中 checkpoint（P2.1 B4 口径），未选中按 D25 回收；按组清理，不保留全过程权重；单 run 峰值按参数×快照估算。仅个别方法超限时再优化。 | 2026-10-21 |
| D30 | **PN-CNN 5 seed 暂缓** | 用户要求 P3.2 中 PN-CNN 的 5 seed 实验等协议定稿后再安排；不视为完成或永久取消，不影响其他方法/support 的五 seed 口径，不改冻结协议及历史结果。 | 2026-10-22 |
| D31 | **v3 冻结即执行协议** | `survey_protocol_v3.json`（`survey-v3.0`）转正为执行协议，runnable 40/64；`survey-v2.0` 保留为 645-run pilot 的历史绑定；`survey_comparison_v4.json` 仍绑 v2，v3 结果在版本化 profile 映射落地前不进比较矩阵。`runnable` 只表示实现能力，正式准入另需清 `formal_blockers` 与 P4.1-B 冻结。 | 2026-10-24 |
| D32 | **clean-support 制品重建口径** | 10-22 的 25 份制品不在本机时，用同一 raw 字节与同一准备入口在本机重建，成员须与旧记录 25/25 一致；重压缩与 provenance 时间戳会改变摘要，因此记为新收据、不冒充旧批次的跨机副本；主 CLI 以制品目录作 `data_dir`、`--split-ref` 指向制品自身 manifest，用 `data/splits/*` 配对会被拒。 | 2026-10-24 |
