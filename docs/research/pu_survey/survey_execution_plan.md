# PU 调研实验执行计划（pilot → 主榜）

> 定位：本文件是**执行路线与状态**的协调层，只保留需人工阅读与决策的内容——当前状态、
> 任务分工与验收、交叉验证对照预注册、风险与留痕约定。按时间追加的决策账本（D1–D26）置于
> 文末**附录 A**，按决策号查阅；账本内的状态类补充（例如「接线状态更新（同日）」）是**当时**
> 的记录，**不代表当前状态**，现状一律以 §1 为准。实现细节见
> [pu_survey_protocol.md](pu_survey_protocol.md)（要求纲要）与
> [experiment_layer.md](../../dev/experiment_layer.md)（实验层实现）；交付证据见对应交付记录、
> 复核包与批次快照（`p2_0a/b/c_delivery.md`、`p2_0a/c_review.md`、`p2_1_b*_snapshot.md`）。
> 状态日期：**2026-10-04**。最新代理技术复核见 [Pilot 独立复核](pilot_independent_review_20261004.md)；不替代合作者本人签署。

## 1. 当前状态

本节是现状的**唯一**叙述；批次级明细见各批次快照，不在此复述。

### 1.1 运行制品

| 批 | 数据集 / 训练路径 / 方法 | runs | 状态 | 执行 commit |
|---|---|---|---|---|
| B1 | spambase / native_2d / 6 方法 + `pn_oracle` | 215 | ✅ 完成 | `7f445be` |
| B2 | imdb / native_2d / 6 方法 + `pn_oracle` | 215 | ✅ 完成 | `7f445be` |
| B3a | cifar10 / cnn_feature_adapter / `lbe` `pusb_kernel` `upu` + `pn_oracle` | 110 | ✅ 完成 | `7f445be` |
| B3b | cifar10 / cnn_feature_adapter / `dist_pu` `self_pu` | 70 | ✅ 完成 | `7f445be` |
| B4 | cifar10 / native_cnn / `nnpu` | 35 | ✅ 完成 | `ee54b5b` |

- **协议身份**：`survey-v1.2`，摘要
  `287c2f45387f02714e5925b35dcb04f3f64e1e3faf8f740cab861fc5747dd84a`；B1–B4 的 645 份
  manifest **单值一致**。`uv.lock` 的跨平台身份（git blob `cf5772f4…`）自 B0 起未变。
- **执行平台**：**AutoDL 云平台**——P2.1 跑批统一在该平台执行，跑批期间逐 run 记录实例与卡型，
  不设跨方 GPU 排期本（P2.1 与 P3.3 不共享单卡）。B0–B3b 于 `7f445be`，B4 起为 `ee54b5b`。
- **模型复核层**：B3a/B3b 保留完整批次归档，在数据盘暂存与文件存储各一份；B4 起启用逐 epoch
  回收（D25），其可持久复现范围随之收窄，见 §1.3 第 3 条。B4 完成时按此保留 **38 个选中权重**
  （逐 run 回收 199/198 个未选 epoch 权重，账目经 `reclaimed + on_disk == refs` 逐 run 核对）。
- **本文不回填批次实测值**：B4 的收尾验收与实测统计见
  [`p2_1_b4_snapshot.md`](p2_1_b4_snapshot.md) §7。原 §1 状态格与决策记录中重复的现状叙述已并入
  本节与附录 A，主文不再各写一份。

### 1.2 门禁与聚合现状

以下为**正式五批**（B1–B4，645 份 manifest）跑通后的实测值：

- 645 份 manifest 逐份通过身份预检，**0 份不可复现**、**0 份被拒收**；
- 公平性门禁分为 **48 个组**，全部通过，0 个组被拒；
- 批次级 `formal_ready = False`：**3 个组含被门禁阻断的单元**——即 B1/B2/B3a 各 5 个
  `pn_oracle` 带 `protocol_deviation: ['c_grid']`。这是 D24 保留该阻断位要标的真实偏差，
  属**正确行为**，不是缺陷。按单批边界，B3b 与 B4 均无 oracle 阻断；B3a+B3b 合并分析组仍为
  `formal_ready = False`。原「B4 唯一 True」措辞与 B3b 快照冲突，已按粒度更正，原始报告待独立对账；
- 汇总出 **183 行**：180 `formal` / 3 `partial`（三条全部是上述 oracle 行），无缺失 seed，
  门槛拒收 0 条；
- 行级 `formal` 与门禁级 `formal_ready` 是**两个层次**：前者说这一条可比条件自身没有阻断，
  后者说整棵树能否整体作为正式榜。两者须同时给出，否则读者会以为其中一个写错了。

复现入口：`scripts/audit_survey_batches.py`（审计）、`scripts/summarize_survey_results.py`（汇总）、
`scripts/compare_survey_results.py`（对照附着），由同一份批次白名单 JSON 经 `--config` 驱动、
输出目录经 `--out-dir` 指定。本次运行的原始制品（`01_audit` / `02_summary` / `03_comparison` /
`04_evidence`）留在执行机的分析工作区，**在仓库外**；分发以 P2.2 交付/制品索引为准（直接交付 Excel、分析可自行重放，不另建云端分析归档）。本节数字来自产出方记录，本轮尚未取得原始结果树独立复算。

### 1.3 不可越过的口径边界

以下三条当前**未解除**，任何呈报不得越过：

1. **PA 行不能数值裁决**：对照矩阵中 **54 条**映射为 `blocked_pending_pa_criterion`，PA 准则的
   合作者复核未获得（D24 ④(a)）；
2. **对照附录必为定性**：矩阵 194 条映射中 `numeric` 为 **0 条**（111 `no_direct_anchor`、
   54 `blocked_pending_pa_criterion`、22 `magnitude_and_trend`、4 `non_runnable`、
   3 `background_only`），故预注册的 `|Δ| ≤ max(3pp, 2·SE_pooled)` 当前**无对象可施**；
3. **回收后的可持久复现范围**是「**选中权重 + 全部逐 epoch 选择记录**」，**不是**「全部 epoch
   权重」（D25 ③）。回收改变的是存储义务，不是选择能力。

### 1.4 尚未获得的合作者复核

以下复核**一律未获得**；相关单元的呈报**不得读作已签署**。各任务行只引本表，不复述：

| 复核项 | 出处 |
|---|---|
| 数据制品的本人复核签署 | 决策 D12 ①；[P1.4 接收端技术验收](p1_4_review.md)已通过，不再缺取件 |
| P2.0b 标签语义门禁的签署 | 决策 D12 ① |
| P2.0c 对照矩阵：36 个锚点 / 7 条映射仍待复核 | 决策 D12 ①、D24 ② |
| PA 正式选模准则的签署 | 决策 D24 ② |
| `self_pu` clean-validation 元重加权的复核 | 决策 D24 ② |
| P2.0e 五个方法 `ts` 接线的方法学复核 | 决策 D20 ① |
| P3.1 / P3.2 各方法的方法学复核 | 决策 D21 ⑥、D25 ⑤(d) |

对照矩阵自身的 `review_status` 仍为 `pending_collaborator_review`、`formal_blockers` 仍为
`["collaborator_review"]`。

## 2. 任务分工与验收

实施主体：**shuidisjtu**（数据、实验编排、结果留痕与文档）和 **HENG958**（算法接入、深度训练与
GPU 执行）。每项只有一名**主责**；协作者须在交付前完成复核。任务完成必须有测试、manifest、
运行记录或 PR 链接等可复核证据，不能只以口头或代码存在判定完成。

### 2.1 已完成（编号 + 任务 + 证据指针）

| 编号 | 任务 | 证据 |
|---|---|---|
| P1.1 | 环境与 GPU 验证 | `uv.lock` 可复现；目标环境 GPU smoke 与版本/设备记录可追溯。HENG958 的 GPU 能力复核已完成；正式跑批需 frozen-lock 环境 |
| P1.2 | 数据获取与版本审计 | pilot 三数据集的来源/版本/标签映射/许可入 manifest（2026-09-19，按官方页原文逐条记录）。协议全量 8 数据集与 ADNI 准入见决策 D1，属后续阶段——本行的余项是**范围**而非遗漏 |
| P1.3a | 方法台账 | 矩阵口径为 7 个 PU 方法 + `pn_oracle`；远程台账另收 `pusb` 线性基线与 `vpu`，为 9 条 PU 条目。本地 2026-09-28 预集成修改另补 `cvir`、`robust_pu`、`split_pu`、`pulda`、`puet`、`gradpu`、`lagam`，现有 16 条；新增七条已于 2026-10-04 提交（`7e779d2`）、尚未经方法负责人复核，均不进冻结执行矩阵（`pusb`/`pusb_kernel` 刻意分开，issue #42）。HENG958 已复核 nnPU、Self-PU（2026-09-16） |
| P1.3b | 官方 Survey 脚本 | 四路输入、PA/OA、结果归档与 oracle 入口均有脚本级测试 |
| P1.4 | Pilot 数据产物 | 三数据集 × 5 seed 统一重建（2026-09-19），三个归档 2026-09-20 发送；2026-10-04 本机确认收到，15 split / 75 文件 / 3 归档摘要全部通过，四角色契约、预处理统计及真实脚本 smoke 通过。见[P1.4 复核](p1_4_review.md)；本人签署未代填 |
| P2.0a | Pilot 共享规格与 oracle 对齐（阶段 A） | 2026-09-17 签署验收，shuidisjtu 复核签署；见[交付记录](p2_0a_delivery.md)、[复核包](p2_0a_review.md) |
| P2.0b | 标签语义门禁（阶段 A） | P1+P2 已完成；按决策 D12 ① 单方技术验收放行。见[交付记录](p2_0b_delivery.md) |
| P2.0c | 交叉验证对照预注册（阶段 A） | 锚点数值与判定规则已冻结并落矩阵；按决策 D12 ① 单方技术验收放行。见[复核包](p2_0c_review.md) |
| P2.0d | SAR-OA 执行路径（issue #43） | 已完成；HENG958 已复核（2026-09-16） |
| P2.0e | TS-OS 校准接入训练执行链 | 五个适用方法（`nnpu`、`upu`、`pusb_kernel`、`dist_pu`、`self_pu`）全部接线并按决策 D20 ① 单方技术验收放行。逐方法口径见决策 D16 ③ / D17 / D18 / D19，**不得**笼统写作「其余 4 个」或「五个都接线**口径相同**」 |
| P2.1 | Pilot 跑批与运行制品 | 五批 **645/645** 全部完成（B1 215 / B2 215 / B3a 110 / B3b 70 / B4 35），退出码 `0`、失败记录 0；批次明细见各批快照，交接清单 13 项全部为「是」（[`p2_1_handoff_checklist.md`](p2_1_handoff_checklist.md)）。编排载体 `scripts/run_survey_pilot.py`（`--datasets`/`--methods`/`--training-paths` 三轴作用域；B3a/B3b/B4 同为 cifar10，只按数据集切不开）。续跑判定按视图收紧、磁盘口径按存储 profile 修正、正式资格阻断位按 D24 放行（同一 `(dataset, seed, c)` 单元须整组重跑、新旧摘要不得混用）。**显式 `--os-or-ts ts` 的合法域随作用域变化**，「合法」不得读作「应当」。B4 起启用 D25 回收，可持久复现范围的收窄见 §1.3 第 3 条。**未获**：HENG958 对深度结果的复核 |

### 2.2 未完成（前置 / 验收标准 / 状态与主责）

| 编号 | 任务 | 前置 | 验收标准 | 状态 / 主责 |
|---|---|---|---|---|
| P2.2 | Pilot 聚合与审计 | P2.1 | 发布 `pilot / partial benchmark` 分层结果；检查路径隔离、复现字段和异常单元；不得生成跨数据集总排名 | 🚧 **正式五批汇总已产出**：见交付记录 [`p2_2_delivery.md`](p2_2_delivery.md) 与 §1.2——645 份 manifest、48 组全过公平性门禁、183 行（180 `formal` / 3 `partial` / 0 `diagnostic`）、0 份不可复现。对照附录裁决 0 条（矩阵 `numeric` 为 0，必为定性，§1.3 第 2 条）。**剩余为合作者复核**（§1.4）及原始结果树独立重放；分发以 P2.2 交付与制品索引为准。**PA 行受 §1.3 第 1 条限制，不能数值裁决**（决策 D24 ④(a)） |
| P3.1 | 缺失方法接入（经典/B 类） | P2.0a、P2.0b | 每方法完成实现、方法卡、台账、原文可追溯、冒烟与公开行为对照；使用已锁定的共享规格 | 🚧 技术预集成 / shuidisjtu：VPU 完成独立组件、采样假设裁决（D21）、台账与视图接线（D23）；PULDA 已补台账及 LDA/margin 风险 TS-OS 校准；CVIR 已有固定 α_U 表格适配器、方法卡与台账，但总体 π 不能无条件代替 α_U，故校准未接线、显式 ts fail-loud。三者均未进入冻结矩阵；共享 backbone/原生图像路径、公开结果对照、多 seed 资源记录及双人复核仍未完成，不得读作 P3.1 验收。PAN、RP、PULNS 待办 |
| P3.2 | 缺失方法接入（深度/C 类） | P2.0a、P2.0b | 同 P3.1，另需 GPU smoke、设备/随机性与保存加载验证 | 🚧 技术预集成 / HENG958：PUET、Grad-PU、Robust-PU、Split-PU、LaGAM 的独立组件、方法卡与台账均已有；Robust-PU/Split-PU 仅 nnPU warm-up/teacher 校准，PUET 节点 U 风险与 Grad-PU 的 U 风险/插值池已接 TS-OS；LaGAM 需要独立干净 support set，当前 PA-ineligible 且无可安全替换的同形未标记风险项，校准不适用。均未进入冻结矩阵，公开结果对照、共享 backbone/原生图像路径、多 seed GPU/资源记录及合作者复核仍待办；表格适配不等于论文图像数值复现。PUET 是 CPU 树方法，其 P3.2 分组与 GPU 条款适用性须复核。GEN-PU、Holistic-PU、P3MIX 待办 |
| P3.3 | 深度 GPU 验证与调度 | P3.2 | GPU 预约、显存预算、失败/OOM 重试及结果路径均有记录，且逐 run 记录实例与卡型；与 P2.1 已不共享单卡，无跨方窗口竞争 | ⏳ 待办 / HENG958 |
| P4.1 | 中心超参数注册表 | 各方法候选参数已确定 | 候选池预注册、版本化；版本写入 artifact 并受 manifest 校验 | **阶段 0 已落地**（2026-10-02，`cb2fe0f`/PR #94）：模块 `pu_toolbox/experiment/survey_recipe_registry.py`、门禁 `scripts/check_survey_recipe_registry.py`（registry JSON 未物化时**声明跳过**、未接入 CI）；**阶段 1+ 待办** / shuidisjtu：冻结 registry JSON 未物化、`validate_manifest_binding` 未接入 runner 与 pilot，故验收标准「版本写入 artifact 并受 manifest 校验」未达成。 |
| P4.2 | 主榜聚合与分析 | P3.1、P3.2、P3.3、P4.1 | 22 项全部通过门禁后，按四组结果和训练路径分层；结论区分文献事实、实验观测与推断 | ⏳ 待办 / shuidisjtu；HENG958 复核 C/A 深度结论 |

> 本地预集成的台账、风险视图接线和 2026-09-28 单卡 CUDA smoke 证据，见 [P3 技术预集成交接](p3_preintegration_handoff.md)；技术预集成已提交（`7e779d2`），不改变正式验收状态。

### 2.3 依赖与升级规则（三种门槛）

**依赖与升级规则（三种门槛）**：① **技术 smoke**（单方法链路验证）仅需基础设施可用，可随时执行；② **正式 pilot 跑批**须 P2.0a/P2.0b/P2.0c 全绿；③ **与 oracle 或跨方法结果混排**须阶段 A 验收全部满足。未达门槛而提前执行的结果，产物必须标记为“需按 P2.0 规格重跑”。P2.1 与 P3.3 已不再共享单卡（跑批平台分离），故**不设跨方 GPU 排期本**；改为**逐 run 记录实例与卡型**——每份 manifest 的 device/资源字段须能回答该 run 跑在哪张卡、是否独占，作为资源与复现证据；数据许可、共享规格、标签语义或资源不足造成阻塞时，主责须在“风险”中记录影响与下一步，并由两位实施主体共同决定升级、拆分或降级。Survey 分工在其范围内覆盖 ADR-0008 中较早的论文分配。

## 3. 交叉验证对照（预注册，2026-09-14）

试验结果须与两篇参考文献（Wang et al., ICLR 2026 = PUBench；Chen et al., 2026 = PU-Bench）
及各方法原论文交叉验证。本节为**预注册**：锚点数值与判定规则在 pilot 启动前冻结，执行后按
规则裁决，禁止事后调整（阈值、锚点、对照来源均以本节为准；确需修订须记录理由并重发
预注册版本）。

**锚点来源三类**：① PUBench（PA/PAUC/OA 三准则并列；我方 OA↔其 OA 列、我方 PA↔其 PA 列）；
② PU-Bench（**论文正文**说选模用真实验证标签的 macro-F1，**发布代码**却用 PU-only 的
`val_proxy_acc`；两者的矛盾登记为矩阵的 `pu_bench_selection_criterion` 条目、`resolution` 取
代码。本格重放实验站在代码一侧：按代码口径可复现锚点（|Δ|=0.75pp），按正文 macro-F1 复现
则高 8pp。**无 PA 机制**——我方 PA 结果与其数值无直接可比性）；③ 各方法原论文（以方法卡
`paper` 字段与官方实现为准）。

**PU-Bench Table 1（p6）：cc/SCAR、c=0.1、10 seeds，Accuracy% ± std**

| 方法 | CIFAR-10 | Spambase | IMDB |
|---|---|---|---|
| nnPU | 85.30±2.63 | 81.66±2.73 | 77.37±4.82 |
| PUSB | 87.68±2.85 | 81.56±2.95 | 77.86±3.82 |
| Dist-PU | 88.09±3.85 | 85.71±3.95 | 77.88±5.02 |
| LBE-PU | 83.98±3.62 | 68.53±3.71 | 76.25±4.73 |
| Self-PU | 76.73±2.43 | 72.10±2.92 | 74.05±3.23 |
| PN oracle | 94.88±0.57 | 91.03±0.66 | 79.89±0.83 |

> 注：uPU、KLDCE 未评估。其 "PUSB" 行对应仓库 `nnpusb` 实现，与**本项目 PUSB 同属
> Kato PUSB/nnPUSB 来源家族**（Kato/Teshima/Honda，ICLR 2019，官方上游
> MasaKat0/PUlearning），但 estimator 目标、模型族、先验语义与训练协议不同，**不能直接数值
> 等价比较**。**PN 行不参与数值裁决**（降级为背景参考：其 `pn` 用 PU-only proxy 选模
> （`val_proxy_acc`），我方 oracle 以 `clean_val` 真实 Accuracy 选模，口径不同——
> `pn_oracle_integration.md` 既有决策；PN ≥ 各 PU 方法仅作协议内诊断预期）。主要协议差异：
> 其验证比例 0.01（我方 5%+5%）、其 CIFAR-10 仅 /255.0（我方 train-only 通道归一化）、
> 其 SBERT 特征 L2 归一化（**已结清**：我方 IMDB 特征同为 L2 单位化，两侧无此差异，
> 见决策 D12 与对照矩阵 `v3`）。

**PUBench Table 1/2（p8-9）：CIFAR-10，正例率 30%，Accuracy%，PA / PAUC / OA 三列**；
Case 1 正类 {0,1,2,8,9}、Case 2 {2,3,5,7,9}——与我方 {0,1,8,9} 划分不同，仅判量级+趋势。
uPU 80.24/76.07/82.04（Case 1）、66.21/69.03/70.46（Case 2）；nnPU 82.03/75.56/82.40、
74.27/62.67/77.62；PUSB 81.53/82.49/82.91、75.74/78.80/78.35；Dist-PU 81.64/79.31/83.56、
73.46/74.83/74.69；LBE 82.71/73.60/85.03、72.47/63.54/75.96（"-c" 校准版各行从略，
见论文原表）。无 PN 基线行、无 Self-PU。

**原论文锚点（方法卡转录）**：nnPU→CIFAR-10（可行，backbone 差异登记）；PUSB→Spambase
（Kato et al. ICLR 2019 Table 2，扫 π∈{0.2,0.4,0.6,0.8} 与我方 c 轴不同，协议轴差异登记）；
Dist-PU→CIFAR-10（clean-room 档，仅量级）；Self-PU→CIFAR-10（13 层 CNN，我方 mlp-only +
adapter，仅量级）；uPU/KLDCE 无 pilot 对照点；LBE 论文为深度 Adam/EM 形态、我方线性近似，
无直接数值对照——无锚点单元显式记录"无直接对照点"，不得静默跳过。

**判定规则**：

1. **数值裁决仅用于"协议高度一致"的锚点**（同选模口径、同 backbone 家族、同 c 档、同指标），
   标准误感知：`SE_pooled = √(SD_anchor²/n_anchor + SD_ours²/n_ours)`；量级一致 ⟺
   `|Δ mean| ≤ max(3pp, 2·SE_pooled)`。超限 → 人工排查（实现级 `-m paper` 复现测试/官方源码
   对照/台账证据 → 协议级 c 抽样/划分/验证比例/backbone 预处理 → 数据口径版本/标签映射/类先验），
   结论分三档：**实现错误 / 协议差异可解释 / 无解释**；仅"无解释"升级为正确性警报。
2. **协议差异不可消除的锚点**（backbone 不同、正类划分不同等）：只记录方向与量级摘要，
   不做数值裁决。
3. **诊断预期**（非正确性硬门禁；违反 → 人工审阅并记录，不自动判实现错误）：PN oracle ≥
   各 PU 方法；同方法 c 增大时 Accuracy 不明显下降；组内相对排序与锚点排序秩相关（软提示）。

对照结论写入聚合报告，按协议 §5 第 7 条区分「文献事实 / 本实验观测 / 推断」；对照矩阵版本与 resolved 单元写入 manifest（manifest 侧 2026-09-19 已接线：runner 按选择协议分列写入，入口脚本开跑前校验覆盖；聚合报告侧由 P2.2 工具链承担，见 §1.2）。

## 4. 风险与留痕约定

### 4.1 风险

1. **GPU 算力/显存**：shuidisjtu 本机 T600（4GB）不足以承担正式跑批，故重活统一在 AutoDL 云平台
   执行，本地只做轻量批与开发验证。显存不足时（批大小/并行）需在实验记录中说明资源限制。
2. **数据集获取确认**：20News/IMDB/Connect-4/Spambase 的版本与标签编码需与协议锁定映射核对；
   数据版本或标签编码变化时必须提供映射转换审计记录（协议 §2.2 要求）。
3. **结果可解释性**：pilot 结果必须区分「文献事实 / 本实验观测 / 推断」（协议 §5 第 7 条要求，
   P2 起执行）。
4. **磁盘容量**：pilot 全 645 run 的累计 checkpoint 预算为 348,443,368,000 B（≈324.5 GiB）、
   跑前门禁 18,874,368,000 B（≈17.58 GiB）（决策 D15）。两者均为**全 pilot 累计**口径而非单批，
   且**数值以代码输出为准**。该数字只含 checkpoint：数据集、日志、manifest、临时文件与操作系统
   余量都不在内——**门禁通过不等于余量充足**（20 GiB 的主机过门禁后只剩 ≈2.4 GiB）；并发跑多个
   run 时总瞬时需求会超过单 run 门禁，调度层须按并发数另加余量。B4 起启用逐 epoch 回收
   （决策 D25），单 run 峰值由「全量 epoch 快照」变为「选中权重 + 全部逐 epoch 选择记录」。

### 4.2 留痕约定

- **每个运行 artifact 必须固化**（协议 §5 第 6 条）：代码 commit、依赖锁文件、Python/PyTorch/CUDA/
  GPU 信息、方法和数据配置、中心超参数注册表版本、seed、split/label manifest、选择 artifact 和
  结果 schema 版本。缺失任一项的结果标记为**不可复现**，不进入汇总主表。
  **注意口径**：其中「代码 commit」与「依赖锁文件」由**批次级**记录承载（批次计划与证据包），
  manifest 本身**不含**这两项——逐 run 索要会把合法制品误判为缺陷。
- 结果按四组存储（SCAR-PA / SCAR-OA / SAR-OA / PN oracle），跨数据集只比较趋势，不生成总排名。
- 交叉验证对照结论随聚合报告归档（§3 的三档结论），区分文献事实/本实验观测/推断。
- 本计划随执行更新：每完成一个阶段、每产生一个决策，更新 §1 与 §2，并把新决策追加到附录 A。

## 附录 A. 决策账本（D1–D26）

> 本账本按时间追加，每条记录的是**写下当时**的裁决与其理由；其后兑现或修正的状态以「补强」
> 「状态更新」等形式挂在原记录上，**不改写**原措辞——这是账本的留痕要求。因此**账本内的状态
> 类补充不代表当前状态**，现状一律见 §1。决策号 D1–D26 是外部引用契约，**不重排**。

| # | 决策 | 内容 | 日期 |
|---|---|---|---|
| D1 | **ADNI 数据获取** | **该数据集的获取似乎比较麻烦**，我会咨询一下学长。我目前的查证结论是（2026-09-08）：需通过 ADNI LONI 官网（adni.loni.usc.edu）在线申请——科研机构身份 + 接受数据使用协议（DUA）+ 研究用途描述，由 ADNI 数据共享与出版委员会（DPC）评审约 1-2 周，批准后经 LONI IDA 下载；限制：不得商用/重新分发、年度更新。决定后若申请通过，ADNI 加入后续实验矩阵；届时矩阵按"7+1"处理 | 2026-09-08 |
| D2 | 协议分工执行口径 | 切分与预处理由研究团队（工具箱用户）完成，工具箱 `fit` 只接受切好的四路数据；§2.4"工具箱不负责切分"的字面矛盾已澄清（第 3 条补充 + 第 7 条备注） | 2026-09-08 |
| D3 | 依赖锁与 CUDA 环境落地 | **uv.lock 已入库**（2026-09-08，chore(deps)）：替代 requirements.txt，PR 快层 CI 用 lock 确定性、nightly `--no-lock` 重新解析验证"最新可解析"（ADR-0012 修订）。**torch CUDA 配置**：pyproject `[tool.uv.index] pytorch-cu(cu126)` + `[tool.uv.sources]` 仅 `sys_platform=='win32'` 生效（CI Linux/macOS 保持 PyPI CPU 版；win 上 torch 2.14.0+cu126）；torchvision 并入 torch extra。多环境（T600/HENG958 主力机）由此保持一致 | 2026-09-08 |
| D4 | issue #41 阶段 A/B 拆分 | 阶段 A（P2 pilot 前置，P2.0a/b/c）：版本化执行矩阵 + runner 强制消费 + manifest 扩展 + CIFAR adapter 接线 + label_semantics_plan P1+P2 提前 + 对照矩阵预注册；阶段 B（P3 前置）：label_semantics P3+P4（pipeline 层检查 + 文档收口）。issue #41 于阶段 B 完成后关闭 | 2026-09-14 |
| D5 | Self-PU `input_ndims` 恢复 `{2,4}` | 审阅 P1#1：模板定义该字段为"支持输入维度"，4D 展平是 fit 的实际公共行为；"非原生 CNN"由 `native_architectures={"mlp"}` 承载（修正 issue #38 的收窄，代码随 fix 分支 PR） | 2026-09-14 |
| D6 | PU-Bench PN 行降级为背景参考 | 其 `pn` 用 `val_proxy_acc` 选模、我方 oracle 用 `clean_val_accuracy`，数值不可直接对比（`pn_oracle_integration.md` 既有决策）；不参与「交叉验证对照」判定规则第 1 条的数值裁决 | 2026-09-14 |
| D7 | SAR 标记频率口径修正 | 复核 PU-Bench 论文与锁定代码 `2d95a19`：`config/datasets_vary_e/*.yaml` 均使用 `c_values: [0.05, 0.5]`；原协议 `{0.1,0.5}` 与参考实现不一致，修正为 `{0.05,0.5}`。SCAR 主实验 `{0.1,0.3,0.5}` 不变；issue #43 按修正后口径实施 | 2026-09-15 |
| D8 | SAR 执行路径设计（issue #43） | `--labeling-mechanism` 与 `--method` **正交**（机制是实验自变量，SAR 行可跑任意 survey 方法）；SAR 强制 OA-only（协议 §2.3 下 PA 仅可诊断，v1 不产出 PA 日志）；SAR c 只接受**规范 token** `{0.05,0.5}`，目录按用户输入 token 命名（`c_0.05` 不得被格式化为 `c_0.1`），同值异拼写（`0.05`/`5e-2`）拒绝；`c_requested_token` 由脚本在运行成功后回写 manifest（runner manifest schema 为固定白名单，不改 runner，降低与 P2.0a 冲突） | 2026-09-15 |
| D9 | SCAR 标记数取整口径 | PU-Bench `n_labeled = int(n_pos · labeled_ratio)` 为**向下取整**，协议 §2.1 采用 `round(c·n₊)`；实现时以协议口径为准并记录实际 c | 2026-09-06 |
| D10 | **对照矩阵修订（v1 → v2）** | 审计 nnPU/Spambase 锚点时发现 v1 的两处登记缺陷：① 15 个 PU-Bench 映射把 CIFAR-10 的预处理差异复制到了全部数据集行，而 Spambase 行两侧同为 train-only z-score，该差异并不存在；② 15 个映射都缺**训练视图**维度——PU-Bench 的 case-control 把已标注正例放回 U（`data/data_utils.py:640-699`），与我方 `ts` 视图等价、与 `os` 视图不等价，而矩阵没有这一维。按预注册规则「确需修订须记录理由并重发预注册版本」新建 `survey_comparison_v2.json`：锚点数值、判定规则、资格判定均未变，仅 `protocol_differences`；`v1` 原样留档 | 2026-09-23 |
| D11 | **split 分母口径与 c 取整的补登记** | 两处协议字面与实现的差异，经审计补登记：① 协议 §2.4(二)3 写「留 10% 验证池、等分 5%+5%，其余 90% 为 train」，实现先在**全量池**上切走 20% test、再在剩余 80% 上做 90/10，故 Spambase 实测 `role_sizes = 3312/184/184/921`（占全量 72/4/4/20，占剩余即 90/5/5）；分层与泄漏隔离无误，但「5%」的分母口径此前未在文档显式；② c 标注数取整，协议 §2.1 定 `round(c·n₊)`，PU-Bench 发布代码为向下取整（`data/data_utils.py:423-425`），本格 n₊=1305、c=0.1 时两者同为 130，别格会分叉 | 2026-09-23 |
| D12 | **P2.0b/P2.0c 单方放行 + 对照矩阵 v3（F8）** | ① **放行 P2.0b/P2.0c**——工程实现经 shuidisjtu 单方技术验收，合作者未签署；为不使单人阻塞进度，从 `formal_blockers` 移除 `P2.0b_label_semantics_acceptance` 与 `P2.0c_cross_validation_acceptance`。**移除的是“阻断正式资格”，不等于合作者已复核**（同 D20/D24）：对照矩阵 `review_status` 仍为 `pending_collaborator_review`、`formal_blockers` 仍为 `["collaborator_review"]`，呈报不得读作已签署；② **F8**——v2 中 IMDB 的 5 条映射登记的“预处理差异”**不成立**（我方 IMDB 特征同为 L2 单位化，与 PU-Bench 锁定代码同口径）；D10 只把该行由 CIFAR 换成 IMDB、未校验替换后的差异是否成立，本条是同类缺陷的收尾。新建 `survey_comparison_v3.json`，锚点数值、判定规则、资格判定均不变，仅 `protocol_differences`；`v1`/`v2` 原样留档；③ 协议摘要由 `c15b0c9e…` 变为 `c019a87d…`（**版本串保持 `survey-v1.2`**，摘要才是绑定锚点），`v1`/`v2`/`v3` 的 `bound_survey_protocol.protocol_sha256` 一次性同步重绑 | 2026-09-25 |
| D13 | **F10 修复：聚合按实际运行视图分区** | 聚合分组键由 `comparability_group` 改为 `(comparability_group, run_view)`，同一方法的 OS/TS 两跑各成一榜。**分区放在聚合层**（`scripts/aggregate_survey_runs.py`），**不动** `partition_fair_leaderboard_runs`——交到它手里的每个 unit 已是单视图，其公开契约不变，继续作为下层 fail-closed 防御。**概念边界**：`run_view` 是逐 run 的**实际**视图，不是台账的 `native_sampling_assumption`（声明原生 TS 但未接线的方法回落 `os-compatible`），故本次**不实现**协议 §5.1 的“按原生假设分层”，报告与文档中也**不得**把该字段读作原生假设。`run_view` 与 `calibration_applied` 不一致即 fail-closed；PN oracle 恒为 `os-compatible`，带 `ts` 视图的 oracle 被拒绝且不广播进 TS 分区。报告 `SCHEMA_VERSION` 升至 `1.1`。**遗留**：P2.2 的期望单元覆盖率/完整性格网须先定口径（**复合分区内检查**还是**对原 group 另做总览**），否则按视图分开后会产生假缺失；该格网属 P2.2 | 2026-09-26 |
| D14 | **F11 修复：续跑状态按实际训练视图限定** | 续跑扫描的键由 `manifest_identity` 单键改为 `(identity, run_view)`：旧实现把同一单元的两个视图折叠成一份，于是“默认应跑 TS 的单元已有一份 OS 结果”被判为完成，TS 那份永不执行且无日志——重跑只是浪费，矩阵空洞事后不可补。`pending_runs` 改为接收**逐单元**的预期视图映射；`resolve_training_view` 下沉到 `pu_toolbox/experiment/training_views.py`，Pilot 与单元脚本**共用同一份**。manifest 视图校验同址建立、聚合器与 Pilot 共用，**两者对非法制品的处置故意不同**——聚合器让异常终止发布，续跑扫描记为未完成并继续（一个无关旧文件不应阻断整个 Pilot）。显式 `--os-or-ts ts` 的非法方法现在在**启动任何 batch 前**被拒。另注：该参数是**整矩阵**请求，计划中含任何原生 OS 方法或 oracle 时无法使用（已写入 CLI help）。**补强（复核后）**：显式 `ts` 分支**必须提供 estimator class**，缺省时原先会跳过接口校验直接返回 `ts`（未经核实的视图承诺），现改为 fail-loud；返回 `os` 同样不可接受——操作员要的是校准视图，却得到一个从未校准的 run，正是 F11 要消灭的静默降级 | 2026-09-26 |
| D15 | **F12 修复：checkpoint 体积按训练路径与架构三元组分派** | `unit_checkpoint_bytes` 原按 `backbone.startswith("resnet18")` 给**所有**图像行返回同一常量，但写入器保存的是 `fitted.model_` 的 state_dict：`native_cnn` 确实含 ResNet，`cnn_feature_adapter` 只训练并保存 MLP head——adapter 侧因此被高估约两个数量级。**签署边界**：P2.0a 绑定的是常量 `RESNET18_COMPONENT_BYTES = 45 MiB` 及其 `native_cnn` 路径，**本次不改该常量值**；被修正的是 adapter 路径引入时才出现、**从未进入任何签署制品**的前缀推广。**判据**改为 profile 身份 `(training_path, backbone, model_family)`：常量只对它被测量过的架构成立，未登记的有 epoch 组合一律 fail-loud，新增 adapter 方法必须连同序列化证据补一行；`native_2d` 保持参数化。`None` 的含义随之收紧为“该预算不产生逐 epoch checkpoint”，尺寸与身份不可知都不再静默取值。**数值以代码输出为准**：口径由此前的高估值修正，明细与旧值换算见 `p2_0a_delivery.md §6`；旧记录的 8.34 GiB 经查不可复现，已在那里加注。**三层 fail-closed**：估算层抛错、Pilot 在**启动第一个 subprocess 之前**统一预检失败、单元脚本以 `error:` + 返回码 1 呈现。**不改变** epoch/candidate/seed/attempt 与 checkpoint 保存频率 | 2026-09-26 |
| D16 | **P2.0e 逐方法接线口径** | ① 剩余原生 TS 条目区分**适用接线**与**适用性裁决**：`pusb_kernel`/`dist_pu`/`self_pu` 待逐个接线，线性 `pusb` 待书面裁决（其训练信号就是 `LogisticRegression().fit(X, y_pu)`，「U ≡ 负类」，不存在与风险估计器同形的"未标记损失输入"，且不在 Pilot 矩阵内），故 P2.0e 的完成标准是前三者接线 + `pusb` 形成明确可审计的处置，而不是"五个方法都接线"；② **UPU 的校准范围**：对 `upu`，TS 下无标签风险项及 RBF 中心候选池使用 `X_U ∪ X_P`；该结论**不自动推广**到其他方法，其他方法须独立推导并审阅；③ 首个落地为 `upu`（D-U 并集 + RBF 候选池跟随；中心数量不随视图变化，避免 OS/TS 对照混入容量变化——机制是中心数一律截断到校准前 `n_U`，显式 `n_centers` 与默认值同受此上限约束，且上限不取自候选池大小）；④ `ts` 合作者复核仍未获得，manifest 继续挂 `ts_view_collaborator_review`（**该阻断位已由 D20 ① 移除；复核本身仍未获得**） | 2026-09-26 |
| D17 | **P2.0e：`pusb_kernel` 接线口径** | 只冻结 `pusb_kernel`，**不推广**（承接 D16 ②）。① `ts` 下无标签风险项 `D_U` → `D_U ∪ D_P`（分母取 `n_P + n_U`），只作用于**内部 CV 训练折**与**最终 refit**；正例项、class prior、正则项不变。② 验证折保持 OS，故超参数在 **OS 目标**下选出——依据是**协议的角色隔离**而非分布论证；否决“验证折并入 P”的理由是**消融干净**（选参准则随视图变化就不再是单变量对照），代价是 TS 跑不是“按 TS 最优调参”的基线。③ RBF 中心**候选池**本已取自完整 `X`，故**不**照搬 D16 ③ 的 upu 规则（照搬会抬高 P 被选为中心的权重）；中心数、CV fold、阈值池不随视图变化。④ 角色构造 helper 只接收显式布尔、**不接收**视图字符串，验证折**不经**该 helper——结构上 fail-closed。⑤ 本口径是**协议选择而非论文事实**。⑥ `ts` 合作者复核仍未获得（阻断位已由 D20 ① 移除）。**逐项细节见[方法卡 PUSB](../method_cards/PUSB.md)** | 2026-09-27 |
| D18 | **P2.0e：`dist_pu` 接线口径** | 只冻结 `dist_pu`，**不推广**（承接 D16 ②）。① `ts` 下 **alignment 与 entropy** 的**角色集合** $`X_U`$ → $`X_U \cup X_P`$（alignment 分母取 $`n_P + n_U`$）；正例 BCE 只消费原始 $`P`$、class prior 保持总体 $`\pi`$ 不重估、Mixup 池与先验决策不变。② alignment 的依据是可核验的 **population identity**（OS 下 $`X_U`$ 不排除已标记正例、不代表完整 marginal），不是约定；审计期观测到的偏离量级支持该结论，但**该数字是本次制品观测、不是本方法的一般性质**，**引用必须带此限定**（数值与一般恒等式见[方法卡 Dist-PU](../method_cards/Dist-PU.md)）。③ entropy 只有协议一致性依据，**不是**数学必然；按 D17 ⑤，本口径整体是**协议选择而非论文事实**。④ **不照搬 upu 的“池子跟随视图”**（Mixup 池本已消费完整 X）；更硬的理由是**单变量消融**。⑤ 随机侧不变量只锁 RNG API 的调用次数/顺序/`randperm` 长度与 `lam` 抽样位置，**不**锁后续 loss 或梯度。⑥ `sample_weight` 处置是 `IGNORED`，**与 `nnpu` 的互斥门、`pusb_kernel` 的整体拒绝三种措辞不可互抄**。⑦ 未接线槽位改用测试内注册的合成方法永久钉住。⑧ `ts` 合作者复核仍未获得（阻断位已由 D20 ① 移除） | 2026-09-27 |
| D19 | **P2.0e：`self_pu` 接线口径** | 只冻结 `self_pu`，**不推广**（承接 D16 ②）。① `ts` 下负 PU 项的**角色集合** $`D_U^{role}`$（未信任 U 行）→ $`D_U^{role} \cup D_P^{k}`$；P 同时供正例项与校正项又进入负项——**角色叠加，非身份改写**，无行复制、无额外前向。② 混合按行数质量，分母取**未信任行数**；**$`\alpha_P`$ 不是 $`\pi`$**，$`\pi`$ 仍只在校正项、不重估。与已合并的 `nnpu` 是**同一个并集估计量**，差异只在负角色集合排除 trusted 行。③ 质量混合 **≠** 并集均值，仅在 U 侧权重均匀时相等——该等价性是 **Gate A**；clean-meta 分支是**质量守恒的扩展**，**不是原论文给出的 TS→OS 公式**。④ trusted 人口与 pace、伪标签、meta influence 维度、consistency 缓存与 hard-negative mask **一律不跟随视图，且不得跟随**（身份安全 + 单变量消融）。⑤ 实现只对 U 批求一致性，与论文两个损失的求和范围不一致——**视图无关的既有收窄**，本次不改，也**不得**据此声称论文的一致性项是 U-only。⑥ 验证/选模保持 OS（clean validation、PU validation、best-epoch 恢复、无 validation 时的 ablation teacher selection；与 D17 ② 同形）。⑦ 随机侧不变量只锁 RNG 调用次数/顺序/抽样范围、**前向次数**（0 次额外前向与 RNG）与 checkpoint/callback 契约。⑧ **Pilot 实际走消融分支**（`self_pu` 恒为 `calibration_mode_="ablation"`），引用 Pilot 结果**不得**读成“含 meta 的完整 Self-PU”。⑨ `ts` 合作者复核仍未获得；**本行口径已由 D20 单方放行取代**，其余条款不变。**逐项细节见[方法卡 Self-PU](../method_cards/Self-PU.md)** | 2026-09-27 |
| D20 | **P2.0e 单方放行（承接 D12 的做法）** | ① **放行 P2.0e**——五个适用方法（`nnpu` #68、`upu` #74、`pusb_kernel` #76、`dist_pu` #77、`self_pu` 本切片）的 `ts` 接线各自独立设计、独立成文并留下 D 记录（D16 ③ / D17 / D18 / D19），本切片另附四项硬证据（OS 基线逐位 0 偏差、变异检验 9/9 被抓住、真实跑批双视图重放逐位重现、CIFAR adapter smoke 成立，明细见交付记录）。据此按 shuidisjtu **单方技术验收放行**，并从 `survey_protocol.py` 的 manifest 阻断逻辑中**移除** `ts_view_collaborator_review`。**移除的是“阻断正式资格”，不等于合作者已复核**：该路径的方法学复核**仍未获得**，口径保留在方法台账 `uncertainty`（`nnpu`/`upu`）与三张方法卡的视图条目中，呈报时**不得**读作已签署；历史制品的 `formal_blockers` 仍如实记录当时状态，**不改写**。② **线性 `pusb` 的处置**：经 2026-09-27 裁决为**不适用校准**（理由见 D16 ①），**不产出**书面的“适用性裁决”文件；D16 ① 完成口径中的“+ `pusb` 明确处置”由本记录落地，故 P2.0e 可标完成。③ **状态与遗留**：P2.0e 行转 ✅；`self_pu` 的两处已知覆盖边界留在交付记录，属**可接受的证据边界**而非阻断项 | 2026-09-27 |
| D21 | **P3.1 局部：VPU 采样假设裁决与训练视图接入口径** | **只冻结 VPU，不推广**（承接 D16 ②）。① **裁决** `native_sampling_assumption = ts`，四层依据**已全部亲验**：(a) 论文式 (6) 的变分目标要求 `f` 是训练总体边缘分布；(b) 作者 docstring 明写 `x_loader` 覆盖 `training data (including positive and unlabeled)`，且池构造让被标记正例**同时留在 X 池**（标签抹为 `-1`），故 P 池 ⊂ X 池、**不是互斥划分**；(c) PU-Bench 把“分离的 P loader 与全训练数据 X loader”列为从源实现 **retained** 的组件（该套件自声明 `source_reproduction: false`，**不替代**作者仓库）；(d) OS 底料下 `D_U` 排除已标记正例，`D_U ∪ D_P` 才还原 `p(x)` 样本（同源论证见 D18②）。**映射集合级精确**：作者 X 池 = 训练分区（除验证集）= 工具箱 OS 分区的 `D_P ∪ D_U`，故“完整 `X` 作边缘池”与作者实现**逐集合等价**、等价于恒在 ts 视图。**定性强度高于 D18③**，两处措辞**不得互抄**。② OS 对照有效但**不是原生视图**（是**协议定义的消融**：边缘池被限制到 `p(x|s=0)`）。③ **不是 `both`**：registry 只登记 `CASE_CONTROL`，而测试要求台账值与 scenario **严格等价**；且 `resolve_training_view` 对 `ts`/`both` 行为相同。**词汇澄清**：本项目 `ts` 指协议 §2.3 的**视图语义**，**不**要求数据按两样本独立抽样生成。④ **验证路径保持 OS，是对上游的显式分歧**：上游对整份验证分区连同其正例子集求值，本工具箱按 **D17②** 否决（D19⑥ 同向）。**引用 VPU 验证变分风险数值时不得声称与上游同定义**；该量仅为 source diagnostic（见 D22）。⑤ 与视图无关的既有适配缺口**登记不改**（学习率衰减与默认值、`max_epochs`、`batch_size`、每 epoch 迭代数），逐条数值见审计稿。⑥ **边界**：单方技术审计、**合作者复核未获得**；视图接线之外的 P3.1 项均未完成，不进冻结矩阵，**不得**读作 P3.1 完成。**接线状态更新（同日）**：`os_or_ts` 视图接线已落地（`fit` 接受该参数、默认 `ts`，显式 `os` 与 `ts` 均能抵达；路由缺口由 D23 修好），见[方法卡 VPU](../method_cards/VPU.md)；此注为**状态更新而非改写**本记录——写成本行时该接线尚未完成，⑥ 中原列的「视图接线未完成」自本注起不再成立，其余条款不变；未完成的项因此指**其余** P3.1 工作（共享 backbone、图像路径、公开数值对照、多 seed GPU/资源记录）。审计全文见 [`vpu_sampling_audit.md`](vpu_sampling_audit.md) | 2026-09-28 |
| D22 | **VPU 验证变分风险的定位：source diagnostic，不作选模准则** | **只针对 VPU**，不推广。① 上游与本工具箱都把该量定义在**合并边缘池** $`U_{val} \cup P_{val}`$ 上（代码位置见审计稿），故 VPU 在此**与上游一致**、**不存在**需要登记的口径分歧——D21 审计稿曾误把“否决上游做法”写成既成事实，已更正。② 协议 §2.3 的“验证集保持原始 OS 协议”约束的是**参与选参或裁决**的视图；本裁决**不修改该条**，而由协议 §2.3 新增“适用范围”段明确：source diagnostic 量不受约束，但**不得**充当 PA/OA 选模准则——这与 D17②/D19⑥ **同向而非相抵**（两者的验证折都用于**选参**）。③ 裁决：(a) 允许按上游定义计算、记录并作 os/ts 对照；(b) **禁止**充当 VPU 的 PA/OA 选模准则；(c) 若将来 VPU 进矩阵且需 PA 选模，**必须另行裁决**（或改用符合 §2.3 的 OS 验证视图并自行成文，或为该行指定其它 PA 机制）。④ **不得引用 D17②/D19⑥**：两者均明文只冻结各自方法，**不覆盖 VPU**（本条的直接触发原因之一）。⑤ 本条**不改代码、不改 `survey_protocol_v1.json`**——改后者会变更协议摘要并触发 D12 式重绑级联，而 VPU 不在矩阵内，故只落在协议 `.md` 与本文档。⑥ 不构成 P3.1 完成 | 2026-09-28 |
| D23 | **`route_training_view` 的非对称转发规则（协议级）** | 承接 D21 接线时暴露的缺口。① **变更**：`ts` 照旧（目标未声明则 fail-loud，**消息逐字不变**）；显式 `os` 在目标把 `os_or_ts` **具名为 `fit` 参数**时转发，否则不动；`None` 一律不转发。② **理由**：旧规则“只有校准请求会改变训练”对默认值 `"os"` 的方法成立，但对**默认视图就是 `ts` 的方法**（VPU 首个），显式 `os` 会被丢弃、静默跑成 ts——即本模块要防的不一致以**反方向**重现。③ **非对称是刻意的**：未声明的 `ts` 是静默**降级**（跑 OS 而 manifest 说 ts）故拒绝；未声明的 `os` 是**无操作**（这类目标本就是 OS）故丢弃。`**kwargs` **不**视为 `os` 的声明；`ts` 路径沿用既有判定，**不触碰已冻结行为**。④ 影响面（实测）：全 registry 中声明 `os_or_ts` 的方法默认值一律 `"os"`、无一在 `fit` 上收 `**kwargs`，故转发 `os` 对既有链路等价。⑤ 被否决的替代：(a) 三态参数——不能避免路由变更且把 `"legacy"` 泄漏进公开属性；(b) 嗅探签名默认值——脆弱不可读；(c) 改 VPU 默认值为 `"os"`——裸 `fit` 会静默交出边缘项被错设的目标，且与 registry `scenario` 冲突；(d) 本切片不碰路由——仍需另行封堵静默反演，且 VPU 无法被显式持于 OS；(e) 按 `pusb` 先例登记——对 `pusb` 诚实，对 VPU 是**虚假陈述**（不改代码就在用并集）。⑥ 不改变任何方法的默认视图，不进冻结矩阵，不构成 P3.1 完成 | 2026-09-28 |
| D24 | **放行三个 manifest 正式资格阻断位（承接 D12/D20 的做法）** | **① 触发与前提**：P2.1 跑批移至云平台后，干跑报告的每个计划单元都带三个阻断位、`formal_eligible` 恒为 `False`，使聚合报告 `formal_ready` 恒为 `False`——跑批与聚合都能执行，但只能进 partial/技术层。三个阻断位的依据按最新事实**均已不成立**：(a) `linux_frozen_lock_environment_deviation`——来源是旧 Linux 服务器驱动低于 CUDA 13 所需版本（方案 3），换平台后不再存在；(b) `SelfPU_clean_validation_meta_reweighting_OA_integration`——2026-09-28 裁决“现有消融变体纳入 pilot 范围”，故 pilot 的 `self_pu` **就是**消融口径，与该阻断位的语义相反；(c) `PA_criterion_pending_collaborator_acceptance`——准则本体已实现并合并（PR #65，见 R9 补记），残留的只是合作者签署，与 D12/D20 已放行的两类**同构**。**② 改动**：从 `survey_protocol_v1.json` 的 `formal_blockers` 移除 `linux_frozen_lock_environment_deviation`（**保留** `per_epoch_independent_PA_OA_checkpoint_selection`——它按实际轨迹是否完整动态解除，不是静态阻断）；`survey_protocol.py` 移除对 PA 与 `self_pu` 两处运行时追加，并在删除点留墓碑注释（对齐 D20 的做法）。**保留** `protocol_deviation`（真实偏差仍要标）与 `method_variant: without_clean_validation_meta_reweighting`（消融**范围记录**，不是阻断位）。**移除的是“阻断正式资格”，不等于合作者已复核**：PA 准则与 `self_pu` 元重加权的合作者复核**仍未获得**，口径保留在决策记录、方法台账 `uncertainty` 与方法卡中，呈报时**不得**读作已签署；对照矩阵的 `review_status` 仍为 `pending_collaborator_review`、`formal_blockers` 仍为 `["collaborator_review"]`，本次**不动**。**③ 摘要变更**：由 `c019a87d…` 变为 `287c2f45…`（**版本串保持 `survey-v1.2`**，摘要才是绑定锚点），`survey_comparison_v1/v2/v3.json` 的 `bound_survey_protocol.protocol_sha256` 与 `README.md` 的摘要串一次性同步重绑。**④ 明确边界**：(a) 对照矩阵中 54 条 `blocked_pending_pa_criterion` 映射**未动**，P2.2 对 PA 行**仍不能数值裁决**；(b) 历史制品的 `formal_blockers` 仍如实记录当时状态，**不改写**（D20 口径）；(c) 因 `validate_comparable_manifests` 要求同一比较组内 `protocol_sha256` 一致，**同一 `(dataset, seed, c)` 单元须整组重跑**，新旧摘要不得混用——混用会在聚合期抛 `comparison mismatch: protocol_sha256`，而分组一致性检查只比 `protocol_version`、拦不住它；(d) 本记录**不放行任何方法的方法学复核** | 2026-09-28 |
| D25 | **逐 epoch checkpoint 回收：保留选中、回收其余（容量治理）** | **① 触发**：B4（`nnpu`/`cifar10`/`native_cnn`，35 runs）的逐 epoch 全量快照约 280 GB（项目内部估算，非租用容量保证），而 AutoDL 数据盘可用约 25 G、文件存储默认上限 200 GB 且当前未开通。`pilot_plan.py:805-811` 已记载 `Nothing deletes them`——峰值门禁（`guard_*`）是按「跑完即回收」设想的，但实现中没有任何回收，故 B4 的需求是全量而非峰值。该设计取舍的初衷是可审计与可复现，其占用量已**反噬该目标**。**② 裁决**：回收**非选中**的逐 epoch 权重，保留每个 run 被 PA/OA 选中的权重（两套协议可能选中不同候选与不同 epoch，故保留集至多两组，同一文件被同时选中时只留一份）。回收时机为 PA/OA 选择与 test 评测**全部完成之后**、manifest 构造**之前**，逐文件 `unlink`（不删整个 `attempt-*` 目录）；逐 epoch 的 epoch/component/sha256/验证指标等**元数据全部保留**在 `candidate_runs[].epoch_checkpoints`。manifest 语义随之变更：被回收项**保留原 `path` 与 `sha256` 并新增 `reclaimed` 标记**，**不得**写成 `path=null`——后者是「从未持久化」的临时目录语义（`epoch_checkpoint_delivery.md:38-39`），与事实不符。**③ 与「完整快照可供 PA/OA 独立选择」的关系（关键口径澄清）**：`epoch_checkpoint_delivery.md` §3 规定实际训练满声明 epoch 且完整快照可供 PA/OA 独立选择才解除 `per_epoch_independent_PA_OA_checkpoint_selection` 阻断。门禁判定发生在 **run 执行期**，回收在其后，故该条件在执行期**完全成立**。本条**不采纳**「选择完成后完整快照仍须在盘上」的读法——那是把**选择能力**偷换成**存储义务**，而后者正是本条要治理的对象。据此，本方案**不是**「关闭 capture」（`config['capture_epoch_checkpoints']=False` 会退回旧单点路径并保留阻断，见 `api.md:1438-1439`），也**不是**「缺少持久化」（`checkpoint_dir=None` 的临时目录语义）。**表述收窄**：此后解除阻断的单元，其可持久复现范围是「**选中权重 + 全部逐 epoch 选择记录**」，**不是**「全部 epoch 权重」；该收窄须在 `epoch_checkpoint_delivery.md` 同步增补一节。**④ 可比性边界**：回收不改变任何 run 的选模结果、阈值、selected epoch 与 test 指标，故**不构成口径变更、不破坏组内可比性**，**B3a 无需重跑**；须如实披露的差异是 B3a 为全量快照留存、B3b 起为选中权重留存（写入批次验收记录与 P2.2 交接清单）。**⑤ 明确不做**：(a) 不改选模路径、不改保存粒度——不采用「只记逐 epoch 验证分数向量」的替代方案，其收益与本条相同但改动面更大且需证明数值等价；(b) 不动 `per_epoch_independent_PA_OA_checkpoint_selection` 的**动态解除**逻辑（该阻断位按实际轨迹动态解除，不是静态阻断位，见 D24 ②）；(c) 开关默认**关闭**，`-m paper`/技术 probe 与既有测试行为不变，仅 pilot 正式跑批启用；(d) 本条**不放行任何方法的方法学复核**。**⑥ 前置与状态**：方案文本见 [`epoch_checkpoint_reclaim_plan.md`](epoch_checkpoint_reclaim_plan.md)，本记录为其决策登记；实现经 `feature/` 分支 + 测试 + PR 合入，须**赶在 B3b 启动前**完成（B3a 运行中进程使用已加载的旧代码，不受影响） | 2026-09-30 |
| D26 | **F13 修复：聚合分组补上 mechanism 维度** | B1 结果树（215 manifests）聚合被拒于 `comparison label-view mismatch: mechanism`。**根因**：分组键为 `(comparability_group, run_view)`（D13）、单元键为 `(seed, c_requested)`，两者都不含 mechanism；而同一 `comparability_group + run_view` 下并存 `scar` / `sar_lbe_a` / `sar_lbe_b` 三种机制，SCAR 的 `c ∈ {0.1, 0.3, 0.5}` 与 SAR 的 `{0.05, 0.5}` 在 **c=0.5 重叠**，于是三者的 c=0.5 run 落进同一个 `(seed, 0.5)` 单元，`validate_comparable_manifests` 取 `manifests[0]` 与后续逐一比对、首个 mechanism 不同的即抛错。**修复**：新增 `group_key()`，把 `generation.train.mechanism` 提升为第三个分组维度，组条目与文本输出带出 `mechanism`。**分区仍放在聚合层**（承接 D13），**不动** `partition_fair_leaderboard_runs` 与 `validate_comparable_manifests`——交到它们手里的每个 unit 已是单视图**且单机制**，公开契约不变。**概念边界**：`comparability_group` 是协议字段（pins dataset / training path / budget family），`run_view` 与 `mechanism` 是 manifest 字段被提升为公平性分组维度，协议组值本身**不**加后缀。**测试缺口的成因**：`tests/unit/experiment/_aggregate_script_helpers.py` 的 fixture 其 mechanism 只有 `scar` 与 `pn_oracle` 两态，后者因 `c_independent` 被分到独立单元、**恰好避开了同 c 冲突**；本次让 fixture 支持 `mechanism` 参数并补复现测试 `test_param_mechanisms_sharing_a_c_value_are_gated_apart`。**影响面**：B2 及后续含 SAR+SCAR 的批次（B3a/B3b/B4）此前均无法聚合，而 `--diagnostic` 只放宽正式资格、绕不过。**验证**：aggregate 相关 35 个测试通过；B1 真实结果树由拒收变为 16 个组、全部单元 `ok`；`check_format` 与 `check_test_quality` 通过。**遗留**：D13 登记的 P2.2 期望单元覆盖率/完整性格网口径问题不变，本次不涉 | 2026-09-30 |
