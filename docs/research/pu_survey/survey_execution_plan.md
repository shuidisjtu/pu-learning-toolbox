# PU 调研实验执行计划（pilot → 主榜）

> 定位：本文件是**执行路线与状态**的协调层，只保留需人工阅读与决策的内容——任务分工、
> 交叉验证对照预注册、决策记录、风险与留痕约定。实现细节与现状见
> [pu_survey_protocol.md](pu_survey_protocol.md)（要求纲要）与
> [experiment_layer.md](../../dev/experiment_layer.md)（实验层实现）；各 Phase 的交付证据
> 见对应交付记录与复核包（`p2_0a/b/c_delivery.md`、`p2_0a/c_review.md`）。
> 状态日期：2026-09-27。

## 1. 任务分工与验收

实施主体：**shuidisjtu**（数据、实验编排、结果留痕与文档）和 **HENG958**（算法接入、深度训练与 GPU 执行）。每项只有一名**主责**；协作者须在交付前完成复核。任务完成必须有测试、manifest、运行记录或 PR 链接等可复核证据，不能只以口头或代码存在判定完成。

| 编号 | 任务 | 前置 | 验收标准 | 状态 / 主责 |
|---|---|---|---|---|
| P1.1 | 环境与 GPU 验证 | — | `uv.lock` 可复现；目标环境完成 GPU smoke；版本、设备与验证记录可追溯 | ✅ 已完成 / shuidisjtu；HENG958 GPU 能力复核完成，正式跑批需 frozen-lock 环境（见独立复核记录） |
| P1.2 | 数据获取与版本审计 | — | 数据来源、版本、标签映射与许可记录入 manifest；ADNI 的准入状态明确 | 🚧 manifest 侧已补齐（2026-09-19，三 pilot 数据集；许可按官方页面原文逐条记录——UCI 为 CC BY 4.0，另两个来源未声明）；ADNI 不在 pilot 范围，准入路线见 D1。本项 🚧 的口径是**范围**而非遗漏：协议 §2.1 列 8 个数据集，pilot 阶段先取 3 个易得且分属三种模态的（CIFAR-10 / Spambase / IMDB），故 pilot 范围内已完成、按协议全量仍待办 / shuidisjtu |
| P1.3a | 方法台账 | — | 7 个已实现方法的 6 槽（另有 paper/code_version/implementation_status 身份与来源字段）已填写，evidence 覆盖其中 3 槽；每项经对应方法负责人复核。**「7 个」为执行矩阵口径**——`survey_protocol_v1.json` 的 `method_profiles` 列 7 个 PU 方法 + `pn_oracle`；台账 `method_ledger.json` 另收 `pusb` 线性基线与 `vpu`（P3.1 局部前置，决策 D21），实为 **9 条 PU 条目**（`pusb` / `pusb_kernel` 刻意分开，issue #42）；**两者都不进执行矩阵**——矩阵口径仍是 7 个 PU 方法 + `pn_oracle`，本行「7 个」不变 | ✅ 已完成 / shuidisjtu；HENG958 已复核 nnPU、Self-PU（2026-09-16） |
| P1.3b | 官方 Survey 脚本 | — | 四路输入、PA/OA、结果归档与 oracle 入口均有脚本级测试 | ✅ 已完成 / shuidisjtu |
| P1.4 | Pilot 数据产物 | P1.1、P1.2 | CIFAR-10、IMDB、Spambase 各 5 个 seed 的四路 split、预处理与 manifest 均通过合同验证 | ✅ shuidisjtu 侧已完成（三数据集统一重建 2026-09-19）；传输/校验两端工具已就位（`scripts/survey_splits_archive.py`）；**载体已定（网盘带外传），三个归档已于 2026-09-20 发送，逐文件索引与归档摘要先于发送入库**（`docs/research/pu_survey/data/split_artifacts_index.json`）；⏸ HENG958 可执行性复核等待其取件校验 |
| P2.0a | Pilot 共享规格与 oracle 对齐决策（阶段 A） | P1.3a、P1.3b | 版本化执行矩阵（`survey_protocol_v1.json`：预算定义表 + 执行单元行）、runner 强制消费、manifest 扩展（protocol_version/backbone/budget/representation/comparability_group + adapter manifest 合并）、CIFAR adapter 接线、训练路径分组与 PN oracle 对齐方式书面锁定 | ✅ 已签署验收（2026-09-17）/ HENG958 交付；shuidisjtu 复核签署；**不放行正式 P2.1**（R5/R8 记为 P2.1 前置条件，两者已于 2026-09-19 工程兑现）；见 [交付记录](p2_0a_delivery.md)、[复核包](p2_0a_review.md) |
| P2.0b | 标签语义门禁（阶段 A） | P1.3a、P1.3b | `label_semantics_plan` P1+P2 提前完成：声明位 + registry 同步 + experiment 层检查，错误组合 fail-loud；pipeline 层检查属阶段 B | ✅ 已完成（P1+P2）/ shuidisjtu 单方技术验收放行（2026-09-25，决策 D12）：19/19 注册分类器显式声明、8 组语义组合 fail-loud 行为正确、红绿对照实测（修复前 3 FAILED `DID NOT RAISE` → 当前 3 passed）。**HENG958 签署未获**；见 [交付记录](p2_0b_delivery.md) |
| P2.0c | 交叉验证对照预注册（阶段 A） | P2.0a | 对照矩阵与判定规则冻结入本文档「交叉验证对照」节；锚点数值预注册 | ✅ 已完成 / shuidisjtu 单方技术验收放行（2026-09-25，决策 D12）：判定规则常量与 `decision_rules` 一致、**48/48 锚点值逐值对账通过**、D10 修订复算完整；F8（IMDB 预处理差异不成立）已由 `survey_comparison_v3.json` 修订。**36 锚点/7 映射的合作者复核仍未获得**；见 [复核包](p2_0c_review.md) |
| P2.0d | SAR-OA 执行路径（issue #43） | P1.3b | 官方脚本可选标记机制（SCAR / SAR LBE-A / SAR LBE-B）；SAR 仅 `{0.05,0.5}` 且强制 OA-only；生成器审计字段入 manifest；脚本级端到端测试 | ✅ 已完成 / shuidisjtu；HENG958 已复核（2026-09-16） |
| P2.0e | TS-OS 校准接入训练执行链（协议 §2.3） | P1.3a | 视图默认由台账 `native_sampling_assumption` 推导、CLI 可覆盖；`ts` 视图逐训练 mini-batch 执行 `D_U^k ← D_U^k ∪ D_P^k`（验证/测试保持 OS）；逐 run 实际视图入 manifest（`run_view`/`calibration_applied`）并成为公平性分组维度；pilot 驱动透传与重跑判定收紧；未接线方法默认回落 `os`、显式请求 `ts` fail-loud | ✅ 已完成 / shuidisjtu 单方技术验收放行（2026-09-27，决策 D20）：`nnpu`、`upu`、`pusb_kernel`、`dist_pu`、`self_pu` 已全部接线并各产出 os/ts 对照（逐方法口径见 D16 ③ / D17 / D18 / D19）；五个适用方法**无剩余待接线项**。线性 `pusb` 经 2026-09-27 裁决为**不适用校准**（D16 ① 已给出理由：训练信号即「U ≡ 负类」，不存在同形的未标记损失输入），D16 ① 完成口径中的「+ `pusb` 明确处置」由 D20 ② 落地，故本行可标完成；仍**不得**笼统写作"其余 4 个"或"五个都接线"。放行证据与边界见 D20 ①：**移除 `ts_view_collaborator_review` 阻断位不等于合作者已复核**——ts 路径的方法学复核仍未获得，该口径保留在方法台账 `uncertainty`（`nnpu`/`upu`）与三张方法卡的视图条目中，呈报不得读作已签署。聚合侧按**实际运行视图**分区已落地（D13）；续跑判定按视图收紧已落地（D14） |
| P2.1 | Pilot 跑批与运行制品 | P1.4、P2.0a、P2.0b、P2.0c、P2.0e | 每个计划单元产生完整 manifest、选择 artifact、资源/失败记录；oracle 按 `(dataset, seed)` 去重 | ⏳ 待办 / HENG958；编排载体与磁盘预算已就位（`scripts/run_survey_pilot.py`），**按数据集跑子集的 `--datasets`/`--plan-json` 机制亦已就位**（2026-09-27，PR #79；用途不限于拆分——批次按 `(dataset, …)` 字典序执行，cifar10 排在最前，故不经此机制无法先跑最便宜的数据集验证链路），跑批主机与环境路线、GPU 窗口排定仍待定；续跑判定已按视图收紧（D14）；**磁盘预算口径已按存储 profile 修正（D15）：以 ≈324.5 GiB 累计、≈17.58 GiB 跑前门禁为准，不得再用 1280.6 GiB / 35.16 GiB** |
| P2.2 | Pilot 聚合与审计 | P2.1 | 发布 `pilot / partial benchmark` 分层结果；检查路径隔离、复现字段和异常单元；不得生成跨数据集总排名 | ⏳ 待办 / shuidisjtu；HENG958 复核深度结果。聚合入口已按**实际运行视图**分区（D13）；覆盖率/完整性口径须在实施前定夺 |
| P3.1 | 缺失方法接入（经典/B 类） | P2.0a、P2.0b | 每方法完成实现、方法卡、台账、原文可追溯、冒烟与公开行为对照；使用已锁定的共享规格 | 🚧 技术预集成 / shuidisjtu：**VPU 已完成独立组件、Gate 0 采样假设裁决（D21）、方法台账条目与训练视图接入（D23）**，但共享 backbone、图像路径、公开行为对照、多 seed GPU/资源记录与双人复核**均未完成**，且**不进执行矩阵**——不得读作本行完成；PULDA 仍只有独立组件，台账/矩阵与正式验收未做；其余 PAN、RP、CVIR、PULNS 待办 |
| P3.2 | 缺失方法接入（深度/C 类） | P2.0a、P2.0b | 同 P3.1，另需 GPU smoke、设备/随机性与保存加载验证 | 🚧 技术预集成 / HENG958：PUET、Grad-PU、Robust-PU、Split-PU、LaGAM 独立组件已完成；Robust-PU、Split-PU、LaGAM 的组件分别采用 self-paced、hardness-aware、clean-support gate 设计（本地提交 `25467bb`、`8efadd4`、`f7a850a`，Split-PU loss 权重修正 `4fbafa0`）。这些仍未完成方法台账/执行矩阵登记、GPU 实跑及正式验收；后三者为表格子集，未做论文数值复现；LaGAM 依赖额外干净 support set，当前 Survey v1/runner 不可用且 PA-ineligible，协议扩展待合作者审阅。GEN-PU、Holistic-PU、P3MIX 待办；Grad-PU/PUET 独立组件已完成，PUET 为 CPU 树方法，分组/GPU 条款须复核 |
| P3.3 | 深度 GPU 验证与调度 | P3.2 | GPU 预约、显存预算、失败/OOM 重试及结果路径均有记录；不与 P2.1 竞争同一窗口 | ⏳ 待办 / HENG958 |
| P4.1 | 中心超参数注册表 | 各方法候选参数已确定 | 候选池预注册、版本化；版本写入 artifact 并受 manifest 校验 | ⏳ 待办 / shuidisjtu |
| P4.2 | 主榜聚合与分析 | P3.1、P3.2、P3.3、P4.1 | 22 项全部通过门禁后，按四组结果和训练路径分层；结论区分文献事实、实验观测与推断 | ⏳ 待办 / shuidisjtu；HENG958 复核 C/A 深度结论 |

**依赖与升级规则（三种门槛）**：① **技术 smoke**（单方法链路验证）仅需基础设施可用，可随时执行；② **正式 pilot 跑批**须 P2.0a/P2.0b/P2.0c 全绿；③ **与 oracle 或跨方法结果混排**须阶段 A 验收全部满足。未达门槛而提前执行的结果，产物必须标记为”需按 P2.0 规格重跑”。P2.1 与 P3.3 共享单卡时，HENG958 负责排定并记录 GPU 窗口；数据许可、共享规格、标签语义或资源不足造成阻塞时，主责须在“风险”中记录影响与下一步，并由两位实施主体共同决定升级、拆分或降级。Survey 分工在其范围内覆盖 ADR-0008 中较早的论文分配。

## 2. 交叉验证对照（预注册，2026-09-14）

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

对照结论写入聚合报告，按协议 §5.7 区分"文献事实 / 本实验观测 / 推断"；对照矩阵版本与
resolved 单元写入 manifest（manifest 侧 2026-09-19 已接线：runner 按选择协议分列写入，
入口脚本开跑前校验覆盖；聚合报告侧仍属 P2.2）。

## 3. 决策记录

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
| D12 | **P2.0b/P2.0c 单方放行 + 对照矩阵 v3（F8）** | 三件事合并为一次摘要变更：① **放行 P2.0b/P2.0c**——两者工程实现经 shuidisjtu 单方技术验收（P2.0b 含红绿对照实测；P2.0c 含 48/48 锚点逐值对账），合作者未同步、未签署，为不使单人阻塞进度，从 `survey_protocol_v1.json` 的 `formal_blockers` 移除 `P2.0b_label_semantics_acceptance` 与 `P2.0c_cross_validation_acceptance`。**移除的是"阻断正式资格"，不等于合作者已复核**：对照矩阵的 `review_status` 仍为 `pending_collaborator_review`、`formal_blockers` 仍为 `["collaborator_review"]`，逐 run 写入 manifest 的 `comparison` 块，呈报时不得读作已签署；② **F8（对照矩阵 v3）**——v2 中 IMDB 的 5 条 PU-Bench 映射登记「preprocessing differs (the benchmark L2-normalises the SBERT features)」，但该差异**不成立**：我方 IMDB 特征同为 L2 单位化（split manifest 记 `effective_output_normalization="l2_unit_norm"`，实测 `data/splits/imdb/split_0/train.npz` 每行 L2 范数 0.99999988–1.00000012），与 PU-Bench 锁定代码的 `_l2_normalize` 同口径。D10 已正确删去 spambase 上错挂的 CIFAR 差异，却只把 IMDB 那行由 CIFAR 换成 IMDB、未校验替换后的差异是否成立——本条是同一类缺陷的收尾。按预注册规则新建 `survey_comparison_v3.json`，锚点数值、判定规则、资格判定均不变，仅 `protocol_differences`；`v1`/`v2` 原样留档；③ ①引发的协议摘要由 `c15b0c9e…` 变为 `c019a87d…`（**版本串按项目先例保持 `survey-v1.2`**，摘要才是绑定锚点），`v1`/`v2`/`v3` 的 `bound_survey_protocol.protocol_sha256` 一次性同步重绑 | 2026-09-25 |
| D13 | **F10 修复：聚合按实际运行视图分区** | 落实 P2.0e 的「逐 run 实际视图成为公平性分组维度」：聚合分组键由 `comparability_group` 单字符串改为 `(comparability_group, run_view)`，同一方法的 OS/TS 两跑各成一榜，不再整体报 `method 'nnpu' appears twice with different specs`。**分区放在聚合层**（`scripts/aggregate_survey_runs.py`），**不动** `partition_fair_leaderboard_runs`——聚合层分开后交给它的每个 unit 已是单视图，其公开契约与 `tests/unit/experiment/test_leaderboard_run_view.py` 的 4 个测试均不变，它继续作为下层 fail-closed 防御。**概念边界**：`run_view` 是逐 run 的**实际**视图，不是台账的 `native_sampling_assumption`（声明原生 TS 但未接线的方法会回落 `os-compatible`），故本次**不实现**协议 §5.1 的「按原生假设分层」，报告与文档中也**不得**把该字段读作原生假设。新增 `manifest_run_view()` 校验 `run_view` 与 `calibration_applied` 一致，矛盾或缺失即 fail-closed；PN oracle 恒为 `os-compatible`：带 `ts` 视图的 oracle 被聚合层直接拒绝（同命令行 `--oracle --os-or-ts ts` 规则的聚合侧一半），且不被广播进 TS 分区，两者均有测试锁定；报告 `SCHEMA_VERSION` 升至 `1.1`（同一 root 的 group 数可能变多）。**遗留**：P2.2 的期望单元覆盖率/完整性格网须先定口径——**复合分区内检查**（每视图各自完整）还是**对原 comparability group 另做总览**——否则按视图分开后会产生假缺失；该格网当前尚不存在（属 P2.2） | 2026-09-26 |
| D14 | **F11 修复：续跑状态按实际训练视图限定** | 续跑扫描的键由 `manifest_identity` 单键改为 `(identity, run_view)`：旧实现以 `setdefault` 折叠，**字典序靠前的那份胜出**，于是"默认应跑 TS 的单元已有一份 OS 结果"被判为完成，TS 那份永远不会跑且无任何日志——重跑只是浪费，矩阵的空洞事后不可补。`pending_runs` 改为接收**逐单元**的预期视图映射，由调用方解析；解析逻辑 `resolve_training_view` 下沉到 `pu_toolbox/experiment/training_views.py`，Pilot 与单元脚本**共用同一份**（原先 Pilot 不读台账，默认路径直接放弃视图判断）。manifest 视图校验（枚举、`calibration_applied` 一致性、oracle 不可有校准视图）同址建立，聚合器与 Pilot 共用；**两者对非法制品的处置故意不同**——聚合器让异常终止发布，续跑扫描记为未完成并继续（一个无关旧文件不应阻断整个 Pilot）。`_ran_on_current_split` **不改**：split 校验在构造 resume key 之前、用基础 identity 完成，它继续收 6 元组。显式 `--os-or-ts ts` 的非法方法现在在**启动任何 batch 前**被拒（原先运行到中途才失败）；registry 无该方法的 estimator class 时报带 dataset/method/path 的可读错误、返回码 1、不启动 subprocess。另注：该参数是**整矩阵**请求，计划中含任何原生 OS 方法或 oracle 时无法使用（Pilot 无方法过滤器），此边界已写入 CLI help。**补强（复核后）**：显式 `ts` 分支要求**必须提供 estimator class**——缺省时原先会跳过接口校验直接返回 `ts`，等于给出未经核实的视图承诺，现改为 fail-loud（返回 `os` 同样不可接受：操作员要的是校准视图，却得到一个从未校准的 run，正是 F11 要消灭的静默降级）；默认路径保持保守回落 `os`（重跑可恢复，矩阵空洞不可恢复）。Pilot CLI 的三条拒绝理由——原生 OS 方法、原生 TS 未接线、oracle——现由参数化测试锁定「返回码 1、未启动任何 subprocess、stderr 以 `error:` 开头、不含 traceback」 | 2026-09-26 |
| D15 | **F12 修复：checkpoint 体积按训练路径与架构三元组分派** | `unit_checkpoint_bytes` 原按 `backbone.startswith("resnet18")` 给**所有**图像行返回同一个常量，但写入器保存的是 `fitted.model_` 的 state_dict（`checkpoints.py`）：`native_cnn` 的 `model_ = Sequential(encoder_, head)` 确实含 ResNet，`cnn_feature_adapter` 只训练并保存 MLP head——两行同值即本缺陷，adapter 侧被高估 **≈178×**（45 MiB / 265,700 B）。**签署边界**：P2.0a 绑定的是常量 `RESNET18_COMPONENT_BYTES = 45 MiB` 及其 `native_cnn` 路径（成本口径见 `epoch_checkpoint_delivery.md`「约 44 MB…约需 8–9 GB/候选/seed」），**本次不改该常量值**；被修正的是 adapter 路径引入时才出现、**从未进入任何签署制品**的那个前缀推广——即交付记录自己挂出的「作为协议问题上报」那笔账的结清。**判据**改为 profile 身份 `(training_path, backbone, model_family)` 三元组：常量只对它被测量过的架构成立，未登记的有 epoch 组合一律 fail-loud，新增 adapter 方法必须连同序列化证据补一行；`native_2d` 保持参数化（字节数由该行自己的 `hidden_dims` 推出，更宽的网络由构造自动覆盖）。`None` 的语义随之收紧为「该预算不产生逐 epoch checkpoint」，尺寸不可知与身份不可知都不再静默取值——静默猜的数正是门禁随后会要求主机的那一个。**实测**：3 个 `runnable + epochs` 的 adapter 行（cifar10 的 `dist_pu`/`self_pu`/`pn_oracle`）经生产构造路径（`assemble_model` + 真实 writer）序列化，最大组件文件 265,695 B，故 `ADAPTER_HEAD_COMPONENT_BYTES = 512 KiB` 留 1.97× 余量，由真实序列化测试锁定（含 native CNN 侧「确实含 encoder 参数」的反向断言）。**数值（精确字节）**：全 Pilot 累计由 **1,374,999,272,000 B（1280.57 GiB）修正为 348,443,368,000 B（324.51 GiB）**，其中 cifar10 341,835,776,000 B（318.36 GiB），且 **307.62 GiB（占 cifar10 的 96.6%）来自 `cifar10/nnpu/native_cnn` 的 35 次真实 ResNet 训练**——估算由「被错误路径主导」变为「由真实训练路径主导」；per-run 峰值 9,437,184,000 B（8.7890625 GiB）、跑前门禁 18,874,368,000 B（17.578125 GiB），峰值单元 `cifar10/nnpu/native_cnn`（修正前是 adapter 行的 35.1562 GiB）。**旧记录中的 8.34 GiB 不可复现**：8.34 GiB = 8.955 GB，恰落在「8–9 GB/候选/seed」区间内，疑为 GB 数值误标为 GiB，已在 `p2_0a_delivery.md` 按未复现加注。**三层 fail-closed**：估算层抛错（消息含 dataset/method/path/backbone/model_family/budget）、Pilot 在**启动第一个 subprocess 之前**统一预检失败（新增；此前正式路径根本不计算估算）、单元脚本以 `error:` + 返回码 1 呈现。**不改变** epoch/candidate/seed/attempt 与 checkpoint 保存频率——仅校正存储容量模型。dry-run 输出新增各单位所用 profile 与「不含数据/日志/临时文件/主机余量」说明 | 2026-09-26 |
| D16 | **P2.0e 逐方法接线口径** | ① 剩余原生 TS 条目区分**适用接线**与**适用性裁决**：`pusb_kernel`/`dist_pu`/`self_pu` 待逐个接线，线性 `pusb` 待书面裁决（其训练信号就是 `LogisticRegression().fit(X, y_pu)`，「U ≡ 负类」，不存在与风险估计器同形的"未标记损失输入"，且不在 Pilot 矩阵内），故 P2.0e 的完成标准是前三者接线 + `pusb` 形成明确可审计的处置，而不是"五个方法都接线"；② **UPU 的校准范围**：对 `upu`，TS 下无标签风险项及 RBF 中心候选池使用 `X_U ∪ X_P`；该结论**不自动推广**到其他方法，其他方法须独立推导并审阅；③ 首个落地为 `upu`（D-U 并集 + RBF 候选池跟随；中心数量不随视图变化，避免 OS/TS 对照混入容量变化——机制是中心数一律截断到校准前 `n_U`，显式 `n_centers` 与默认值同受此上限约束，且上限不取自候选池大小）；④ `ts` 合作者复核仍未获得，manifest 继续挂 `ts_view_collaborator_review`（**该阻断位已由 D20 ① 移除；复核本身仍未获得**） | 2026-09-26 |
| D17 | **P2.0e：`pusb_kernel` 接线口径** | 只冻结 `pusb_kernel`，**不推广**到其他方法（承接 D16 ②）。**① 校准范围**：`ts` 下无标签风险项由 `D_U` 换成 `D_U ∪ D_P`（分母随之取 `n_P + n_U`），只作用于**内部 CV 训练折**与**最终 refit**；正例项、class prior、正则项不变。**② 验证折保持 OS，且这是协议驱动的跨目标选参**：内部 CV 验证折的两个角色由该折自己的 P/U 就地构造（不经视图开关），故超参数是在 **OS 目标**下选出的。规范依据是**协议的角色隔离**（「验证/测试保持 OS」），**不是**「OS validation 与 TS training risk 同分布」——两者本就不是同一泛函（OS 侧无标签经验分布是 `p(x|s=0)`，不是 population-marginal surrogate）。**被否决的替代**：验证折并入该折自己的 P（`U_val ∪ P_val`）同样 held-out，且在「选参准则与训练风险同形」上其实**更自洽**；否决理由是**消融干净**——若选参准则也随视图变化，OS/TS 就不再是单变量对照，`cv_scores_` 的跨视图可比性一并失去。**代价**：TS 那一跑不是「按 TS 最优调参」的基线，选出的 `(σ, λ)` 可能偏向 OS 部署表现；该限制与 `nnpu` 对称（`nnpu.py:365` 的 `calibrated_view` 只作用于 `:417` 的训练批损失，早停验证风险 `:470-483` 不随视图变化）。**③ 容量与池子不变**：RBF 中心**候选池**本已取自完整 `X`（非 `X_U`），故**不**照搬 D16 ③ 的 upu 规则——照搬会人为抬高 P 被选为中心的权重；中心数量、CV fold、先验分位数阈值池（仍是完整训练 `X` 的 scores）一概不随视图变化。**④ 实现边界**：角色构造 helper 只接收显式布尔 `include_positive_in_unlabeled`，**不接收** run 级视图字符串，且验证折**不经**该 helper——使「验证保持 OS」在代码结构上 fail-closed，而非靠调用者记得传 `"os"`。**⑤ 落纸定性**：本口径是**协议选择（protocol choice）而非论文事实（paper fact）**——PUSB 原论文不涉及 TS-OS 视图，只有项目协议依据。**⑥** `ts` 合作者复核仍未获得，manifest 继续挂 `ts_view_collaborator_review`（**该阻断位已由 D20 ① 移除；复核本身仍未获得**） | 2026-09-27 |
| D18 | **P2.0e：`dist_pu` 接线口径** | 只冻结 `dist_pu`，**不推广**到其他方法（承接 D16 ②）。**① 校准范围**：`ts` 下 **label-distribution alignment 与 entropy** 的**角色集合**由 $`X_U`$ 换成 $`X_U \cup X_P`$（alignment 均值的分母随之取 $`n_P + n_U`$）；正例 BCE 只消费原始 $`P`$（分母仍为 $`n_P`$）、class prior 保持总体 $`\pi`$ 不重估、Mixup 物理池与先验决策不变。**② alignment 的依据是可核验的 population identity，不是约定**：$`R_{lab}`$ 约束 $`E_{p(x)}[f(x)] = \pi`$，而 OS 下 $`X_U`$ 排除了已标记正例、不再代表完整 marginal，约束落在错的值上；并集把它搬回正确落点。**量化数字（本次 Spambase `split_0` 制品观测，非本方法的一般性质——换 seed/数据集/`c` 会变）**：OS 角色集真实正例率 0.369265 vs $`\pi`$=0.394045（偏离 −0.024780），TS 角色集 0.394022（偏离 −0.000023，分层切分舍入残差），前者约为后者 1076 倍；一般成立的是恒等式 $`\left((n_P+n_U)\pi - n_P \bar f_P\right)/n_U`$（本制品上等于 U 行真实正例率 0.369）。**引用该数字必须带「本次制品观测」限定**。**③ entropy 只有协议一致性依据**（协议 §2.3 对「未标记损失输入」的整体规则 + PUBench 先例），**不是**数学必然：P 的概率已被权重 1.0 的正例损失推向 1，entropy 权重仅 0.05，方向一致不冲突。按 D17 ⑤ 的口径，本口径整体是**协议选择（protocol choice）而非论文事实**——Dist-PU 原论文不涉及 TS-OS 视图。**④ 不照搬 upu 的「池子跟随视图」**：Mixup 池本已消费完整 X，保持原样；更硬的理由是**单变量消融**——池子若随视图缩回 $`X_U`$，OS/TS 的差异就不止「正则项角色集」一个变量（与 D17 否决「验证折并入 P」同源）。**⑤ 随机侧不变量只锁 RNG API 的调用次数、顺序、`randperm` 长度、`lam` 的形状与抽样位置**，**不**锁后续 epoch 的 loss 或梯度（模型参数首轮后即分化，那是消融目的而非缺陷）。**⑥ `sample_weight` 处置与相邻方法都不同**：本估计器是 `IGNORED`——既非 `nnpu` 的「需要互斥门」，也非 `pusb_kernel` 的「整体拒绝」；并集引入的无权重 P 行因权重整体被忽略而无害，三种措辞**不可互抄**。**⑦ 未接线槽位改用测试内注册的合成方法永久钉住**（原钉在 `dist_pu` 上；接线后该槽位改钉 `self_pu` 会在其落地时再次腐烂，故直接换成合成方法，见交付记录）。**⑧** `ts` 合作者复核仍未获得，manifest 继续挂 `ts_view_collaborator_review`（**该阻断位已由 D20 ① 移除；复核本身仍未获得**） | 2026-09-27 |
| D19 | **P2.0e：`self_pu` 接线口径** | 只冻结 `self_pu`，**不推广**到其他方法（承接 D16 ②）。**① 校准范围**：`ts` 下负 PU 项的**角色集合**由 $`D_U^{role}`$（本 epoch **未信任**的 U 行——trusted 行已由伪标签监督，两个视图下都不入负角色）换成 $`D_U^{role} \cup D_P^{k}`$；P 在同一 epoch 内既供正例项 $`\pi R_P^+`$ 与校正项 $`\pi R_P^-`$，又进入负项——**角色叠加，非身份改写**，无行复制、无额外前向。**② 混合按行数质量，且 $`\alpha_P`$ 不是 $`\pi`$**：$`\alpha_U=n_U^{role}/(n_U^{role}+n_P)`$、$`\alpha_P=n_P/(n_U^{role}+n_P)`$，分母取**未信任行数**（取整批 U 行数会把已由伪标签监督的 trusted 行重复计入质量）；$`\pi`$ 仍只出现在校正项，P 不按 $`\pi`$ 重加权、$`\pi`$ 不重估。这与已合并的 `nnpu`（`nnpu.py:417` 的 `cat` + `:436` 的 `mean`）是**同一个并集估计量**，跨方法可比性要求如此；差异只在该方法负角色集合排除 trusted 行。**③ 质量混合 ≠ 并集均值，且这是项目协议定义**：U 侧若被元权重重整，$`\alpha_U E_U^{eff}+\alpha_P E_P`$ **不等于**普通经验并集均值；仅在 U 侧为均匀权重时二者相等（Pilot 恰是这一配置）——该等价性是 **Gate A**，只在无 meta 配置成立。clean-meta 分支是**质量守恒的扩展**，**不是原论文给出的 TS→OS 公式**，其正确性证据是质量守恒、身份安全、OS 冻结与 clean validation 测试，**不是**声称与普通 union 完全等价。$`\alpha_U`$ 无需进入 meta 探针：列 1 的逐列归一化会抵消一致缩放（有测试锁定）。**④ trusted/meta/consistency 一律不跟随视图，且不得跟随**：trusted 人口、pace 与 trusted 容量仍由原始 $`n_U`$ 决定（$`\lfloor pace \cdot n_U \rfloor`$），伪标签、meta influence 维度、consistency 缓存与 hard-negative mask 全部不动——这既是身份安全的来源（已知正例在任何得分下都拿不到伪标签），也是单变量消融的要求。**⑤ 论文保真度的既有收窄**：论文的 $`L_{students}`$ 在 $`D-D_{trust}`$、$`L_{teachers}`$ 在 $`D`$ 上求和（方法卡 §3.4），实现只对 U 批求一致性；这是**视图无关**的既有收窄，本次不改，也**不得**据此声称论文的一致性项是 U-only。**⑥ 验证/选模保持 OS**：clean validation、PU validation、best-epoch 恢复，以及无 validation 时的 ablation teacher selection 分支，都不随视图变化；TS 训练目标由 OS validation 选择（与 D17 ② 同形）。**⑦ 随机侧不变量**只锁 RNG API 的调用次数/顺序/抽样范围、**前向次数**（P 的 logits 复用 `positive_logits`，0 次额外前向、0 次额外 RNG）与 checkpoint/callback 契约，**不**锁后续参数或 trusted 成员（那是消融目的）；OS 路径与改造前一致，本切片冻结基线实测**逐位 0 偏差**。**⑧ Pilot 实际走的是消融分支**：实验层不向 PU 方法传 clean validation（协议 §2.4 把 `clean_val` 留给 OA oracle），故 Survey 中的 `self_pu` 恒为 `calibration_mode_="ablation"`（无 self-calibrated 元重加权，选模走 PU-validation nnPU risk）；含 meta 的分支只由单元测试覆盖，引用 Pilot 结果**不得**读成「含 meta 的完整 Self-PU」。**⑨** `ts` 合作者复核仍未获得（D20 放行后 manifest 不再挂该阻断位，但复核本身仍未获得）。**本行口径已由 D20 单方放行取代**：`manifest 继续挂 ts_view_collaborator_review` 的表述自 D20 起不再成立，其余条款不变 | 2026-09-27 |
| D20 | **P2.0e 单方放行（承接 D12 的做法）** | 三件事合并为一次摘要变更：① **放行 P2.0e**——五个适用方法（`nnpu` #68、`upu` #74、`pusb_kernel` #76、`dist_pu` #77、`self_pu` 本切片）的 `ts` 接线各自独立设计、独立成文并留下 D 记录（D16 ③ / D17 / D18 / D19），本切片另附四项硬证据：OS 基线四组跑与改造前**逐位 0 偏差**；变异检验 **9/9** 被抓住（隔离方案，全程**未**改写工作树里的生产文件）；真实跑批双视图且**重放审计逐位重现**制品（`max_abs_delta = 0.0`，已揭示正例 ∩ trusted = 0、`meta_influence_calls = 0`）；CIFAR adapter smoke 双 teacher checkpoint 与 hook 转发成立。据此按 shuidisjtu **单方技术验收放行**，并从 `survey_protocol.py` 的 manifest 阻断逻辑中**移除** `ts_view_collaborator_review`（原 `:486-490` 的运行时追加）。**移除的是「阻断正式资格」，不等于合作者已复核**：`ts` 路径的方法学复核**仍未获得**，该口径保留在方法台账的 `uncertainty`（`nnpu`/`upu`）与三张方法卡的视图条目中，呈报时**不得**读作已签署；历史制品的 `formal_blockers` 仍如实记录它们当时挂过该阻断位，**不改写**。② **线性 `pusb` 的处置**：经 2026-09-27 裁决为**不适用校准**（其训练信号即 `LogisticRegression().fit(X, y_pu)`、「U ≡ 负类」，不存在与风险估计器同形的未标记损失输入，且不在 Pilot 矩阵内——理由见 D16 ①），**不产出**书面的「适用性裁决」文件；D16 ① 完成口径中的「+ `pusb` 明确处置」由本记录落地，故 P2.0e 可标完成。③ **状态与遗留**：P2.0e 行转 ✅；`self_pu` 的两处已知覆盖边界（M4 类缺陷只被下游形状报错抓住、无 validation 的消融分支无法直接 spy 输入行数）留在交付记录 §7.4/§9，属**可接受的证据边界**而非阻断项 | 2026-09-27 |
| D21 | **P3.1 局部：VPU 采样假设裁决与训练视图接入口径** | 公共训练视图（`pu_toolbox/core/training_views.py`）的首个真实待接入算法。**只冻结 VPU，不推广**到其他方法（承接 D16 ②）。**① 裁决**：`native_sampling_assumption = ts`，四层依据且**已全部亲验**——(a) **论文事实**：式 (6) 的变分目标 `log E_{x~f}[Φ] - E_{x~f_P}[log Φ]` 要求 `f` 是训练总体边缘分布，既非类条件、也非「仅未标记行」；(b) **作者实现**：`vpu.py:79`/`:158` 的 docstring 明写 `x_loader` 覆盖 `training data (including positive and unlabeled)` 与 `the whole training set (positive and unlabeled)`，而池构造 `dataset/dataset_cifar.py:32-40` 让被标记正例**同时留在 X 池**（标签抹为 `-1`，`:102`），故 P 池 ⊂ X 池、**不是互斥划分**（cifar10 下 X 池 45000 行，其中 3000 行为已标记正例）；(c) **对照实现**：PU-Bench 把 `separate labeled-positive P loader and all-training-data X loader` 列为从源实现 **retained** 的组件（`config/methods/vpu.yaml:16`，不在 `benchmark_adaptations` 清单内），训练侧 `vpu_x_loader` 取整个训练 dataset（`train/vpu/trainer.py:90-98`）——独立旁证，但该套件自声明 `source_reproduction: false`，**不替代**作者仓库；(d) **数据事实**：OS 底料下 `D_U` 排除已标记正例、经验分布是 `p(x|s=0)` 而非 `p(x)`，`D_U ∪ D_P` 才还原 `p(x)` 样本（同源论证见 D18②）。**映射是集合级精确的**：作者 X 池 = 训练分区（除验证集）= 工具箱 OS 分区的 `D_P ∪ D_U`，故现有「完整 `X` 作边缘池」与作者实现**逐集合等价**、等价于恒在 ts 视图，只是此前未显式记录。**定性强度高于 D18③**：`dist_pu` 的 entropy 项只有协议一致性依据，本条目是「论文事实 + 数据事实」双重依据，两处措辞**不得互抄**。**② OS 对照有效但不是原生视图**：它是**协议定义的消融**（边缘池被限制到 `p(x|s=0)`），不得读作 VPU 在其原生假设下的运行。**③ 不是 `both`**：`both` 可辩（genuine OS 数据集定义下 `X` 本身即 `p(x)` 样本），但 registry 对 VPU 只登记 `CASE_CONTROL`，而 `test_native_sampling_assumption_matches_registry_scenario` 要求台账值与 scenario **严格等价**，选 `both` 须改 registry、属方法学定位变更、超出本切片；且 `resolve_training_view` 对 `ts`/`both` 行为完全相同，`both` 不带来额外能力。**词汇澄清**：本项目 `ts` 指协议 §2.3 的**视图语义**（边缘池包含正例），**不**要求数据本身按两样本独立抽样生成——§2.3 第 1 条明写基础数据恒为 OS。**④ 验证路径保持 OS，且这是对上游的显式分歧**：上游的验证变分风险对整份验证分区连同其正例子集求值（`U_val ∪ P_val`；作者 `vpu.py:210`、`dataset/dataset_cifar.py:114`/`:133-134`，PU-Bench `train/vpu/trainer.py:112-138`/`:165-189`），本工具箱按 **D17②** 否决该做法（消融干净），D19⑥ 同向。故**引用 VPU 验证变分风险数值时不得声称与上游同定义**；该量在本工具箱内仅为 source diagnostic，不参与 PA/OA 选模。**⑤ 与视图无关的既有适配缺口（登记，本切片不改）**：上游每 20 epoch 将学习率减半并重建优化器（`vpu.py:39-41`）而本实现**无此衰减**；学习率默认值本实现 `3e-4`、上游 `3e-5`（`run.py:11`），相差一个数量级；`max_epochs` 100 vs 50、`batch_size` 128 vs 500，且每 epoch 迭代数由数据量推导而上游固定 `val_iterations=30`。**⑥ 边界**：本记录为 **shuidisjtu 单方技术审计，合作者复核未获得**；VPU 的方法台账条目、`os_or_ts` 视图接线、共享 backbone、图像路径、公开数值对照与多 seed GPU/资源记录均**未完成**，不进入冻结执行矩阵，**不得**读作 P3.1 完成。审计全文与逐行引用见 [`vpu_sampling_audit.md`](vpu_sampling_audit.md) | 2026-09-28 |
| D22 | **VPU 验证变分风险的定位：source diagnostic，不作选模准则** | 承接 D21 暴露的未决问题，**只针对 VPU**，不推广。**① 事实前提**：上游与本工具箱都把 VPU 的验证变分风险定义在**合并边缘池** `U_val ∪ P_val` 上——作者 `vpu.py:210`、`dataset/dataset_cifar.py:114`/`:133-134`（`cal_val_var` 同时消费 `val_x_loader` 与 `val_p_loader`）；PU-Bench `train/vpu/trainer.py:112-138`/`:165-189`；工具箱 `pu_toolbox/estimators/risk/vpu.py:152-158`/`:222-231`。故 VPU 在此**与上游一致**，**不存在**需要登记的口径分歧——D21 审计稿曾误把「否决上游做法」写成既成事实，已更正。**② 与协议 §2.3 的关系**：该条要求「验证集和测试集保持原始 OS 协议，不进行该合并」，VPU 的验证量不满足。本裁决**不修改该条**，而是由协议 §2.3 新增「适用范围」段明确其边界：该要求约束**参与选参或裁决**的验证视图，source diagnostic 量不受约束、但不得充当 PA/OA 选模准则。该适用范围与 D17②/D19⑥ **同向而非相抵**——两者的验证折都用于**选参**，正落在受约束的那一侧，其「验证保持 OS」的依据不因本段而松动。**③ 裁决（走「保持与上游一致」一路）**：(a) 允许按上游定义（合并边缘池）计算、记录并作 os/ts 对照；(b) **禁止**充当 VPU 的 PA/OA 选模准则；(c) 若将来 VPU 进入执行矩阵且需要 PA 选模，**必须另行裁决**——或改用符合 §2.3 的 OS 验证视图（此时即成为对上游的显式分歧，须按 D17② 的论证结构自行成文，并承担「选参准则与训练风险不同形」的代价），或为该行指定其它 PA 机制。**④ 不得引用 D17②/D19⑥**：前者明文「只冻结 `pusb_kernel`，不推广到其他方法」、后者只冻结 `self_pu`，**均不覆盖 VPU**；把它们的「验证保持 OS」口径搬来是对 D 记录的误用（本条的直接触发原因之一）。**⑤ 实现边界**：本条**不改任何代码**——当前行为即目标行为；也**不改 `survey_protocol_v1.json`**——改它会变更协议摘要并触发 D12 式的 `bound_survey_protocol.protocol_sha256` 重绑级联，而 VPU 不在执行矩阵内、无执行影响，故只落在协议 `.md` 的「适用范围」段与本记录中。**⑥ 边界**：本裁决不构成 P3.1 完成，不改变 VPU 不进入冻结矩阵的现状，也不改变「台账条目、视图接线、共享 backbone、公开数值对照均未完成」的口径 | 2026-09-28 |
| D23 | **`route_training_view` 的非对称转发规则（协议级）** | 承接 D21 接线时暴露的缺口。**① 变更**：路由器由「只转发 `ts`」改为——`ts` 照旧（目标未声明则 fail-loud，**消息逐字不变**）；显式 `os` 在目标把 `os_or_ts` **具名为 `fit` 参数**时转发，否则不动；`None` 一律不转发。**② 理由**：旧规则依据「只有校准请求会改变训练」，这对默认值为 `"os"` 的方法成立；但对**默认视图就是 `ts` 的方法**（VPU 首个），显式 `os` 请求会被丢弃、静默跑成 ts——即本模块要防的「manifest 声称的视图与估计器实际应用的视图不一致」以**反方向**重现。**③ 非对称是刻意的**：未声明的 `ts` 请求是静默**降级**（该目标会跑 OS 而 manifest 说 ts），故拒绝；未声明的 `os` 请求是**无操作**（这类目标本就是 OS），故丢弃。`**kwargs` **不**视为 `os` 的声明——它不是「该参数应当被承载」的证据；`ts` 路径继续沿用既有的「具名或 `**kwargs`」判定，**不触碰已冻结行为**。**④ 影响面（实测）**：全 registry 21 个方法中只有 5 个在 `fit` 声明 `os_or_ts`、默认值一律 `"os"`、且**无一**在 `fit` 上收 `**kwargs`，故转发 `os` 对既有链路行为等价；`tests/` 中精确断言 fit-kwargs 的地方只有 `test_training_view_routing.py` 三处，全在改动文件内。**⑤ 被否决的替代**：(a) 三态 `os_or_ts: RunView \| None = None`——不能避免路由变更，且把实现年代的 `"legacy"` 泄漏进公开属性，而它并不是一种视图；(b) 路由里嗅探「签名默认值是否为 `os`」——路由行为依赖另一个函数的默认值，脆弱不可读；(c) 把 VPU 的默认值改成 `"os"`——裸 `fit` 会静默把一个边缘项被错设的目标交给用户，且与 registry `scenario=CASE_CONTROL` 冲突；(d) 本切片不碰路由——则仍需另行封堵 `os` 的静默反演，同样是路由相邻改动，且 VPU 无法被显式持于 OS、与五个已接线方法不对称；(e) 按 `pusb` 先例把 VPU 登记为 `native=ts` + `run_view=os-compatible` + `calib=false`——对 `pusb` 诚实（确实未接线、确实拿 `X_U` 当负类），对 VPU 则是**虚假陈述**（不改代码就在用并集）。**⑥ 边界**：本变更不改变任何方法的默认视图，不进入冻结执行矩阵，不构成 P3.1 完成 | 2026-09-28 |

## 4. 风险

1. **GPU 算力/显存**：shuidisjtu 本机 T600（4GB）不够强，所以主要进行轻量批与开发验证的工作，
   显存不足时（批大小/并行）需在实验记录中说明资源限制；
2. **数据集获取确认**：20News/IMDB/Connect-4/Spambase 的版本与标签编码需与协议锁定映射核对；
   数据版本或标签编码变化时必须提供映射转换审计记录（协议 §2.2 要求）
3. **结果可解释性**：pilot 结果必须区分"文献事实/本实验观测/推断"（协议 §5.7 要求，P2 起执行）
4. **磁盘容量**：P2.1 的累计 checkpoint 预算为 **348,443,368,000 B（≈324.5 GiB）**、跑前门禁
   **18,874,368,000 B（≈17.58 GiB）**（D15），**不得再使用旧估算 1280.6 GiB / 35.16 GiB**。
   该数字只含 checkpoint：数据集、日志、manifest、临时文件与操作系统余量都不在内——**门禁通过
   不等于余量充足**（20 GiB 的主机过门禁后只剩 ≈2.4 GiB）；并发跑多个 run 时总瞬时需求会超过
   单 run 门禁，调度层须按并发数另加余量

## 5. 留痕约定

- 每个实验产物：manifest 固化（协议 §5.6：代码 commit、依赖锁文件、Python/PyTorch/CUDA/GPU、
  数据与方法配置、注册表版本、seed、split/label manifest、选择 artifact、schema 版本）
- 结果按四组存储（SCAR-PA / SCAR-OA / SAR-OA / PN oracle），跨数据集只比较趋势，不生成总排名
- 交叉验证对照结论随聚合报告归档（「交叉验证对照」节三档结论），区分文献事实/本实验观测/推断
- 本计划随执行更新：每完成一个阶段、每产生一个决策，更新任务分工表与决策记录相应条目
