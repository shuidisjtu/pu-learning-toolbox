# PU 调研实验执行计划（pilot → 主榜）

> 本文件维护 Survey 的实验能力、缺口、预算/资源口径与执行边界，以及仍影响执行的决策摘要。
> 实验协议和比较矩阵分别以 [`pu_survey_protocol.md`](pu_survey_protocol.md)、
> `pu_toolbox/experiment/survey_protocol_v2.json` 与 `survey_comparison_v4.json` 为准；
> `survey_protocol_v1.json`/`survey_comparison_v1..v3.json` 保留为历史冻结绑定。
> 状态核对日期：**2026-10-23**（已补 GEN-PU 配方与 PULDA/GradPU 独立图像 runner、真实 CIFAR 小子集120/200快照；PN-CNN 5 seed 继续暂缓）

## 1. 实验能力与缺口

> 本节按“PU Survey 实验设施是否满足实验要求”组织；任务编号只作关联。判定标准：命令可执行
> 并产出合规 manifest 才算“可用”，只有代码存在或测试通过不算。基础状态日期：2026-10-21；本日工程增量见 §1.5。

### 1.1 能力矩阵

| 能力 | 现在能用？ | 缺什么 | 补完后怎么算可用 |
|---|---|---|---|
| 8 数据集四路 split | **是** | — | `run_survey_pilot.py --dry-run` 能列出单元并读取 split |
| clean support（5 数据集） | **是（本机）** | 跨机器传递须按收据核验；正式消费另需 v3/预算 | 25 份五角色制品已生成/回读验证，见本日交付 |
| 协议内 8 方法单跑 | **是**（接口就绪） | 本轮未实跑 | 单方法跑通并产出 manifest |
| 13 个 P3.1 方法单跑 | **否** | 已有来源绑定的 v3 接入请求；仍缺可执行 profiles/units、预算与 runner 绑定 | runner 接受 `--method <13 方法>` |
| 图像/共享 backbone | **部分（独立工程链已通）** | 正式 v3/预算/阶段资格与对应验收；GradPU 私有无BN单列 | 小子集120/200快照及双轨 manifest 已实测，非正式主榜 |
| GEN-PU 独立工程 recipe | **是** | 正式 v3/预算冻结与对应全量资源、数值验收 | 已有真实 CIFAR 固定小子集、标准宽度 ResNet-18 双 batch GPU manifest/回放，不作正式主榜 |
| PA/OA 选模 | **是**（8 方法）；Self-PU/LaGAM/PULNS support 工程双轨另已接通，Self-PU 含显式 TS | 正式 support units/预算与对应验收；真实 Spambase 短 CPU 已验 | 选模产物与 manifest 对账 |
| 分组跑 + 汇总 | **是** | — | `--methods/--datasets/--training-paths` 分批 + 汇总入口 |
| 22 方法主榜 | **否** | 上述缺口 + 预算冻结 | P4.2 门禁 |

### 1.2 缺口任务

| 缺口 | 影响 | 任务 | 负责人 | 完成判据 |
|---|---|---|---|---|
| clean support 正式消费 | PULNS、LaGAM、Self-PU 完整版 | 本机 25 份已齐；待跨机收据核验、正式 units/预算绑定 | shuidisjtu | 正式消费 manifest 与五角色制品对账 |
| 13 方法不在协议 | 13 方法不能进 survey | protocol v3：补 13 个 `method_profiles` 与 `execution_units` | shuidisjtu 起草，HENG958 核对 | `--method` 可跑 |
| 13 方法预算未冻结 | 可比性 | 冻结 epochs/batch/candidates/seeds | shuidisjtu | P4.1-B 记录 |
| 图像正式单元未接 | PULDA/GradPU 不能进主榜 | 独立 runner/小子集120/200峰值已交付；待 v3/预算/阶段绑定 | HENG958 | 正式单元与对应 smoke/profile |
| CVIR 正式接线 | TS 主表 | 已补独立校准/先验换算；待 v3 units/runner、预算与真实验收 | HENG958 | 正式单元 + 跑通 |
| Self-PU 完整版 | 方法完整 | 显式 SCAR OS/TS full-meta 工程双轨已实跑；待正式 clean-support 单元、适用视图/预算与验收 | HENG958 | 正式单元 + 跑通 |
| GEN-PU recipe | 方法完整 | 四项已有工程规格、单候选命令及短 GPU 收据；正式 v3/预算与全量验收待闭合，不自动关 #126 | HENG958 | 正式绑定；当前工程交付见 §1.5 |
| `PA+clean-support` | PULNS/LaGAM | 两方法独立工程路径及真实 Spambase CPU 已通；待正式 units/预算与对应验收 | HENG958 | 正式单元命令与 manifest 对账 |
| PN-CNN | oracle | 5 seed 实验暂缓，协议定稿后再安排；独立真实 CIFAR GPU smoke 仍未交付，本轮不启动 | HENG958 | 定稿后另行安排，不标完成 |
| v2_p4 / v2_p10 / #122 / RV5 / RV6 | 协议与 Pilot 复核 | 先回复五项复核；已有技术意见见本日记录，人工接受/裁决未代填 | HENG958 | 结论记录；v2 review_items 由 shuidisjtu 更新 |

### 1.3 预算与资源

**预算（必须冻结，P4.1-B）**

- `budgets`：每方法 epochs、optimizer steps、batch size、outer candidates；
- `seeds`：0–4；
- `c_tokens`：SCAR 0.1/0.3/0.5，SAR 0.05/0.5；
- 标签预算：|P|、clean support 大小（Spambase 100、其余 500）。

这些参数保持统一与不变是为了维护实验的公平性。冻结后不按运行批次临时改。

**资源**

- CPU/GPU 时间默认不限，不作门禁；
- 单 run 峰值磁盘/内存按方法实测或按参数×快照估算；P2.1 pilot 的 B4 任务参考为 8.34 GiB → 选后
  42.5 MiB；
- 只保留选中 checkpoint，其余按 D25 回收；按组清理，不保留全过程权重；
- P3.3 因此只做“峰值估算 + 保留规则 + 一次 smoke”，不再作为“资源批准”门。

**分批执行**

- 按 method、dataset、training-path（含 backbone）或 labeling mechanism 分组跑，
  `run_survey_pilot.py --methods/--datasets/--training-paths` 原生支持；
- 每组 manifest 绑定同一协议/比较摘要，汇总时按
  `(comparability_group, run_view, mechanism)` 分区；
- 不要求同批跑完，也不要求一次性占用全部磁盘；汇总只在最后统一做。

### 1.4 执行边界

#### 1.4.1 冻结与版本

- 协议以 `survey_protocol_v2.json`（`survey-v2.0`）为准；`survey_protocol_v1.json` 与
  `survey_comparison_v1..v3.json` 仅作历史绑定，对照矩阵以 `survey_comparison_v4.json` 为准。
- 冻结文件只在版本化提案中修改；新数据、方法或配方登记新协议，不覆盖旧 645-run 结果、
  旧 manifest 或历史签署。
- **P3.1 完成 ≠ 正式准入**；正式准入仍待预算冻结与主榜门禁。

#### 1.4.2 数据、角色与标签预算

- train / pu_val / clean_val / test 四角色严格隔离；test 不参与先验、候选、阈值或
  checkpoint 选择。
- clean support 从 train 切出并移除，与其余四路不相交；规模 Spambase 100、其余 500，
  正负标签计入训练期标签预算；覆盖 Spambase、Connect-4、MNIST、F-MNIST、CIFAR-10。
- PULNS、LaGAM 与 Self-PU 完整分支需要该 support；无 support 的 Self-PU 路径只作消融。
- ADNI 槽位为替代数据，subject 级隔离与许可限制保留；未声明/未知许可不阻塞研究实验，
  公开发表或再分发前重新确认。

#### 1.4.3 选模与 PA/OA

- PA 只在 `pu_val` 上用 proxy accuracy 选配置；OA 只在 `clean_val` 上选；两者不混用。
- `PA+clean-support` 与纯 PU PA 同表另列，不参与纯 PU 优势比较；PULNS/LaGAM 只报 OA
  或 `PA+clean-support`。
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
| CVIR | 原始 U 的 α_U 与总体 π 不同；已补全局风险池校准与视图先验换算的独立工程入口 | 正式 v3/预算、真实数据/GPU 仍待，不把工程对照当论文数值复现 |
| Grad-PU | 论文要求移除 BN | `cnn13_no_bn` 单列 method-private，附 SHOULD 共享对照 |
| PULNS / LaGAM | clean support 是额外监督；PA 不适用 | 只报 OA 或 `PA+clean-support`；P3.2 |
| GEN-PU / Holistic-PU | 作者归档未发现 LICENSE | `license=unknown`；不复制、执行或再分发 |
| KLDCE / PUSB 线性 | 求解器或来源身份与论文/官方路径不同 | 方法卡与台账已披露，不改求解策略 |

### 1.5 推进顺序（2026-10-22 对齐任务交接）

以用户提供的 `heng958_tasks_20261021.md` 为本轮任务交接依据；逐项技术意见、证据与
未解除的边界见[20261021 任务复核回复](reviews/heng958_task_review_20261022.md)。

1. **先回复五项 review**（HENG958）：`v2_p4`、`v2_p10`、#122、RV5、RV6；
   前两项接受后，由 shuidisjtu 更新 v2 `review_items`，本轮不代签或修改冻结 JSON。
2. **Issue 收尾**：#126 四项 GEN-PU recipe/blocker 逐项给结论；#88 对照 RV8 区分已接受
   的证据与 #122 等残余项；#122 按明确裁决处理，不能只改 JSD 参数或因测试通过而关闭。
3. **P3.2 剩余实现**：GEN-PU recipe、PULDA/GradPU 图像 runner 与 120/200 快照峰值、
   Self-PU 完整分支及 `PA+clean-support` 正式接线。CVIR 独立校准、25 份 support、
   三方法真实 Spambase 短 CPU 已交付，不重复做；正式 units/预算/对应验收仍待。
4. **protocol v3**（shuidisjtu + HENG958，与上项可并行）：13 方法 profiles/units，
   含 Self-PU 完整分支的新单元和正式 support 消费；跨机数据按已有收据核验。
5. **P4.1-B**：冻结预算/配方；随后 P3.3 核对单 run 峰值、选后保留和 smoke。
6. **分组跑批 + 汇总、P4.2 主榜门禁**：只运行满足协议/预算/配方前置的单元。

**PN-CNN 的 5 seed 实验单独暂缓**：按用户 2026-10-22 指令，协议定稿后再安排；
不启动、不补造结果、不标记完成。该指令不取消其他方法或 support 制品的 seeds 0–4，
也不删除 PN-CNN 已有实现、历史证据或协议中的 oracle 角色要求。

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

2026-10-22 P3.2 工程补记：[clean-support checkpoint 双轨](delivery/clean_support_checkpoints_20261022.md)。
Self-PU/LaGAM 的显式 SCAR OS 路径已接通真实 meta、完成 meta epoch/teacher 快照、
`PA+clean-support`/OA 独立选择、容量估算和选后回收；旧终点 OA/默认消融不变。
CLI 可显式执行单候选 CPU 工程链，不代表正式单位、真实数据/GPU/预算验收；
同日第二批补齐 PULNS episode 回调和同一双轨 CLI；只保存完成 episode 的实际 classifier，
不把 reward probe/内部训练当作额外选择 epoch。v2 两个 pending review 项保留，
正式 units、数据与 P3.2 其他缺口仍待。

同日后续：[CVIR 校准与先验](delivery/cvir_calibration_20261022.md)。新增显式训练
U∪P 全局排序/截断和按 PU 计数换算先验，保留默认 OS 与旧显式 α_U adapter。
独立 SCAR 工程入口可记录 PA/OA checkpoint 选择，仍拒绝 versioned Survey/SAR；
尚未解锁正式 CVIR 单元、配方预算或真实数据/GPU 结果。

同日数据补齐：[五数据集 clean-support 制品](delivery/clean_support_materialization_20261022.md)。
Spambase/Connect-4/MNIST/F-MNIST/CIFAR-10 的 seeds 0–4 共 25 份已在本机生成并按 raw
逐行/五角色摘要/剩余 train 预处理验收（100/500，support_seed=split_seed）。
Self-PU/LaGAM/PULNS 的真实 Spambase 单 seed 短 CPU 双轨 CLI 也已跑通并回放，
不等于正式配方、多 seed/GPU 或论文效果验收。制品不入 Git，只登记
[摘要清单](data/clean_support_materialization_20261022.json)，跨机器传递仍须核验；
v2 pending review、正式 v3/预算边界保持。

## 2. 风险与留痕约定

### 2.1 风险

1. **运行环境**：正式运行仍须 frozen-lock 环境、兼容的驱动/依赖和逐 run 身份记录；合成探针不能代替真实 smoke。
2. **数据与标签**：公开原始数据不等于许可、正式 split 或四角色验收；新数据和版本变化必须留来源、映射、授权范围与摘要，ADNI 官方数据未获批不得纳入；`adni` 槽位使用的是替代数据，须按 D1 的修订标注。
3. **方法准入**：代码级接入、方法卡、smoke、资源 profile 或阶段批准均不等于正式准入；P3.1 收口与 RV8 签署仍不等于正式准入，PULNS/LaGAM 等额外标签预算单独裁决。
4. **存储峰值**：按 D29 只保留选中 checkpoint 并按组清理；D25 回收不降低训练期峰值，单 run 峰值仍按全部 epoch × component × candidate × retry 估算，不外推缩短探针。

### 2.2 留痕约定

- Artifact 固化代码/依赖身份、Python/PyTorch/CUDA/GPU、方法/数据配置、注册表版本、seed、split/label manifest、选择 artifact 与结果 schema；缺必要身份的结果不得进入主表。
- 历史 Pilot 的代码 commit 和依赖锁身份由批次计划/证据包承载，不要求逐 run manifest 重复记录；P4.1-A 未完成前不把 draft 当 locked，不回填历史 manifest。
- 工程资源探针记录实际源码摘要；旧读数对应测量时源码，后续修复不自动构成重测证据。
- 2026-10-07 扩展 CNN receipt 已绑定 `4bfa59b` 基线及逐文件 SHA256；其 21 条 profile 仅证明当前合成 CPU 路径可复核，不解除 P3.3 的峰值估算与保留规则验证。
- 复核结论记录主责人、复核人、代码/制品身份、证据、日期和整改状态；`接受`、`有条件接受`、`退回`不得混写。
- 新阶段进展更新 §1 能力矩阵与缺口任务；新增或修订决策追加附录 A，不重写历史提交或冻结协议。

## 附录 A. 决策账本（D1–D29）

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
