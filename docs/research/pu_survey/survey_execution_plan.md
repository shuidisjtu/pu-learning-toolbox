# PU 调研实验执行计划（pilot → 主榜）

> 本文件维护 Survey 的实验能力、缺口、预算/资源口径与执行边界，以及仍影响执行的决策摘要。
> 实验协议和比较矩阵分别以 [`pu_survey_protocol.md`](pu_survey_protocol.md)、
> `pu_toolbox/experiment/survey_protocol_v3.json`（执行协议）与 `survey_comparison_v6.json`（比较矩阵，v3 运行自动使用）为准；
> `survey_protocol_v2.json` 是 645-run pilot 的历史绑定（绑 `survey_comparison_v4.json`），
> `survey_protocol_v1.json`/`survey_comparison_v1..v3.json` 仅作历史留档。
> 交付物定义、完成线与分工以 [README](README.md) 为准，本文不重复。
> 状态核对日期：**2026-10-26**（v3 协议 72 个单元，runnable 48 / blocked 24，覆盖 5/8 数据集；25 份 clean-support 制品本机重建；GEN-PU 配方与 PULDA/GradPU 图像 runner 已补；PN-CNN 5 seed 已分配 HENG958，见 D45；profile → 比较矩阵映射已落地，见 D40）

## 1. 实验能力与缺口

> 本节按“PU Survey 实验设施现在能做什么”组织；任务编号只作关联。状态分三层，避免把工程接线
> 当结果：**runnable**（`survey_protocol_v3.json` 的 `execution_units` 标 `runnable=true`，
> 有预算/runner，能产出合规 manifest，当前 **48 / 72**）、**formal**（清掉 `formal_blockers`
> 并绑进比较矩阵，当前 **0**；配方注册表已于 D36 冻结、profile 映射已于 D40 落地，但阻断位未清，
> 两者都不等于准入）、
> **evidence**（已有真实数据读数，目前只有单 seed 工程 smoke；32 GiB vGPU 上已跑完一次
> GEN-PU 正式预算 100+100，2026-10-25，仍无正式多 seed 批次）。
> 判定标准：命令可执行并产出合规 manifest 才算“可用”，只有代码存在或测试通过不算。
> 状态核对日期：2026-10-26。

### 1.1 能力矩阵

| 能力 | 现在能用？ | 缺什么 | 补完后怎么算可用 |
|---|---|---|---|
| 8 数据集四路 split | **是** | — | `run_survey_pilot.py --dry-run` 能列出单元并读取 split |
| clean-support 制品（5 数据集 × seeds 0–4） | **是（本机，2026-10-24 重建）** | 无；收据属新批次，跨机传递仍须按收据核验 | 25 份逐份 `load_support_dataset` 通过并登记[机器清单](data/clean_support_materialization_20261024.json) |
| 协议内单跑（v3 48 个 runnable 单元） | **是** | 覆盖 spambase/imdb/cifar10/connect_4/twenty_newsgroups 五个数据集；mnist/fashion_mnist/adni 尚无单元 | `--protocol survey_protocol_v3.json` 跑通并产出 manifest |
| 5 方法阶段预算（GEN-PU/PULDA/Robust/Split/Holistic） | **是（已冻结）** | 逐 seed 成本收据与 manifest 对账 | 阶段快照/成本收据与运行 manifest 对账 |
| 图像三条路径 | **部分**：GEN-PU/PULDA 原生 CNN 与 Robust/Split/Holistic 的 `cnn_feature_adapter` 已标 runnable；GradPU 私有 `cnn13_no_bn` 装配已通、行仍 blocked | GradPU 首次真实运行的 checkpoint 契约；全量预算 | 行翻 runnable 并出真实 CIFAR manifest |
| clean-support 主链（PULNS/LaGAM/Self-PU full-meta） | **是**（spambase seed0 三方法已跑通） | 5 数据集正式 units/预算 | 正式 support 批次 manifest 与制品对账 |
| PA/OA 选模（含 `PA+clean-support` 双轨） | **是** | 准则本体已实现（2026-09-20，PR #65）；54 条 PA 映射的裁定已定并应用（D41：全部 `no_direct_anchor`——两篇基准都没有「本数据集 + proxy accuracy 选模」的数字；Chen 的选模是真标签验证集 macro-F1，对应 OA） | v5 中 54 条 PA 单元全为 `no_direct_anchor`，`blocked_pending_pa_criterion` 已无映射使用 |
| 分组跑 + 汇总 | **是** | — | `--methods/--datasets/--training-paths` 分批 + 汇总入口 |
| v3 ↔ comparison/registry 绑定 | **是** | profile 映射 v3 已生效（16 mapped / 7 not_in_matrix）；比较矩阵按**协议版本**解析，v3 运行自动用 `survey_comparison_v6.json`（2026-10-25 接线，P4.2b 升到 v6）| v3 运行 manifest 的 `comparison_version` / `comparison_sha256` 是 v6 摘要，覆盖率门禁在 v3 上也真跑 |
| 22 方法主榜 | **否** | 24 个 blocked 单元 + 8 个数据集中 3 个（mnist/fashion_mnist/adni）无单元 + connect_4/twenty_newsgroups 只有 4 个方法；P4.2a（4 个基准方法锚点）与 P4.2b（3 个定性交代）均已落地，剩余缺口是数据集覆盖与 blocked 单元 | 数据集补齐 + blocked 单元清零后出主表 |

### 1.2 缺口任务

只列仍未关闭的缺口；已完成项并入 §1.1，不再留在这里。

| 缺口 | 影响 | 任务 | 负责人 | 完成判据 |
|---|---|---|---|---|
| 24 个 blocked 单元 | CVIR/KLDCE/P3MIX/PAN/PUET/RP/VPU 全行、GradPU 全行、LaGAM/PULNS 的 cifar10 行不能跑 | 逐方法补预算与 runner，按各自 `non_runnable_reason` 一条条翻 runnable | HENG958 | 单元 runnable 且产出 manifest |
| P4.2a：4 个有基准覆盖的方法无比较条目 | `pulda`、`robust_pu`、`holistic_pu`、`lagam` 在 v3 可跑，但矩阵里没有锚点 | **已完成**：论文核对确认 PU-Bench Table 1 四个方法全报；10 条锚点已写入 `survey_comparison_v5.json`（PULDA / Robust / Holistic 各 3，LaGAM 1）（OA 单元、`magnitude_and_trend`），profile map v2 把四个 profile 转 `mapped` | shuidisjtu | 已满足：v5 于 2026-10-25 接管运行链路，随后由 v6 接替（锚点不变） |
| P4.2b：3 个无基准方法只做定性交代 | `genpu`、`pulns`、`split_pu` 不在任何统一基准里，各自论文的数据集/指标口径不同 | **已完成**：71 个单元在 `survey_comparison_v6.json` 记为 70 `no_direct_anchor` + 1 `non_runnable`（pulns/cifar10），profile 映射 v3 转 `mapped`；不为凑锚点改数据集/指标 | shuidisjtu | 已满足：三条 profile 有明确 eligibility 与非空理由，摘要中只作背景展示 |
| support 正式 units（5 数据集） | LaGAM/PULNS 在 v3 只有 spambase（runnable）与 cifar10（blocked）两行，Self-PU full-meta 只有 spambase | 为三方法补 connect_4/mnist/fashion_mnist 单元；LaGAM/PULNS 的 cifar10 行并入上面的 24 个 blocked 单元 | shuidisjtu | 5 数据集 support manifest 与制品对账 |
| 数据集覆盖（v3 有 5 个，缺 3 个） | 主榜缺 mnist/fashion_mnist/adni；connect_4 与 twenty_newsgroups 只有 4 个方法，其余 18 个待补 | **进行中**：connect_4 与 twenty_newsgroups 已各加 4 个可运行方法（原地扩充）；mnist/fashion_mnist/adni 要先补 `backbone_specs.image.per_dataset` 与图像预处理。这 4 个方法是按 spambase 已有的 native_2d runnable 行选的，只换数据集、不新造预算；`backbone_specs.image` 目前是 32×32×3 的平铺规格，补 `per_dataset` 时 cifar10 / mnist / fashion_mnist 三条都要写，漏 cifar10 会使其被规格校验拒绝 | shuidisjtu | 每数据集至少一行 runnable（5 个中已满足 2 个） |
| GPU 机时实测 | 无法排 AutoDL 正式批次 | 已完成：GEN-PU 正式预算 100+100（2.666 h / 2.54 GiB）、三个 adapter 行真实 CIFAR 单 seed（0.8–17 min，见[工作簿](data/PU_Survey_results_20261025.xlsx)）；仍缺 GradPU 首跑与 PN-CNN 单 seed 计时 | shuidisjtu | 实测表反推机时 |
| PN-CNN 5 seed | `pn_oracle` cifar10 / native_cnn 行 | 已解除暂缓并分配（D45 取代 D30） | HENG958 | 5 seed 合规 manifest + 峰值记录 |
| PN oracle 其余数据集 | 现有 oracle 行只有 spambase/imdb/cifar10 | 补 connect_4/twenty_newsgroups/mnist/fashion_mnist/adni | shuidisjtu | 每数据集一行 runnable |
| xlsx 导出器与诊断卡工具 | 全库只有单 seed 工程核对表，没有入库的四组拆分/±std/成本导出器，也没有 §1.4.5 信号的诊断卡工具 | 写入库的导出器与诊断卡生成器 | shuidisjtu | 能从 manifest 一条命令生成交付物 A 的工作簿与诊断卡 |

### 1.3 预算与资源

**预算（runnable profile 已由 P4.1-B 冻结，未纳入 methods 仍是 draft）**

- 已冻结（5 方法阶段预算，绑进 `survey-recipes-v1.0`）：GEN-PU 100、PULDA 120、Robust-PU 30、
  Split-PU 40、Holistic-PU 130 份快照，逐组件 8–64 MiB；
- 已冻结（显式 budget 令牌）：PULNS、LaGAM、Self-PU full-meta（spambase 行）；
- 仍是 draft：CVIR、GradPU、KLDCE、P3MIX、PAN、PUET、RP、VPU 只有 `draft_*` 令牌，未冻结，
  在注册表里标 `excluded` 并带理由；
- `seeds`：0–4；`c_tokens`：SCAR 0.1/0.3/0.5，SAR 0.05/0.5；
- 标签预算：|P|、clean support 大小（Spambase 100、其余 500）。

这些参数保持统一与不变是为了维护实验的公平性。冻结后不按运行批次临时改；
已冻结的 profile 以 `survey-recipes-v1.0` 的 `resolved_snapshot` 为准，未纳入的方法不把
draft 令牌当锁定预算。

**资源**

- CPU/GPU 时间默认不限，不作门禁；
- 单 run 峰值磁盘/内存按方法实测或按参数×快照估算；P2.1 pilot 的 B4 任务参考为 8.34 GiB → 选后
  42.5 MiB；
- 只保留选中 checkpoint，其余按 D25 回收；按组清理，不保留全过程权重；
- GEN-PU 正式预算（100 GAN + 100 PN、batch 64）实测：32 GiB vGPU 上每 GAN 轮 78.1 s、每 PN 轮
  19.0 s，合计约 2.7 h；峰值 2.55 GiB，逐组件 43.4 MiB（64 MiB 上界内），100 份快照 4.55 GB；
  本机 T600 一个 GAN 轮超过 20 分钟，正式预算不在本机跑。见容量记录。
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

- 执行协议为 `survey_protocol_v3.json`（`survey-v3.0`，`accepted`；runnable 48/72）。
  `survey_protocol_v2.json`（`survey-v2.0`）继续是 645-run pilot 的历史绑定，不被覆盖；
  `survey_protocol_v1.json` 与 `survey_comparison_v1..v3.json` 仅作历史绑定。
- 比较矩阵按**协议版本**解析：`survey-v2.0` → `survey_comparison_v4.json`，`survey-v3.0` →
  `survey_comparison_v6.json`（各自钉住自己的矩阵，历史版本可重读；v5 仍在盘上）。profile →
  比较条目的解析由生效的 `survey_comparison_profile_map_v3.json`（`survey-profile-map-v3`，绑 v3 协议摘要
  与 v6 矩阵摘要）给出：v3 运行据此记录 `comparison_registration_status`（`mapped_inherited` /
  `mapped_with_profile_deviations` / `not_in_matrix`），`mapped` 的 profile 按继承条目解析每个结果单元；
  P4.2b 后 `genpu`/`pulns`/`split_pu` 已按 `no_direct_anchor` 登记（16 mapped / 7 not_in_matrix），
  余下 7 条都是 v3 不可运行的 profile。
- `runnable` 只表示实现能力，不是正式准入。v3 的 `formal_blockers` 未清；配方注册表已冻结
  （D36，`survey-recipes-v1.0` 绑 `survey-v3.0`），但冻结的是**配方绑定**，15 个 locked profile
  之外的方法仍无配方、正式 GPU 预算与效果结论都不在这里。
  `formal_blockers`、`draft_scope`、`decisions` 与 split_pu 的 `configuration_review` 曾在冻结后
  落后于接线与 #122 裁决；这些措辞已于 2026-10-24 原地修正（版本号仍为 `survey-v3.0`，未改
  `execution_units` 与预算），正式准入状态不变。
- 冻结文件只在版本化提案中修改；新数据、方法或配方登记新协议，不覆盖旧 645-run 结果、
  旧 manifest 或历史签署。
- **P3.1 完成 ≠ 正式准入**；正式准入仍待 `formal_blockers` 清除与主榜门禁（版本化 profile 映射
  已于 2026-10-25 落地，见 D40）。

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
| Split-PU | 论文 Eq. (6) 与源码 JSD 权重方向相反；TS 均衡 batch ≠ 总体 π | 已裁决（2026-10-24）：保持源码 0.7、batch 记 benchmark 近似、台账指针待版本化修正，见裁决记录 |
| PAN | 论文 Eq. (7) 与源码目标、采样不一致 | 保留 paper-faithful；正式采样跟随论文 OS |
| CVIR | 原始 U 的 α_U 与总体 π 不同；已补全局风险池校准与视图先验换算的独立工程入口 | v3 已声明 3 个单元但全 blocked；预算/runner 与真实数据/GPU 仍待，不把工程对照当论文数值复现 |
| Grad-PU | 论文要求移除 BN | `cnn13_no_bn` 单列 method-private，附 SHOULD 共享对照 |
| PULNS / LaGAM | clean support 是额外监督；PA 不适用 | 只报 OA 或 `PA+clean-support`；主 CLI 已接通，正式批次待 P4.1-B |
| GEN-PU / Holistic-PU | 作者归档未发现 LICENSE | `license=unknown`；不复制、执行或再分发 |
| KLDCE / PUSB 线性 | 求解器或来源身份与论文/官方路径不同 | 方法卡与台账已披露，不改求解策略 |

#### 1.4.5 结果汇总与异常诊断

**汇总的目的不是对照论文数值，而是用我们自己的读数判断"某方法在哪、为什么异常"**，并据此佐证或反驳
论文声称的优势。比较矩阵的锚点是辅助信号之一，不是验收目标；因此没有锚点的单元并不因此失去诊断价值。
主产物是我们自己跑出来的结果。

每个进入汇总的单元按**协议已固定的维度**归因，这样异常才有可检查的落点；每行都给出协议依据：

| 信号 | 协议依据 | 异常的形态 | 归因方向 |
|---|---|---|---|
| 机制对比 SCAR→SAR | §2.3（SAR 是独立压力测试、只用 OA，方法分"原生适用"与"假设违背"） | 假设违背组的下降幅度不高于原生适用组；或 SAR 反而更好 | 标注机制处理、权重/score 更新是否真的随机制变化 |
| 视图对比 OS ↔ TS | §2.3（TS-OS 校准只对原生 TS 方法，验证/测试保持 OS） | 原生 TS 方法在 OS 反而更好；或校准前后几乎无差 | 校准实现、未标记风险项是否被真正替换 |
| backbone / 表征 | §2.5（同数据集统一表征；私有网络例外须单列；SHOULD 另给共享架构对照） | 例外方法与共享架构结果排名颠倒；adapter 与原生 CNN 混排 | 表征适配、私有网络的架构影响 |
| class prior 语义 | §3（π_population / π_train / π_U 三者分开记录，视图换算，known 与 estimated 分列） | `c_realized ≠ c_requested`；先验与视图不匹配；known 与 estimated 结果异常接近 | 先验换算与传入是否有误 |
| `c` 敏感性 | §2.3（SCAR c ∈ {0.1,0.3,0.5}） | 曲线非单调或断崖；对 c 完全不敏感 | 标记频率机制、风险估计对 n_L 的依赖 |
| PN oracle 对齐 | §5（四组不混排；PN oracle 同划分同表征） | 超过同组 oracle；与 oracle 的差距随数据集剧烈变化 | 泄漏、评测口径或划分一致性 |
| PA / OA 差距 | §1、§5（两套独立选模，PA 只用 `pu_val`） | OA 高显著而 PA 低；PA 选中 epoch/阈值呈退化模式 | 选模健康度、是否依赖真实标签 |
| seed 间方差 | §1、§5（5 seed 均值±std） | 方差大于方法间差异 | 初始化与调参稳定性 |
| 跨数据集趋势 | §2.5、§5（跨数据集只比较趋势，不合并绝对分） | 论文声称的泛化方向与我们的趋势相反 | 泛化性判断（仍不生成总排名） |
| 外部锚点偏差 | 比较矩阵（当前仅 22 条 `magnitude_and_trend` + 3 条 `background_only`） | 与锚点方向相反或量级差过大 | 仅作辅助：不等于实现有错 |

规则：

- 每个单元必须标出它依据的是上表哪几类信号；没有标依据的单元不出结论。
- 汇总表按 §5 的四组拆分（SCAR–PA 主榜、SCAR–OA 对照、SAR–OA 压力测试（LBE-A/B 分开）、
  PN oracle），并保留按数据集、训练路径、原生假设的分层；不合并总排名。
- `no_direct_anchor` 的单元（当前 111 条）**不得**用"与论文一致/不一致"表述，只能引用内部信号；
  否则"没有参照"会被读成"没有问题"。
- 单 seed 读数不进结论表；5 seed 的均值±标准差才是主表单位。
- 分区键不变（`comparability_group`、`run_view`、`mechanism`，D13/D26）。
- 异常标记只是提示，不自动判定"实现有错"；标记后要回看该方法的实现、选模与数据视图。
- 每条结论都要按 §5 第 7 条区分**文献事实 / 本实验观测 / 推断**，三者不得混写。

产物：方法 × 数据集主表（四组分开）+ 每方法的诊断卡（各信号值、异常标记、待查方向）。

### 1.5 推进顺序与推进记录

**当前推进顺序**（2026-10-26，与 §1.2 的缺口一一对应；分工同 [README](README.md) §3）：

1. **24 个 blocked 单元翻 runnable**（HENG958）：KLDCE/CVIR/P3MIX/PAN/VPU/GradPU 各 3 行
   （spambase、imdb、cifar10）、PUET/RP 各 2 行（spambase、imdb）、LaGAM/PULNS 各 1 行（cifar10）；
   含 GradPU 首次真实运行。**PN-CNN 5 seed**（HENG958）。
2. **数据集与方法补齐**（shuidisjtu）：mnist/fashion_mnist/adni 补 `backbone_specs.image.per_dataset`
   与图像预处理后加单元；connect_4/twenty_newsgroups 补其余 18 个方法；PN oracle 补五个数据集。
3. **汇总工具**（shuidisjtu）：入库的 xlsx 导出器（四组拆分、附加指标、±std、成本）与诊断卡工具。
4. **正式 5 seed × c 网格跑批 → 汇总 → 写结论**（合流）：只运行满足协议/预算/配方/映射前置的单元；
   汇总按 §1.4.5 的诊断口径出"方法 × 数据集主表 + 诊断卡"，不以"与论文数值一致"作为验收标准。
   GPU 机时不设限，按 method / dataset / training-path / 机制分批（D28）。

profile 映射（D40）、P4.2a/b（D37）、GPU 机时实测、P4.1-B 注册表冻结（D36）均已完成，不再是待办项；
此后新增 runnable 单元须同批更新注册表。

**PN-CNN 的 5 seed 实验**曾按用户 2026-10-22 指令暂缓（D30），现已解除并分配给 HENG958（D45）。
该实验仍不得补造结果；它也不影响其他方法或 support 制品的 seeds 0–4。

以下按日期记录推进过程（最新在下）；逐项技术意见与未解除的边界见各日复核文档，
2026-10-21 交接的技术复核回复见 heng958 任务复核（`heng958_task_review_20261022.md`，已归档于 tag `survey-docs-archive-20261025`）。

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

同日数据补齐：五数据集 clean-support 制品。
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
#126 未关闭；本日后续已补PULDA/GradPU 图像 runner 与120/200峰值：
标准宽度 ResNet-18/CNN13-no-BN、固定 CIFAR 小子集各1 seed，0告警/0重试，
双轨选择回放、选后2份快照；完整快照数量不冒充全量数据/默认batch/正式预算验收。
同日后续已补[Self-PU 显式 TS/full-meta 工程双轨](delivery/selfpu_ts_support_20261023.md)，
真实 Spambase 同制品/标记/参数的短 OS/TS 运行均0告警/0重试并回放；
不修改原生风险、不宣称校准效果改善。下一步正式 full-meta/support units 与13方法 profiles/units
继续同 v3 对接，冻结正式预算/阶段资格；PN-CNN 5 seed 暂缓不变。

本日另补 v3 profile/unit 接入请求与校验（`survey_v3_proposal_20261023.md`，已归档于 tag `survey-docs-archive-20261025`）：
范围按 RV8 对齐（13方法含P3MIX、不含VPU；VPU另补，TEDn不计13），22目标算法、
Self-PU 完整版单列后23 profile、8数据集184个阻断请求。该 planning schema 不可执行，
不改 v2 基线/默认，未填正式预算或签署；下一步仍是 profile identity、正式五角色/配方、
多阶段预算校验和新版本 registry/comparison 的实际接线，不能把生成草案标成13方法可跑。

2026-10-24 已补profile 与算法身份分离：
显式 draft schema 下的模型、先验/视图、CLI、批次规划及完整来源摘要续跑已接通；
同算法多 profile 不污染 registry、不覆盖输出。旧 v2 保持，新扩展强制工程验证，
不挪用旧 comparison/registry；full-meta/support 与 CVIR 在缺正式适配时仍提前拒绝。
下一步转入方法族预算/阶段资格、五角色适配和新版本 registry/comparison；
未把这些工程测试或184个请求标成正式准入，PN-CNN 5 seed 仍暂缓。

同日补GEN-PU/PULDA 显式阶段预算：
二维 draft 的预算参数、原生架构身份、实际快照容量和阶段/优化器成本收据已接通；
GEN-PU 不再把 GAN 轮数当 PN 快照数，PULDA 计入两段但不把 TS 并集误计为额外 batch。
其他方法族、native CNN、正式预算/阶段资格及五角色/registry/comparison 仍未完成；
不据此把 P4.1-B 或13方法正式可跑标成完成。

同日后续补Robust-PU/Split-PU/Holistic-PU 阶段预算：
schema 2 已接通三方法 native_2d draft 的实际 CLI/pilot 与成本收据，
Robust episode 内多轮不多造快照；Split 预留 cap、核验原生早停和逐 round 覆盖并可续跑；
Holistic LZO 回选状态仍完整计入已执行 warmup 和前向，OS/TS 边界保持。
阶段预算汇总不再投影成占位1；同快照数也须核对内层/GAN等全部显式参数。
连同前批共5方法具有阶段预算工程接线，不等于5份正式配方批准；
五角色、其他方法族、图像正式绑定、新 registry/comparison 与数值验收仍待。

同日已冻结 v3 协议与 clean-support 主链：
`survey_protocol_v3.json`（`survey-v3.0`，accepted）runnable 40/64，并补 25 份
clean-support 制品本机重建。
重建用同 raw 字节、同准备源码，成员 25/25 与 10-22 记录一致，但收据是新的（本机 numpy
重压缩与 provenance 时间戳），不能冒充旧批次的跨机副本。主 CLI 消费时把制品目录同时作为
`data_dir`、`--split-ref` 指向制品自身 manifest；用 `data/splits/*` 配对会因 train 已移除
support 而被拒。三个 clean-support 方法已在重建制品上跑通单 seed CPU 链路，仍非正式批次。

同日 #122 与 #126 相继裁决：Split-PU JSD/TS batch/台账指针
与 [GEN-PU 四项配方冻结](admission/genpu_126_adjudication_20261024.md)。两者都只冻结决定与已登记差异，
正式准入、全量资源与数值 parity 仍按 §1.2 的缺口处理；v3 里对应的陈旧措辞已原地修正。

2026-10-25 完成 P4.1-B：先做GEN-PU 配方宽度对齐
（协议 128 是构造器默认值，冻结配方与 `GenPURecipeTrainer` 要求 256，128 这条链从未可执行；
对齐后 profile/预算/backbone 规格都是 256，标签改 `mlp256`，64 MiB 逐组件上界按 256 补测仍在界内），
再落地[注册表冻结](admission/p4_1b_registry_freeze_20261025.md)：`survey-recipes-v1.0` 绑
`survey-v3.0`，15 个 runnable profile 锁 `locked`、8 个带理由 `excluded`，覆盖 40 个 runnable 单元；
gate 改按注册表自身 `protocol_basis` 选协议，CI 两道草稿 + 一道生产锁。运行链在本机 T600 上用
`nnpu` 与 `pusb_kernel` 跑通（绑定字段齐备、错误摘要/篡改/无期望聚合均被拒），并有集成测试固化。
仍未改变的是：`formal_blockers` 未清、8 个方法族无配方、比较矩阵仍绑 v2、正式 GPU 批次与效果结论未开始。

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

## 附录 A. 决策账本（D1–D45）

> 本附录只保留仍影响执行的决策摘要；当前任务与责任以 §1 为准，实验规则与比较矩阵以
> `pu_survey_protocol.md` 和机器可读 JSON 为准。实现细节以对应交付文档、代码与提交历史为准。

| # | 决策 | 内容 | 日期 |
|---|---|---|---|
| D1 | **ADNI 数据获取** | ADNI 官方数据仅在获批持有人提供授权材料后纳入，未取得不记作官方 ADNI 获取完成。**2026-10-12 修订**：经与学长、导师讨论，为避免流程过于耗时，`adni` 槽位改用 Kaggle 4 类 Alzheimer MRI 的 GitHub 转存（论文 2 的 Table B.1 称其 Alzheimer MRI，附录称 ADNI），固定在转存仓库 commit 与逐文件摘要；它不是 ADNI 数据，所有登记、报告与快照须写明「替代数据」；Kaggle 页面许可为 Unknown（2026-10-12 核对），同一受试者切片是否跨集未核实，见[替代数据记录](data/adni_substitute_receipt_20261012.json)。 | 2026-09-08（修订 2026-10-12） |
| D2 | 协议分工执行口径 | 研究团队准备四角色数据，工具箱 `fit` 消费已划分输入；工具箱不替代数据划分。 | 2026-09-08 |
| D3 | 依赖锁与 CUDA 环境落地 | `uv.lock` 入库，PR 使用提交的锁，nightly 用 `--upgrade` 重新解析；正式 GPU 运行另核驱动与依赖身份，详见 [ADR-0012](../../adr/0012-dependency-release-policy.md)。 | 2026-10-05 |
| D4 | issue #41 阶段 A/B 拆分 | 标签语义阶段 A 服务 Pilot 门禁，阶段 B 覆盖 pipeline/CLI/UI 收口；实现细节见P2.0b 交付。 | 2026-09-14 |
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
| D20 | **P2.0e 单方放行（承接 D12 的做法）** | ① 五个适用方法的 TS 接线经单方技术验收，移除 `ts_view_collaborator_review`；方法学复核仍未获，历史制品不改写。证据见P2.0e 交付。② 线性 `pusb` 不适用校准，以本记录完成明确处置，不另产裁决文件。③ P2.0e 工程完成，不等于 HENG958 对既有接线的签署或新方法准入。。 | 2026-09-27 |
| D21 | **P3.1 局部：VPU 采样假设裁决与训练视图接入口径** | ① VPU 原生视图裁决为 `ts`，边缘池用训练分区 `P ∪ U`；② OS 是协议消融，③ 不登记为 `both`，此处 TS 是视图语义，不要求独立两样本生成。④ **验证口径由 D22 更正**，不再采用“与上游分歧”的旧判断。⑤ 既有适配缺口见[VPU 审计](admission/vpu_sampling_audit.md)。⑥ 接线及显式 OS 路由随后落地（D23），不代表新方法准入或 §1 列示复核完成。 | 2026-09-28 |
| D22 | **VPU 验证变分风险的定位：source diagnostic，不作选模准则** | ① 更正 D21④：VPU 验证风险保持上游 `U_val ∪ P_val` 定义。② 协议的 OS 验证约束用于选参/裁决，source diagnostic 不受该约束。③ 允许记录和视图对照，但禁止作 PA/OA 选模；未来进入矩阵且需 PA 选模时须另行裁决。④ D17②/D19⑥不能推广到 VPU。⑤ 仅补协议文档，不改代码或协议 JSON；⑥ 不构成 P3.1 完成。⑦ 2026-10-18 复核补注：上游 `val_p_loader` 由全量 clean 类标签抽样，并非 `val_x_loader` 的正例子集；D22 的 source diagnostic 裁决不变，采样差异见[复核](admission/vpu_d21_d22_review_20261018.md)。 | 2026-09-28 |
| D23 | **`route_training_view` 的非对称转发规则（协议级）** | ① `ts` 请求对未声明目标仍 fail-loud；显式 `os` 仅向 `fit` 具名声明 `os_or_ts` 的目标转发，`None` 不转发，`**kwargs` 不算 OS 声明。② 防止默认 TS 的 VPU 丢失显式 OS 请求；③ 非对称是刻意设计。⑤ 替代方案及来源见原文和P2.0e 交付 §11。⑥ 不改方法默认视图或冻结矩阵，不构成 P3.1 完成。 | 2026-09-28 |
| D24 | **放行三个 manifest 正式资格阻断位（承接 D12/D20 的做法）** | ①② 移除旧 Linux 环境偏差、Self-PU clean-meta OA 集成与 PA 待签署三个静态正式阻断；保留逐 epoch 独立选模的动态阻断、`protocol_deviation` 和 Self-PU 消融标记，不代替 HENG958 对 PA/Self-PU 协议的复核。③ 摘要 `c019a87d…` → `287c2f45…`，版本仍为 `survey-v1.2`，矩阵与 README 同步重绑。④(a) 54 条 PA 映射仍不能数值裁决；(b) 历史制品不改写；(c) 同一 `(dataset, seed, c)` 单元整组重跑，不混新旧摘要；(d) 不放行方法学复核。 | 2026-09-28 |
| D25 | **逐 epoch checkpoint 回收：保留选中、回收其余（容量治理）** | ② PA/OA 选模与 test 评测后、manifest 前，仅逐文件回收未选中权重；保留选中权重与全部 epoch 元数据，原 `path`/`sha256` 加 `reclaimed`，不改成 `path=null`。③ 复现范围为“选中权重＋全部选择记录”，不关闭 capture，不降低训练期峰值。④ 选模、阈值与 test 指标不变，B3a 无需重跑；**B3a/B3b 全量留存、B4 起启用回收**，见[交付记录](architecture/epoch_checkpoint_delivery.md) §4。⑤ 不动动态阻断，默认关闭，不放行方法学复核。 | 2026-09-30 |
| D26 | **F13 修复：聚合分组补上 mechanism 维度** | 聚合键由 D13 扩展为 `(comparability_group, run_view, mechanism)`，避免 SCAR/SAR 在相同 c 下混组。只改聚合分区，下层公平性与可比性校验不变；实际视图/机制作为分组维度，不改协议 group 值。实现与验收见[P2.2 交付](delivery/p2_2_delivery.md)；D13 的覆盖率格网问题未由本修复解决。。 | 2026-09-30 |
| D27 | **分批执行与统一汇总** | 最终实验按 method、dataset、training-path 或 labeling mechanism 分组跑，不要求同批或一次性跑完；每组 manifest 绑定同一协议/比较摘要，汇总按 `(comparability_group, run_view, mechanism)` 分区。分组不改变可比性。 | 2026-10-21 |
| D28 | **计算资源不设上限** | CPU/GPU 时间默认不限，不作为跑批门禁；正式运行仍须 frozen-lock 环境与逐 run 身份记录。 | 2026-10-21 |
| D29 | **存储与 checkpoint 保留** | 只保留选中 checkpoint（P2.1 B4 口径），未选中按 D25 回收；按组清理，不保留全过程权重；单 run 峰值按参数×快照估算。仅个别方法超限时再优化。 | 2026-10-21 |
| D30 | **PN-CNN 5 seed 暂缓** | 用户要求 P3.2 中 PN-CNN 的 5 seed 实验等协议定稿后再安排；不视为完成或永久取消，不影响其他方法/support 的五 seed 口径，不改冻结协议及历史结果。**已由 D45 取代。** | 2026-10-22 |
| D31 | **v3 冻结即执行协议** | `survey_protocol_v3.json`（`survey-v3.0`）转正为执行协议，runnable 40/64；`survey-v2.0` 保留为 645-run pilot 的历史绑定；`survey_comparison_v4.json` 仍绑 v2，v3 结果在版本化 profile 映射落地前不进比较矩阵。`runnable` 只表示实现能力，正式准入另需清 `formal_blockers` 与 P4.1-B 冻结。冻结时另有两点：Robust-PU / Split-PU / Holistic-PU 在 cifar10 上只走 `cnn_feature_adapter`，因为 CNN13 / ResNet 的原生实现不存在，adapter 是唯一可比的图像路径；clean-support 类 checkpoint 含双 teacher 或 meta 头，按参数量估算的 MLP 体积偏低，故 `clean_support_two_student`、`lagam`、`pulns` 显式声明 8 MiB 逐组件上界（MLP），最宽的 CNN13 用 64 MiB。 | 2026-10-24 |
| D32 | **clean-support 制品重建口径** | 10-22 的 25 份制品不在本机时，用同一 raw 字节与同一准备入口在本机重建，成员须与旧记录 25/25 一致；重压缩与 provenance 时间戳会改变摘要，因此记为新收据、不冒充旧批次的跨机副本；主 CLI 以制品目录作 `data_dir`、`--split-ref` 指向制品自身 manifest，用 `data/splits/*` 配对会被拒。CIFAR 制品回读后有约 1e-11 的浮点差异，验收按原始 HWC 布局重算；25 份合计约 1.04 GiB。 | 2026-10-24 |
| D33 | **Split-PU #122 三项裁决** | ① JSD 保持锁定源码 `teacher=0.7` 与现有归一化分母，论文 Eq. (6) 方向作已登记差异，不另立变体（只改权重会引入约 1.4466 倍缩放）；② TS 均衡 batch 的并集比例登记为 benchmark 近似，本轮不按 π 加权；③ 台账指针授权修正但推迟到下一次版本化台账更新。Split-PU 两行翻牌、#122 可关。见裁决记录。台账 `prior_semantics` 指针的目标应由 `split_pu.py:223-231` 改为 `:221`，待下一次版本化台账更新时修正。 | 2026-10-24 |
| D34 | **GEN-PU #126 配方冻结** | 认定 2026-10-23 单候选工程 recipe 的四项工程规格即正式配方：论文先验加权 minimax/anti-GAN（`Du` 假样本 `π/(1−π)`、PN 先验加权 softplus）、`genpu` 阶段预算 100 快照、identity 输出 + 两层 256/latent 100 生成器 + `Dp`/`Dn` 线性头 + `Du`/PN 两层 256 + 四份 seeded ResNet-18、G 步冻结 D 的 BN/dropout 保留输入梯度；λ=1 与先验加权不冲突。全量资源实测与数值 parity 不在冻结范围，#126 关单。见[裁决记录](admission/genpu_126_adjudication_20261024.md)。 | 2026-10-24 |
| D35 | **GEN-PU 配方宽度对齐（128 → 256）** | v3 里 128 是构造器默认值被留在冻结时，不是任何一次运行的参数，且 `GenPURecipeTrainer` 会把它判为配方变更；不改就等于用不可执行的数字冒充冻结配方，registry 锁定时 `resolved_snapshot` 也会记 128。就地改协议三处 `hidden_dim` 为 256（profile params、`genpu` 阶段预算、`genpu_native_mlp` 规格），profile 标签 `mlp128` → `mlp256`，`configuration_status` 标为冻结；`execution_units` 的 `genpu_native*` 身份不变。64 MiB 逐组件上界原本就按 256 实测声明，补测 spambase/imdb 两条 native-2d 行后仍在界内（cifar10 45.5 MB、imdb 646 KiB、spambase 319 KiB）。clean-support 清单刷新 `protocol_policy_sha256`，制品未重建。见对齐记录与[机器收据](data/genpu_width_measure_20261025.json)。`model_family` 分两层：执行单元层必须是 `算法名_native`（`survey_staged_budget.py:158-162` 强制），profile 层是 `mlp256`，二者不要求与 `backbone_specs` 一一同名。 | 2026-10-25 |
| D36 | **P4.1-B 配方注册表冻结** | `pu_toolbox/experiment/survey_recipe_registry_v1.json`（`survey-recipes-v1.0`，schema 2.0）绑定 `survey-v3.0`；按既定范围，15 个有 runnable 单元的 profile 锁 `locked`、8 个标 `excluded` 并保留理由，覆盖全部 40 个 runnable 单元。每个 candidate 的 `resolved_snapshot` 由生成器展开、gate 重算比对，手改参数即报 `resolved_snapshot_mismatch`；`review.reviewers` 保持空数组，项目不要求签署。gate 改为按注册表自身 `protocol_basis` 选协议，CI 生产步骤由"待物化则跳过"转为真跑（`--require-present --require-locked`），并新增 v0.2 草稿步骤。不构成方法准入、不回溯绑定历史制品、不代表数值 parity。运行链已用本机 T600 跑通 spambase 的 `nnpu` 与 `pusb_kernel` 绑定（含 `inner_search_id`），错误摘要、篡改与无期望聚合均被拒，并有集成测试固化。见[冻结记录](admission/p4_1b_registry_freeze_20261025.md)。阶段预算 schema 1 / 2 的分工见 `survey_staged_budget.py` 与 `survey_episodic_budget.py`；选择范围沿用原生可部署快照全集，尚未实现任意 stage 子集裁剪。 | 2026-10-25 |
| D37 | **P4.2 拆两档：4 个补基准条目，3 个只做定性交代** | v3 有 7 个 runnable profile 在 `survey_comparison_v4.json` 里没有条目，按原料分两档。① P4.2a：`pulda`、`robust_pu`、`holistic_pu` 在两个已引用基准（`pu_bench_2026`、`pubench_2026`）的代码里都有实现，`lagam` 在 PU-Bench 里有；因此可复用现有 source 补 anchor 与 mapping，不新引入论文。前置是核对基准**论文表格**确实报了这些行——仓库只有代码与 README，没有结果表，代码实现不等于论文报告。② P4.2b：`genpu`、`pulns`、`split_pu` 不在任何统一基准里（PU-Bench 的 `cgenpu` 是变体，不是 genpu；`nnpusb` 不是 Split-PU），各自论文的数据集/指标口径与矩阵不一致，按用户 2026-10-25 决定记 `no_direct_anchor`（不用 `background_only`：它的含义是「存在已发表数字，但不得据此告警」，而这三个方法在统一基准里没有数字），只做定性交代，不为凑锚点改数据集或指标。两档都归我方，队友任务单不含这两项，避免分散其 blocked 单元的工时。重新裁定的触发条件：统一基准发布这三个方法在我方数据集、c 网格、选模口径下的数字，或我们另行跑通论文原始设定并登记为单独的行。 | 2026-10-25 |
| D38 | **PA 准则的账面更正：不再作为外部依赖** | P2.0c（54 条 PA 映射的对照预注册）早已交付，R9 的准则本体也已于 2026-09-20 实现（proxy accuracy，Wang et al. 2026 Def 1 的 OS 分支 + 与 OA 同构的阈值网格与归一化 + π 必传预检，PR #65），当时遗留的只有"合作者签署"与"54 条映射裁决"。签署要求已于 2026-10-24 取消，因此剩下的仅是**我方重新裁定这 54 条映射**并把 `per_epoch_independent_PA_OA_checkpoint_selection` 的措辞改成现状；不再把"PA 准则放行"列为队友编码任务或外部依赖。裁定前 PA 行仍不得做数值裁决，这一约束不变。 | 2026-10-25 |
| D39 | **汇总口径：诊断优先，锚点为辅** | 用户 2026-10-25 定：实验的最终产物是**我们自己的读数**，汇总用于判断"某方法在哪、为什么异常"——例如论文声称多种模态上都稳定，而我们的实现只在单一数据集成立，就提示估计器实现可能有问题——**不强制对照论文数值**。因此新增 §1.4.5：诊断信号按**协议已固定的维度**设计——机制（SCAR/SAR，§2.3）、视图（OS/TS，§2.3）、backbone/表征（§2.5）、class prior 语义（§3）、`c` 敏感性（§2.3）、PN oracle 对齐与四组不混排（§5）、PA/OA 差距（§1）、seed 方差、跨数据集趋势（§2.5）、外部锚点——每行标注协议依据与"异常的形态"，使异常有可检查的落点；每个单元必须标出依据哪几类信号，`no_direct_anchor` 单元只用内部信号表述，禁止写成"与论文一致/不一致"。环境按"固定镜像与规格的实例组"实现（平台文档保证同镜像同规格环境一致），正式批次登记 `frozen_lock_environment_verified=true` 与镜像/驱动/依赖版本，不视为跑批前置障碍。 | 2026-10-25 |
| D40 | **profile → 比较条目走"继承 + 显式偏差"；`not_in_matrix` 只管矩阵，不管可运行性** | v3 的 `execution_units.method` 是 profile ID，而 `survey_comparison_v4.json` 用 method 选择器、没有 profile 维度，因此新增 `survey_comparison_profile_map_v1.json`（`survey-profile-map-v1`）作为 profile 的全函数映射：23 profile = 9 mapped / 14 not_in_matrix（7 个 P4.2 缺口 + 7 个非 runnable）。mapped profile 继承其算法的既有条目，profile 特有差异以 `deviations` 记录并标 `against` 锚点族——论文 `chen_table7` 与受控实现 `pu_bench_table1` 的偏差可以是相反方向的，`self_pu_full_meta` 就是"无 meta 偏差于论文、有偏差于受控实现"，混写会让读者分不清"和谁不同"。`not_in_matrix` 只陈述矩阵没有条目：`kldce` 被注册表 `excluded` 却在 v4 有三条 non-runnable 标记行，故记 mapped；若按"不可运行"把它记为 not_in_matrix，这三行在反查门禁里就成了没有任何 profile 可达的孤立条目。本文件不改 `eligibility`——能否数值裁决仍归 v4（D38 的 PA 阻断不受影响），映射只回答"继承哪条"。协议或矩阵摘要变更必须同步刷新双绑定，CI 以 `scripts/check_survey_profile_map.py --require-present` 强制；历史 manifest 不改写。 | 2026-10-25 |
| D41 | **54 条 PA 映射裁定：全部 `no_direct_anchor`；v3 的静态阻断位改为按运行证据动态判定** | 54 条 `blocked_pending_pa_criterion` 是 6 方法 × 3 数据集 × 3 个 c 的全部 `pa` 选模单元。读两篇基准 PDF 后核定选模口径：Chen 等（PU-Bench）Table 1 确实报了 `pulda`/`robust_pu`/`holistic_pu`/`lagam`（8 数据集、`c=0.1`、SCAR、accuracy mean±std），但其 checkpoint 是「**带真标签的验证集上 macro-F1 最优**」（正文 §3 与 §D.2.1 两处，Table B.2 的 `Validation` 列属 PN 统计；Wang 等附录 A 亦指出该验证集带真标签），属 oracle 家族，对应我们的 **OA** 而非 PA；Wang 等只报 `Robust-PU`/`Holistic-PU`，且其 CIFAR-10 用 20k 子集、π=0.3、正类子集与我们不同，PA/PAUC 数字 v4 已用在 `row_*_cifar10` 行级映射。因此 54 条 PA 单元在两篇里都没有「本数据集 + proxy accuracy 选模」的数字，**全部记 `no_direct_anchor`**；18 条 c=0.1 的 OA 单元维持 `magnitude_and_trend` 与各自锚点不动。附带发现：这 18 个 Chen 锚点在 v4 里标 `selection_protocol: proxy_acc` 属**元数据错误**（实为真标签验证 + macro-F1），v5 修正并把准则/验证集差异写进 `protocol_differences`。**应用载体是 comparison v5**：v5 还要绑 v3 并收录 P4.2a/b 条目（`self_pu_full_meta` 与 7 个 runnable profile 在 v3 的单元枚举里必须有映射，否则覆盖率门禁直接失败），因此本次不新建中间版本，也不原地改 v4（那会追溯改写 645-run 的裁决，违反"历史制品不改写"）。同一笔清理 v3 的静态阻断位：删掉 `per_epoch_independent_PA_OA_checkpoint_selection`（准则本体 2026-09-20 已实现，且该项已由 `runner._checkpoint_context` 依据每次运行的真实 checkpoint 轨迹动态加减；静态列表再留一条会让 runner 永远摘不掉），并把"等 P4.1-B 冻结与 profile 映射"改写成现状（两者均已于 2026-10-25 完成，剩下 v5 与多 seed 批次）。协议就地改、版本仍 `survey-v3.0`，规范摘要 `44128fcb…` → `50a96eb1…`，连带刷新 recipe registry 与 v0.2 草稿的 `protocol_basis`、clean-support 清单的 `protocol_policy_sha256`（原始字节 `57abbcb1…` → `b3ea7e5d…`）与其 `receipt_sha256`；25 份制品本身未重建、其自身收据未改。**本记录第一版曾裁定把 18 个 `proxy_acc` 锚点迁到 PA 单元；读完两篇论文原文后更正为上文的裁定，错因（拿矩阵自身的元数据当证据起点）记在交付记录 §4。** 见裁定记录与论文核对。两处更正与补记：① 上文「12 条锚点」「v4 已用在 `row_*_cifar10` 行级映射」均与 v6 不符——v6 的 P4.2a 锚点实为 10 条（PULDA / Robust / Holistic 各 3，LaGAM 1），行级映射只有 upu、nnpu、pusb_kernel、dist_pu、lbe、self_pu。② Wang 等 CIFAR-10 的 Robust-PU / Holistic-PU 读数（PA / PAUC / OA）未登记为锚点，其 c 值未核：Robust-PU Case1 85.57 / 85.61 / 85.91，Case2 78.94 / 78.43 / 79.60；Holistic-PU Case1 50.20 / 50.00 / 81.81，Case2 55.60 / 56.04 / 71.18。 | 2026-10-25 |
| D42 | **协议与矩阵原地演进；绑定只校验版本** | 用户 2026-10-25 定：`survey_protocol_v3.json` 与生效比较矩阵（现 `survey_comparison_v6.json`）**原地扩充**，不再新增版本号产物；历史版本文件留在盘上、可显式加载，不重建。摘要绑定随之简化为**只校验版本**（矩阵 ↔ 协议、profile 映射 ↔ 协议/矩阵、recipe registry ↔ 协议）：摘要字段保留为来源记录，不再要求相等，因此原地改不再需要刷新任何摘要（生成式 registry 只在内容真正变化时重建）。profile 映射仍要求覆盖协议声明的 profile 集合，否则视为「不是这份矩阵」。运行约束：改协议要避开在飞的批次，否则同批 manifest 的 `protocol_sha256` 会分成两半、`check_protocol` 判不一致。已知残留：`comparison_for_protocol` / `comparison_for_version` 双入口可并为一个。 | 2026-10-25 |
| D43 | **P3.1 宽口径范围；图像 runner 120/200 快照的实测口径** | ① 范围：P3.1 宽口径为 13 个方法 = RV8 评审的 14 条减 TEDn（TEDn 只走 RV8、不计入 13），含 P3MIX、不含 VPU（VPU 另补）；对应 22 个目标算法、23 个 profile（含 `self_pu_full_meta`）。RV8 于 2026-10-21 由 shuidisjtu 签署 14 条全部「有条件接受」，2026-10-24 起不再要求签署。② PULDA 120 份 / GradPU 200 份快照在真实 CIFAR 24 行子集上的工程实测（各 1 seed，数字在 `data/candidate_image_validation_20261023.json`）：单份 44,900,758 / 40,974,252 B，回收前合计约 5.02 / 7.63 GiB，选后各留 2 份；fit 的 CUDA allocated 峰值约 437 / 254 MiB，RSS 峰值约 2.72 / 2.29 GiB；磁盘预检 1 候选 × 2 次尝试，分别 15 / 25 GiB。口径：是 allocated 不是 reserved，RSS 为进程生命周期而非 fit 增量，**不代表全量数据、默认 batch 或多 seed 的峰值**；所有已完成的 warmup 与 pu_mixup 快照都进入选择；该子集上 proxy accuracy 可超出 [0,1]。 | 2026-10-26 |
| D44 | **终点线与交付物定义** | 最终交付物两部分：A 结果文件（四组工作簿，5 seed 均值±标准差，加按 §1.4.5 的结论分析）；B 代码功能（协议 §2.4（四）第 7–11 条逐条对应可运行命令或文件，加从四路 split 到汇总表的复现命令序列）。两者共同过协议 §5 验收线；22 个目标方法未全部过四项门禁时，标 `pilot / partial benchmark` 发布。mnist / fashion_mnist / adni 为必做数据集。§5.6 的代码 commit 与依赖锁由批次层承载（每批一份证据包，run 引用批次身份）；协议 §5.6 第 6 条措辞待改，并须确认批次层确有写入 commit 与 `uv.lock` 哈希的代码。交付物、完成定义与分工见 [README](README.md)。 | 2026-10-26 |
| D45 | **PN-CNN 5 seed 解除暂缓，分配 HENG958** | 取代 D30：协议已定稿，`pn_oracle` 的 cifar10 / native_cnn 5 seed 由 HENG958 承担，与 24 个 blocked 单元并行。验收同其他单元：合规 manifest + 峰值记录；不补造结果。 | 2026-10-26 |
