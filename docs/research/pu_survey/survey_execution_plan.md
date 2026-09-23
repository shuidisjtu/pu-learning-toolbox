# PU 调研实验执行计划（pilot → 主榜）

> 定位：本文件是**执行路线与状态**的协调层，只保留需人工阅读与决策的内容——任务分工、
> 交叉验证对照预注册、决策记录、风险与留痕约定。实现细节与现状见
> [pu_survey_protocol.md](pu_survey_protocol.md)（要求纲要）与
> [experiment_layer.md](../../dev/experiment_layer.md)（实验层实现）；各 Phase 的交付证据
> 见对应交付记录与复核包（`p2_0a/b/c_delivery.md`、`p2_0a/c_review.md`）。
> 状态日期：2026-09-20。

## 1. 任务分工与验收

实施主体：**shuidisjtu**（数据、实验编排、结果留痕与文档）和 **HENG958**（算法接入、深度训练与 GPU 执行）。每项只有一名**主责**；协作者须在交付前完成复核。任务完成必须有测试、manifest、运行记录或 PR 链接等可复核证据，不能只以口头或代码存在判定完成。

| 编号 | 任务 | 前置 | 验收标准 | 状态 / 主责 |
|---|---|---|---|---|
| P1.1 | 环境与 GPU 验证 | — | `uv.lock` 可复现；目标环境完成 GPU smoke；版本、设备与验证记录可追溯 | ✅ 已完成 / shuidisjtu；HENG958 GPU 能力复核完成，正式跑批需 frozen-lock 环境（见独立复核记录） |
| P1.2 | 数据获取与版本审计 | — | 数据来源、版本、标签映射与许可记录入 manifest；ADNI 的准入状态明确 | 🚧 manifest 侧已补齐（2026-09-19，三 pilot 数据集；许可按官方页面原文逐条记录——UCI 为 CC BY 4.0，另两个来源未声明）；ADNI 不在 pilot 范围，准入路线见 D1。本项 🚧 的口径是**范围**而非遗漏：协议 §2.1 列 8 个数据集，pilot 阶段先取 3 个易得且分属三种模态的（CIFAR-10 / Spambase / IMDB），故 pilot 范围内已完成、按协议全量仍待办 / shuidisjtu |
| P1.3a | 方法台账 | — | 7 个已实现方法的 6 槽（另有 paper/code_version/implementation_status 身份与来源字段）已填写，evidence 覆盖其中 3 槽；每项经对应方法负责人复核 | ✅ 已完成 / shuidisjtu；HENG958 已复核 nnPU、Self-PU（2026-09-16） |
| P1.3b | 官方 Survey 脚本 | — | 四路输入、PA/OA、结果归档与 oracle 入口均有脚本级测试 | ✅ 已完成 / shuidisjtu |
| P1.4 | Pilot 数据产物 | P1.1、P1.2 | CIFAR-10、IMDB、Spambase 各 5 个 seed 的四路 split、预处理与 manifest 均通过合同验证 | ✅ shuidisjtu 侧已完成（三数据集统一重建 2026-09-19）；传输/校验两端工具已就位（`scripts/survey_splits_archive.py`）；**载体已定（网盘带外传），三个归档已于 2026-09-20 发送，逐文件索引与归档摘要先于发送入库**（`docs/research/pu_survey/data/split_artifacts_index.json`）；⏸ HENG958 可执行性复核等待其取件校验 |
| P2.0a | Pilot 共享规格与 oracle 对齐决策（阶段 A） | P1.3a、P1.3b | 版本化执行矩阵（`survey_protocol_v1.json`：预算定义表 + 执行单元行）、runner 强制消费、manifest 扩展（protocol_version/backbone/budget/representation/comparability_group + adapter manifest 合并）、CIFAR adapter 接线、训练路径分组与 PN oracle 对齐方式书面锁定 | ✅ 已签署验收（2026-09-17）/ HENG958 交付；shuidisjtu 复核签署；**不放行正式 P2.1**（R5/R8 记为 P2.1 前置条件，两者已于 2026-09-19 工程兑现）；见 [交付记录](p2_0a_delivery.md)、[复核包](p2_0a_review.md) |
| P2.0b | 标签语义门禁（阶段 A） | P1.3a、P1.3b | `label_semantics_plan` P1+P2 提前完成：声明位 + registry 同步 + experiment 层检查，错误组合 fail-loud；pipeline 层检查属阶段 B | 🚧 工程实现与回归完成，HENG958 独立复核/签署待办；见 [交付记录](p2_0b_delivery.md) |
| P2.0c | 交叉验证对照预注册（阶段 A） | P2.0a | 对照矩阵与判定规则冻结入本文档「交叉验证对照」节；锚点数值预注册 | 🚧 技术审计修订完成，36 锚点/7 行映射待审；HENG958 正式复核未签署，见 [复核包](p2_0c_review.md) |
| P2.0d | SAR-OA 执行路径（issue #43） | P1.3b | 官方脚本可选标记机制（SCAR / SAR LBE-A / SAR LBE-B）；SAR 仅 `{0.05,0.5}` 且强制 OA-only；生成器审计字段入 manifest；脚本级端到端测试 | ✅ 已完成 / shuidisjtu；HENG958 已复核（2026-09-16） |
| P2.0e | TS-OS 校准接入训练执行链（协议 §2.3） | P1.3a | 视图默认由台账 `native_sampling_assumption` 推导、CLI 可覆盖；`ts` 视图逐训练 mini-batch 执行 `D_U^k ← D_U^k ∪ D_P^k`（验证/测试保持 OS）；逐 run 实际视图入 manifest（`run_view`/`calibration_applied`）并成为公平性分组维度；pilot 驱动透传与重跑判定收紧；未接线方法默认回落 `os`、显式请求 `ts` fail-loud | 🚧 部分完成 / shuidisjtu：`nnpu` 已接线并产出 os/ts 对照；其余 5 个原生 TS 方法校准形态各异（闭式 / EM / 双学生 / LR），待各自设计。`ts` 视图尚未经合作者复核，manifest 挂 `ts_view_collaborator_review` 阻断正式资格 |
| P2.1 | Pilot 跑批与运行制品 | P1.4、P2.0a、P2.0b、P2.0c、P2.0e | 每个计划单元产生完整 manifest、选择 artifact、资源/失败记录；oracle 按 `(dataset, seed)` 去重 | ⏳ 待办 / HENG958；编排载体与磁盘预算已就位（`scripts/run_survey_pilot.py`），跑批主机与环境路线、GPU 窗口排定仍待定 |
| P2.2 | Pilot 聚合与审计 | P2.1 | 发布 `pilot / partial benchmark` 分层结果；检查路径隔离、复现字段和异常单元；不得生成跨数据集总排名 | ⏳ 待办 / shuidisjtu；HENG958 复核深度结果 |
| P3.1 | 缺失方法接入（经典/B 类） | P2.0a、P2.0b | 每方法完成实现、方法卡、台账、原文可追溯、冒烟与公开行为对照；使用已锁定的共享规格 | 🚧 技术预集成 / shuidisjtu：VPU、PULDA 已完成独立组件，台账/矩阵与正式验收未做；其余 PAN、RP、CVIR、PULNS 待办 |
| P3.2 | 缺失方法接入（深度/C 类） | P2.0a、P2.0b | 同 P3.1，另需 GPU smoke、设备/随机性与保存加载验证 | ⏳ 待办 / HENG958：PUET、Grad-PU、Robust-PU、Split-PU、LAGAM、GEN-PU、Holistic-PU、P3MIX；Grad-PU/PUET 独立组件已完成，台账/矩阵及正式验收仍待前置项；PUET 为 CPU 树方法，分组/GPU 条款须复核 |
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
> 其 SBERT 特征 L2 归一化（我方口径待核对）。

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

## 4. 风险

1. **GPU 算力/显存**：shuidisjtu 本机 T600（4GB）不够强，所以主要进行轻量批与开发验证的工作，
   显存不足时（批大小/并行）需在实验记录中说明资源限制；
2. **数据集获取确认**：20News/IMDB/Connect-4/Spambase 的版本与标签编码需与协议锁定映射核对；
   数据版本或标签编码变化时必须提供映射转换审计记录（协议 §2.2 要求）
3. **结果可解释性**：pilot 结果必须区分"文献事实/本实验观测/推断"（协议 §5.7 要求，P2 起执行）

## 5. 留痕约定

- 每个实验产物：manifest 固化（协议 §5.6：代码 commit、依赖锁文件、Python/PyTorch/CUDA/GPU、
  数据与方法配置、注册表版本、seed、split/label manifest、选择 artifact、schema 版本）
- 结果按四组存储（SCAR-PA / SCAR-OA / SAR-OA / PN oracle），跨数据集只比较趋势，不生成总排名
- 交叉验证对照结论随聚合报告归档（「交叉验证对照」节三档结论），区分文献事实/本实验观测/推断
- 本计划随执行更新：每完成一个阶段、每产生一个决策，更新任务分工表与决策记录相应条目
