# PU 调研实验执行计划（pilot → 主榜）

> 本文件只维护 Survey 的任务分工、验收边界、风险与留痕约定，以及仍影响执行的决策摘要。
> 实验协议和比较矩阵分别以 [`pu_survey_protocol.md`](pu_survey_protocol.md)、
> `pu_toolbox/experiment/survey_protocol_v1.json` 与 `survey_comparison_v3.json` 为准。
> 状态核对日期：**2026-10-07**；合并基线：远程 `e85f4ff`、本地工程 `25845c3`。
> 复核责任明确归属 **shuidisjtu** 或 **HENG958**；代理检查不能代替二人的签署。

## 1. 任务分工与验收

**HENG958** 负责新增算法、CNN 适配与训练、资源测量；**shuidisjtu** 负责数据、实验协议、
注册表与结果审计，并复核 HENG958 的新增实现。HENG958 复核 shuidisjtu 交付的既有 Pilot 材料。
每项任务只有一名主责；工程 smoke、单方放行或代码存在不等于正式准入。

### 1.1 已完成的工程交付

| 编号 | 任务 | 证据 |
|---|---|---|
| P1.1 | 环境与 GPU 工程验证 | 已有目标设备 smoke；正式运行仍须 frozen-lock 环境 |
| P1.2 | Pilot 三数据集获取与版本审计 | 来源、版本、标签映射与许可已登记；剩余数据见 P1.2-R |
| P1.3a | 方法台账 | 冻结 Pilot 为 7 个 PU 方法 + PN oracle；新增方法状态见 P3.1/RV8 |
| P1.3b | Survey 脚本 | 四路输入、PA/OA、归档与 oracle 入口已有脚本级证据 |
| P1.4 | Pilot 数据产物 | 15 split / 75 文件 / 3 归档技术核验通过；[证据](delivery/p1_4_review.md)，HENG958 复核见 RV1 |
| P2.0a | Pilot 共享规格与 oracle 对齐 | shuidisjtu 已签署；[交付](delivery/p2_0a_delivery.md)、[复核](delivery/p2_0a_review.md) |
| P2.0b | 标签语义门禁 | 单方技术放行；[交付](delivery/p2_0b_delivery.md)，HENG958 复核见 RV2 |
| P2.0c | 对照预注册 | 锚点与规则已冻结；[复核记录](delivery/p2_0c_review.md)，HENG958 复核见 RV3 |
| P2.0d | SAR-OA 路径 | HENG958 已复核（2026-09-16） |
| P2.0e | 五方法 TS-OS 校准接入 | 工程已接线；[交付](delivery/p2_0e_delivery.md)，HENG958 方法学复核见 RV6 |
| P2.1 | Pilot 跑批 | 645/645 完成；[交接](delivery/p2_1_handoff_checklist.md)，HENG958 结果复核见 RV7 |

### 1.2 未完成任务

| 编号 | 任务 | 主责 / 复核 | 前置与当前状态 | 完成条件 |
|---|---|---|---|---|
| P1.2-R | 剩余数据集获取与许可 | **shuidisjtu**；HENG958 核对来源与交付 | MNIST、F-MNIST、20News、Connect-4 已有[原始收件记录](data/public_data_receipt_20261004.json)，须确认执行端可取用及许可；ADNI 尚需获准材料 | 五数据集逐项登记来源、版本、授权/使用范围、文件摘要与可访问位置；ADNI 未取得获批材料不得记作完成 |
| P1.4-R | 新数据预处理与正式 split | **shuidisjtu**；HENG958 复核 | 依 P1.2-R 逐数据集启动，旧 Pilot split 不覆盖 | 按协议准备五 seed 四角色、train-only 统计及索引隔离；锁定 20News 的 20→7 映射、Connect-4 编码和 MNIST/F-MNIST 输入规格，输出数据/划分 manifest |
| P2.2 | Pilot 交付与审计收尾 | **shuidisjtu**；HENG958 独立重放与 C1–C6 复核 | Excel 对账已交付；原始计划、日志、manifest/权重与归档证据待核验 | 补齐证据和说明勘误，完成 O1–O10，按[复核包](delivery/p2_2_review.md)登记双方结论及输入/产物摘要；复核范围见 RV3/RV4/RV7 |
| P3.1 | P3MIX 完整集成与新增方法准入证据 | **HENG958**；shuidisjtu 复核 | 除 `p3mix` 外，Survey 目标方法均已有代码级接入；五方法[来源与预算证据](p3_admission_evidence_20261005.md)、[公开对照草稿](p3_public_comparison_20261005.md)已补，均待审；`p3mix` 仅有数学组件，未注册完整 estimator | HENG958 完成 P3MIX 来源核验、完整 estimator/训练流程、注册与测试；并补齐已接入方法的来源、公式/行为对照、先验/标签预算与校准适用性；shuidisjtu 按 RV8 逐项复核 |
| P3.2 | CNN/共享特征适配与算法收尾 | **HENG958**；shuidisjtu 复核 | 多方法已有 CNN/特征工程路径；Holistic-PU、PULNS、GEN-PU 新路径及[阶段预算证据](p3_stage_budget_review_20261007.md)已补；正式共享规格、增强/调度差异与执行证据仍缺 | 接通获批图像/特征路径、分批训练预测、阶段预算及 checkpoint；明确与原论文适配差异，交付可追溯工程证据；资源验收见 P3.3-A/B |
| PN-CNN | 端到端 CNN PN 全监督 oracle baseline | **HENG958** 实现/训练；shuidisjtu 登记协议并复核 | `PilotOracleCNN` 已有代码和合成验证；旧协议 native CNN oracle 仍不可运行，真实 CIFAR 五 seed 未交付 | 按协议用真实 train PN 标签端到端训练 ResNet-18 与分类头，与对应 PU 共享 split/表征/预算；仅用 clean_val accuracy 作 OA 选模，每 dataset/seed 一次、五 seed；先完成新协议/存储预算登记与真实 CIFAR GPU smoke，再交付权重、指标和 manifest |
| P3.3-A | 批准结构的资源 profile | **HENG958**；shuidisjtu 复核预算依据 | 指定候选工程就绪，结构/表征/测量范围获准；[扩展资源探针](extended_cnn_resource_review_20261007.md)支持 7 配置×3 seed，但完整实测记录尚未交付；现有合成探针不能外推 | 测真实批准结构的序列化、内存/磁盘及保守配额；PUET 按 CPU 条款处理，不计入 GPU 覆盖 |
| P3.3-B | 正式资源与持久化验收 | **HENG958**；shuidisjtu 复核制品 | 对应候选 P4.1-B 冻结绑定，正式协议批准，frozen-lock 环境与资源可用 | 多 seed CPU/GPU、OOM/重试、保存加载、selected checkpoint 对账及逐 run 环境/成本记录；仅验收批准范围 |
| P4.1-A | 注册表绑定工程 | **shuidisjtu**；HENG958 复核接口与方法参数映射 | schema 与 draft 已有，可与算法、数据和 Pilot 复核并行 | 完成 runner/pilot 训练前绑定、manifest/审计/CI 消费及拒收边界；不把 draft 当 locked |
| P4.1-B | 配方审阅与冻结 | **shuidisjtu**；HENG958 核对配置与实现一致性 | P4.1-A 就绪；候选/选模/标签预算与 P3.3-A 资源依据齐备 | 记录双方配置审阅结论、实际批准范围、正式 registry/resolved snapshot 与版本摘要；此为候选/预算批准，不代替 RV8 最终方法复核，不回填旧 manifest |
| P4.2 | 主榜运行结果聚合与分析 | **shuidisjtu**；HENG958 复核深度结论 | 拟纳入数据、PN-CNN、方法准入、P3.3-B/P4.1-B 与正式运行制品均具备 | 22 项通过门禁，按机制、PA/OA、预算族及训练路径分层；区分文献事实/观测/推断，不生成跨数据集总排名 |
| RV1 | Pilot 数据制品复核 | **HENG958**；shuidisjtu 提供材料 | P1.4 技术验收已通过 | 确认数据身份、四角色隔离和预处理证据，记录 HENG958 结论、索引摘要与基线 |
| RV2 | 标签语义门禁复核 | **HENG958**；shuidisjtu 提供材料 | P2.0b 已单方技术放行 | 核对主输入语义、oracle 拒绝路径与角色边界，登记 HENG958 独立复核结论 |
| RV3 | 对照矩阵复核；C1/C2/C4 | **HENG958**；shuidisjtu 提供当前矩阵 | 对象为当前 v3，不沿用历史 v2 签署 | 回查来源、映射与论文/代码矛盾，闭合 36 个锚点/7 条映射待审项，记录逐项结论及绑定身份 |
| RV4 | PA 准则裁决；C3 | **HENG958**；shuidisjtu 提供冻结准则 | 54 条 PA 映射仍受阻 | 给出维持或解除的依据并登记 HENG958 裁决；未解除前不得数值裁决，矩阵变更另留痕 |
| RV5 | Pilot Self-PU 元重加权/消融口径复核 | **HENG958**；shuidisjtu 提供既有 Pilot 协议证据 | Pilot 实际为 ablation，不是完整版 | 确认 clean 标签监督、消融范围与 PA 资格，记录 HENG958 方法学结论，不将消融说成完整论文复现 |
| RV6 | 五方法 TS 接线复核 | **HENG958**；shuidisjtu 提供既有接线证据 | `nnpu`/`upu`/`pusb_kernel`/`dist_pu`/`self_pu` 工程接线已有 | 逐方法确认校准作用范围与不变量，记录 HENG958 结论；路由回归不代替方法学复核 |
| RV7 | Pilot 深度结果与 C5/C6 复核 | **HENG958**；shuidisjtu 提供原始制品 | Excel 不代替结果树、报告及权重核验 | 独立重放、核对深度结果与定性对照边界，复用 RV3/RV4 证据，登记 HENG958 结论与摘要；P2.2 签署按复核包落地 |
| RV8 | 新增算法正式准入复核 | **shuidisjtu**；HENG958 提供实现证据 | 台账 12 条新增方法待复核，需适用的来源/规格/资源证据齐备 | shuidisjtu 逐方法确认 HENG958 的来源、适配、先验/标签预算、选模和资源证据，记录接受、有条件接受或退回及整改项；HENG958 不自审其新增算法 |

### 1.3 执行边界

- 当前推进顺序：P1.2-R/P1.4-R、P2.2/RV1–RV7 与 P4.1-A 并行；随后按候选逐项执行 P3.1/P3.2 → P3.3-A → P4.1-B → P3.3-B；满足全部前置后才执行 P4.2。
- 新数据与 PN-CNN 在新协议中登记，不修改旧 645-run 矩阵或历史 manifest，不混用协议摘要。
- 正式训练前锁定数据、配置与比较组；test 不参与先验、候选、阈值或 checkpoint 选择。
- 工程 smoke、资源测量和配置批准均不等于算法准入；未通过表中前置不得提前主榜。
- 复核由表中指定的 shuidisjtu 或 HENG958 记录证据、结论、身份与日期；退回项另记整改，不代签。

## 2. 风险与留痕约定

### 2.1 风险

1. **GPU 与正式环境**：正式资源验收须使用批准配置、frozen-lock 环境、兼容的驱动/依赖和逐 run 资源记录；合成 CPU/CUDA 探针不能代替 P3.3-B。
2. **数据与标签**：公开原始数据不等于许可、正式 split 或四角色验收；新数据和版本变化必须留来源、映射、授权范围与摘要，ADNI 未获批不得纳入。
3. **方法准入**：代码级接入、方法卡、smoke、资源 profile 或阶段批准均不等于正式准入；P3MIX 必须先完成完整 estimator，PULNS/LaGAM 等额外标签预算单独裁决。
4. **存储峰值**：D25 回收发生在选模和 test 评测之后，只降低最终留存，不降低训练期间峰值；新候选按全部 epoch × component × candidate × retry 估算，不外推缩短探针。

### 2.2 留痕约定

- Artifact 固化代码/依赖身份、Python/PyTorch/CUDA/GPU、方法/数据配置、注册表版本、seed、split/label manifest、选择 artifact 与结果 schema；缺必要身份的结果不得进入主表。
- 历史 Pilot 的代码 commit 和依赖锁身份由批次计划/证据包承载，不要求逐 run manifest 重复记录；P4.1-A 未完成前不把 draft 当 locked，不回填历史 manifest。
- 工程资源探针记录实际源码摘要；旧读数对应测量时源码，后续修复不自动构成重测证据。
- 复核结论记录主责人、复核人、代码/制品身份、证据、日期和整改状态；`接受`、`有条件接受`、`退回`不得混写。
- 新阶段进展只更新本节任务表；新增或修订决策追加附录 A，不重写历史提交或冻结协议。

## 附录 A. 决策账本（D1–D26）

> 本附录只保留仍影响执行的决策摘要；当前任务与责任以 §1 为准，实验规则与比较矩阵以
> `pu_survey_protocol.md` 和机器可读 JSON 为准。实现细节以对应交付文档、代码与提交历史为准。

| # | 决策 | 内容 | 日期 |
|---|---|---|---|
| D1 | **ADNI 数据获取** | ADNI 仅在获批持有人提供授权材料后纳入；未取得不记作数据获取完成，执行任务见 P1.2-R。 | 2026-09-08 |
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
| D22 | **VPU 验证变分风险的定位：source diagnostic，不作选模准则** | ① 更正 D21④：VPU 验证风险保持上游 `U_val ∪ P_val` 定义。② 协议的 OS 验证约束用于选参/裁决，source diagnostic 不受该约束。③ 允许记录和视图对照，但禁止作 PA/OA 选模；未来进入矩阵且需 PA 选模时须另行裁决。④ D17②/D19⑥不能推广到 VPU。⑤ 仅补协议文档，不改代码或协议 JSON；⑥ 不构成 P3.1 完成。 | 2026-09-28 |
| D23 | **`route_training_view` 的非对称转发规则（协议级）** | ① `ts` 请求对未声明目标仍 fail-loud；显式 `os` 仅向 `fit` 具名声明 `os_or_ts` 的目标转发，`None` 不转发，`**kwargs` 不算 OS 声明。② 防止默认 TS 的 VPU 丢失显式 OS 请求；③ 非对称是刻意设计。⑤ 替代方案及来源见原文和[P2.0e 交付](delivery/p2_0e_delivery.md) §11。⑥ 不改方法默认视图或冻结矩阵，不构成 P3.1 完成。 | 2026-09-28 |
| D24 | **放行三个 manifest 正式资格阻断位（承接 D12/D20 的做法）** | ①② 移除旧 Linux 环境偏差、Self-PU clean-meta OA 集成与 PA 待签署三个静态正式阻断；保留逐 epoch 独立选模的动态阻断、`protocol_deviation` 和 Self-PU 消融标记，不代替 HENG958 对 PA/Self-PU 协议的复核。③ 摘要 `c019a87d…` → `287c2f45…`，版本仍为 `survey-v1.2`，矩阵与 README 同步重绑。④(a) 54 条 PA 映射仍不能数值裁决；(b) 历史制品不改写；(c) 同一 `(dataset, seed, c)` 单元整组重跑，不混新旧摘要；(d) 不放行方法学复核。 | 2026-09-28 |
| D25 | **逐 epoch checkpoint 回收：保留选中、回收其余（容量治理）** | ② PA/OA 选模与 test 评测后、manifest 前，仅逐文件回收未选中权重；保留选中权重与全部 epoch 元数据，原 `path`/`sha256` 加 `reclaimed`，不改成 `path=null`。③ 复现范围为“选中权重＋全部选择记录”，不关闭 capture，不降低训练期峰值。④ 选模、阈值与 test 指标不变，B3a 无需重跑；**B3a/B3b 全量留存、B4 起启用回收**，见[交付记录](architecture/epoch_checkpoint_delivery.md) §4。⑤ 不动动态阻断，默认关闭，不放行方法学复核。 | 2026-09-30 |
| D26 | **F13 修复：聚合分组补上 mechanism 维度** | 聚合键由 D13 扩展为 `(comparability_group, run_view, mechanism)`，避免 SCAR/SAR 在相同 c 下混组。只改聚合分区，下层公平性与可比性校验不变；实际视图/机制作为分组维度，不改协议 group 值。实现与验收见[P2.2 交付](delivery/p2_2_delivery.md)；D13 的覆盖率格网问题未由本修复解决。。 | 2026-09-30 |
