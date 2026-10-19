# PU 调研实验执行计划（pilot → 主榜）

> 本文件只维护 Survey 的任务分工、验收边界、风险与留痕约定，以及仍影响执行的决策摘要。
> 实验协议和比较矩阵分别以 [`pu_survey_protocol.md`](pu_survey_protocol.md)、
> `pu_toolbox/experiment/survey_protocol_v1.json` 与 `survey_comparison_v3.json` 为准。
> 状态核对日期：**2026-10-19**；RV1/RV2 验收基线为 PR #110 的 `8b4e751`。
> 复核责任明确归属 **shuidisjtu** 或 **HENG958**；代理检查不能代替二人的签署。
> RV1/RV2 与 P1.4-R 已由用户接受技术复核证据并确认完成，不再要求额外人工签署，见对应验收记录。
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
`input_sampling="ts"`，但不做 OS→TS 校准，也不自动验证采样设计。真实数据数值验收、
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
| RV1 | Pilot 数据制品复核 | **2026-10-16 完成（用户确认）**；接受数据身份、四角色隔离及预处理技术证据，见[验收记录](delivery/p1_4_review.md)；不再另要求 HENG958 签署 |
| RV2 | 标签语义门禁复核 | **2026-10-16 完成（用户确认）**；170 项相关测试及 PNU 三值直调通过，见[验收记录](delivery/label_semantics_stage_b_20261015.md)；Issue #41 可关闭 |
| RV3 | PA 阻断映射 eligibility；C2 | **2026-10-16 完成（shuidisjtu 裁决）**；54 条 `blocked_pending_pa_criterion` 解除，归类与其 OA 对应映射一致，不新增数值裁决，见[裁决记录](reviews/p2_2_c2_c3_ruling_20261016.md)。**机器矩阵 v3 暂不更新**，以裁决记录为准，正式使用对照前须另发 v4；36 个锚点、7 条映射、4 条 contradictions 保持 `pending_review`，对照陈述继续视为未复核 |
| RV4 | PA 准则裁决；C3 | **2026-10-16 完成（shuidisjtu 裁决）**；锁定 `pu_val` proxy accuracy 准则（Def. 1 OS 分支），依据为试点 54 个配对单元的 PA/OA 差异，见[裁决记录](reviews/p2_2_c2_c3_ruling_20261016.md)；锁定不等于 PA 单元获得数值裁决 |
| RV7 | Pilot 深度结果与 C5/C6 复核 | **2026-10-16 按范围调整关闭（用户决定）**；C5/C6 按通过处理，**未做独立重放或逐项核查**，也无 HENG958 签署，见 P2.2 与[复核包](delivery/p2_2_review.md) §2.3。深度结果仍以 P2.2 的三个分析入口重跑为证据 |

### 1.2 未完成任务

「当前状态」列先写已有基础，再以 **待办：** 显式列出尚需完成的事项；「完成条件」只写验收口径与边界。

| 编号 | 任务 | 主责 / 复核 | 前置 | 当前状态 | 完成条件 |
|---|---|---|---|---|---|
| P3.1 | P3MIX 完整集成与新增方法准入证据 | **HENG958**；shuidisjtu 复核 | 无（与 P3.2 并行） | 2026-10-19 根据用户提供的 14 页 ICLR 2022 全文完成 P3Mix/P3Mix-E/P3Mix-C 来源核对、可训练 estimator、注册、OS-only 台账与方法卡。CIFAR-10 文献读数另有[机器可读来源锚点](data/p3mix_literature_anchors_20261019.json)，绑定论文摘要与表格页码，保持待审且不改冻结矩阵。专项 estimator、registry、候选扩展、台账一致性及准入契约测试通过；P3 admission evidence/public comparison 门禁通过。recipe registry 属冻结阶段产物，当前文件不存在；`--require-present` 运行报告 deferred/skipped，不能记作已验证。验收记录见 §1.3 当日记录及[逐项原文交叉表](admission/p3mix_paper_review_20261019.md)。2026-10-19 已把 TS 校准作用范围审计链接纳入台账；本轮新增补核 11 个方法，VPU 保留既有单方采样审计，当前 12/12 个 `calibration_hooked=true` 方法均有审计引用。对照全文后补正 γ=0.5 时边缘集合必须为空（原文式 (6) 闭区间与紧随其后的文字说明存在退化点张力）；实现、方法卡和精确 0.5 回归测试已同步。全部仍待 HENG958 逐方法确认，不代表签署。正式配方/预算、资源验收、准入和其他新增方法逐项复核也未完成。<br>**待办：** ① 方法负责人按 RV8 核对论文公式、未指定工程细节及实现；② 补齐批准协议的公开对照与资源证据；③ 其余已接入方法逐项复核 | P3MIX 工程集成经测试，适配与论文未指定项透明记录，负责人按 RV8 完成复核；新增准入另需协议/预算/资源批准，不由代码接入自动触发 |
| P3.2 | CNN/共享特征适配与算法收尾 | **HENG958**；shuidisjtu 复核 | 无（与 P3.1 并行） | 多方法已有 CNN/特征工程路径；Holistic-PU、PULNS、GEN-PU 新路径及[阶段预算证据](p3_stage_budget_review_20261007.md)已补；2026-10-19 复跑 19 个 CNN/encoder、图像协议、候选训练收据及分批预测测试文件，退出码 0；9 项 CUDA-only 测试因本机无 CUDA 跳过，不能据此声称 GPU 验收。各方法的补记见 §1.3。<br>**待办：** 确定正式共享规格；获批图像/特征路径、分批训练预测、阶段预算及 checkpoint 的跨方法统一验收；说明增强/调度与原论文的适配差异；完成执行证据与合作者复核 | 交付可追溯工程证据；资源验收见 P3.3-A/B |
| PN-CNN | 端到端 CNN PN 全监督 oracle baseline | **HENG958** 实现/训练；shuidisjtu 登记协议并复核 | 新协议与存储预算登记 | `PilotOracleCNN` 已有代码和合成验证；旧协议 native CNN oracle 仍不可运行；真实 CIFAR 五 seed 未交付。<br>**待办：** ① shuidisjtu 完成新协议/存储预算登记；② HENG958 完成真实 CIFAR GPU smoke；③ 交付五 seed 权重、指标和 manifest | 用真实 train PN 标签端到端训练 ResNet-18 与分类头，与对应 PU 共享 split/表征/预算；仅用 clean_val accuracy 作 OA 选模；每 dataset/seed 一次、五 seed |
| P3.3-A | 批准结构的资源 profile | **HENG958**；shuidisjtu 复核预算依据 | P3.2 的批准结构 | [扩展资源 receipt](data/candidate_extended_cnn_storage_probe_20261007.json)已完成 7 recipe×3 seed 的独立 CPU 技术测量并通过 checker；仍是合成输入、非批准结构/正式上界，GPU 未测，不能外推正式数据集。<br>**待办：** 测真实批准结构的序列化、内存/磁盘及保守配额 | 测量覆盖批准结构；PUET 按 CPU 条款处理，不计入 GPU 覆盖 |
| P3.3-B | 正式资源与持久化验收 | **HENG958**；shuidisjtu 复核制品 | 对应候选 P4.1-B 冻结绑定；正式协议批准；frozen-lock 环境与资源可用 | 未开始（前置未满足）。<br>**待办：** 多 seed CPU/GPU 运行；OOM/重试、保存加载、selected checkpoint 对账；逐 run 环境/成本记录 | 仅验收批准范围 |
| P4.1-A | 注册表绑定工程 | **shuidisjtu**；HENG958 复核接口与方法参数映射 | 无（与 RV5/RV6 并行） | runner/CLI、pilot、外部白名单/审计/汇总、draft CI 及[版本化候选池身份](delivery/p4_1_pool_binding_20261014.md)已接通，含全字段删除拒收、PA/OA 独立候选与 trajectory 对账；配方未冻结。<br>**待办：** ① 激活正式产物 CI；② HENG958 复核接口与方法参数映射 | 正式使用须采用外部 v2 白名单，不把 draft 当 locked 或代签；新增参数搜索仍需协议/registry 语义与预算批准 |
| P4.1-B | 配方审阅与冻结 | **shuidisjtu**；HENG958 核对配置与实现一致性 | P4.1-A 就绪；候选/选模/标签预算与 P3.3-A 资源依据齐备 | 未开始（前置未满足）。<br>**待办：** 双方审阅配置；登记实际批准范围、正式 registry/resolved snapshot 与版本摘要 | 此为候选/预算批准，不代替 RV8 最终方法复核，不回填旧 manifest |
| P4.2 | 主榜运行结果聚合与分析 | **shuidisjtu**；HENG958 复核深度结论 | 拟纳入数据、PN-CNN、方法准入、P3.3-B/P4.1-B 与正式运行制品均具备 | 未开始（前置未满足）。<br>**待办：** 聚合并分析主榜运行结果 | 22 项通过门禁；按机制、PA/OA、预算族及训练路径分层；区分文献事实/观测/推断，不生成跨数据集总排名 |
| RV5 | Pilot Self-PU 元重加权/消融口径复核 | **HENG958**；shuidisjtu 提供既有 Pilot 协议证据 | 既有 Pilot 协议证据 | Pilot 实际为 ablation，不是完整版。<br>**待办：** 确认 clean 标签监督、消融范围与 PA 资格 | 记录 HENG958 方法学结论；不将消融说成完整论文复现 |
| RV6 | 五方法 TS 接线复核 | **HENG958**；shuidisjtu 提供既有接线证据 | 既有接线证据 | `nnpu`/`upu`/`pusb_kernel`/`dist_pu`/`self_pu` 工程接线已有；2026-10-19 已补[逐方法作用范围/不变量技术核验](reviews/rv6_ts_calibration_technical_review_20261019.md)，五方法及公共路由专项测试通过。**待办：** HENG958 确认 mini-batch 校准、PUSBKernel 的 TS 训练/OS 验证选择口径，以及分布损失/加权负风险的适配结论 | 记录 HENG958 逐方法结论；工程技术建议和路由回归均不代替方法学复核 |
| RV8 | 新增算法正式准入复核 | **shuidisjtu**；HENG958 提供实现证据 | 适用的来源/规格/资源证据齐备 | 台账 14 条新增/扩展方法待复核（含独立 native-TS TEDn 与 P3MIX）。<br>**待办：** 逐方法确认 HENG958 的来源、适配、先验/标签预算、选模和资源证据 | 记录接受、有条件接受或退回及整改项；HENG958 不自审其新增算法 |

### 1.3 执行边界

2026-10-19 续批收尾见[任务/发布验收与后续](reviews/candidate_followup_summary_20261019.md)：
逐项修复评分模式、15候选失败拟合和机器证据绑定；另修临时LBE脚本的Git/cwd溯源。
短预算Spambase wheel回归不判方法效果，不覆盖旧产物或正式跑批。

2026-10-19 用户补入 `30_who_is_your_right_mixup_partne.pdf` 后完成 P3MIX 全文核对与
[完整实现交付](../method_cards/P3MIX.md)，并建立[逐项原文复核交叉表](admission/p3mix_paper_review_20261019.md)：式(1)–(8)、Algorithm 1、pool epoch 更新和独立
P3Mix-E/C 已接入训练类、registry、ledger、方法卡与专项测试。文中没有作者仓库链接，也未
指定完整优化器/批次/backbone/teacher decay 配方；当前默认均标作工程选择。只声明代码级
NATIVE 与 OS 训练，不启用 TS、不改冻结矩阵/候选准入/历史结果，RV8 负责人复核及正式预算仍待。

2026-10-19 对冻结来源中的 Dist-PU 作者仓库补做训练行为交叉核对：toolbox 当前全量张量
clean-room estimator 与作者锁定程序差异不只在 backbone/batch size，还包括两阶段 warm-up/PU、
持久批次伪标签、histogram distribution loss、分阶段 Adam/余弦日程及熵系数调度。来源状态
`official_exact` 仅表示作者来源已锁定，不表示实现行为忠实；详细审计见
[Dist-PU 来源行为复核](admission/dist_pu_source_behavior_review_20261019.md)。随后新增显式 `author_staged`
适配路径（见下方补记）；默认 clean-room 训练目标/参数更新不变，但公开 `batch_size` 默认值从 128 调为 256，
仅 staged 路径使用该值，clean-room 仍忽略它。两种路径均不改校准原则、冻结结果或历史 Pilot，也不照搬作者
test accuracy 选模；正式数据/结构/预算仍待协议审批。
该差异还通过拟合后的 `training_provenance_` 输出到 run manifest，显式披露 batch_size 未生效和实际优化步数。
补充核对 `utils.train()` 与 `train_mixup()` 后更正 scheduler cadence：warm-up 每个 epoch 末 step 一次，PU 阶段每个 mini-batch 后 step；两者均设 `T_max=pu_epochs`。详见来源审计中的锁定 commit 行号链接。此前“两阶段都逐 batch step”的记录已更正。
另补 Dist-PU `author_histogram` 目标组件及显式 `training_recipe="author_staged"` 适配：覆盖来源两阶段 mini-batch、持久伪标签、分阶段 Adam、warm-up epoch/PU batch scheduler cadence、动态熵与双目标 Mixup；TS 角色映射仍遵循项目校准规则。默认 clean-room 路径与历史结果保持不变；专项测试覆盖 histogram golden、两阶段步数/epoch checkpoint 身份、校准映射和失败重训失效。适配采用 CPU 常驻输入与 batch transfer，仍使用 toolbox MLP/调用方 prior；不移植作者数据/图像模型/预处理/split，不照搬 test accuracy 选模，也不构成正式 recipe、数值复现或新准入。其余数据、结构、PA/OA 选择、预算和资源仍须协议审批。

同日验收补记：P3Mix estimator、registry、候选扩展脚本、台账一致性与准入契约专项测试通过；
`check_p3_admission_evidence.py` 与 `check_p3_public_comparison.py` 通过。
`check_survey_recipe_registry.py --require-present` 因冻结阶段文件不存在而报告 deferred/skipped，
没有验证任何配方。训练/生命周期测试已纳入本地验收，
不改变上一段列明的负责人复核、正式配方/预算、资源验收与准入前置。

2026-10-19 门禁范围澄清：`check_p3_admission_evidence.py` 的 `methods_checked` 是
PULDA、PUET、GradPU、Robust-PU、Split-PU；`check_p3_public_comparison.py` 检查的是
既有 12 条文献读数/15 个覆盖单元，**两者均不校验 P3MIX 文献锚点**。P3MIX 的六条
CIFAR-10 锚点由 `test_p3mix_literature_anchors.py` 单独核对，候选 fit 生命周期由
`check_candidate_fit_contracts.py` 纳入；以上均是技术一致性证据，不替代 RV8 或正式准入。

2026-10-19 依据全文新增[P3MIX 文献锚点机器记录](data/p3mix_literature_anchors_20261019.json)，绑定上传 PDF 摘要、Table 2/3 页码、CIFAR 类别映射、1000 个标记正例、500 个验证样本和五次运行的 accuracy 均值/标准差。该记录只用于来源审计，E/C 重复报告不视作独立实验；保持待负责人复核，不写入冻结矩阵或实验结果，也不改变准入状态。

2026-10-19 P3MIX 生命周期同步：将完整 estimator 纳入16候选 `check_candidate_fit_contracts.py`
静态身份门禁；补未知 fit 参数绑定失败和 epoch 中断后的失效回归，更新台账工程引用及机器草稿。
另加强显式 CNN encoder 回归：确认拟合会更新 estimator 内 encoder 权重、但不改调用方模板。
59项候选门禁、P3MIX 与台账一致性测试通过，正式准入仍 false；历史实验与冻结协议未改。

2026-10-19 来源锁/方法台账交叉契约补齐：Dist-PU 与 PUSB kernel 的已锁定作者 commit
此前仅见于来源锁/行为审计或 Dist-PU 说明文字，现同步为台账结构化 `upstream_commit`；
契约测试逐项绑定 nnPU、Dist-PU、PUSB kernel、Self-PU 的锁定 commit 与行为审计，并断言线性
`pusb` 不继承 kernel 来源 pin。RECPE 与先验估计来源仍在来源锁中，但不是算法台账方法，不强行映射。
台账 JSON、来源/台账契约、Dist-PU/PUSB kernel estimator 测试通过；不表示逐行复现或正式准入。
后续对照最新 VPU D21/D22 复核，又将已有完整作者 commit `603bcc1e…` 同步至 VPU 台账，
并以调度来源 JSON、方法卡和只读复核记录约束其一致性；该 pin 只标识来源快照，不解除方法复核或准入。
随后只读核对 `t-sakai-kure/pywsl`，解析并登记 uPU bundle commit `79a5bf104a53cb6d3498918e435bd88ac5331543`
（2020-04-03；README 声明 ICML 2015 uPU；`PU_SL` 是固定二次闭式路径）。来源锁、台账与来源审计已同步，
明确 pin 不表示其他损失分支/实验等价；未执行或复制上游源码，uPU 方法学复核与配方前置不变。

原文复审补记：Algorithm 1 第 1 行明确将 student Θ 与 teacher eΘ 均随机初始化，但未说明
二者是否独立；当前实现复制 student 初始化 teacher。已把这一点从“论文未说明初始化”更正为
“初始化关系未定义、实现选择待方法负责人确认”；同时登记式(8)在 agreement 处以 `1−1e−7`
截顶来避免 `log(0)`，并补测饱和目标有限性、float state EMA 与整型 BN buffer 同步；端到端
回归确认 epoch 末 candidate pool 正好对应训练 P 中预测熵最高的 top-k。另补两 epoch 前向轨迹
回归，锁定候选池 eval 刷新后 student 每轮恢复 train 模式，避免 dropout/BatchNorm 在后续 epoch
沿用 eval 状态。来源/准入门禁保持通过且 `formal_admission=false`。

2026-10-19 全仓契约回归补齐 P3MIX/TEDn 集成缺口：两者纳入所有可训练方法的基准 factory，
同步 CNN UI 候选声明；P3MIX 加入权威 API 参考，GradPU 预算草稿源摘要同步。另修隔离 CNN
资源子进程未安装工作树时的导入路径，并澄清 `calibration_hooked=false` 的 TS 拒绝说明，
不把 TEDn 原生 TS 描述为 OS 原生。针对最初发现的失败用例、资源子进程回归均复验通过，
API/预算/目录/文档/格式门禁通过；不改正式预算、准入、冻结矩阵或历史结果。

2026-10-19 P3.2 oracle 推理状态修复：`PilotOracleCNN` 预测原先只恢复根模块模式，递归 `train()`
会覆盖 encoder 内 BN/Dropout 的混合状态。现按模块记录/恢复，正常前向、模型前向异常和 `eval()`
切换异常均加回归；不改训练目标、权重、选模、预算或冻结协议。真实 CIFAR/GPU 五 seed 仍待
新协议与存储预算获批。

2026-10-19 P3MIX 参数边界加固：estimator 与公开 batch/loss helper 统一拒绝 bool、字符串及
非有限数值，避免 Python 将 bool 隐式当作整数或非法输入泄漏为原始类型异常；方法卡和 API
同步声明范围，候选 fit 合约草稿中的源码摘要随实现更新。专项 estimator/math 回归、格式与
候选 fit 合约门禁通过；不改变论文锚点、实验配方、正式准入或既有结果。

2026-10-19 KLDCE 来源/行为独立核查：确认生产 `fit` 调用原生广义 pair-SMO，SLSQP 仅保留为
固定质心联合 QP oracle；发现并修正模块/类 docstring 的过期 solver 描述。源码在拼接 `[α;γ]`
上全局选对，而非附录 Algorithm 1 按两组分别选对，且 ACS 块更新顺序不同；方法卡、来源审计和
ledger 已披露，契约测试锁定该差异。KLDCE/台账测试、JSON、文档链接及 diff 检查通过；不改
求解策略、冻结结果或准入，仍待 RV8 方法负责人复核。

2026-10-19 PUSB 线性项来源/行为核查：确认 `PUSBClassifier` 是直接拟合观测 P/U 来源标签的
`StandardScaler + LogisticRegression` 附加基线，`predict_proba` 是来源标签概率，非论文后验排序
算法或校准后验；官方论文来源与 PUSBKernel RBF 适配不应被误读为该线性 estimator 的 exact 实现。
将其 source status 从 `official_exact` 调整为 `official_related`，同步代码、ledger、方法卡和契约测试；
PUSBKernel 与 Pilot 行不变，冻结结果/矩阵/准入均不变。

2026-10-19 PUSBKernel 来源/行为交叉核查：为官方锁定 RBF 路径登记中心池、风险/梯度、网格 CV、
分位数阈值及 TS 校准的实现映射和测试范围；明确官方正则公式/梯度 2 倍矛盾的 clean-room 裁决、
冻结预测阈值与批次分位数 API 的区别、TS 训练/OS 验证导致的跨目标选参，以及 IJCNN1/45-cell
结果的 `paper_claim=false` 边界。契约审计链接写入台账；不改算法、Pilot 数值、矩阵或准入。

2026-10-19 uPU 来源/行为核查：逐层登记 ICML 2015 风险结构、pywsl 官方 bundle 的能力边界、toolbox
Squared/C-DH/C-LL 路径与 TS-OS 角色扩展；确认 `official_sources.lock.json` 尚无 pywsl immutable
revision，审计中保留该 pin 缺口而不臆造版本。公式/梯度/视图测试覆盖实现不变量，不证明官方代码
逐行一致或论文数值复现；未改 loss 默认、冻结结果或正式准入。

2026-10-19 nnPU 来源/行为核查：对照论文 Algorithm 1、官方 Chainer `pu_loss.py`/`train.py` 与
toolbox PyTorch branch/backward，确认校正分支 `−γr` 与报告量分离；同时指出论文成对 P/U mini-batch、
官方合并 `SerialIterator` 和 toolbox 双 loader 的采样/更新轨迹并非同一实现；官方 revision
`a92c5365c07490c871e338c07f6e31d3eae2ca37` 现已登记来源锁，该锁不构成逐行适配或数值复现。
审计引用入台账并补契约测试；不改算法、默认超参、TS 审阅状态或正式结果。

2026-10-19 Self-PU 一致性范围补齐：按正式论文式 (13)/(16) 将 clean-support full-meta
探索路径显式设为 student `D-D_trust`（P + 未信任 U）及 teacher 全 `D`（P + 全部 U）；
估计器默认仍 `untrusted_u`，冻结 Pilot 无 meta 消融/历史轨迹不变。同步更新训练 history、
metadata、方法卡、API 与 ledger，并加行数范围回归。论文和剩余工程差异、额外 clean-support
预算/真实制品/正式准入阻断见[来源交叉审计](admission/selfpu_consistency_scope_review_20261019.md)。
官方仓库/MIT 与 `master` commit `a0e332ae4f8110e2490d597876e36bf837e1060f` 已登记来源锁，
详见[Self-PU 来源版本核对](admission/selfpu_source_pin_review_20261019.md)；该 pin 不等于工具箱实现逐行等价。

2026-10-19 Self-PU 固定作者源码行为审计：只读核查已锁定 commit `a0e332ae4f8110e2490d597876e36bf837e1060f`；
作者 `train_2s2t_mix.py` 将 `split=test` loader 同时用于 meta 与逐 epoch validate，meta 外层目标是测试批输入上的负预测熵，
随后又按测试真值准确率选择最佳 checkpoint。该发现不等同于论文定义/报告协议；工具箱 clean-support adapter 继续隔离外部 OA/test，
不为源码逐行贴合而引入测试集训练/选模。台账、方法卡和契约已同步；论文数值复现仍需原始数据划分/验证与选择记录，RV5 仍待 HENG958。
详见[Self-PU 固定源码行为复核](admission/selfpu_source_behavior_review_20261019.md)。

2026-10-19 来源证据索引补齐：扫描发现 PULDA、PUET、GradPU、Split-PU、VPU 已有的论文/作者源码行为材料
未统一绑定到 ledger `source_behavior_review_ref`。现分别链接对应 method card、GradPU 来源复核及 VPU 采样审计，
并用契约测试核对字段与证据文件身份；该同步只改善可追溯性，不把来源材料升级为方法签署、数值复现或正式准入。
同轮候选 fit 生命周期门禁也发现 P3Mix `implementation_sha256` 落后于已提交的 γ=0.5 边界修复；只刷新该草稿摘要，
实现本身及冻结协议/实验结果不变。

2026-10-19 P3.2 CNN feature adapter 模式恢复：[专项记录](admission/feature_adapter_inference_modes_20261019.md)。
修复特征抽取只保留根模块模式且 eval 转换位于异常保护外的问题；39 项相关测试中 34 通过、
5 项 CUDA 因环境无设备跳过。只验证模式与异常安全，不代表 GPU/真实数据资源验收或正式准入。

2026-10-19 补[16候选生命周期机器门禁](reviews/candidate_fit_contracts_20261019.md)：
只读AST/源码摘要绑定公开fit参数、共享guard、工程审阅引用和校准口径；
静态CI核对工程草稿，删guard/漂移/伪准入均拒收，不替代行为测试或正式配方冻结。
已将 16 项 `calibration_hooked` 显式同步进台账与草稿，拒绝缺失或与
`calibration_applied` 矛盾的声明，并核对声明为 true 的方法公开 `os_or_ts` fit 参数；
TEDn 的 native-TS 不校准状态保留为独立例外。
同轮再审计了其余冻结方法，将整个 24 项台账的 hook 状态补为显式布尔值；新旧方法均由
registry/fit 接口契约回归验证，默认 TS/OS 路由未改变。

2026-10-19 生命周期再补[VPU/神经LBE/PAN/CVIR/RP失败拟合保护](reviews/remaining_candidate_fit_failure_20261019.md)：
参数绑定前失效及取消后禁用公开预测；RP在禁止Torch导入的独立进程验证，
不改算法/校准/先验/预算、不注册新正式方法、不删除用户制品。

2026-10-19 P3.2 续批补[VPU/神经LBE评分状态保护](admission/vpu_lbe_scoring_modes_20261019.md)：
公开预测、神经LBE内部E-step和标记概率两分支评分均恢复逐层模式；不改原线性LBE、
VPU归一化/校准或神经LBE的OS联合似然/更新预算，正式准入和资源证据不变。

2026-10-19 LBE 用户源码包静态复核补记：[机器记录](data/lbe_source_review_20261012.json)
确认作者发表页当前直链包与既有官方 SHA-256 锁一致，用户上传包为不同变体；固定了此前未登记的
`LBE_LF` Adam/GD 更新器摘要，并核对 breast 与 toydata 主入口。
两入口实际调用 Adam、注释 GD；被调用的 Adam 每次覆盖传入参数，且 breast 与 toydata
初始化规则不同。该发现仅描述未执行的用户归档控制流，更新方法卡与 `lbe_neural` 台账的不确定项；
网页 research-only/商业许可声明不等于明确软件许可证；不改变当前 clean-room 算法、旧官方来源锁、
校准状态或历史实验，方法负责人复核仍待。现将该收据同时登记为原线性 `lbe` 的
`source_behavior_review_ref`，契约测试核对官方锁摘要与用户包身份分离；不表示代码等价。

2026-10-19 独立 TEDn 补[内部/公开评分模式保护](admission/tedn_scoring_modes_20261019.md)：
空输入及评分成功/异常均不覆盖调用者逐层模式；真实BBE/时序测试保留，
不改 α_U 目标、留出规则、采样、CVIR 或冻结结果，不解除正式准入前置。

2026-10-19 算法生命周期续批补[五扩展候选失败拟合保护](reviews/extended_candidate_fit_failure_20261019.md)：
GEN-PU/Holistic-PU/LaGAM/PULNS/TEDn 的未知/缺失参数、异常和取消均撤销公开拟合状态；
仅复用共享guard，不改训练方法体、目标、支持集、校准、预算或历史结果。

2026-10-19 P3.2 续批补[LaGAM resident/CPU-stream 推理模式保护](admission/lagam_inference_modes_20261019.md)：
恢复逐层模式并让空输入不触发 eval；不改元更新、虚拟步、支持集/PA 边界、采样或目标。
来源记录与台账补工程引用，正式配方/资源及合作者审阅仍待。

2026-10-19 P3.2 续批补[GEN-PU/Holistic-PU/PULNS eval 转换异常保护](admission/candidate_eval_transition_20261019.md)：
原有逐层恢复覆盖到 eval 本身抛异常的路径，目标/正常训练/校准/支持集和批准边界不变；
台账新增工程证据指针，不代填正式准入或重测历史资源。

2026-10-19 P3.2 续批补[Split-PU 评分状态恢复](admission/splitpu_inference_modes_20261019.md)：
分批评分成功/前向异常/eval 切换异常均恢复逐层模式；不裁决 #122 的 JSD、prior 或
混合比例，不改历史结果/资源 probe，不代签准入。

2026-10-19 #122 来源补核：[Split-PU 现状记录](admission/splitpu_122_characterization_20261018.md)
新增 arXiv 正文 Eq. (6) 的独立核对：`rho=0.7` 权重落在 student，teacher 系数为0.3，
仅反转实现参数仍会更换归一化分母。该结论确认论文侧公式；锁定作者源码的字节本轮无法重取，
既有源码摘要保留为历史证据，须负责人对照并裁决；没有改 JSD、prior、采样或校准行为。

2026-10-19（用户补入全文前）暂跳过待补原文的 P3MIX，推进 P3.2 的
[Robust-PU 推理模式恢复](admission/robustpu_inference_modes_20261019.md)：
成功及异常路径均恢复逐层 BN/Dropout 模式；不改 fit、校准、目标、预算或冻结结果，
来源/台账与当前草稿摘要同步，历史资源证据不冒充重新测量。其后收到 P3MIX PDF 并完成
全文核对和实验级实现，当前状态见本节前文及原文交叉表。

2026-10-18 图像技术入口补[实际fit收据](admission/candidate_image_training_receipt_20261018.md)：
训练前绑定像素/模板/seed/增强/prior，记录fit参数、实际epoch/更新数与代码摘要。
不接冻结runner，不证明四角色/标签预算/锁环境或快照持久化，正式准入仍false。

2026-10-18 独立修复[PULDA/GradPU推理逐模块模式恢复](admission/pulda_gradpu_inference_modes_20261018.md)：
修复预测覆盖混合Dropout/BN模式的实际缺陷，含前向/eval切换异常恢复，不改变训练
策略、默认数值、校准/预算或正式准入；旧资源probe不回填为新源码实测。

2026-10-18 独立核查 #122：[Split-PU现状反例与决策清单](admission/splitpu_122_characterization_20261018.md)
补JSD权重反转会连带改变分母的标量核对，以及真实teacher批次插桩：全量union的
population比例不保证均衡batch并集恢复该比例。只补测试与文档，不擅自改目标、
prior、采样或台账；正式方向/归一化、比例采样或近似接受仍待负责人决定。

2026-10-18 优先整改 #121：[GradPU 调度与真实batch运行声明](admission/gradpu_training_recipe_review_20261018.md)
保留常量LR默认，补显式线性/余弦、实际batch/更新成本与SAR/TS边界的每候选manifest，
完善CNN/回放/fold证据。正式曲线/总batch语义/候选与阶段仍待确认，不改冻结协议、
矩阵、台账准入位，不替代200快照资源实测。

2026-10-19 GradPU 参数 UI 类型同步：将 `lr_schedule` 声明为 `Literal` 三选项，
参数面板现显示合法选项而非自由文本；默认 `constant`、训练轨迹及正式搜索空间不变。
UI catalog 与候选契约摘要已同步测试/更新，详见 #121 记录；不代表调度或预算获得方法负责人批准。

2026-10-18 独立续批：[合作者图像配方待确认清单](admission/candidate_image_owner_decisions_20261018.md)
已按初始化/BN/增强/MLP/warmup 分项，可直接回复；新增只读资源预检按源码默认
120/200快照和候选/重试/并发估算文件空间，缺少实测样本不放行。
不训练、不加载制品，完整 CPU/GPU 峰值 profile、正式配方/接线及准入仍待办。

2026-10-18 优先核对远程 #119/#120 并补[PULDA/GradPU 图像技术接线](admission/pulda_gradpu_image_review_20261018.md)：
PULDA 增加逐训练 batch 增强与共享 seeded ResNet-18 回归；GradPU 固定技术入口为
cnn13_no_bn/private，均记录初始化/归一化/训练路径。未改冻结矩阵或台账准入位；
#119 尚待 BN/增强/MLP 决定，正式 recipe/runner 分层和完整120/200快照资源实测仍待办。

2026-10-18 RP 新增[PU 特例来源与实际 OOF 拟合证据](admission/rp_source_review_20261018.md)：
补四文件双摘要、独立公式/排序和逐次拟合隔离/重权/成本测试；仅修正正例均值
上溢舍入导致的空 confident/NaN，数学算法、OS/先验/校准、默认值不改；
台账与扩展草稿同步，正式 noise 来源、预算及准入仍待审。

2026-10-18 PAN 新增[来源与逐层推理状态证据](admission/pan_source_inference_review_20261018.md)：
九文件来源绑定、实际交替梯度/采样及 CNN 混合模式恢复；不改 `fit`、论文目标或
负责人 OS 决定，同步台账/校准与扩展草稿；历史探测和正式准入不变。

2026-10-18 五候选方法新增[失败重训状态整改](reviews/candidate_fit_failure_20261018.md)：
错误参数、训练异常或回调取消后禁用旧/部分模型的公开预测，保留成功路径不变；
同步台账、工程证据与当前实现/共享 guard 的 SHA，不改历史探测、冻结结果或准入状态。

2026-10-18 按最新合作者复核补交 Robust-PU
[来源/行为证据](data/robustpu_source_review_20261018.json)：完整记录采样/网络文件、
许可未发现、真实梯度与原始行移动权重测试，同步先验调用契约和草稿绑定。
不改训练、已冻结结果、负责人签署或正式准入；Split-PU 待裁决项保持。

2026-10-17 按合作者 `adb00c4` 的非阻断审核补齐 PUET
[风险质量登记与适配回归](reviews/puet_adaptation_completion_20261017.md)：不改训练代码，
同步台账/校准证据/预算草稿并锁住边界行为；不合并审核分支或代填正式准入。

2026-10-17 独立支持集制品与 CPU 探索入口已补，见
[交付与命令](delivery/clean_support_workflow_20261017.md)：五数据集 raw 重建、显式
support 大小/seed、剩余 train-only 预处理与三方法单候选 OA 实验回放。仅合成测试，
真实支持集/标签预算、文本/分组和正式 CNN/多 seed 前置未获自动批准；旧制品不改。

2026-10-17 算法收尾新任务：线性 LBE 可解释性诊断 → P3MIX 完整来源/实现 →
完整 Self-PU/独立支持集接线与验收，见[推进与前置清单](reviews/algorithm_completion_20261017.md)。
未获批准的标签预算/正式 GPU 配方不因用户要求工程收尾而自动放行。

2026-10-19 PULNS 独立来源复核补记：重新核对 AAAI 正式全文未发现作者代码仓库链接，保持
`not_found` 而不外推源码不存在；补录 CIFAR-10/MNIST 类别映射、P/U 与 γ 协议以及 Table 3
公开 error-rate 锚点。括号 SD 尺度仍有原文单位歧义，锚点仅用于来源审计，不并入冻结矩阵；
台账保持 `ts` 原生采样语义、无 TS-OS 校准及 clean-support/PA blocker。详见
[PULNS 来源与 CNN 复核](pulns_source_cnn_review_20261007.md) §2026-10-19。

2026-10-17 新增算法整改：#114 PUET bootstrap 风险归一化、#115 LaGAM OS 元数据、
#117 CVIR 先验/路径及 #116 PULNS 支持集/额外监督独立适配，见
[整改与执行边界](reviews/algorithm_issues_114_117_20261017.md)。
不改旧冻结协议/结果，不自动准入；PULNS 正式支持集/标签预算与 CVIR 正式配方仍待批准。

2026-10-16 用户调整任务：本阶段不再主动推进 P2.2/RV3–RV7 review，转向算法与
[线性/神经 LBE 临时对照](admission/lbe_comparison_exploration_20261016.md)。历史待审和原始制品事实
原样留痕，不将取消本阶段复核工作等同于已验收或更改冻结协议；临时实验不并入旧 645-run。
（此处的“用户”指 HENG958，调整的是其复核任务；P2.2 收尾及 C2/C3 裁决改由 shuidisjtu 推进，以 §1.2 为准，RV7 已关闭。）

2026-10-16 P4.1-A/P2.2/RV3–RV7 补记：[推进记录](reviews/p41_p22_progress_20261016.md)。
正式 registry CI 条件接线与声明锁门禁已补，文件尚未物化时明确跳过；冻结时须改为必需存在，
HENG958 接口复核/P4.1-B 仍待。原结果树分批接收工具已交付，但本机未收到完整正式树，
独立重放未完成。nnPU 官方 supplement 已完整检查；PUBench 增强消费/搜索预算候选补记待审，
v3 的 36/7/4 待审与 54 PA 阻断保持，RV3–RV7 不据工程检查签署。
（同日稍后的 P2.2 范围调整见 §1.2：RV7 关闭，RV3/RV4 已由 shuidisjtu 于 10-16 裁决完成，C1/C4/C5/C6 按通过处理；以 §1.2 为准。）

2026-10-15按远程Issue #41优先补记：[阶段B公共入口](delivery/label_semantics_stage_b_20261015.md)
补齐pipeline、比较/调优、CLI/UI及敏感性分析语义门禁、展示和ADR；2026-10-16 用户接受复核证据，RV2 完成，Issue #41 可关闭。
Issue #40的P4.1-A本地工程接线不等于P4.1-B冻结；Issue #88成本分类明确延期，不抢占Survey工作。

2026-10-15 P3.1/P3.2 工程补记：[LaGAM 来源与 CNN/CPU-stream](admission/lagam_source_cnn_review_20261015.md)
补 weight-only 范围、图像分批、BN/dropout 隔离和阶段成本；发布 OS 与 registry TS 的
采样冲突明确待审，不改签该定位，不解除独立 clean support 标签预算/PA 禁入或正式准入。
（2026-10-17 #115 已按锁定源码将 registry/台账纠正为 OS；support/PA 限制不变。）

2026-10-15 P3.1/P3.2 工程补记：[CVIR 来源差异与 CNN/CPU-stream](admission/cvir_source_cnn_review_20261015.md)
保留论文加权目标与旧二维默认，披露发布代码差异，补图像/分批/两阶段成本快照；
α_U 来源、正式共享规格与准入仍未批准，HENG958 复核待办。
（2026-10-17 #117 明确本 Survey 走 OS adaptation + 显式 α_U；OS→TS 校准不属于本实验，不是运行前置。）

2026-10-14 P3.2 独立工程补记：[神经 LBE 变体](delivery/lbe_neural_integration_20261014.md)
以 `lbe_neural` 另立双网络 Adam/CNN/分批及 OS-only 接线，原线性 `lbe` 和冻结矩阵不变。
此为 shuidisjtu 侧独立技术补齐，HENG958 接口/方法学复核待办；不替代 P3.2 正式共享规格、
P3.3 资源、P4.1 配方或当前 14 项 RV8 待审方法的责任与准入条件。

2026-10-15 P3.1/P3.2 来源缺口补记：[VPU 源调度与固定步数](admission/vpu_schedule_review_20261015.md)
已提供显式工程选项并同步台账/准入草稿；默认训练行为、D21/D22 与正式准入阻断不变。

2026-10-15 P3.2 工程补记：[VPU 注入 encoder 与 CPU-stream](delivery/vpu_encoder_stream_20261015.md)
接通四维图像、分批验证/归一化/预测与快照；仅 CPU 合成验证，不替代作者图像实验、
正式共享规格/资源验收或 D22 PA 裁决，合作者复核仍待办。

- 当前推进顺序（10-16）：HENG958 侧先做算法工程及独立 LBE 对照，按候选执行 P3.1/P3.2 → P3.3-A → P4.1-B → P3.3-B；shuidisjtu 侧 P2.2 已收尾（含 C2/C3 裁决），转 P4.1-A。P4.2 的既有正式前置不因临时实验自动满足。
- 当前优先级更正（2026-10-19；上项保留为历史截点）：①继续 P3.1 的独立逐方法来源/行为证据与台账/校准一致性，P3MIX 全文核对和代码级实现已完成，但 RV8 复核、正式配方/预算及资源验收未完成；②继续 P3.2 已有 CNN/共享特征路径的技术核验，正式共享规格、图像配方与预算须先获批准；③PN-CNN 真实数据/GPU smoke、P3.3-A 真实结构资源 profile 均受前述协议/资源前置约束；④待 P4.1-A 的正式 registry/接口审阅及 P3.3-A 证据齐备后再做 P4.1-B 冻结，随后才进入 P3.3-B 正式资源/持久化验收与 P4.2 主榜汇总。任何代码 smoke、独立来源审计或技术 profile 都不自动解除审批、签署、旧结果接收或正式准入阻断。
- 新数据与 PN-CNN 在新协议中登记，不修改旧 645-run 矩阵或历史 manifest，不混用协议摘要。
- 正式训练前锁定数据、配置与比较组；test 不参与先验、候选、阈值或 checkpoint 选择。
- 工程 smoke、资源测量和配置批准均不等于算法准入；未通过表中前置不得提前主榜。
- 复核由表中指定的 shuidisjtu 或 HENG958 记录证据、结论、身份与日期；退回项另记整改，不代签。

## 2. 风险与留痕约定

### 2.1 风险

1. **GPU 与正式环境**：正式资源验收须使用批准配置、frozen-lock 环境、兼容的驱动/依赖和逐 run 资源记录；合成 CPU/CUDA 探针不能代替 P3.3-B。
2. **数据与标签**：公开原始数据不等于许可、正式 split 或四角色验收；新数据和版本变化必须留来源、映射、授权范围与摘要，ADNI 官方数据未获批不得纳入；`adni` 槽位使用的是替代数据，须按 D1 的修订标注。
3. **方法准入**：代码级接入、方法卡、smoke、资源 profile 或阶段批准均不等于正式准入；P3MIX 必须先完成完整 estimator，PULNS/LaGAM 等额外标签预算单独裁决。
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
