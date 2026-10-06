# Pilot 验收以外的后续优先级（2026-10-04）

本页回答「除去本轮 Excel/制品复核，接下来先做什么」。任务编号仍以
[执行计划](survey_execution_plan.md) 为准。该计划不撤销既有复核阻断，不授权改写历史结果或提前主榜跑批。

## 1. 实施顺序

| 顺序 | 对应编号 | 子任务 / 完成标准 | 当前可独立推进的范围 |
|---|---|---|---|
| 1 | P3.1/P3.2 准备 + P4.1 准备 | 把已有实现的参数、预算、来源、视图、选择准则及阻断统一成准入交接包；实际物化可验证的注册表草稿 | 本批已完成草稿生成器、机器草稿及准入清单；不宣称正式准入 |
| 2 | P3.1 优先，P3.2 随后 | 先闭合 PULDA/PUET 的预算/存储/选模与共享特征规格，再 Grad-PU、Robust-PU、Split-PU；公开行为对照与各自适配差异必须留证 | 来源检查、组件与规格草案可独立做；进入新冻结矩阵及方法学决定需负责人复核 |
| 3 | P3.1 / P3.2 缺失算法 | PAN、RP、PULNS 优先；再 GEN-PU、Holistic-PU、P3MIX。每项先核论文/官方来源与标签预算，再实现、方法卡、台账、校准机制及 golden/smoke | 不仅新增类名；不得绕过已识别的先验/视图问题，也不把技术 smoke 当数值复现 |
| 4 | P3.3 | frozen-lock 隔离环境、多 seed CPU/GPU、显存/磁盘、OOM/重试、保存加载与 selected checkpoint 对账 | 有资源且候选规格已批准后执行；PUET 为 CPU 树，GPU 条款适用性单列 |
| 5 | P4.1 正式完成 | 候选确定后正式审阅/冻结，生成 resolved snapshot；训练前绑定身份、manifest 校验及审计消费闭环 | 不把草稿校验通过写成 locked，不回填历史 manifest |
| 6 | P4.2 | 22 项门禁通过后主榜聚合与分析，按机制/PA-OA/预算族/训练路径分层 | 不是现在跑全主榜；不生成跨数据集总排名 |

第 1 项是 P4.1 的**准备切片**，不是在 P3 完成之前决定全部新方法的候选池。
已有 pilot 参数确定且已有注册表校验代码，因此现在物化其草稿有明确依据；新增方法的正式参数仍留待第 2–5 项。
VPU 的 D22 PA 准则、CVIR 的 α_U 口径、LaGAM 的独立 clean-support 预算属于单独决策，
不为了凑齐算法数而静默假定。

## 2. 本批已完成：P4.1 注册表准备

- 生成器：`scripts/prepare_survey_recipe_registry.py`，只读协议、输出 JSON，不训练、不改参数、不写冻结 registry。
- 草稿：[survey_recipe_registry_v0_1_draft.json](data/survey_recipe_registry_v0_1_draft.json)。
- 范围：协议全部 8 个 profile；6 个 runnable PU + 1 个 oracle 为 draft，KLDCE 为 excluded 且无 candidate。
- candidate：每个 runnable profile 1 个 empty-override 外部候选，预算与选模只引用冻结协议。
  PUSB-kernel 内部 72 组搜索以独立 search identity / grid reference 登记，不当成 72 个外部候选。
- 草稿规范摘要：`54ecef2a150ac874ac2578874700d89e5574c615479128a92390fb7ade90e275`。
- draft 保持 `resolved_snapshot=null`、reviewers 为空；不写入正式冻结路径、不升级 lifecycle。
- 历史 B1–B4 仍为 `legacy_unbound`，不重新贴标签或绑定身份。

复核命令（从仓库根；只输出，不写文件）：

```bash
PYTHONPATH=. python scripts/prepare_survey_recipe_registry.py
PYTHONPATH=. python scripts/check_survey_recipe_registry.py \
  --registry docs/research/pu_survey/data/survey_recipe_registry_v0_1_draft.json
```

生成草稿必须与入库草稿结构逐字一致；候选参数非空 override、预算数量矛盾或 method 无执行 scope
均 fail-closed。原默认门禁仍报告正式 registry 未物化；检查草稿不意味着冻结门禁已经完成。

## 3. 本批已完成：8 个现有方法准入缺口清单

[机器清单](data/p3_candidate_admission_v1_draft.json) 固定准备优先级、台账指针、当前视图/校准事实与逐方法 blocker。
公共条件为：新版本执行/对照预注册、公开行为或数值对照、预算/存储/选模契约、方法负责人复核。
每方法的 candidate protocol、budget、candidate pool 目前均为 null，`admitted=false`；不编造已经确定的参数。

| 方法 | 下一步要解决的关键缺口 |
|---|---|
| PULDA | 两阶段预算、共享表征或原生图像规格、选模预注册 |
| PUET | CPU 树预算与存储 profile；GPU 条款豁免的负责人决定 |
| Grad-PU | 官方源码状态 not_found 的来源复核；梯度插值/训练预算 |
| Robust-PU | warm-up-only 校准及未复现官方 scheduler/图像增强的披露 |
| Split-PU | teacher-only 校准；特征扰动不等于作者图像增强/SimSiam |
| VPU | D22 的 PA 决定及既有学习率/调度等上游差异 |
| CVIR | α_U 来源、有限样本转换与显式 ts 不支持 |
| LaGAM | 独立 clean support、support/selection 隔离与 PA 标签预算 |

契约测试检查清单与台账同步、八方法均不进入旧协议、null 待决字段及特殊限制不被移除。
本批不是八方法正式验收；不更新合作者本人签署状态。

## 4. 紧接着的交接

以下按最新进展在前保留历史交接记录；较早记录中的“下一项/待实现”是当时状态，
当前工程完成情况以最上方更新及所链接的最新复核文档为准，正式审批阻断不因此消失。

2026-10-07 继续[新 CNN 资源证据准备](extended_cnn_resource_review_20261007.md)：
复用探针覆盖 PULNS/GEN-PU/Holistic-PU，七变体独立子进程、阶段成本/内存高水位/快照与模型
回放、来源指针和只读检查均已接入；默认宽度21组实测待执行，不以小宽度单测代替。
PULNS快照明确null/独立奖励预算，LZO后段恢复不回退付出，正式候选/资源/签署仍待负责人。

2026-10-07 继续核对 [P3MIX 来源身份](../method_cards/P3MIX.md)：
按 ICLR 官方链接更正论文 ID 为 `NH29920YEmj`（数字0），保留旧误记/403的历史上下文。
正确入口当前要求浏览器验证，作者论文/代码页未确认对应源码，未把摘要冒充全文或注册完整方法。
全文训练/pool 更新与 E/C 变体仍待核；不依赖其全文的下一独立项是新 CNN 路径合成资源证据。

2026-10-07 继续 [GEN-PU 像素生成/CNN 来源复核](genpu_source_cnn_review_20261007.md)：
确认原论文/作者归档为稠密像素网络；补双 generator 同形 NCHW、三判别器与新建 PN 的四份
独立可训练 CNN、G 步状态冻结、显式 tanh 域校验、六网络成本与 PN 快照回放。
台账/能力/校准与单独准入草稿同步，默认 MLP 保留；不借真值、不改历史、不批准源配方或签署。
下一独立项为 P3MIX 完整来源/训练组件，以及新 CNN 路径的合成资源与选模交接证据；
真实数据数值、正式候选与资源上界仍须新协议及方法负责人复核。

2026-10-07 继续 [PULNS 原生 CNN 路径及来源复核](pulns_source_cnn_review_20261007.md)：
补独立编码器端到端预训练/probe/实际分类器更新、分批CPU状态、模板/BN/最佳模型隔离及全成本。
论文确用CNN，但未确认作者代码，不把通用backbone当原CNN数值复现；台账/校准/能力/草稿同步。
需要独立真值奖励集、PA-ineligible、TS拒绝及正式资源/全角色协议阻断保留；公共工作流不偷偷
借用selection/test标签。下一独立项为GEN-PU图像架构来源与新路径资源准备。

2026-10-07 继续 [Holistic-PU LZO 工程终点选择](holistic_lzo_selection_20261007.md)：
只用训练 P 固定类内 mixup、全预算 CE argmin/最早平局、趋势前缀和模型/Adam/随机流恢复；
统计完整已执行成本和额外验证，不将正例 CE 当总体 accuracy 或外部 PA/OA 选模。
默认 fixed 路径保留，台账/准入草稿新增 recipe/外部角色隔离复核，签署与正式候选仍空。
下一独立项继续 GEN-PU/PULNS 原生路径与来源差异、Holistic-PU 新变体资源/选模交接；
完整图像/增强/精确源配方及数值验收未据工程接口清空。

2026-10-07 继续[Holistic-PU 后段初始化](holistic_stage_initialization_20261007.md)：
显式 `reinitialize` 新建模型/Adam、随机重置 CNN 参数和 BN，默认 `continue` 保留兼容；
分阶段更新和全局成本不混淆，快照跨重建回放，台账/准入草稿同步但不决定正式变体。
下一项继续 LZO 选预热终点接口及资源证据，完整源 fine-tune/图像配方、数值与负责人审批仍保留。

2026-10-07 继续缺失算法图像路径：[Holistic-PU 来源/端到端 CNN 核查](holistic_source_cnn_review_20261007.md)。
已补编码器两阶段训练、分批 CPU→设备、模板/BN/fold 隔离及阶段快照回放，台账和扩展草稿同步。
新增来源核查明确论文 LZO 与锁定代码固定 warming_steps 的差异，且补充 Algorithm 2/作者代码
均要求后段新建模型；当前继续同模型/Adam 保留为显式适配，LZO/重初始化变体仍待实现。
下一独立项是两项来源支持的训练语义补齐；P3MIX 全文 API 仍 HTTP403，不注册空壳。
不改冻结结果，不以 CNN 技术接线清空预算/公开数值/负责人准入阻断。

2026-10-07 第2项继续：[阶段/选模/预算准备](p3_stage_budget_review_20261007.md)
已实现新方法快照阶段、round/local epoch、累计更新数留痕，选中恢复与回收保留元数据；
五方法规格由 AST/来源生成并与台账同步，全部正式候选/阶段资格/资源决定仍为 null。
这使负责人能逐阶段审阅，但不批准 recipe 或启动新正式实验；旧 pilot 引用与冻结矩阵不改。
GenPU/Holistic-PU 同时补阶段技术留痕，P3MIX 完整来源与图像训练源配方缺口仍待继续。

2026-10-05 P1-2 更新：[公开结果对照准备](p3_public_comparison_20261005.md)
已登记12条原表读数、五方法×三数据集15格范围、来源单位/重复数/采样与选模差异，
新增只读检查器核 frozen bytes、读数单位和 pending 边界。PULDA 全文访问仍未获得；
其余读数也不与尚不存在的新方法正式结果配对，不清空公开数值验收/负责人阻断。
下一步先完善并审阅方法特定选模、预算、表征与采样规格，以及未取得的来源材料；
批准新版协议和资源后才执行源配方或 benchmark-adapted 数值实验。

2026-10-05 P1-1 更新：第 2 项中五方法的自主工程准备已补齐，详见
[来源行为、预算与选模证据包](p3_admission_evidence_20261005.md)。PULDA/PUET/Grad-PU/
Robust-PU/Split-PU 的默认参数、预算计数、校准范围、来源差异和机器引用已同步；
四个锁定作者仓库的 17 个文件有摘要核验，Grad-PU 仍仅论文来源。
这不关闭正式预算/存储 profile、阶段选模资格、公开数值复现与负责人复核的 blocker。
下一独立优先项是完善公开对照可比性/差异证据；正式候选池与新版协议待负责人决定，
不跳到主榜训练，也不继续以新增类名替代验收闭环。

2026-10-05 状态更新：[独立推进记录](independent_progress_20261005.md)。PULDA 两阶段 checkpoint
计数已接入磁盘预检，PULDA/PUET 已有可复跑的合成存储/恢复探针；三项新增算法 PAN、RP、PULNS
已完成独立组件、台账和方法卡。下文磁盘警告为 10-04 历史状态，不再表示技术估算接口缺失；
正式预算/存储 profile 与负责人验收仍未完成。继续顺序为 GEN-PU、Holistic-PU、P3MIX 的来源核查与组件。

同日后续：GEN-PU、Holistic-PU 的二维组件、方法卡、台账和回归也已完成；台账现 21 条。
P3MIX 仅完成作者 slides 可核的 batch 组件，原论文/作者源码核查与完整训练仍未完成，不注册空壳。
新方法的四角色 runner / 共享特征 smoke 和独立短时 GPU 验证已补，7 方法 × 3 seed
合成存储记录已有；正式预算/签署不代填。五新增方法的单独[准入交接草稿](data/p3_preintegration_extension_v1_draft.json)
已生成，原八方法清单范围保持不变。下一步是审批所需证据完善，不是自动正式准入。

同日工程补齐：PAN 已支持独立训练分类器/判别器编码器的原生 CNN 注入路径，
不同 CV 折和两网络的 BatchNorm/权重隔离通过；不是仅冻结特征的替代。
Holistic-PU 已接两阶段逐轮推断快照，GEN-PU 已接合成 PN 阶段逐轮推断快照，
两者磁盘预检分别计两阶段总轮数/PN 轮数，而训练成本仍包含全部阶段。
新合成存储记录为 [v2 探针](data/candidate_storage_probe_20261005_v2.json)，旧记录保留。
后续独立顺序：其它方法原生图像路径与来源行为补验 → 方法特定选模/预算草案；
P3MIX 完整来源可用前不强行注册，正式阶段资格与签署仍交负责人决定。

继续按第 2 项优先补 PULDA：已接可训练注入 CNN、分批 GPU 图像、eval 分批伪标签刷新；
两阶段损失/TS 风险角色、预算与快照保持原接口，台账/原八方法准入草稿同步但 admitted=false。
工具箱 CNN 适配仍非作者 CIFAR 完整训练复现；正式图像规格与资源上界继续待办。

Robust-PU / Split-PU 继续补资源接口：前者快照为 pretrain+episodes（inner epochs 只增更新），
后者上界为 teacher+split+rounds×student（早停可减少实际数）。两者累计全部阶段 optimizer
更新次数、refit 重置，已用真实 step 插桩与 snapshot 回放核验；台账/准入草稿同步，
正式预算和图像/来源行为缺口未据此清空。

GradPU 已补 [原始来源与实验协议核查](gradpu_source_review_20261005.md)：CNN13 去 BatchNorm、
理论边缘 U 与实验剩余 U 分层、灰度 MLP/学习率/选模差异；作者源码仍 not_found。
下一步显式无 BN 的原图像梯度路径，不静默改默认骨干，也不将 TS 数值冒充论文原流程。

GradPU 已继续实现注入无 BN CNN 的原图像插值/二阶梯度路径，显式 `cnn13_no_bn` 骨干
供 pipeline/CLI/UI 选择，默认 cnn13 保留；编码器 deepcopy/训练、原图梯度、fold 隔离、
种子/pickle/epoch 回放及短时 CUDA 证据已补。正式图像规格、来源源码、调度、预算与
数字对照仍需后续准备/审核，未清空准入 blocker。

Robust-PU 已继续到可训练 CNN 图像路径：probe/权重扫描/推断 eval，原行权重和数据留 CPU，
优化批次上设备；保留 warmup-only TS，零 warmup 的 TS 拒绝。两阶段、pipeline/fold 隔离
与回放技术证据补齐，不称作者完整 CIFAR 训练复现。下一项继续 Split-PU 的表征/预算规格。

Split-PU 已先修正全数据驻 GPU / 整批教师扫描：数据及目标保留 CPU，教师、splitter、
学生优化及预测分批；原始 U 至少两条的边界明确拒绝，预测恢复模式。
能力仍仅二维 MLP；CNN 的多层表征接口、官方增强/SimSiam 和正式预算未据此完成。

Split-PU 已继续接真实 CNN 中间层 MSE/最终编码向量余弦，教师/splitter/各轮学生独立
deepcopy；默认首池化层（否则 Conv/Linear）或显式模块路径，单行 BN 暂用运行统计。
公共 pipeline/fold 隔离与全阶段回放纳入测试；Gaussian 增强、无 SimSiam 投影/predictor、
同模板各阶段初始化仍是已登记适配，正式资源 profile 与选模/预算规格继续待办。

已补[默认宽度 CNN 存储技术实测](candidate_cnn_storage_review_20261005.md)：PULDA、GradPU、
Robust-PU、Split-PU、PAN 各三种子 CPU 快照/完整对象恢复，另有 PULDA 单组 CUDA allocator
峰值；记录实际源摘要。不使用二维探针外推图像配额，不以这些缩短合成任务替代 P3.3。

本批验证：注册表 schema/source/manifest/draft、P3 清单契约及两方法 runner smoke 合计 **55 passed**。文档一致性与 diff 空白检查通过。原注册表模块已有的 7 条 UP038 风格告警未改动，本轮增量 lint 排除该既有规则后通过；不是全仓库 lint 验收。

第 2 项已继续到 [PULDA/PUET 规格草案](p3_pulda_puet_admission_spec_draft.md)：预算单位、标签/视图、选模及存储缺口已分别登记，并新增两方法四角色 PA/OA runner smoke（2 passed）。PULDA 的磁盘估算缺口明确保留，不称正式存储验收。

接下来审阅 PULDA、PUET 预算/选模/资源草案并补来源行为及存储实测证据；审批前不选定新正式候选池。
源码审计与缺失算法集成可以按第 3 项单独继续，但不能占用完整主榜资源提前跑批。
本轮 Excel/manifest 复核仍是并行工作流，不因本页排序而被宣布完成。
