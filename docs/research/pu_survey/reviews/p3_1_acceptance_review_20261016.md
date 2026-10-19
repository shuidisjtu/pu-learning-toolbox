# P3.1 技术验收复核（2026-10-18 汇总）

> 本文是 P3.1 逐方法复核的压缩记录，替代此前按时间追加的流水账。工程细节、
> 整改证据与历史测试记录保留在链接文档和 git 历史中。
> 本文是代理技术审核建议；除明确标注“授权代录”的事项外，不代表负责人签署，
> 不修改正式准入位。

## 结论与范围

- **P3.1 暂不整体验收通过。** 八方法草稿与后增五方法草稿共覆盖 13 个方法，
  正式准入均未签署；P3MIX 已于 2026-10-19 补齐全文核对与完整 estimator，但尚未完成方法负责人复核、正式配方/预算、资源验收或准入（见本文补记）。既有 VPU 另列，不新增准入。
- 复核区分四层：**论文目标、作者源码行为、工具箱实现、P3.1 验收结论**。
  四者不一致时明确采用哪一层；适配实现统一标 `benchmark-adapted`，
  不称论文复现或源码逐行复现。
- 有官方源码的方法区分 `official_exact` / `official_related`；
  Grad-PU 未找到作者代码，保持 paper-only。
- 共同阻断：PA/OA 阶段资格、批准训练预算、表征与资源 profile、公开数值对照、
  负责人审阅。CNN 适配、资源 profile、配方冻结分别留给 P3.2、P3.3、P4.1。

## 方法状态

下表“有条件接受”仅指当前披露的工程证据可继续进入后续审阅，不是正式方法准入。

| 方法 | 复核结论 | 关键边界与未决项 |
|---|---|---|
| P3MIX | 2026-10-19 追加工程交付，负责人复核待办 | 原始汇总时只有 slides 组件；2026-10-19 用户补入全文后，已完成原文核对、P3Mix/P3Mix-E/P3Mix-C estimator、训练时序、registry/ledger 与专项测试。工程复核和正式准入前置仍未完成，详见本文末尾追加记录 |
| PUET | 有条件接受工程证据 | #114 修复已验证；CPU 树预算、候选/选模、公开数值与负责人审阅待定 |
| PULDA | known-prior PULDA，非 PULDA\* | 损失/EMA 与作者源码一致；MixUp 与 TS 池差异按适配登记；stage/budget 已定；共享表征（#119）与正式准入待定 |
| Grad-PU | 有条件接受工程证据 | paper-only；Eq. (5)–(7) 与实现一致；TS 池与配方为适配；图像走 method-private `cnn13_no_bn`（#120）；证据边界见 #121 |
| Robust-PU | 有条件接受适配子集 | 默认 SPL 与官方实现一致；正式 `ts` 路径先验正确；预训练 batching 与预算为适配；来源/行为证据已于 2026-10-18 补交，仍待负责人审阅 |
| Split-PU | 有条件接受适配子集 | TS 只作用于 teacher；JSD 权重方向与 `class_prior` 的 OS/TS 语义待裁决（#122）；其他适配保留 |
| PAN | 有条件接受工程证据 | `official_related`；保留 paper-faithful 实现；论文/源码差异见 #123；对抗训练预算与快照规格待定 |
| RP | 有条件接受 PU 特例证据 | 固定 `rho0=0`，不称一般双向噪声算法；OS-only；`frac_pos2neg` 须登记来源或固定默认估计，禁止由 test 真值提供 |
| PULNS | 接口/审计已交付，正式 support 待批 | 仅 SCAR OS + 终点 OA；标准 PU-only PA 不接受；八集 support 来源/规模/标签预算待定 |
| GEN-PU | 论文目标已定，有条件接受 | 正式目标采用论文 prior-weighted minimax；作者 demo 仅差异对照；先验来源、TS 池、PN 阶段资格与正式 recipe 待定；归档未发现 LICENSE，登记 `unknown` |
| Holistic-PU | 当前实现保留，有条件接受 | 论文 pairwise/variance 路径；默认 fixed + continue；作者变体仅对照；`license=unknown`；LZO/初始化/存储 recipe 与公开数值待定 |
| CVIR | OS adaptation 已接，准入待定 | 显式 α_U；population π 只给 PA；OS→TS 校准不属于本实验；α_U 有限样本口径、候选/预算与公开数值待定 |
| LaGAM | OS 身份已裁决，准入暂缓 | clean support 是额外训练监督，标准 PU-only PA 不适用；support/标签预算、许可与公开数值待定 |
| VPU（既有） | 沿用 D21/D22 | 不新增准入；原生 TS、OS 消融与验证变分风险仅作 diagnostic |

BBE/TEDⁿ 是 main 合并后新增的独立 native-TS 扩展，不在上述 13 方法表，
不纳入 P3.1 方法准入计数。

## 2026-10-19 P3MIX 全文与工程交付补记

用户提供 `30_who_is_your_right_mixup_partne.pdf`（ICLR 2022，14页）后，逐页核对 §2、
式 (1)–(8)、Algorithm 1、§3.1 与附录。当前完整训练路径覆盖三种互斥 variant：基础
P3Mix、EMA mean-teacher 正则 P3Mix-E、以及高置信伪负例标签修正 P3Mix-C；另有每 epoch
top-k 熵候选池刷新、初始随机 P pool、OS-only 训练与显式不支持 TS 校准。实现和短拟合
测试见 `pu_toolbox/estimators/deep/p3mix.py` 与 `tests/unit/estimators/test_p3mix.py`。

核查发现论文正文列出 Adam、alpha=1、k=100、beta/gamma/eta 搜索范围及五次运行，但未
指定完整学习率、batch/backbone训练细节或 mean-teacher EMA decay；论文未给作者代码链接。
因此本地实现标为 NATIVE 独立适配，作者代码状态为 not_found，工程默认不冒充论文 recipe。
尚未据此添加公开结果复现值或修改 Survey 冻结矩阵；OS-only/未校准登记，P3.1 整体验收、
shuidisjtu 的 RV8 复核、正式协议预算、资源 profile 与 GPU 多 seed 结果仍待完成。

## 近期五项复核的关键差异

### PULDA

- 实现身份是 known-prior（oracle-prior）PULDA；固定真实 `π_population` 符合
  Survey 主实验约定，PULDA\* 的 BBE 先验估计属单独变体。
- LDA、two-way margin 与 EMA 和锁定作者源码一致，未发现公式缺陷。
- 论文 MixUp 目标用当前模型预测，作者源码用持久伪标签并强制 P 目标为 1，
  工具箱跟随作者源码；TS 风险损失逐批使用 `U∪P`，MixUp/伪标签仍用原始 U，
  作者则把 P 副本物理并入 U。均按适配保留。
- 方法身份与上述适配由 shuidisjtu 于 2026-10-18 授权代录；代录范围不含
  PULDA\*、阶段资格、训练预算或正式准入，后者由同日负责人决定另行登记。
- 已定：PA/OA 只接受 `pu_mixup`，`warmup` 仅作诊断；原生 60+60 epoch、
  P/U batch 16/128、120 个快照、不早停；单一默认配置且先验固定；阈值固定 logit 0；
  图像路径采用端到端 ResNet-18，标为 `end-to-end CNN` / `benchmark-adapted`。

### Grad-PU

- 未找到作者代码，按论文目标独立实现；Eq. (5)–(7) 的实现要素一致，
  未发现新的公式缺陷，`class_prior` 不参与目标。
- TS 的 `U∪P` 风险池是 Survey 适配；原论文实验使用剩余 U。
- 图像路径使用 method-private `cnn13_no_bn`，不进入标准 ResNet-18 主榜，
  以避免与共享表征的 BN 约束冲突（#120）。
- 网络宽度、batch、学习率调度、epoch 与验证集均与论文配方不同，
  统一按 `benchmark-adapted` 披露；调度/配方与证据边界见 #121。

### Robust-PU

- SPL 权重、阈值调度与加权 BCE 和官方默认实现一致，未发现公式缺陷。
- 官方 `class_prior` 的语义是 U 内正类比例；正式 CLI 按台账走 `ts`，runner 传
  population π，U 风险池为 `U∪P`，因此正式协议路径一致。裸 estimator 或显式
  `os` 覆盖时只应传 `π_U`，当前 estimator 不自动换算，这是调用边界。
- 预训练 batching 与 episode 尾批处理、预算 10/20（论文 100/20，官方 parser
  400/100）保留为 `benchmark-adapted`；moving weights 绑定原始行索引是有意修复，
  默认 `phi=0` 不受影响。
- 上轮要求的 `utils.py`、`models.py`、`paper_review_ref`、行为对照与许可证观察
  已由代理在 2026-10-18 [补交来源证据](../data/robustpu_source_review_20261018.json)，
  含实际优化步梯度/权重身份测试；训练不改，工程补交仍待负责人审阅，
  不自动替换本节“有条件接受适配子集”结论或正式准入。

### Split-PU

- teacher、easy/hard 划分与两轮训练路径已核对；TS 只作用于 teacher，
  学生阶段仍使用原始 U。
- JSD 权重方向、`class_prior` 的 OS/TS 语义、证据指针与 TS batch 比例问题
  已登记 #122，待负责人裁决，本摘要不展开。
- quantile 0.15/0.85 的 hard→easy 复制、增强与双视图、模板初始化等
  按 `benchmark-adapted` 保留，尊重当前实现，不要求整改。
- teacher epoch 论文 50 / 源码 20 / 本实现 10，按不同配方分别披露，
  不合并为统一配方。

### PAN

- 原 `source_status=not_found` 是检索遗漏。作者相关源码为 `morning-dews/PAN`，
  README 声称是论文代码但仓库无 LICENSE、论文正文未直接链接，因此定为
  `official_related`，不称 `official_exact`。
- 论文 Eq. (7) 与作者实现的目标、平滑项、求和/期望、C 更新 batch 及优化器设置
  不一致；作者实验还是 overlap/case-control 采样。按用户决定保留当前
  paper-faithful 实现，正式 Survey 采样跟随论文的 single-training-set/OS，
  差异登记在方法卡、台账、扩展草稿与 #123。
- 未决：对抗训练预算、C-only 快照/D 优化器续训恢复、当前源码资源重测、
  PA/OA、公开数值与正式准入。

## 已关闭整改与专项记录

- **#114 PUET**：TS bootstrap 风险池归一化已修复并独立验证；回归、证据同步与
  适配边界见 [PUET 审核补项](puet_adaptation_completion_20261017.md)。
- **#115 LaGAM**：发布代码先移除 validation、再从 U 移除 labeled P，采样身份
  裁决为 OS；registry/台账/类 `scenario` 已同步。
- **#117 CVIR**：本 Survey 走 `CVIRKnownPriorTrainer` + 显式 α_U；population π
  只给 PA，OS→TS 校准不属于本实验。
- **#116 PULNS**：额外 support 的接口/审计/披露已交付；实际 support 制品、
  标签预算与正式准入仍待批准。
- 上述整改的工程证据与调用边界见
  [algorithm_issues_114_117_20261017.md](algorithm_issues_114_117_20261017.md)。
- **GEN-PU**：2026-10-18 决定正式目标采用论文 Eq. (3)–(11)，作者 demo 仅作差异
  对照；未发现新的实现 bug。作者归档未发现 LICENSE，登记 `license=unknown`，
  不复制、执行或再分发。正式 recipe（网络/归一化、生成器输出域、四份 CNN、
  六网络预算/选模）与公开数值另行验收。
- **Holistic-PU**：2026-10-18 决定在未发现明确缺陷的前提下尊重当前实现；
  保留论文 trend 路径、默认 fixed warmup 与 continue 初始化，作者变体仅作对照。
  作者仓库未发现 LICENSE，登记 `license=unknown`；LZO/finetune、初始化 recipe、
  LZO 选模隔离与 trajectory/storage blocker 仍保留。
- **LBE / Self-PU / 独立 support / BBE / TEDⁿ** 的收尾状态与预算边界见
  [algorithm_completion_20261017.md](algorithm_completion_20261017.md)。
- 首轮流水账中的测试数量、commit 与临时 XML 不再保留；需要时从 git 历史追溯。

## 跨方法口径

1. **身份口径**：有官方源码的方法区分 `official_exact` / `official_related`；
   无源码的方法保持 paper-only；所有适配统一标 `benchmark-adapted`。
2. **先验口径**：正式 TS 路径使用 population π，并要求 U 风险池为 `U∪P`；
   显式 OS 是独立调用边界，需 `π_U`，不应与 population π 混用。
3. **验收口径**：代码级“未发现缺陷”只说明当前证据下可继续审阅，不等于方法准入；
   PA/OA、预算、资源、公开数值与负责人审阅仍需逐项完成。
