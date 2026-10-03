# PU Survey 文档索引

PU 调研实验（工具箱首次实际应用）的全部文档。本目录自包含：外层 `docs/README.md` 只保留到本目录的目录级入口，文件级索引在此维护。

**机器真相源**（数值/状态以它们为准，文档只作解释）：

- `pu_toolbox/experiment/survey_protocol_v1.json` — 执行协议，当前 `survey-v1.2`，摘要 `287c2f45387f02714e5925b35dcb04f3f64e1e3faf8f740cab861fc5747dd84a`
- `pu_toolbox/experiment/survey_comparison_v3.json` — 外部对照矩阵，当前 `survey-comparison-v3`
  （`v1`/`v2` 原样留档为历史版本；修订理由见 `survey_execution_plan.md` 决策 D10、D12）

**签署状态速览**：P2.0a 已签署（`accepted`）。P2.0b、P2.0c 的合作者签署**仍未获得**
（`pending_collaborator_review`）；两者由 shuidisjtu 于 2026-09-25 基于单方技术验收**放行**
（已从协议 `formal_blockers` 移除，见决策 D12）。放行不等于对方已复核。

## 权威源

| 文档 | 定位 |
|---|---|
| [`pu_survey_protocol.md`](pu_survey_protocol.md) | 协议要求纲要（8 数据集、OS/TS 视图、四份数据接口、PA/OA 双选模） |
| [`survey_execution_plan.md`](survey_execution_plan.md) | 执行协调层：当前状态、任务分工与验收、交叉验证对照预注册、风险与留痕约定；决策账本见文末附录 A |

## 交付记录

| 文档 | 定位 |
|---|---|
| [`p2_0a_delivery.md`](p2_0a_delivery.md) | P2.0a 共享规格、执行矩阵、CIFAR 接线与 CPU/GPU 验证记录 |
| [`p2_0b_delivery.md`](p2_0b_delivery.md) | P2.0b 标签语义 fail-loud 门禁交付（未签署） |
| [`p2_0c_delivery.md`](p2_0c_delivery.md) | P2.0c 外部对照预注册：资格闭集、覆盖推导、来源口径差异判读与签署顺序 |
| [`p2_0e_delivery.md`](p2_0e_delivery.md) | P2.0e TS-OS 校准接线：各方法校准形态调查、接线范式、`nnpu`/`upu` 切片证据与未决项 |
| [`p2_2_delivery.md`](p2_2_delivery.md) | P2.2 Pilot 聚合与审计：两套键与两套状态、分层计数、审计可判项与未接线项、复现入口；原始制品在仓库外，**未获合作者复核** |
| [`epoch_checkpoint_delivery.md`](epoch_checkpoint_delivery.md) | 逐 epoch 权重保存、PA/OA 独立恢复与验证证据 |
| [`pn_oracle_integration.md`](pn_oracle_integration.md) | PN oracle 接入决策与 PU-Bench 口径对照记录 |
| [`vpu_sampling_audit.md`](vpu_sampling_audit.md) | P3.1 局部：VPU 采样假设审计（Gate 0 四层证据、原生 `ts` 裁决、与上游的验证口径分歧）；对应决策 D21，**未完成 P3.1 验收** |

## 批次快照

| 文档 | 定位 |
|---|---|
| [`p2_1_b1_snapshot.md`](p2_1_b1_snapshot.md) | P2.1 B1（`spambase`，215 runs）：清单层与门禁层快照；不含排名与数值裁决（属 P2.2） |
| [`p2_1_b2_snapshot.md`](p2_1_b2_snapshot.md) | P2.1 B2（`imdb`，215 runs）：同构快照，含与 B1 合并聚合的独立性验证 |
| [`p2_1_b3a_snapshot.md`](p2_1_b3a_snapshot.md) | P2.1 B3a（`cifar10`/adapter，110 runs）：快照，含 checkpoint 形态与 `ConvergenceWarning` 已知现象 |
| [`p2_1_b3ab_snapshot.md`](p2_1_b3ab_snapshot.md) | P2.1 B3a+B3b **合并分析组**（`cifar10`/adapter，180 runs）：§12.1 要求的两批合并快照，含 adapter cache 审计、量化指标逐 run 罗列、B 层策略与 checkpoint 容量对账；不含组级方法排名 |
| [`p2_1_b4_snapshot.md`](p2_1_b4_snapshot.md) | P2.1 B4（`cifar10`/`native_cnn`，35 runs）：**草稿**，运行中。已填身份、前置门禁与对账、§7A 探针、D25 回收实测；清单层与门禁层统计及验收命令见其 §7 |

## P2.1 交接

| 文档 | 定位 |
|---|---|
| [`p2_1_b5_review_checklist.md`](p2_1_b5_review_checklist.md) | P2.1 B5 跨批复核清单：14 号 §8 十三项拆成的复核动作，以及备料阶段查出的四条既有记录缺陷 |
| [`p2_1_handoff_checklist.md`](p2_1_handoff_checklist.md) | P2.1 → P2.2 交接清单的执行侧记录：14 号 §8 各项的证据与缺口，以及「未使用测试集真值选模」的代码级证据 |

## 方案

| 文档 | 定位 |
|---|---|
| [`artifact_storage_architecture.md`](artifact_storage_architecture.md) | 制品存储架构：L1 决策记录 / L2 选中权重 / L3 轨迹权重的分级，面向 8 数据集 × 22 方法的规模推演与三段迁移路径 |
| [`epoch_checkpoint_reclaim_plan.md`](epoch_checkpoint_reclaim_plan.md) | 上者的**阶段 1**：保留选中权重、回收其余（**已实施**，D25 见 commit `e7c6f21`；含协议口径收窄的决策登记要求与可比性影响） |

## 复核 / 签署

| 文档 | 定位 |
|---|---|
| [`p2_0a_review.md`](p2_0a_review.md) | P2.0a 合作者逐条复核决定、未决项与签署模板（已签署） |
| [`p2_0c_review.md`](p2_0c_review.md) | P2.0c 复核包：自审结论、判读留痕、合作者复核与签署栏（未签署） |
| [`p2_2_review.md`](p2_2_review.md) | P2.2 复核包：双栏清单（合作者管文献与对照、实验负责人管交付面）、复现与比对步骤、签署模板（未签署） |
| [`heng958_independent_review.md`](heng958_independent_review.md) | HENG958 环境、nnPU/Self-PU 台账、split 可执行性与 P2.0d SAR-OA 复核（独立复核，不替代签署） |

## 其他

- [`data/split_artifacts_index.json`](data/split_artifacts_index.json) — P1.4 的 15 份 split 制品索引（归档摘要与逐文件 sha256），接收端 `verify` 的比对基准
- [`data/p2_2_artifacts_index.json`](data/p2_2_artifacts_index.json) — P2.2 交付面基准：输入树的 645 份 manifest 树摘要与逐文件摘要（先比树、再比产物）、十份分析产物与三份证据的 sha256，以及 645 行交付表的摘要
- `assets/` — 协议配图
