# PU Survey 文档索引

PU 调研实验（工具箱首次实际应用）的全部文档。本目录自包含：外层 `docs/README.md` 只保留到本目录的目录级入口，文件级索引在此维护。

**机器真相源**（数值/状态以它们为准，文档只作解释）：

- `pu_toolbox/experiment/survey_protocol_v1.json` — 执行协议，当前 `survey-v1.2`，摘要 `c15b0c9e0677729d749c79af2d563849d5b70be395e65fd3ea2b2cfa0eaff529`
- `pu_toolbox/experiment/survey_comparison_v1.json` — 外部对照矩阵，`survey-comparison-v1`

**签署状态速览**：P2.0a 已签署（`accepted`）；P2.0b、P2.0c 未签署（`pending_collaborator_review`），见各自复核/交付记录。

## 权威源

| 文档 | 定位 |
|---|---|
| [`pu_survey_protocol.md`](pu_survey_protocol.md) | 协议要求纲要（8 数据集、OS/TS 视图、四份数据接口、PA/OA 双选模） |
| [`survey_execution_plan.md`](survey_execution_plan.md) | 执行路线、状态与决策记录（§4 的 0a–0j 条），本目录的状态权威源 |

## 交付记录

| 文档 | 定位 |
|---|---|
| [`p2_0a_delivery.md`](p2_0a_delivery.md) | P2.0a 共享规格、执行矩阵、CIFAR 接线与 CPU/GPU 验证记录 |
| [`p2_0b_delivery.md`](p2_0b_delivery.md) | P2.0b 标签语义 fail-loud 门禁交付（未签署） |
| [`p2_0c_delivery.md`](p2_0c_delivery.md) | P2.0c 外部对照预注册：资格闭集、覆盖推导、来源口径差异判读与签署顺序 |
| [`epoch_checkpoint_delivery.md`](epoch_checkpoint_delivery.md) | 逐 epoch 权重保存、PA/OA 独立恢复与验证证据 |
| [`pn_oracle_integration.md`](pn_oracle_integration.md) | PN oracle 缺陷修复接入记录，决策仍被引用，开放问题见其 §8 |

## 复核 / 签署

| 文档 | 定位 |
|---|---|
| [`p2_0a_review.md`](p2_0a_review.md) | P2.0a 合作者逐条复核决定、未决项与签署模板（已签署） |
| [`p2_0c_review.md`](p2_0c_review.md) | P2.0c 复核包：自审结论、判读留痕、合作者复核与签署栏（未签署） |
| [`heng958_independent_review.md`](heng958_independent_review.md) | HENG958 环境、nnPU/Self-PU 台账、split 可执行性与 P2.0d SAR-OA 复核（独立复核，不替代签署） |

## 其他

- [`data/split_artifacts_index.json`](data/split_artifacts_index.json) — P1.4 的 15 份 split 制品索引（归档摘要与逐文件 sha256），接收端 `verify` 的比对基准
- `assets/` — 协议配图
