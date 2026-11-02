# PU Survey

> 一页纸：交付什么、什么算完成、还剩什么、哪些文档算数。
> 实验规则以 [`pu_survey_protocol.md`](pu_survey_protocol.md) 与版本化执行 JSON 为准；当前最终候选是 `survey-v3.2` / `survey-comparison-v8` / `survey-recipes-v1.2`。
> 当前实验进度、完成数与队列状态统一见 [`survey_execution_plan.md`](survey_execution_plan.md) §1；本页不维护进度数字。

## 1. 交付物与完成定义

**A. 结果文件**

- 工作簿由 `scripts/export_survey_results.py` 生成（`survey_results.xlsx`），按协议 §5.1 分四组、不混排名：
  SCAR–PA 主榜、SCAR–OA 对照、SAR–OA 压力测试（LBE-A / LBE-B 分开）、PN oracle。
- 每格为 5 seed 的 Accuracy 均值±标准差；AUC / F1 / Precision / Recall 作附加指标；含单配置训练成本、
  完整调参成本、峰值显存。单 seed 读数不进结论表。
- 结论分析（独立文档，或并入工作簿）：按 [执行计划](survey_execution_plan.md) §1.4.5 的信号表归因，
  每条标明依据哪几类信号，并区分「文献事实 / 本实验观测 / 推断」。

**B. 代码功能**

- 协议 §2.4（四）第 7–11 条逐条对应一个可运行的命令或文件：`fit(model, train, pu_val, clean_val, test)`
  接口、可 DIY 的训练与造数据函数、官方示例脚本（现为 `scripts/run_survey_experiment.py`）、PN oracle、
  seed / split manifest 留痕；加上从四路 split 到汇总表的复现命令序列。

**共同验收线**：协议 §5。22 个目标方法全部过四项门禁则发完整主榜；否则标 `pilot / partial benchmark`。
§5.6 的代码 commit 与依赖锁**由批次层承载**：每批一份证据包落在 `<results>/batch_evidence/`，
run 在 manifest 的 `batch` 块里引用它（D44 定口径、D52 定实现，命令见 [runbook](runbook.md) 第 4 节）。

## 2. 剩余工作

只列仍未关闭的事项；完成数、排队数和批次状态以 [执行计划](survey_execution_plan.md) §1 为准。
分工只在本表维护，执行计划的缺口表不再重复负责人。

| 工作 | 负责 | 完成判据 |
|---|---|---|
| v3.1/v3.2 缺口批（1845 + 280 fit）按 canary → 全网格跑完并对账；D54 完整批次审计（15 个 support 单元、120 次修正替代关系） | HENG958 / 合流 | 完整网格 + 批次审计通过 |
| GradPU 首次完整真实运行及资源记录；PN-CNN 5 seed 现有产物验收 | HENG958 | 合规 manifest + 峰值记录 |
| v3.1/v3.2 复核与 formal blockers 清零（新增比较映射仍 `pending_review`） | HENG958 / 合流 | review accepted、formal 0 |
| 正式 5 seed 结果工作簿、诊断卡五类信号、四组结论 | 合流 | 交付物 A |
| 实验结束后清理 v3.0/v3.1 旧协议、矩阵与注册表，只保留最终正式版并验证复现命令 | shuidisjtu / 合流 | 最终版文档 + 从零复现通过 |

## 3. 文档地图

### 权威源

| 绑定 | 协议 | 比较矩阵 | profile 映射 | 配方注册表 | 定位 |
|---|---|---|---|---|---|
| v3.0 基线 | `survey_protocol_v3.json` | `survey_comparison_v6.json` | `survey_comparison_profile_map_v3.json` | `survey_recipe_registry_v1.json`（15 locked / 8 excluded） | 历史基线；完成后清理 |
| v3.1 缺口批 | `survey_protocol_v3_1.json` | `survey_comparison_v7.json` | `survey_comparison_profile_map_v4.json` | `survey_recipe_registry_v1_1.json`（23 locked） | 历史缺口批；完成后清理 |
| v3.2 当前最终候选 | `survey_protocol_v3_2.json` | `survey_comparison_v8.json` | `survey_comparison_profile_map_v5.json` | `survey_recipe_registry_v1_2.json`（23 locked） | 保留为最终复现版本（待 HENG958 复核） |

- [`survey_execution_plan.md`](survey_execution_plan.md)：能力矩阵、缺口、预算与资源口径、执行边界、**决策账本（D1–D55）**，也是当前进度与批次状态的唯一记录。
- `pu_toolbox/experiment/method_ledger.json`：方法台账；身份、来源、视图、校准、适配与待复核状态。
- [`runbook.md`](runbook.md)：从 `data/raw` 到工作簿的命令序列，含 split / support 制品 / 批次 / 证据包 / 汇总 / 复现，以及 v3.1/v3.2 队列与监控入口。
- [`../method_cards/`](../method_cards/)：各方法的公式、复现状态与实现边界。
- [`admission/p4_1b_registry_freeze_20261025.md`](admission/p4_1b_registry_freeze_20261025.md)：v3.1/v3.2 项目配方、版本化例外与验证边界。
- 队列与交接：`scripts/run_survey_backlog_campaign.py`、`run_survey_parallel_queue.py`、`monitor_survey_campaign.py`、`audit_survey_queue.py`、`export_survey_queue_handoff.py`。

v1 / v2 协议与 v1–v5 比较矩阵仅作历史绑定；v6–v8 是 v3.0–v3.2 的当前绑定，由协议版本自动解析。

### 结果与凭证（`data/`）

- 当前数据制品：[`clean_support_materialization_20261024.json`](data/clean_support_materialization_20261024.json)（25 份 clean-support 制品；`formal_admission=false`，不代表正式预算或方法准入）。
- 正式/工程凭证：[`survey_recipe_formal_run_20261025.json`](data/survey_recipe_formal_run_20261025.json)（100+100，单 seed，约 2.7 h GPU）、
  [`genpu_formal_capacity_20261025.json`](data/genpu_formal_capacity_20261025.json)、
  [`genpu_width_measure_20261025.json`](data/genpu_width_measure_20261025.json)、
  [`genpu_recipe_validation_20261023.json`](data/genpu_recipe_validation_20261023.json)、
  [`candidate_image_validation_20261023.json`](data/candidate_image_validation_20261023.json)（PULDA / GradPU 120 / 200 快照实测，口径见 D43）、
  [`survey_recipe_binding_20261025.json`](data/survey_recipe_binding_20261025.json)。
- 数据来源凭证：`adni_substitute_receipt_*`、`adni_split_artifacts_index_*`、`public_data_receipt_*`、`p1_2_r_raw_receipt_*`、
  `split_artifacts_index.json`。
- 旧工程核对表、2026-10-22 重建前清单与旧 TS 机器摘要已按 D55 清理；20261024 的 `rebuild_of` 仅保留来源路径文字，不再提供旧制品。
- 这些 JSON 带自身摘要哈希，**不要改内容**。

### 冻结的证据文档

`admission/`、`reviews/`、`delivery/`、`architecture/` 与本目录根部的 `*_review_*.md` 是**冻结的证据文档**：
`method_ledger.json` 和测试按路径引用并断言其中一部分的内容，`audit_survey_batches.py` 读取 `delivery/p2_1_b*_snapshot.md`，
所以它们留在原位，不要重命名、不要改字。更正写进执行计划的决策账本，不回改这些文档。

### 已归档

此前 93 份交付、复核、交接、审批文档与数据（另有已退役的 P3 / 存储探针 / 收据审计脚本）已从工作树移除，
没有丢失：`git show survey-docs-archive-20261025:<原路径>` 即可取回。

## 4. 不做清单

- 不再新增跨文件摘要绑定或门禁脚本；新增须能指出一次已发生的具体事故。
- 不再为已冻结文件新开版本号（D42 默认）；2026-11-01 的 v3.1/v3.2 是 **D55 例外**。实验完成后统一清理旧协议/矩阵/注册表，只保留最终正式版供工具箱实验脚本复现。
- 不补写历史交接和复核文档。
- 不追求与论文数值一致（沿用 D39）；`no_direct_anchor` 单元只用内部信号表述。
- 不做 π 估计器敏感性实验（协议 §3 第 3 条的「后续」项）。
- 不使用 ADNI 官方数据（沿用 D1 的替代数据口径）。

## 5. 协作约定

- 一任务一 Issue：写明 assignee 和验收标准（命令跑通 + 合规 manifest + 峰值记录），群聊只发链接。
- 开工前先看自己和对方名下未关闭的 Issue，避免重复做、或做完没通知。
- 任务完成：临时文档（任务单、复核、交接）蒸馏成决策账本一行，或直接删除；不入库的任务单放 `tmp/`。
- 能一句话在群里说清的，不写文档。
