# PU Survey

> 一页纸：交付什么、什么算完成、还剩什么、哪些文档算数。
> 实验规则以 [`pu_survey_protocol.md`](pu_survey_protocol.md) 和 `survey_protocol_v3.json` 为准，本页不重复。

## 1. 交付物与完成定义

**A. 结果文件**

- 工作簿（形式同 [`PU_Survey_results_20261025.xlsx`](data/PU_Survey_results_20261025.xlsx)），按协议 §5.1 分四组、不混排名：
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
§5.6 的代码 commit 与依赖锁**由批次层承载**（每个批次一份证据包，run 引用批次身份）。

## 2. 当前进度（2026-10-26，按协议 JSON 实测）

- `survey_protocol_v3.json`：130 个执行单元，106 个 runnable，24 个 blocked，8 个数据集都已有单元。
  22 方法 × 8 数据集 = 176 格，其中 126 格有单元、102 格 runnable。
- 每个数据集都带齐其模板数据集的全部 runnable 行（见 D46）：mnist、fashion_mnist、adni 对 cifar10，
  connect_4 对 spambase，twenty_newsgroups 对 imdb；每个方法族已在 GPU 上真跑过 1 个 seed、接线核对通过。
  blocked 方法在这些数据集上还没有行，随其模板行一起翻 runnable。
  LaGAM / PULNS / Self-PU full-meta 的 support 策略不覆盖 twenty_newsgroups，只落在 connect_4。
- 现有 xlsx 是 31 个 run、单 seed 的**工程核对表**，不是交付物 A。
- 现有 runnable 单元的正式批次（5 seed × 全部 c 与机制）约 3350 个 run（其中三个图像数据集 1185 个，`run_survey_pilot.py --dry-run --datasets mnist,fashion_mnist,adni` 实测）；GPU 机时不设限，按 method / dataset /
  training-path / 机制分批跑，最后统一汇总。

## 3. 剩余工作

| 工作 | 负责 |
|---|---|
| 24 个 blocked 单元翻 runnable：KLDCE / CVIR / P3MIX / PAN / VPU / GradPU 各 3 行（spambase、imdb、cifar10）、PUET / RP 各 2 行（spambase、imdb）、LaGAM / PULNS 各 1 行（cifar10）；含 GradPU 首次真实运行 | HENG958 |
| PN-CNN 5 seed（`pn_oracle` cifar10 / native_cnn） | HENG958 |
| LaGAM / PULNS / Self-PU full-meta 补 mnist、fashion_mnist 的 support 单元（support 策略覆盖这两个数据集，目前只有 spambase 与 connect_4 的行） | shuidisjtu |
| 正式 5 seed × c 网格跑批 → 汇总 → 写结论；协议 §5.6 第 6 条措辞改为「批次证据包 + run 引用批次身份」，并确认批次层确有写入 commit 与 `uv.lock` 哈希的代码 | 合流 |
| 官方示例脚本归位或在 `api.md` 登记；写一页从原始数据到汇总表的命令序列 | shuidisjtu |

## 4. 文档地图

### 权威源

| 文档 / 数据 | 定位 |
|---|---|
| [`pu_survey_protocol.md`](pu_survey_protocol.md) | 协议：数据集、四角色输入、OS / TS 视图、PA / OA 选模、PN oracle 与主榜门禁 |
| [`survey_execution_plan.md`](survey_execution_plan.md) | 能力矩阵、缺口、预算与资源口径、执行边界、**决策账本（D1–D48）** |
| `pu_toolbox/experiment/survey_protocol_v3.json` | 机器可读执行协议（`survey-v3.0`，原地演进，见 D42） |
| `pu_toolbox/experiment/method_ledger.json` | 方法台账：身份、来源、视图、校准、适配、待复核状态 |
| `pu_toolbox/experiment/survey_comparison_v6.json` | 比较矩阵（v3 运行自动使用）；`survey_comparison_profile_map_v3.json` 给出 profile → 条目映射 |
| `pu_toolbox/experiment/survey_recipe_registry_v1.json` | 配方注册表（15 个 locked profile，8 个 excluded），CI 强制校验 |
| [`../method_cards/`](../method_cards/) | 各方法的公式、复现状态与实现边界 |

v1 / v2 协议与 v1–v5 比较矩阵留在盘上，只作历史绑定。

### 结果与凭证（`data/`）

- 结果：[`PU_Survey_results_20261025.xlsx`](data/PU_Survey_results_20261025.xlsx)（工程核对表）；
  [`workbook_runs_20261025.json`](data/workbook_runs_20261025.json) 是它**唯一入库的来源**（原始 manifest 与生成脚本在被忽略的 `tmp/`，本机已不存在）。
- GEN-PU 正式运行：[`survey_recipe_formal_run_20261025.json`](data/survey_recipe_formal_run_20261025.json)（100+100，单 seed，约 2.7 h GPU 才能重得）、
  [`genpu_formal_capacity_20261025.json`](data/genpu_formal_capacity_20261025.json)、[`genpu_width_measure_20261025.json`](data/genpu_width_measure_20261025.json)、
  [`genpu_recipe_validation_20261023.json`](data/genpu_recipe_validation_20261023.json)。
- 其它实测与绑定凭证：[`candidate_image_validation_20261023.json`](data/candidate_image_validation_20261023.json)（PULDA / GradPU 120 / 200 快照的实测，口径见 D43）、
  [`survey_recipe_binding_20261025.json`](data/survey_recipe_binding_20261025.json)、[`selfpu_ts_support_validation_20261023.json`](data/selfpu_ts_support_validation_20261023.json)。
- 数据来源凭证：`adni_substitute_receipt_*`、`adni_split_artifacts_index_*`、`public_data_receipt_*`、`p1_2_r_raw_receipt_*`、
  `split_artifacts_index.json`、`clean_support_materialization_20261022 / 20261024.json`（25 份 clean-support 制品的清单）。
  这些 JSON 带自身摘要哈希，**不要改内容**。

### 冻结的证据文档

`admission/`、`reviews/`、`delivery/`、`architecture/` 与本目录根部的 `*_review_*.md` 是**冻结的证据文档**：
`method_ledger.json` 和测试按路径引用并断言其中一部分的内容，`audit_survey_batches.py` 读取 `delivery/p2_1_b*_snapshot.md`，
所以它们留在原位，不要重命名、不要改字。更正写进执行计划的决策账本，不回改这些文档。

### 已归档

此前 93 份交付、复核、交接、审批文档与数据（本目录由 205 个文件降到 112 个；另有已退役的 P3 / 存储探针 / 收据审计脚本）已从工作树移除，
没有丢失：`git show survey-docs-archive-20261025:<原路径>` 即可取回。

## 5. 不做清单

- 不再新增跨文件摘要绑定或门禁脚本；新增须能指出一次已发生的具体事故。
- 不再为已冻结文件新开版本号（沿用 D42：原地扩充）。
- 不补写历史交接和复核文档。
- 不追求与论文数值一致（沿用 D39）；`no_direct_anchor` 单元只用内部信号表述。
- 不做 π 估计器敏感性实验（协议 §3 第 3 条的「后续」项）。
- 不使用 ADNI 官方数据（沿用 D1 的替代数据口径）。

## 6. 协作约定

- 一任务一 Issue：写明 assignee 和验收标准（命令跑通 + 合规 manifest + 峰值记录），群聊只发链接。
- 开工前先看自己和对方名下未关闭的 Issue，避免重复做、或做完没通知。
- 任务完成：临时文档（任务单、复核、交接）蒸馏成决策账本一行，或直接删除；不入库的任务单放 `tmp/`。
- 能一句话在群里说清的，不写文档。
