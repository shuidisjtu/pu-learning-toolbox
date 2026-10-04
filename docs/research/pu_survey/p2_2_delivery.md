# P2.2 Pilot 聚合与审计（交付记录）

> **2026-10-04 复核入口**：当前 v3 与 pilot 的代理技术检查、来源读数、文档纠错及未决输入见 [独立技术复核](pilot_independent_review_20261004.md)。历史结论与本人签署保留；本轮不解除 pending_collaborator_review，也不将工具回归冒报为正式 645 次结果重算。

- 日期：2026-10-03
- 定位：把 P2.1 五批（645 runs）的运行制品推进为可审计、可分层的 P2.2 交付。**不含**跨数据集总排名
  （协议 §5 第 1 条），也不含方法排名——分层结果的逐行数值由汇总产物承载，本文件只写口径、边界与未决项。
- 依据：`survey_execution_plan.md` §2.2 的 P2.2 行与决策 D13 / D24 / D26；实施方案
  `P2.2_B5_B6_implementation_plan.md`（v0.4，执行工作区文档，未入库）。

## 1. 交付物与权威源

| 角色 | 是什么 | 在哪 |
|---|---|---|
| 审计入口 | `scripts/audit_survey_batches.py` | 仓库 |
| 汇总入口 | `scripts/summarize_survey_results.py` | 仓库 |
| 对照附着入口 | `scripts/compare_survey_results.py` | 仓库 |
| 纯函数与 schema | `pu_toolbox/experiment/survey_audit.py`、`survey_summary.py` | 仓库 |
| 公平性门禁 | `scripts/aggregate_survey_runs.py`（被前两者**调用**，不另立实现） | 仓库 |
| 协议 | `survey_protocol_v1.json`（`survey-v1.2`） | 仓库 |
| 对照矩阵 | `survey_comparison_v3.json` | 仓库 |
| 批次白名单 | 五批根目录 + 期望 manifest 数的 JSON，由 `--config` 传入 | 执行机（不含在本仓库） |
| 报告身份块 | `pu_toolbox/experiment/survey_provenance.py`（三份报告共用，见 `api.md` 同名一节） | 仓库 |
| 原始制品 | `01_audit` / `02_summary` / `03_comparison` / `04_evidence` | 执行机分析工作区，**在仓库外** |
| 复核包 | [`p2_2_review.md`](p2_2_review.md)（双栏清单与签署模板） | 仓库 |
| 比对基准 | [`data/p2_2_artifacts_index.json`](data/p2_2_artifacts_index.json) | 仓库 |
| 645 行交付表 | 逐 run 原始数据（含三项成本与环境身份）加 P2.2 派生层 | **在仓库外，由执行方直接发送** |

**制品分发的决定（2026-10-03）**：不打包上网盘。合作者同时具备仓库权限与数据盘权限，
分析产物按 §7 自行复现即可，本文件与基准文件承载口径与比对基准；645 行交付表是唯一需要
送达的实体文件，由执行方直接发送。基准文件里登记了它的摘要，接收端可比对。

**先比输入、再比产物。** 基准文件同时给出 645 份 manifest 的**树摘要**与各产物的逐文件摘要。
复核者用自己的副本重跑后，须先比对树摘要确认读的是同一批字节——树不一致时，结果差异应归因于
输入而非代码。逐行的均值/标准差、分组明细与审计证据都在上表的「原始制品」里，不在仓库。

## 2. 输入覆盖与白名单机制

五批 `215 / 215 / 110 / 70 / 35 = 645` 全部纳入，逐份 manifest 通过身份预检。

白名单不是「一个根目录递归扫」——`discover_manifests` 是 `rglob`，若把父目录交给它，会连同
**合并工作副本**再数一遍（`B3a+B3b` 的合并副本正是这种情形）。因此实现按批给**精确根**，
并显式拒绝工作副本目录名与技术 probe；这两类目录不计入覆盖，也不得被白名单化。

**B4 的纳入是本轮与阶段 0 的差别**：阶段 0 的 610 份（B1–B3b）只是工具链验证，
正式交付以 645 份为准。

## 3. 两套键与两套状态（读报告前必须分清）

- **键分两层**：公平性门禁的组是 `(comparability_group, run_view, mechanism)`，单元是
  `(seed, c_token)`；而**数值报告的行**是 `(comparability_group, run_view, mechanism, method,
  selection_protocol, c_token)`。`seed` 是**聚合维度**，不是行键——把单元键并进行键会按 seed 拆成五行，
  五次重复的均值无从计算。
- **`selection_protocol`（PA/OA）只在报告层分行**，不是门禁维度：门禁不按 PA/OA 再跑一遍。
- **状态有两套词表且必须翻译而非复制**：门禁内部的单元状态是 `comparable` / `blocked`，那是它的
  内部语言；交付状态是闭集 `formal` / `partial` / `technical_probe` / `historical` / `refused` /
  `incomplete` / `not_reproducible`，配 `reasons` 代码列表，判定式 `status == formal` ⟺ `reasons == []`。
  **`blocked` 不得作为 `status` 取值。**
- **行级 `formal` 与批次级 `formal_ready` 是两个层次**：前者说这一行自身齐备，后者说整棵树能否整体
  作为正式榜。两者必须同时给出，否则读者会以为其中一个写错了。

## 4. 分层结果（口径与计数，非数值表）

- 公平性门禁分为 **48 个组**（B1 16 / B2 16 / B3a 7 / B3b 6 / B4 3），全部通过，**0 组被拒**；
- 汇总出 **183 行**：**180 `formal` / 3 `partial` / 0 `diagnostic`**；`not_reproducible` 0、`refused` 0；
- 3 条 `partial` 是三个数据集各一个 `pn_oracle` 单元，带 `protocol_deviation: ['c_grid']`
  （c-independent 行不跑完整 c 网格）——这是 D24 保留该阻断位要标的**真实偏差**，不是缺陷；
- 按单批边界，**B3b 与 B4 均无 oracle 阻断**；B1 / B2 / B3a 的 `formal_ready = False` 由上述 oracle 单元造成。
  B3a+B3b 合并分析组仍为 False；原「B4 唯一 True」与 B3b 快照冲突，已更正，源端报告待独立对账。

分榜边界与三张表的划分依据协议 §5 第 1 条（SCAR–PA 主榜 / SCAR–OA 对照 / SAR–OA 压力测试 /
PN oracle）；跨数据集只比较趋势，不生成总排名。

## 5. 不可越过者

三条边界当前**一律未解除**，逐条出处与口径见 [`survey_execution_plan.md`](survey_execution_plan.md) §1.3，
本文件不复述；此处只点明它们对 P2.2 的约束：

1. PA 行不能数值裁决；
2. 对照附录必为定性（矩阵 `numeric` 为 0 条，预注册的判定式当前无对象可施）；
3. 回收后的可持久复现范围是「选中权重 + 全部逐 epoch 选择记录」，不是「全部 epoch 权重」。

## 6. 审计层：可判项与未接线项

审计共 17 项：**9 项 pass、7 项 not_run、1 项 not_applicable、0 项 fail**。`overall` 为 `partial`，
来源是 not_run 而非不合格。

**7 项 not_run 都是「输入未接线」，逐项如此记账而不报成 pass**（把没跑的检查报成通过，等于把缺口写成覆盖）：

| 检查 | 缺什么输入 |
|---|---|
| A04 计划身份 | 五批参考计划 JSON |
| A05 完成状态 | 退出码与完成行（在运行日志里，不在 manifest） |
| A10 checkpoint 回收 | 盘上路径——B4 的 manifest 记的是源端绝对路径，本机解析不到 |
| A11 归档与恢复 | 备份目录与 `.sha256` |
| A14 阶段性快照 | 属文档检查，不读 manifest |
| A16 消融变体披露 | 同上 |
| A17 无测试集泄露 | 由协议测试套件承载，非 manifest 可判 |

A10 的**实质**已另行验证：B4 的回收守恒式 `reclaimed + on_disk == refs` 在源端逐 run 核对通过
（35/35），见 [`p2_1_b4_snapshot.md`](p2_1_b4_snapshot.md) §7.1。

## 7. 复现

三个入口都是读 manifest 的纯分析，不重训、不改 test 指标：

```
uv run python scripts/audit_survey_batches.py --config <白名单 JSON> --out-dir <01_audit>
uv run python scripts/summarize_survey_results.py --config <白名单 JSON> --out-dir <02_summary>
uv run python scripts/compare_survey_results.py --summary <02_summary/summary.json> --out-dir <03_comparison>
```

白名单 JSON 只需五批的 `name` / `root` / `expected_manifests` / `role`，外加
`excluded_subpaths` 列出必须拒绝的工作副本目录名。跑之前必须确认输入树已稳定（B4 已收尾、
无 runner 在写），协议摘要在一个值上。

**两处按原样照抄会失败，须知道：**

1. 白名单里的 `batches[].root` 记的是执行机路径，直接沿用会找不到输入。改成自己那份副本的路径；
   该改成什么，同一份文件里的 `source_roots` 已经写了。
2. 三份报告会各自写入一个身份块（代码 commit 与其状态、输入结果根、协议与对照摘要）。
   因此**报告是产出它的那次 checkout 的函数**：换了 commit 或输入根，报告就会不同。
   比对时必须先核身份块，再比数值。

**比对基准**在 [`data/p2_2_artifacts_index.json`](data/p2_2_artifacts_index.json)：先比
`inputs.total.tree_sha256`（确认读的是同一批字节），再比 `artifacts` 的逐文件摘要。
Markdown 与 CSV 不含时间戳，故同一输入、同一 checkout 下逐字节可复现；JSON 里带 `generated_at`，
比对时须先移除该字段。

## 8. 未决项

- **合作者复核未获得**：P2.0c 对照矩阵仍为 `pending_collaborator_review`、`formal_blockers` 仍为
  `["collaborator_review"]`；PA 准则与 `self_pu` 元重加权的复核同样未获得。**本文件不得读作已签署**
  （未获项清单见 [`survey_execution_plan.md`](survey_execution_plan.md) §1.4）。
- **期望单元完整性格网口径**（D13 遗留）：复合分区下「缺失单元」如何判定仍未定，本次不涉。
- **A04（计划身份）仍为 `not_run`**：需要五批的参考计划 JSON，目前只有 B1 的在执行机上，
  B2/B3a/B3b/B4 需从源端取。其余 6 项 `not_run` 的缺什么输入见 §6。
