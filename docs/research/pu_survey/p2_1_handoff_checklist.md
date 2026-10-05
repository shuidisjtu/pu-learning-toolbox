# P2.1 → P2.2 交接清单（执行侧记录）

> **2026-10-04 接收侧补记**：Excel 收件身份及 645 次运行覆盖 / 183 行派生层数值对账通过，见 [阶段验收](p21_workbook_review_20261004.md)。原 manifest、日志与权重尚未接收，不改变源端制品核验及本人签署要求。

> **2026-10-04 复核入口**：当前 v3 与 pilot 的代理技术检查、来源读数、文档纠错及未决输入见 [独立技术复核](pilot_independent_review_20261004.md)。历史结论与本人签署保留；本轮不解除 pending_collaborator_review，也不将工具回归冒报为正式 645 次结果重算。

状态日期：2026-10-03。依据：执行手册 14 §8 的交接清单。

本文件逐项记录**证据**与**当前状态**，供 B6 勾选。它**不含**方法排名、跨方法对比与数值裁决——
那些属 P2.2。这里只回答「制品与记录是否齐备、可不可审计」。

## 1. 逐项状态

| # | 清单项 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| 1 | 五批覆盖 215/215/110/70/35 = 645 | 是 | B1–B3b 见各批快照 §1；B4 实测 35（`p2_1_b4_snapshot.md` §7.2），五批合计 **645** |
| 2 | 五批 `selection` / `execution_units` 与参考 JSON 一致 | 是 | 五批均与参考快照的六项身份字段对账，`mismatches= []`；B4 跑后重跑 dry-run 得 `planned 35 / completed 35 / pending 0`（`p2_1_b4_snapshot.md` §7.1 第 6 项） |
| 3 | 所有正式 manifest 协议摘要唯一且为冻结值 | 是 | 五批均单值且等于冻结摘要；B4 实测 35 份单值（`p2_1_b4_snapshot.md` §7.2） |
| 4 | 每个分析单元的阶段性快照与边界措辞已保存 | 是 | B1 / B2 / B3a / B3a+B3b / B4 五份快照齐；B4 见 `p2_1_b4_snapshot.md` |
| 5 | A 层制品、B 层策略与清理记录完整 | 是 | A 层齐备（各批 manifests 树 + 证据包 + 日志；归档双侧各一份、摘要一致）；B 层策略均已选定并记录——B1 有回收记录、B2 / B3a / B3b 保留完整批次归档、**B4 采用每 run selected checkpoint**（快照 §7.7）。B2 / B3a / B3b 的未选中 checkpoint 未清理：16 §4.3 的措辞是「才可**考虑**清理」，清理属可选项而非要求，故不构成缺口 |
| 6 | 五批结果、日志、计划、备份目录相互隔离 | 是 | 目录按批独立，B3a 与 B3b 结果树分开 |
| 7 | B3a adapter cache / frozen feature 来源已审计 | 是 | `p2_1_b3ab_snapshot.md` §5；cache 已于 2026-10-01 清理 |
| 8 | B3b 70 个 TS 视图与校准字段已审计 | 是 | `p2_1_b3ab_snapshot.md` §2 |
| 9 | B4 技术 probe 与正式结果分开，无 native oracle 冒充 | 是 | probe 结果树已删、manifest 与 §7A 测量记录存本批证据目录（`p2_1_b4_snapshot.md` §3）；B4 不含 oracle，`formal_eligible` 35/35 True（同快照 §7.2） |
| 10 | Self-PU 消融变体与未获方法学合作者复核已披露 | 是 | `p2_1_b3ab_snapshot.md` §2 |
| 11 | 所有失败、重试与最终状态可追溯 | 是 | 五批失败记录均为 0；B4 的 `failure_index.txt` 为空，退出码 `0`、`0 still pending` |
| 12 | 正式 / partial / 技术 probe / 历史结果状态标签清楚 | 是 | 词表已由实验负责人 2026-10-02 确认（`P2.2_B5_B6_implementation_plan.md` §11.2）：`formal` / `partial` / `technical_probe` / `historical` / `refused` / `incomplete` / `not_reproducible`，配 `status` + `reasons` 两字段，判定式 `status == formal` ⟺ `reasons == []`。P2.2 汇总已按该闭集分层输出（180 formal / 3 partial / 0 diagnostic） |
| 13 | 未使用测试集真值选择先验、阈值或超参数 | 是 | 代码级证据见 §2 |

## 2. 无测试集泄露：代码级证据

这一项不是靠约定，而是**结构上被挡住**，且可复核：

- **顺序**：`runner.py:480` —— `# 5. independent test evaluation (test never entered selection/training)`。
  选模是步骤 4、测试打分是步骤 5，打分时选谁已经定死。
- **视图隔离**：`runner.py:458-459` —— `val_view = clean_val if isinstance(proto, ProtocolOA) else pu_val_view`。
  两个准则分别拿到 `clean_val` / `pu_val_view`，**`test` 谁都碰不到**。数据按
  `DatasetBundle(train=, pu_val=, clean_val=, test=)` 四角色分开传递。
- **第二道防线**：`strategies.py` 中 `ProtocolOA.select` 要求 `view == "clean"` 且 `for_selection`；
  `ProtocolPA.select` 要求 `view == "pu"` 且 `for_selection`，注释写明 **never clean labels**。
  接错视图会**抛异常**，不会静默算出一个数。
- **阈值**：`strategies.py` 中 OA 的文档字符串说明归一化仿射常数在**验证侧**记录、同一变换应用到
  测试分数；若在测试集上重新归一化会使阈值漂移（代码内称 F1 fix）。
- **先验与超参数不是「选」出来的**：人口先验读自冻结的 `split_manifest.json`
  （`class_prior.population`）；`c`、backbone、epoch 数来自冻结协议 JSON，属先注册，
  不存在调参这一步。

边界说明：`pn_oracle` 的选区是 `clean_val`（真标签），那是 oracle 的定义——它是有监督上界，
其**测试集同样未参与选模**。

## 3. 待办

- **B4 已完成**（2026-10-03）：第 1、2、3、4、5、9、11 项的 B4 行已按实测补齐；§7A 测量记录与
  回收实测见 `p2_1_b4_snapshot.md` §3–§4 与 §7（单 run 占用 8.34 GiB → 42.5 MiB，终态 38 个
  选中权重，判据 `reclaimed + files_on_disk == refs` 对 35 行全部成立）；
- **P2.2 产出方已完成**：五批白名单与三个入口产物见 [交付记录](p2_2_delivery.md)。接收端仍需取得原始制品重放核验，不再将首次产出列为待办；
- **清理未执行**：B2 / B3a / B3b 的未选中 checkpoint 至今未清理；16 号 §9 的门禁项已具备
  （归档双侧各一份、目标端校验通过），是否清理另行决定；
- **B6 阶段**：第 12 项状态标签已于 2026-10-02 确认，剩余独立复核与本人签署。
