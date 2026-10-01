# P2.1 → P2.2 交接清单（执行侧记录）

状态日期：2026-10-01。依据：执行手册 14 §8 的交接清单。

本文件逐项记录**证据**与**当前状态**，供 B6 勾选。它**不含**方法排名、跨方法对比与数值裁决——
那些属 P2.2。这里只回答「制品与记录是否齐备、可不可审计」。

## 1. 逐项状态

| # | 清单项 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| 1 | 五批覆盖 215/215/110/70/35 = 645 | 部分 | B1–B3b 见各批快照 §1；**B4 待跑完后核对** |
| 2 | 五批 `selection` / `execution_units` 与参考 JSON 一致 | 部分 | B4 已做 §6 对账，六项身份字段 `mismatches= []`；B4 待跑完复核 |
| 3 | 所有正式 manifest 协议摘要唯一且为冻结值 | 部分 | B1–B3b 已记；B4 待跑完统计 |
| 4 | 每个分析单元的阶段性快照与边界措辞已保存 | 部分 | B1 / B2 / B3a / B3a+B3b 已有；**B4 待写** |
| 5 | A 层制品、B 层策略与清理记录完整 | **缺口** | B1 有回收记录；B2 / B3a / B3b 的未选中 checkpoint **尚未清理**，归档内容亦未抽查；B4 的策略待写入快照 |
| 6 | 五批结果、日志、计划、备份目录相互隔离 | 是 | 目录按批独立，B3a 与 B3b 结果树分开 |
| 7 | B3a adapter cache / frozen feature 来源已审计 | 是 | `p2_1_b3ab_snapshot.md` §5；cache 已于 2026-10-01 清理 |
| 8 | B3b 70 个 TS 视图与校准字段已审计 | 是 | `p2_1_b3ab_snapshot.md` §2 |
| 9 | B4 技术 probe 与正式结果分开，无 native oracle 冒充 | 是（待写入快照） | probe 结果树已删、manifest 与 §7A 测量记录存本批证据目录；协议载明 `cifar10/pn_oracle/native_cnn` 为 `runnable=false` |
| 10 | Self-PU 消融变体与未获方法学合作者复核已披露 | 是 | `p2_1_b3ab_snapshot.md` §2 |
| 11 | 所有失败、重试与最终状态可追溯 | 部分 | B1–B3b 失败记录为 0；B4 待跑完 |
| 12 | 正式 / partial / 技术 probe / 历史结果状态标签清楚 | 部分 | probe 已分离并留档；标签体系待 B6 明确 |
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

- **B4 跑完后**：补第 1、2、3、4、11 项的 B4 行，并写入 §7A 测量记录与回收实测
  （单 run 占用 8.34 GiB → 42.5 MiB，判据 `reclaimed_true + files_on_disk == refs`）；
- **清理前置**：B2 / B3a / B3b 未选中 checkpoint 的归档内容抽查与可恢复抽查（16 号 §9），
  通过后方可清理；
- **B6 阶段**：明确第 12 项的状态标签体系。
