# B1 批次阶段性快照（P2.1）

状态日期：2026-09-30。定位：P2.1 首个完成批次的**阶段性分析快照**——清单层与门禁层的
可审计记录。**不含**方法排名、跨方法对比与数值裁决，那些属 P2.2。

## 1. 批次身份

| 项 | 值 |
|---|---|
| 批次 | B1（`spambase`，经典表格路径） |
| runs | 215 / 215（与计划一致） |
| 协议 | `survey-v1.2` |
| 协议摘要 | `287c2f45387f02714e5925b35dcb04f3f64e1e3faf8f740cab861fc5747dd84a` |
| 正式资格 | `formal_eligible` True 210 / False 5 |
| 制品 | `B1_spambase.tar.gz`（806 MiB，本地副本已校验 `sha256`）+ `B1_evidence.tar.gz` |

215 份 manifest 的 `protocol_sha256` **单值一致**且等于当前冻结摘要，证据包独立记录了同一值。

## 2. 清单层

**215 个 run 全部成功**，失败记录 0。

| 维度 | 分布 |
|---|---|
| 方法 | `dist_pu` / `lbe` / `nnpu` / `pusb_kernel` / `self_pu` / `upu` 各 35；`pn_oracle` 5 |
| 视图 | `ts-compatible` 175；`os-compatible` 40 |
| `calibration_applied` | True 175；False 40（与视图逐一对应） |
| 机制 | `scar` 90；`sar_lbe_a` 60；`sar_lbe_b` 60；`pn_oracle` 5 |
| `protocol_deviation` | 空 210；`['c_grid']` 5 |

**选择产物的分布与协议规则逐一对上，本身即一条可审计的正面证据**：

- 含 **PA + OA 两套**选择产物的 90 个 run = `scar` 全部——SCAR 是主实验，PA 正式使用；
- **仅 OA** 的 125 个 = `sar_lbe_a` 60 + `sar_lbe_b` 60 + `pn_oracle` 5——SAR 下 PA 仅作
  诊断日志（协议 §2.3），PN oracle 为 OA only、不得伪造 PA 结果（协议 §159-161）；
- `os-compatible` 的 40 个 = SAR 的 os 视图 20 + SCAR 的 15 + `pn_oracle` 5，与 D13
  「PN oracle 恒为 `os-compatible`」一致。

## 3. 门禁层

`scripts/aggregate_survey_runs.py` 按 comparability 与分榜两道门禁输出
（自 D26 起分组含 mechanism）：

| 项 | 值 |
|---|---|
| 组 | 16（`comparability_group` × `run_view` × `mechanism`） |
| 单元 | 180（`(seed, c)`） |
| `comparable` | 175 |
| `blocked` | 5 |
| `refused` | 0 |
| `formal_ready` | **False** |

**唯一的阻断来源**：5 个 `pn_oracle` run 带 `protocol_deviation: ['c_grid']`——它是
c-independent 行、不跑完整 c 网格。这是**真实偏差**，正是 D24 保留 `protocol_deviation`
这一阻断位要标的情形，故 `formal_ready=False` 是**正确行为，不是缺陷**。

复现：

```
uv run python scripts/aggregate_survey_runs.py <B1 结果根>
```

`--diagnostic` 只放宽正式资格，两道公平性门禁在两种模式下都跑。

## 4. 制品状态

解压产物按 **D25** 的 L2 / L3 分级回收过一次：

| 项 | 回收前 | 回收后 |
|---|---|---|
| `.pt` 权重 | 29000 个 / 904.8 MiB | 155 个 / 4.8 MiB |
| `manifest.json` | 215 | 215（未动） |
| 解压产物合计 | 945 MiB | 43 MiB |

保留的是 manifest `selection` 指向的选中权重（PA / OA 去重后 155 个），回收的是非选中的
逐 epoch 权重。回收**未改变任何选模结果**，也未改写 manifest——判据与须披露事项见随制品
存放的 `RECLAIMED.md`。

**已知且有意的不一致**：`candidate_runs[].epoch_checkpoints` 仍记录全部 epoch 的 `path`，
其中绝大多数文件已不在盘上；B1 的 manifest 产出早于 D25 的实现，**没有** `reclaimed` 标记。
**不得**据该字段判断文件是否存在，实际存在性以 `sha256` 校验或 `RECLAIMED.md` 为准。

## 5. 本快照不含

- 方法间排名、对比与数值裁决（P2.2）；
- 与参考文献的交叉验证结论（P2.2）；
- 期望单元完整性格网（口径未定，见 D13 遗留项）。

## 6. 待办

- **B2**：结果在训练主机上（215 份 manifest），取回后并入同一根，生成跨数据集的第二批快照；
- **§9 补验**：dry-run 对账（`completed=215` / `pending=0`）尚未在 B1 上补跑。
