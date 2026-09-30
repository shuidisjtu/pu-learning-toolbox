# B2 批次阶段性快照（P2.1）

状态日期：2026-09-30。定位：P2.1 第二个完成批次的**阶段性分析快照**——清单层与门禁层的
可审计记录。**不含**方法排名、跨方法对比与数值裁决，那些属 P2.2。

## 1. 批次身份

| 项 | 值 |
|---|---|
| 批次 | B2（`imdb`，文本表示路径） |
| runs | 215 / 215（与计划一致） |
| 协议 | `survey-v1.2` |
| 协议摘要 | `287c2f45387f02714e5925b35dcb04f3f64e1e3faf8f740cab861fc5747dd84a` |
| 正式资格 | `formal_eligible` True 210 / False 5 |
| 制品 | `B2_manifests.tar.gz`（3.1 MiB，**仅 manifest 与方法台账条目**，不含权重） |

215 份 manifest 的 `protocol_sha256` 单值一致，与 B1 及当前冻结摘要相同。

## 2. 清单层

**215 个 run 全部成功**，失败记录 0。

| 维度 | 分布 |
|---|---|
| 方法 | `dist_pu` / `lbe` / `nnpu` / `pusb_kernel` / `self_pu` / `upu` 各 35；`pn_oracle` 5 |
| 视图 | `ts-compatible` 175；`os-compatible` 40 |
| `calibration_applied` | True 175；False 40（与视图逐一对应） |
| 机制 | `scar` 90；`sar_lbe_a` 60；`sar_lbe_b` 60；`pn_oracle` 5 |
| 选择产物 | PA + OA 90；仅 OA 125 |

**与 B1 逐项同构**（对照 [`p2_1_b1_snapshot.md`](p2_1_b1_snapshot.md)），选择产物的分布也在内：
90 个含 PA + OA 对应 SCAR 全部 run，125 个仅 OA 对应 SAR 120 + `pn_oracle` 5，
仍与协议 §2.3 及 §159-161 的规则逐一对上。

test 指标的字段结构与覆盖同 B1（OA 全覆盖、PA 覆盖 90 个 SCAR run，其余 125 个无 PA 指标
而非缺失），逐 run 罗列样例见 [`p2_1_b1_snapshot.md`](p2_1_b1_snapshot.md) 的「量化指标现状」一节。

## 3. 门禁层

| 项 | 值 |
|---|---|
| 组 | 16（`imdb/native_2d/` 下四个预算族 × 视图 × 机制） |
| 单元 | 180（`(seed, c)`） |
| `comparable` | 175 |
| `blocked` | 5 |
| `refused` | 0 |
| `formal_ready` | **False** |

唯一阻断来源与 B1 相同：5 个 `pn_oracle` 的 `protocol_deviation: ['c_grid']`——真实偏差，
正是 D24 保留 `protocol_deviation` 要标的情形，故 `formal_ready=False` 是**正确行为**。

**两批合并聚合的结果**（同一根下 430 份 manifest）：

```
32 组 = imdb 16 + spambase 16
360 单元 = 180 x 2
refused: 0
```

即两批**各自成组、互不污染**：`comparability_group` 以数据集为前缀，B1 与 B2 不会落入
同一个公平性单元。

复现：

```
uv run python scripts/aggregate_survey_runs.py <B1 与 B2 结果所在的根>
```

`--diagnostic` 只放宽正式资格，两道公平性门禁在两种模式下都跑。

## 4. 制品状态

**本批未导出权重。** 打包时刻意只取 `manifest.json` 与 `method_ledger_entry.json`
（215 + 210 = 425 个文件，3.1 MiB），避免重演 B1 那 806 MiB 的全量快照负担。
`pn_oracle` 不产出方法台账条目，故台账比 manifest 少 5 份——与 B1 一致，不是缺失。

权重仍留在训练主机的结果树上，**尚未取回、也尚未回收**。若后续需要模型级复核，按 D25 的
L2 语义只取被 `selection` 指向的选中权重即可。

## 5. 本快照不含

- 方法间排名、对比与数值裁决（P2.2）；
- 与参考文献的交叉验证结论（P2.2）；
- 期望单元完整性格网（口径未定，见 D13 遗留项）。

## 6. 待办

- **§9 补验**：已于 2026-09-30 补做（`planned 215 / completed 215 / pending 0`，退出码 0）；
  §6 对账的身份字段与参考计划逐值相同，仅 `totals` 的进度分量不同。汇总见
  [`p2_1_b3ab_snapshot.md`](p2_1_b3ab_snapshot.md) §10。
