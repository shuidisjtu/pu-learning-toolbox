# B3a 批次阶段性快照（P2.1）

状态日期：2026-09-30。定位：P2.1 第三个完成批次的**阶段性分析快照**——清单层与门禁层的
可审计记录。**不含**方法排名、跨方法对比与数值裁决，那些属 P2.2。

## 1. 批次身份

| 项 | 值 |
|---|---|
| 批次 | B3a（`cifar10`，`cnn_feature_adapter` 路径） |
| runs | 110 / 110（与计划一致） |
| 方法 | `lbe` / `pusb_kernel` / `upu` 各 35；`pn_oracle` 5 |
| 协议 | `survey-v1.2` |
| 制品 | `B3a_manifests.tar.gz`（227 KiB，**仅 manifest 与方法台账条目**） |

**收尾证据**：日志末行 `completed 110 of 110 run(s) in cifar10; 0 still pending`，
manifest 计数 110，退出码文件 `B3a_exit_code.txt` 为 `0`。

> **更正（2026-10-01）**：本节原写「**本批未产出退出码文件**（dry-run 那一步有
> `B3a_dry_run_exit_code.txt`），故本快照不引用退出码——验收以日志的 completed 行与
> manifest 计数为准，不假称有退出码」。该判断有误。实测
> `/root/autodl-tmp/pu-survey-logs/B3a_cifar10_adapter_classical_oracle/B3a_exit_code.txt`
> **存在**：2 字节、内容 `0`、mtime 2026-09-30 20:00。更正后 13 号 §9 完成判定的第一条
> （真实退出码为 0）对 B3a 同样成立，不必只依赖日志的 completed 行。

## 2. 清单层

**110 个 run 全部成功**，失败记录 0。

| 维度 | 分布 |
|---|---|
| 视图 | `ts-compatible` 70；`os-compatible` 40 |
| `calibration_applied` | True 70；False 40（**与视图严格对应，无交叉**） |
| 机制 | `scar` 45；`sar_lbe_a` 30；`sar_lbe_b` 30；`pn_oracle` 5 |
| 选择产物 | PA + OA 45；仅 OA 65 |
| `formal_eligible` | True 105；False 5 |

机制分布可逐方法还原：`lbe` / `pusb_kernel` / `upu` 各 `15 scar + 10 sar_lbe_a` +
`10 sar_lbe_b = 35`，加 `pn_oracle` 5 即 110。**`lbe` 的 35 个全在 os 视图**，
两个核方法的 70 个全在 ts。

选择产物的分布同样与协议规则对上：45 个含 PA + OA 的正是 SCAR 全部；65 个仅 OA 的是
SAR 60（PA 仅作诊断日志，协议 §2.3）加 `pn_oracle` 5（OA only，协议 §159-161）。

## 3. 门禁层

| 项 | 值 |
|---|---|
| 组 | **7** |
| 单元 | **75** |
| `comparable` | 70 |
| `blocked` | 5 |
| `refused` | 0 |
| `formal_ready` | **False** |

组的构成：

| `comparability_group` | 视图 | 机制 | 单元 | 方法 |
|---|---|---|---|---|
| `classical` | os | scar / sar_lbe_a / sar_lbe_b | 15 / 10 / 10 | `lbe` |
| `classical` | ts | scar / sar_lbe_a / sar_lbe_b | 15 / 10 / 10 | `pusb_kernel`, `upu` |
| `minibatch` | os | pn_oracle | 5 | `pn_oracle` |

**只有 7 组是方法构成的必然结果**：四个方法只落在两个预算族上（`classical` 与 `minibatch`），
而 `lbe` 与两个核方法虽同族却分属 os / ts，故各成一榜——这正是 D13 的作用。

唯一阻断来源同 B1 / B2：5 个 `pn_oracle` 的 `protocol_deviation: ['c_grid']`。这是真实偏差，
D24 保留 `protocol_deviation` 正为此，故 `formal_ready=False` 是**正确行为**。

**`per_epoch_independent_PA_OA_checkpoint_selection` 命中 0**：该动态阻断位按实际轨迹解除，
而本批确实产出了完整快照（见 §4）。

**三批合并聚合**（同一根下 540 份 manifest）：

```
39 组 = spambase 16 + imdb 16 + cifar10 7
435 单元
refused: 0
```

三批各自成组、互不污染。

## 4. 制品与 checkpoint 形态

| 项 | 值 |
|---|---|
| 结果树 | 257 MiB |
| epoch checkpoint | **1000 个，全部来自 `pn_oracle`**（5 个 run × 200） |
| 选中权重引用 | 5 个，**盘上缺失 0** |

**本批的 checkpoint 只来自 `pn_oracle`**，其余 105 个 run 一个都没有。这不是捕获失效：
`lbe` / `pusb_kernel` / `upu` 未声明 `epoch_components`（在固定特征上拟合，没有 epoch 概念），
本就不产逐 epoch 快照。本地 B1 的对照可证——同样这三个方法在 B1 里也是 0，而
`dist_pu` / `nnpu` / `self_pu` / `pn_oracle` 各 200（`self_pu` 因双 teacher 为 400）。

**对 D25 的含义**：回收在本批上**几乎没有收益**（257 MiB 中只有 oracle 那 5 个 run 的权重可回收）。
B3a 走 adapter 路径、只训练 MLP head，量级本就小；真正的容量压力仍只在 B4 的 `native_cnn`。
这与把 D25 的紧迫性绑定在 B4 上的既有判断一致。

## 5. 已知现象：`ConvergenceWarning`

本批日志出现 **120 次** `ConvergenceWarning`（日志共 1322 行），对应 60 个 SAR run，
约 **2 次/run**（train 与 pu_val 各一次标注生成）。来源是 SAR 标注机制生成器的
`LogisticRegression(max_iter=100)`（`strategies.py:54`），**不影响任何 run 的选模结果**。

B1 / B2 计数为 0 而本批出现，合理解释是特征维度：CIFAR-10 的 adapter 特征维度远高于
spambase / imdb，lbfgs 在该尺度下才不收敛。**预期它在图像数据集上会持续出现**，
后续批次（B3b / B4）与 P2.2 应把它当作已知现象，而不是偶发故障。

## 6. 量化指标现状

字段结构与覆盖口径同 B1（见 [`p2_1_b1_snapshot.md`](p2_1_b1_snapshot.md) 的「量化指标现状」一节）。
本批：OA 指标 110 个全覆盖；PA 指标 45 个（= SCAR 全部）。5 次重复的均值与标准差仍属 P2.2，
**但聚合口径不需再定**——协议 §5 第 2 条已冻结（组内含 `c`、组内跨 5 个 seed），见
[`p2_1_b1_snapshot.md`](p2_1_b1_snapshot.md) §5 的更正。

## 7. 本快照不含

- 方法间排名、对比与数值裁决（P2.2）；
- 与参考文献的交叉验证结论（P2.2）；
- 期望单元完整性格网（口径未定，见 D13 遗留项）。

## 8. 待办

- **B3b**：已完成，与 B3a 的**合并分析**见 [`p2_1_b3ab_snapshot.md`](p2_1_b3ab_snapshot.md)；
- **§9 补验**：已于 2026-09-30 补做（`planned 110 / completed 110 / pending 0`，退出码 0）；
- **B4**：D25 回收逻辑第一个有实质收益的用例，回收已合入 `main`。
