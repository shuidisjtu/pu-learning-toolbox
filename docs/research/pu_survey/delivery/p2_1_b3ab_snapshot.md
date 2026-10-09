# B3a+B3b 合并快照（P2.1）

状态日期：2026-09-30。定位：`cifar10 / cnn_feature_adapter` **分析组**的阶段性快照——
清单层、门禁层、制品与 adapter cache 的可审计记录。

**本快照不输出 adapter 组级方法排名。** 按执行指南 §12.1，B3a 与 B3b 必须作为同一
`cifar10/cnn_feature_adapter` 分析组处理，B3a 单独只能做健康审计。本快照是两批合并后的
首份记录，仍**不含**方法间比较与数值裁决，那些属 P2.2。

## 1. 分析组身份

| 项 | B3a | B3b |
|---|---|---|
| runs | 110 / 110 | 70 / 70 |
| 方法 | `lbe` / `pusb_kernel` / `upu` 各 35；`pn_oracle` 5 | `dist_pu` 35；`self_pu` 35 |
| 视图 | `ts-compatible` 70；`os-compatible` 40 | `ts-compatible` 70 |
| 机制 | `scar` 45；`sar_lbe_a` 30；`sar_lbe_b` 30；`pn_oracle` 5 | `scar` 30；`sar_lbe_a` 20；`sar_lbe_b` 20 |
| 协议 | `survey-v1.2` | 同 |
| 正式资格 | True 105 / False 5 | True 70 / False 0 |

- 协议摘要：`287c2f45387f02714e5925b35dcb04f3f64e1e3faf8f740cab861fc5747dd84a`（两批 180 份 manifest **单值一致**）
- 分析代码身份：`4fbecd1c3668faebf418e6b44c9fdf380a5a91ae`

## 2. 清单层（B3b）

70 个 run 全部成功，失败记录 0。

| 维度 | 分布 |
|---|---|
| `execution_mode` | `versioned_pilot` 70 |
| `run_view` | `ts-compatible` 70 |
| `calibration_applied` | True 70 |
| `protocol_deviation` | 空（`c_grid` 命中 0） |
| 选择产物 | 70 份全有 `selection` |
| `method_variant` | `shared_spec_engineering`（`dist_pu`）；`without_clean_validation_meta_reweighting`（`self_pu`） |

**B3b 是四批里唯一没有阻断位的批次**：B1 / B2 / B3a 各有 5 个 `pn_oracle` 带
`protocol_deviation: ['c_grid']`，而 B3b 不含 oracle。这是构成的必然结果，不是门禁放松。

`self_pu` 用的变体是**无 clean-validation 元重加权的已裁决消融变体**（指南 §1）。
结果呈报必须披露，不能称为完整版论文复现。

> **补记（2026-10-09）**：交接清单第 10 项引用本节，称「未获方法学合作者复核」已披露，但本节原文只披露了消融变体。
> 此处补上：该变体与 `self_pu` 元重加权的方法学合作者复核**未获**，P2.2 复核包仍待签署，
> 见 [`p2_2_review.md`](p2_2_review.md) §3 与 [`p2_2_delivery.md`](p2_2_delivery.md) §8。本快照其余内容未改。

## 3. 门禁层

180 份 manifest 合并聚合（自 D26 起分组含 mechanism）：

| 项 | 值 |
|---|---|
| 组 | **13** |
| 单元 | **145** |
| `comparable` | 140 |
| `blocked` | 5 |
| `refused` | 0 |
| `formal_ready` | **False** |

组的构成（`comparability_group` × `run_view` × `mechanism`）：

| 预算族 | 视图 × 机制 | 组数 | 单元 | 方法 |
|---|---|---|---|---|
| `classical` | os × 3 + ts × 3 | 6 | 75 | `lbe`（os）；`pusb_kernel` / `upu`（ts） |
| `fullbatch` | ts × 3 | 3 | 35 | `dist_pu` |
| `minibatch` | os × 1 | 1 | 5 | `pn_oracle` |
| `two_student_sampled` | ts × 3 | 3 | 30 | `self_pu` |

**一条结构性事实**：`dist_pu` 与 `self_pu` 分属 `fullbatch` 与 `two_student_sampled`
两个**不同的预算族**，两者之间本身就不构成同一张榜。adapter 组内因此不存在可比的
跨方法公平性单元——这与 §12.1「不输出组级排名」的方向一致，但成因在构成层面，
不是口径上的限制。

唯一阻断来源同前三批：5 个 `pn_oracle` 的 `protocol_deviation: ['c_grid']`。真实偏差，
正是 D24 保留 `protocol_deviation` 要标的情形，故 `formal_ready=False` 是**正确行为**。

复现：

```
uv run python scripts/aggregate_survey_runs.py <B3a 与 B3b manifest 的合并根>
```

`--diagnostic` 只放宽正式资格，两道公平性门禁在两种模式下都跑。

## 4. 制品与 checkpoint 形态

| 项 | B3a | B3b |
|---|---|---|
| 结果树 | 257 MiB | 5.3 GiB |
| epoch checkpoint | 1000 个（全部 `pn_oracle`，5 × 200） | **21000 个** |
| checkpoint 实测 | — | **5.196 GiB** |
| 选中权重引用 | 5 | **100**（去重 85） |
| 盘上缺失 | 0 | **0** |

**B3b 的 checkpoint 构成**：`self_pu` 14000 个（35 run × 400，双 student，形态同 B1）、
`dist_pu` 7000 个（35 run × 200）。平均 **265 KB/文件**，与 D15 修正后「adapter 路径只训
MLP head」的量级一致。

**容量对账**：dry-run 预估 **10.3 GiB 是保守上界**（其输出含 `reserving 2 attempt(s) per
candidate` 的重试预留），实际落盘 **5.196 GiB**——`du`（5.3G）与清单逐文件求和
（5.196 GiB）两法互证。D25 的收益按实测值算，不按预估值。

**选中引用 100、去重 85**：100 = 30 个 SCAR run ×（PA + OA）+ 40 个 SAR run × OA；
差额 15 是 15 个 SCAR run 的 PA 与 OA 选中了**同一个** checkpoint（并列规则下校准后
PA/OA 趋同），不是重复落盘。

**归档**：`B3b_cifar10_self_dist_20260930_210009.tar.gz`（4.8 GB，`sha256`
`ed97a3ab2143c40e6554315f369843647d3b2ad572751e050c4cc6cc9bc3ba5b`）；manifest 子集包
`B3b_manifests.tar.gz`（2.2 MiB / 140 文件）已取回本地。

## 5. adapter cache 审计

`/root/autodl-tmp/pu-survey-cache/cifar10_feature_adapter`，**586 MiB**，5 个 key 目录，
每个含 `features.npz` 与 `adapter.json`。

| 项 | 值 |
|---|---|
| `feature_version` | `survey-v1/random-frozen-resnet18` |
| `feature_dimension` | 512 |
| `initialization` / `weights` | `random` / `null` |
| `encoder_fit_scope` | `fixed_external` |
| `adaptation_level` | `benchmark-adapted` |
| key ↔ seed | **5 个 key 对应 seed 0–4，每个 seed 一个独立初始化的 encoder** |

5 个 key 的 `encoder_state_sha256` **互不相同**（`bf068305…` / `2d9935c2…` /
`04e90d49…` / `4c0321d5…` / `7062e2dd…` 各属一个 seed）。因此「两批共用同一份冻结特征」
应精确表述为：**同 seed 下 B3a 与 B3b 命中同一 key，跨 seed 不共用**。B3b 能在约 12 分钟
内跑完 70 个 run，正是直接命中 B3a 已建好的 cache。

归档 `cifar10_feature_adapter_20260930_211528.tar.gz`（`sha256`
`c1d5d01e25378e8a39babed28cc2a2b93acd8f37d4b86c8c023462c05ab6d992`），校验 `OK`。
**cache 已于 2026-10-01 清理**。§12.1 的四项前置条件彼时全部满足：B3a 验收完成、B3b 跑完、
cache key 与特征摘要及两批 manifest 审计完成（见本节）、adapter cache 独立归档并校验通过
（`c1d5d01e…`）。清理前另按 16 号 §9 做了可恢复抽查：`tar -tzf` 确认归档内确为
`cifar10_feature_adapter/<key>/{features.npz,adapter.json}` 结构，且目录 key 与本节记录的
`26c9f820ae…` 一致。释放 586 MiB。

## 6. 量化指标现状（**非裁决**）

本节只**罗列**每份 manifest 已记录的 test 指标，**不做**跨 seed 或跨 c 的聚合，
**不做**方法间比较，**不构成** P2.2 的数值裁决。列在这里是为了回答「数据齐不齐」，
而不是「结果如何」。

**字段结构**（同 B1）：每份 manifest 的 `test_results` 按选择协议分组：

```json
"test_results": {
  "PA": {"accuracy": ..., "auc": ..., "auc_unavailable_reason": null},
  "OA": {"accuracy": ..., "auc": ..., "auc_unavailable_reason": null}
}
```

**覆盖**（B3a 110 + B3b 70 = 180 份）：

| 项 | 值 |
|---|---|
| manifest 总数 | 180 |
| 含 OA 指标 | **180**（全覆盖） |
| 含 PA 指标 | **75** = B3a 45 + B3b 30（两批 SCAR 的全部 run） |
| 无 PA 指标 | **105** = B3a 65 + B3b 40（SAR 全部 + `pn_oracle`） |

无 PA 指标**不是缺失**：SAR 下 PA 仅作诊断日志、不产出正式选择产物，`pn_oracle` 为
OA only。这与门禁层的 `selection` 分布逐一对上。

**逐 run 罗列样例**（各方法 `scar` / `c=0.1`，五个 seed）：

`dist_pu`（B3b，`fullbatch`，`ts-compatible`）：

| seed | PA accuracy | PA AUC | OA accuracy | OA AUC |
|---|---|---|---|---|
| 0 | 0.5961 | 0.5581 | 0.5983 | 0.5731 |
| 1 | 0.6935 | 0.7590 | 0.6978 | 0.7649 |
| 2 | 0.7215 | 0.7729 | 0.7241 | 0.7779 |
| 3 | 0.7392 | 0.8020 | 0.7358 | 0.8027 |
| 4 | 0.6684 | 0.7446 | 0.6852 | 0.7444 |

`self_pu`（B3b，`two_student_sampled`，`ts-compatible`）：

| seed | PA accuracy | PA AUC | OA accuracy | OA AUC |
|---|---|---|---|---|
| 0 | 0.5997 | 0.5266 | 0.5997 | 0.5266 |
| 1 | 0.5999 | 0.3439 | 0.6000 | 0.3439 |
| 2 | 0.6705 | 0.6987 | 0.6722 | 0.6943 |
| 3 | 0.6077 | 0.5830 | 0.6075 | 0.5870 |
| 4 | 0.5998 | 0.3906 | 0.5997 | 0.3906 |

`lbe`（B3a，`classical`，`os-compatible`）：

| seed | PA accuracy | PA AUC | OA accuracy | OA AUC |
|---|---|---|---|---|
| 0 | 0.5824 | 0.3958 | 0.6000 | 0.3958 |
| 1 | 0.5968 | 0.3804 | 0.5993 | 0.3804 |
| 2 | 0.5982 | 0.3869 | 0.5999 | 0.3869 |
| 3 | 0.5999 | 0.3845 | 0.5996 | 0.3845 |
| 4 | 0.6000 | 0.3829 | 0.5999 | 0.3829 |

`pusb_kernel`（B3a，`classical`，`ts-compatible`）：

| seed | PA accuracy | PA AUC | OA accuracy | OA AUC |
|---|---|---|---|---|
| 0 | 0.7338 | 0.8110 | 0.7417 | 0.8110 |
| 1 | 0.7401 | 0.8088 | 0.7422 | 0.8088 |
| 2 | 0.7137 | 0.8140 | 0.7474 | 0.8140 |
| 3 | 0.7510 | 0.8181 | 0.7476 | 0.8181 |
| 4 | 0.7223 | 0.8002 | 0.7253 | 0.8002 |

`pn_oracle`（B3a，`minibatch`，`os-compatible`；`c` 为空，OA only）：

| seed | PA accuracy | PA AUC | OA accuracy | OA AUC |
|---|---|---|---|---|
| 0 | — | — | 0.8546 | 0.9250 |
| 1 | — | — | 0.8499 | 0.9237 |
| 2 | — | — | 0.8638 | 0.9318 |
| 3 | — | — | 0.8503 | 0.9205 |
| 4 | — | — | 0.8494 | 0.9257 |

完整罗列（B3a + B3b 共 180 行 × 12 列）由 manifest 的 `test_results` 逐行摊平得到，
**无聚合、无派生字段**：列为 `batch` / `dataset` / `method` / `training_path` / `run_view` /
`mechanism` / `c` / `seed`，加上 PA、OA 各自的 `accuracy` 与 `auc`。

**5 次重复的均值与标准差尚未计算**——它属 P2.2。聚合口径**不需再冻结**：协议 §5 第 2 条已明文
规定「……仅在同一数据集、`c`、协议和训练路径内，以独立 `test` Accuracy 五次重复均值比较；
同步报告标准差……」，即组内含 `c`、组内跨 5 个 seed 汇总。

> **更正（2026-10-02）**：本节原写「需先冻结口径（哪些 run 进统计、按 `seed` 还是 `(seed, c)`
> 聚合、5 个带 `protocol_deviation` 的单元如何处理），再行汇总」。该措辞与
> [`p2_1_b1_snapshot.md`](p2_1_b1_snapshot.md) §5 同源，同日一并更正：聚合维度已由协议
> §5 第 2 条冻结，不是待决项。

## 7. 已知现象：`ConvergenceWarning`

| 批 | 次数 | SAR run 数 | 比例 | 日志行数 |
|---|---|---|---|---|
| B3a | 120 | 60 | 2.0 / run | 1322 |
| B3b | 80 | 40 | 2.0 / run | 944 |

来源是 SAR 标注机制生成器的 `LogisticRegression(max_iter=100)`（`strategies.py:54`），
**不影响任何 run 的选模结果**。B1 / B2 计数为 0 而两批图像数据稳定出现，比例一致——
预期 B4 与 P2.2 同样会看到，应作为**已知现象**而不是偶发故障。

## 8. 模型复核层（B 层）策略

按 16 号 §4.2 的二选一：

| 批 | 选择 | 含义 |
|---|---|---|
| B3a | 远程保留完整批次归档 | 保留完整 epoch / 模型复核能力 |
| B3b | 远程保留完整批次归档（4.8 GB） | 同上 |

两批的全量归档在**数据盘暂存目录**与**文件存储**各存一份，本地只取回了 manifest 子集。
B3a 的归档为 `B3a_cifar10_adapter_classical_oracle_20261001_161352.tar.gz`
（246197696 B，sha256 `9e7396a2a88d3e8e976e6b12063f63167e6a4f747b359efa47016b319f7277a3`），
已于文件存储端以 `sha256sum -c` 回读校验通过。

> **更正（2026-10-01）**：本节原写「**两批的全量归档目前只存在于 AutoDL 数据盘**，本地只
> 取回了 manifest 子集。实例释放即丢失——这是归档方案里显式承担的取舍，不是疏漏」。
> 两处需更正，完整记录见
> [`p2_1_b5_review_checklist.md`](p2_1_b5_review_checklist.md) §2.1：
>
> - 该句**对 B3a 而言从来就不成立**——实测两侧 B3a 目录此前均为**空目录**，B3a 的 B 层当时
>   只有数据盘结果树一个副本。该缺口已于同日补做归档闭合（见上）。
> - B3b 的全量归档已于同日复制到
>   `/root/autodl-fs/pu-survey-backups/pu-survey-backup-stage/`（文件存储，独立于实例），
>   「实例释放即丢失」不再成立。

## 9. 本快照不含

- **adapter 组级方法排名**（§12.1 禁止）；
- 方法间比较、跨 seed / c 聚合与数值裁决（P2.2）；
- 与参考文献的交叉验证结论（P2.2）；
- 期望单元完整性格网（口径未定，见 D13 遗留项）。

## 10. 完成判定补验与待办

**§9 完成判定（2026-09-30 补验）**：四批的重新 dry-run 全部报 `planned = completed`、
`pending 0`、退出码 0——B1 215/215、B2 215/215、B3a 110/110、B3b 70/70。

**§6 对账**在 B1 / B2 上以「本次 recheck 生成的计划 vs 已验收参考计划」的方式补做：
五个身份字段（`snapshot_schema_version`、`source_protocol_sha256`、`selection`、
`execution_units`、`runs`）**逐值相同**；**仅 `totals` 的进度分量不同**——recheck 为
`completed 215 / pending 0`，参考计划为跑前的 `completed 0 / pending 215`，两侧 `planned`
都是 215。这正是 §11.4「completed/pending 可因进度变化而不同，协议、selection、
execution_units 与 runs 身份不得变化」所预期的差异，不是缺陷。

## 11. 待办

- **B4**：已于 2026-10-01 开跑。AutoDL 仓库已由 `7f445be` 推进到
  `ee54b5ba488674faf3b9e504611518471ab0a31b`（含 D25 回收，默认关闭、本批显式启用）；
  `uv.lock` 与协议摘要两项未变，dry-run 与计划对账（六项身份字段 `mismatches= []`）均通过；
- **adapter cache**：已于 2026-10-01 清理，依据与抽查过程见 §5。
