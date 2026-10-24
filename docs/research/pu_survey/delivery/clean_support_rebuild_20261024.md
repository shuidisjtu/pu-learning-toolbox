# Clean-support 25 份制品本机重建（2026-10-24）

10-22 批次的 25 份制品在队友机器（`v3_freeze_20261024.md` 记「不在本机」）。本轮用公共入口
`scripts/prepare_clean_support.py` 在本机重建同一成员口径，供 v3 正式 support 批次使用。
只做数据准备与 CPU 链路验收，不是正式预算、GPU 批次或方法准入。

## 结果

- 制品根：`data/clean-support-20261024-u5axe92t`，25 份，1,120,586,182 字节（1.044 GiB）。
- 每份仍是 `<dataset>/split_<seed>/` 下 `train/pu_val/clean_val/test/support` 五个 NPZ 加
  `support_manifest.json`（`independent-clean-support-v1`）。
- 机器清单：[`../data/clean_support_materialization_20261024.json`](../data/clean_support_materialization_20261024.json)。
  Git 只登记清单，不含 NPZ、原始数据或模型。
- 参数与 10-22 一致：Spambase support 100、其余 500；`support_seed = split_seed`；seeds 0–4。
- 全量构建 2.1 分钟。本机 python 进程写 `data/` 仍 `EACCES`，因此先建在 `tmp/` 暂存目录，
  再由 shell 移入制品根，之后从最终路径重新逐份校验。

## 与 10-22 批次的关系：新收据，不是传输

| 项 | 结论 |
|---|---|
| raw reader 字节 | 9 个 reader 文件 SHA-256 与 10-22 记录逐位一致 |
| 准备脚本源码 | LF 归一化摘要与 10-22 记录一致（本工作树 `core.autocrlf=true`，裸文件摘要是 CRLF 版本） |
| 成员口径 | **25/25** 一致：角色规模、support 正负标签数逐份相同 |
| 收据 / NPZ 摘要 | **不同**：本机 numpy 2.4.6 重压缩 + 本机 provenance 下载时间戳，属新收据 |
| 公共 CLI 复现 | `spambase seed0` 重建的 receipt 与五个 NPZ 摘要与批量产物逐位一致 |

所以这批不能冒充 10-22 摘要清单里那 25 个收据的跨机副本；两套收据各自完整、各自可验。

## 主 CLI 消费方式

`data/splits/<dataset>/split_<seed>` **不能**直接与制品配对：制品把 support 从 train 移除了
（Spambase 3312→3212），CLI 会在 `_require_same_support_split` 报
`support artifact train does not match the loaded split`。正确用法是把制品目录同时作为
`data_dir`，`--split-ref` 指向制品自身 manifest（其 `dataset`/`seed`/`role_sizes` 正是本次运行所用）：

```bash
prod=data/clean-support-20261024-u5axe92t/spambase/split_0
uv run python scripts/run_survey_experiment.py "$prod" \
  --protocol pu_toolbox/experiment/survey_protocol_v3.json \
  --method lagam --dataset spambase --seeds 0 --c 0.5 \
  --support-dir "$prod" --split-ref "$prod/support_manifest.json" --device cpu
```

`self_pu_full_meta` 另需 `--class-prior 0.39404477287546186`（Spambase 制品的 population π）。
`--support-dir` 支持 `{seed}` 占位，批量时指向 `<root>/spambase/split_{seed}`。

## 冒烟（单 seed、默认预算、CPU；只证明链路）

三个 clean-support 方法都从重建制品读入并完成 PA+clean-support / OA 双轨选择与回放：

| 方法 | 训练视图 | 耗时 | PA+clean-support acc | OA acc |
|---|---|---:|---:|---:|
| self_pu_full_meta | ts | 8.0s | 0.901 | 0.902 |
| lagam | os | 17.4s | 0.926 | 0.927 |
| pulns | os | 30.6s | 0.934 | 0.935 |

读数与 10-22 那批不同是因为走的是 v3 的 profile 预算，不是同一套候选/轮次；单 seed 读数
不当效果结论，manifest 里 `formal_eligible=false`。

## 复现

单份（输出目录必须不存在）：

```bash
uv run python scripts/prepare_clean_support.py --dataset spambase --raw-dir data/raw \
  --seed 0 --support-seed 0 --support-size 100 --group-policy not_applicable \
  --out-dir data/clean-support-new/spambase/split_0 --write
```

全量循环与 10-22 文档相同（`support_size`：spambase 100，其余 500；`--raw-dir data/raw`）。

## 边界

`formal_admission=false`、`formal_budget_approved=false`、无签署。`v2_p4_clean_support_policy`
与 `v2_p10_self_pu_scope` 已在 `survey-v3.0` 置 accepted，新清单因此记
`protocol_pending_review_items: []` 并把协议指向 v3。仍待：正式 units/预算绑定、跨机传输核验、
GPU/多 seed 正式批次。
