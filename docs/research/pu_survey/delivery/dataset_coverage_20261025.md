# 数据集覆盖（第一片）：connect_4 与 twenty_newsgroups（2026-10-25）

计划 §1.2「数据集覆盖」的第一片。口径按用户 2026-10-25 晚的决定：协议与比较矩阵**原地扩充**，
不新增版本号产物——`survey-v3.0` 仍是 `survey-v3.0`，比较矩阵仍是 `survey-comparison-v6`。

## 加了什么

- **执行单元**：64 → 72 行。`connect_4` 与 `twenty_newsgroups` 各加 4 个方法：`upu`（闭式解）、
  `pusb_kernel`（核 CV）、`lbe`（EM）、`nnpu`（mini-batch）。选择依据是「在 spambase 上已有可运行
  `native_2d` 行的方法」——同一方法、同一训练路径、只换数据集，不新造预算。
- **表示**：`connect_4` 用 `train_zscore`（126 维表格特征，48,640 行）；`twenty_newsgroups` 用
  `sbert384`（384 维文本嵌入，10,182 行），与 imdb 一致。
- **比较矩阵**：新增 **80** 个单元映射，全部 `no_direct_anchor`（没有任何统一基准报这两个数据集），
  每条带理由；锚点仍 64 条，判定规则不变。
- **绑定**：协议摘要 `f0334b10…`、v6 `17db3aca…`、map v3 `1cfa80ac…`、clean-support 清单
  `protocol_policy_sha256` `e2132a79…` / `receipt_sha256` `4c93461b…`。历史矩阵（v5、map v1/v2）
  只同步了它们记录的协议摘要，内容未动。

## 跑通证明（不是效果结论）

`connect_4` 上 `upu`、SCAR `c=0.1`、seed 0、CPU：

```
uv run python scripts/run_survey_experiment.py data/splits/connect_4/split_0 \
  --protocol survey-v3.0 --dataset connect_4 --method upu \
  --class-prior 0.6583059210526315 --c 0.1 --seeds 0 --out-dir tmp/connect4_smoke_20261025
```

- manifest：`protocol_version=survey-v3.0`、`comparison_registration_status=mapped_inherited`，
  比较块 `survey-comparison-v6`，PA/OA 均 `no_direct_anchor`；
- 读数：PA accuracy 0.7753、OA accuracy 0.7798（单 seed、单 c，仅证明链路）；
- `formal_eligible=false`，blockers 与其它单元一致（多 seed 批次未跑是其中之一）。

隐私与口径：`--class-prior` 用的是该 split 的 train 正例率，属工程输入，不是从 test 反推；
单 seed 读数不进效果表。

## 覆盖计数

| 项 | 之前 | 现在 |
|---|---:|---:|
| 执行单元行 | 64 | 72 |
| 数据集 | 3 | 5 |
| 结果单元 / 单元映射 | 359 | 439 |
| 锚点 | 64 | 64 |
| `outside_matrix` 行 | 19 | 19 |

## 还没做

- **图像数据集**（`mnist`、`fashion_mnist` 28×28×1；`adni` 128×128×1）：`backbone_specs["image"]`
  目前是 CIFAR-10 的平铺规格（32×32×3），加行之前要补 `per_dataset` 覆盖与对应的图像预处理，
  否则会静默按 32×32 处理。这一片刻意没碰。
- **每数据集的方法数**：现在每个新数据集只有 4 个方法（计划判据是「至少一行 runnable」）。
  扩到全部可运行 profile 只差在同一条规则上继续加行。
- **多 seed 与正式读数**：属 HENG958 的 §1.5 批次，不在本片范围。
