# 我方实验读数工作簿（2026-10-25）

把 2026-10-24 汇报里属于我方的实验任务、以及 10-25 新跑的读数，整理成与
`P2.1_results_645runs_20261008_corrected.xlsx` 相同的表结构，供学长直接查阅。
工作簿：`data/PU_Survey_results_20261025.xlsx`（`runs` / `主表` / `汇总` / `coverage` / `readme`
五个 sheet，31 行 run）。

**全部是单 seed 工程读数，不是效果证据。** `effect_evidence` 列恒为 `False`，
`formal_eligible` 全为 `false`，`missing_seeds` 恒为 `1,2,3,4`。表的作用是让每条读数可回溯到
manifest，不是排方法优劣。

## 最重要的发现：汇报里的数字都对，但没说 `c`

汇报正文没有写 `c`，而读数对 `c` 很敏感。重跑后所有汇报数字都逐位复现，只是分属不同 `c`：

| 汇报 | 条目 | `c` | 重跑 |
|---|---|---|---|
| 二.2 Self-PU 0.904 / 0.897 | `self_pu_full_meta` | 0.1 | 0.9045 / 0.8969 |
| 二.2 LaGAM 0.859 / 0.871 | `lagam` | 0.1 | 0.8588 / 0.8708 |
| 二.2 PULNS 0.844 / 0.878 | `pulns` | 0.1 | 0.8436 / 0.8784 |
| 二.3 uPU 基线 0.872 | `upu` | 0.5 | 0.8719 |
| 二.3 Robust 0.928 / 0.934 | `robust_pu` | 0.5 | 0.9283 / 0.9338 |
| 二.3 Split 0.940 / 0.923 | `split_pu` | 0.5 | 0.9403 / 0.9229 |
| 二.3 Holistic 0.941 / 0.932 | `holistic_pu` | 0.5 | 0.9414 / 0.9316 |

二.2 在 `c=0.1`、二.3 在 `c=0.5`——同一份汇报里混用了两个 `c`。工作簿因此把 `c` 单独成列，
任何一条汇报数字都能按 `c` 对上。

顺带交叉验证了确定性：`c=0.5` 的 support 三值（0.9012/0.9023、0.9262/0.9273、0.9338/0.9349）
与 10-24 存活下来的三份 manifest 完全相同。

## 重跑了什么、为什么

| 批次 | 内容 | 为什么重跑 |
|---|---|---|
| `W-spambase-grid` | spambase seed0 × `c`{0.1,0.3,0.5} × 7 方法 = 21 跑（本机 T600，CPU） | 汇报二.3 的 Robust/Split/Holistic 三组读数**本地没有任何 manifest**；其余为补全 `c` 网格 |
| `W-cifar-adapter` | cifar10 `cnn_feature_adapter` × `c`{0.1,0.3,0.5} × 3 方法 = 9 跑（AutoDL vGPU-32GB） | 汇报二.4 只有 120 行子集的一次链路验证，没有可入库的完整读数 |
| `W-genpu-formal` | GEN-PU 正式预算 100 GAN + 100 PN（cifar10，seed 0） | 10-25 新跑的工程读数，收据已在 `data/survey_recipe_formal_run_20261025.json` |

CIFAR 三行跑的是**完整 split**（train 45,000 / test 10,000），不是汇报里那次 120 行子集，
两者不可互相替代。

## 口径细节

- `spambase` 的 population class prior 用 split manifest 记录的 **0.39404477287546186**。
  此前若用四舍五入的 0.4，PA 的 proxy accuracy 权重会偏（该准则的正项权重就是 π）。
- **PA 一列按选模协议拆开**：support 方法（`lagam` / `pulns` / `self_pu_full_meta`）选的是
  `PA+clean-support`（用 clean 标签打分），与纯 PU 视图的 `PA` 是两个准则，
  `pa_selection_protocol` 列写明，`主表` 里也分成两张表，不混排。
- `GEN-PU` 正式预算行由 `scripts/run_genpu_recipe.py` 跑，不经 Survey 协议上下文，
  因此 `comparability_group` 为空、`protocol_deviation` 记为
  `engineered_recipe_no_survey_protocol_context`。实测 2.67 h、峰值 2.54 GiB、100 份快照均在
  64 MiB 界内。
- `peak_gpu_memory_mb` 为空表示该 run 没有 GPU 读数（CPU 运行），不代表 0。
- 不跨 `training_path` 排名：cifar 三行 adapter 的 OA 在 0.81–0.83，GEN-PU 正式预算行 0.633，
  但一个是冻结特征 + MLP 头、一个是原生 CNN 端到端 GAN 训练，表征与预算都不同，各自只有 1 seed。
  这组差异只能当"需要进一步查"的线索。

## 复现

- 网格：`tmp/run_spambase_grid_20261025.py`（本机 CPU）；远端批次脚本
  `tmp/run_cifar_adapter_20261025.sh`。
- 收据：`data/workbook_runs_20261025.json`（31 条 run，入库）。它由
  `tmp/make_workbook_receipt_20261025.py` 从原始 manifest 生成，保留 selection / resources /
  test_results 等审计字段，去掉占体积 40% 的 `candidate_runs` 逐 epoch 轨迹（只留摘要），
  并把重复 9 次的 clean-support `split_ref` 按摘要去重。每份 manifest 的原路径、字节数与
  `manifest_sha256` 都在条目里，被删去的轨迹与 `split_ref` 各有 `*_sha256`。
- 工作簿：`tmp/build_results_workbook_20261025.py`，**只读这份收据**、不读 manifest、不手抄，
  因此原始 manifest 不入库也能重建；`runs` 的 `evidence` 列就是收据里的 run key。
- 原始 manifest 留在 `tmp/wb_20261025/` 与 `tmp/wb_cifar_20261025/`（本地证据，不入 git）。

## 未完成

- 5 seed 正式批次仍未开始，多 seed 方差表无法给出。
- GradPU 首次真实运行归队友，本表不含；本表也不含队友负责的任何单元。
- 汇报二.4 的 120 行子集读数没有被复现（已被完整 split 取代），如需保留应在旧批次里找。
