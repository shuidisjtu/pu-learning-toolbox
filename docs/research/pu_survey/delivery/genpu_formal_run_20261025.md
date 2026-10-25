# GEN-PU 正式预算首跑（2026-10-25）

第一次按**声明的正式预算**（100 GAN + 100 PN epoch、batch 64、`hidden_dim` 256）在真实
CIFAR-10 split 上跑完 GEN-PU，回答三个工程问题：预算跑得完吗、每份快照是否都在 64 MiB
逐组件上界内、保留规则与 manifest 是否如实记录。

**这不是效果证据。** 单 seed（seed 0）、无多 seed 批次、比较矩阵仍未绑定该方法
（`genpu` 在 `survey-comparison-v4` 里无条目，P4.2b 只做定性交代）；PA/OA 读数只说明链路接通。
收据自身记 `formal_admission: false`、`paper_numerical_comparison: false`、`single_seed: true`。

## 跑的是什么

| 项 | 值 |
|---|---|
| 数据集 | `cifar10` split_0（45,000 训练行），`class_prior` 0.4 |
| 配方 | `recipe_sha256 1abfdf92…`，`plan_sha256 a82e8fe2…` |
| 预算 | 100 GAN + 100 PN epoch，batch 64，`checkpoint_bytes_per_component` 64 MiB |
| 环境 | 远端 32 GiB vGPU，`/root/p21-env`（Python 3.10.8，torch 2.13.0+cu130） |
| 代码 | `726abd2` 的干净 clone（`git bundle` 传输；不含之后合并的 #141/#142） |

## 实测 vs 预测

| 量 | 预测（容量记录） | 实测 | 结论 |
|---|---:|---:|---|
| 总耗时 | 2.70 h | **2.666 h**（9,597.6 s，含选模回放 9,605.2 s） | 成立 |
| 峰值显存 | 2.55 GiB | **2.54 GiB**（2,731,042,304 B） | 成立 |
| 进程峰值 RSS | — | 2.81 GB | 记录 |
| 单份快照 | ≤ 64 MiB | **45,542,453 B**（43.4 MiB），100/100 份等大 | 上界成立 |
| 快照总量 | 4.55 GB | 4,554,245,300 B（4.24 GiB） | 成立 |
| 保留 | 按 D25 | 保留 2、回收 98，manifest 逐条记 `reclaimed` 标记 | 成立 |

0 警告、0 重试、0 失败；`accepted_engineering_run: true`。

## 选模与测试读数

| 视图 | 选中 epoch | 阈值 | 验证侧 | 测试 acc | AUC | P / R / F1 |
|---|---|---|---|---|---:|---|
| PA（`pu_val`，proxy accuracy） | 14 | 1.0 | 1.0196 | 0.6192 | 0.592 | 0.530 / 0.427 / 0.473 |
| OA（`clean_val`，accuracy） | 68 | 0.9 | 0.6304 | 0.6332 | 0.549 | 0.619 / 0.216 / 0.320 |

PA 的 proxy accuracy 大于 1 是按定义来的，不是错误：R9 的准则为

```
PA(theta) = (2*pi/n'_P) * sum_{D'_P} I(f >= theta) + (1/(n'_P+n'_U)) * sum_{D'_P u D'_U} I(f < theta)
```

完美分类器上它等于 `ACC + pi`（`strategies.proxy_accuracy` 的 docstring 写明这一点，也是
Proposition 1 成立的原因）。因此 1.0196 应读作 `≈ ACC + 0.4`，不能当百分比看。

CIFAR-10 上 GAN 生成式 PU 方法的这两个读数本身偏弱，与配方无关：单 seed、`class_prior`
固定 0.4、无任何比较锚点，本跑只用来证明预算、快照与保留链路。

## 收据与门禁

- 收据 `docs/research/pu_survey/data/survey_recipe_formal_run_20261025.json`
  （schema `genpu-formal-run-engineering-v1`，`receipt_sha256 64a0722a…`），由
  `tmp/assemble_formal_run_receipt_20261025.py` 从 `plan`/`environment`/`manifest`/`summary`
  四份原始产物重算字节数并核对，不是手抄。
- 收据内 `files_sha256` 由远端 `summary.json` 给出，逐文件对应 `plan/manifest/environment/
  selected_PA.pkl/selected_OA.pkl`。

## 未完成 / 边界

- **不是效果证据**：要效果结论需要 5 seed 正式批次 + P4.2b 的定性定位，两者都未开始。
- 运行的是 `726abd2`，不含之后合并的 #141（正式预算容量）与 #142（执行计划更正）；
  这两笔只改文档与计划，不影响本次读数，但重跑应以当前 `main` 为准。
- `genpu` 仍在 `survey_comparison_v4.json` 之外，比较矩阵的注册状态为 `not_in_matrix`
  （P4.2 main-table gate pending），本收据不改变这一点。
- 实例按小时计费（约 ¥1.58/h），数据收完即关机。
