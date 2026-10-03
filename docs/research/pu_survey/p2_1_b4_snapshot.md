# B4 批次快照（P2.1）· 草稿

状态日期：2026-10-03。B4 于 2026-10-01 09:34 起跑，2026-10-03 00:03 结束，退出码 `0`、
`completed 35 of 35 run(s)`。§1–§4 为跑前与跑中记录；§5 与 §7 为跑完后的实测统计与收尾验收。

定位同其他批次快照：记录**清单层与门禁层**的可审计情况，**不含**方法排名、跨方法对比与数值裁决
（属 P2.2）。

## 1. 批次身份

| 项 | 值 |
|---|---|
| 批次 | B4（`cifar10`，`native_cnn` 路径，`nnpu`） |
| runs | 计划 35 |
| 协议 | `survey-v1.2`，摘要 `287c2f45387f02714e5925b35dcb04f3f64e1e3faf8f740cab861fc5747dd84a` |
| 执行 commit | `ee54b5ba488674faf3b9e504611518471ab0a31b`（含 D25 回收） |
| `uv.lock` 跨平台身份 | git blob `cf5772f4c0afb2b0c3ede49937bf8c00c6c3fcc5`（**与 B0–B3b 相同，未变**） |
| 峰值单元 | `cifar10/nnpu/native_cnn`，profile `native_cnn_full_resnet18` |
| 人口先验 | `cifar10=0.4`，读自冻结的 `split_manifest.json` |

**身份推进已披露**：B0–B3b 执行于 `7f445be`，B4 起为 `ee54b5b`。B4 必须启用逐 epoch 回收——
不回收时单 run 峰值约 8.3 GiB、35 run 累计约 292 GiB，而跑前门禁要求每个 run 启动时至少
17.58 GiB 空闲。`git diff 7f445be..ee54b5b` 对 `uv.lock`、`pyproject.toml` 与协议 JSON 三项
**均为空**，协议身份未变；变化的只有代码（D25 回收、D26 聚合分组维度）与文档。

## 2. 前置门禁与计划对账

§3 门禁全过：仓库处于分离头指针（`ee54b5b`）、工作区干净；`uv.lock` blob 与协议摘要逐字相符；
splits 校验 `ok: 15 split(s), 75 file(s) match`；GPU `NVIDIA vGPU-32GB`；冻结环境
`/root/p21-env` 的 Python `3.10.8` 在实例重启后仍可用。

§5 dry-run：`planned 35 / completed 0 / pending 35`，退出码 0；`scope: datasets=cifar10;
methods=nnpu; training_paths=native_cnn`；预估 `307.6 GiB`、单 run 峰值 `8.79 GiB`、跑前门禁
`17.58 GiB`。

§6 对账：与控制包内已验收的参考快照 `plans/B4_cifar10_native_cnn.json` 比对六项身份字段
（`snapshot_schema_version` / `source_protocol_sha256` / `selection` / `execution_units` /
`totals` / `runs`），`mismatches= []`。

**容量口径的限定**：`324.5 GiB` 是**全 Pilot 645 run 的累计**（决策 D15 正式口径），
不是 B4 的；B4 自身预估 307.6 GiB、实测外推 **291.8 GiB**。B4 一家占全 Pilot 的九成，
其余四批实测合计约 12.5 GiB。

## 3. 技术探针（§7A）

独立目录 `B4_cifar10_native_cnn_technical_probe`，单元 `scar / c=0.1 / seed=0 / native_cnn`，
**不计入正式 35 runs**。

| 量 | 实测 | 说明 |
|---|---|---|
| checkpoint | **200** | 与 200 epoch 预算一致 |
| 磁盘增量 | **8,951,009,055 B = 8.34 GiB** | `du -sb`；折合 **42.7 MiB/epoch** |
| 耗时 | **4143 秒（69 分 3 秒）** | 与墙钟 22:49:59→00:00:53 一致 |
| 峰值显存 | 约 **8687 MiB** | **人工抽样观测，非连续仪测**（未按 §7A 把 `nvidia-smi` 采样写入 `gpu.csv`） |

与预估 8.79 GiB 相比，实测为 **95%**——预估是小幅保守的上界。原因是每 epoch 产物就是
ResNet-18 的 `state_dict`，尺寸固定（11.7M 参数 × 4 字节 = 42.5 MiB）。

**探针结果树的处置**：manifest 与上述四项测量已复制到本批证据目录
（`probe_manifest.json`、`probe_measurements.txt`），随后结果树删除以释放 8.34 GiB。
正式目录自始至终与探针分开。

## 4. 回收（D25）实测

**第一个 run 完成后的 manifest 证据**：

```
refs 200   reclaimed_true 199   files_on_disk 1
```

三者自洽（`199 + 1 = 200`）：manifest 仍**完整记录**全部 200 个 epoch 权重引用，其中 199 个
被打上「曾落盘、后回收」标记且文件确已删除，仅保留被 `selection` 指向的 1 个。
单 run 磁盘占用因此从 **8.34 GiB 降到 42.5 MiB**。

**盘面观测序列**：25G（起跑）→ 24G（24 个 checkpoint）→ 23G（50 个）→ 23G（run 2 进行中）。
计数呈锯齿（run 内涨到 200、run 结束回落至个位数）。

**一处如实说明**：run 1 结束那一刻的 `df` **没有取样**，因此「回弹到约 25G」是由 manifest
证据与盘面序列**推定**，不是直接观测。若需要直接证据，可在后续 run 边界补取一次。

## 5. 已知现象（实测）

`ConvergenceWarning`（SAR 标注生成器的 `LogisticRegression(max_iter=100)`）：B3a 为
120 次 / 60 个 SAR run、B3b 为 80 次 / 40 个，两批均 **2.0 次/SAR run**。B4 实测 **40 次 /
20 个 SAR run = 2.0 次/run**，与前两批一致（跑前估计区间 20–40）。该警告不影响任何 run 的
选模结果，**不修改**（`max_iter` 属冻结实现，改动即协议变更）。

## 6. 本快照不含

- 方法排名、跨方法对比与数值裁决（P2.2）；
- 与参考文献的交叉验证结论（P2.2）；
- 期望单元完整性格网（口径未定，见 D13 遗留项）。

## 7. 收尾验收与统计（实测，2026-10-03）

验收在**无卡模式**实例上执行。第 6 项的 dry-run 经代码核实不触 GPU：`run_survey_pilot.py:676-706`
的 dry-run 分支在 `_run_batch`（唯一把 `--device` 传进子进程处，`:561`）之前返回，`cuda` 全程
只作字符串存进 config。

### 7.1 十项验收结果

| # | 项 | 实测 | 判定 |
|---|---|---|---|
| 1 | 真实退出码 | `0` | 通过 |
| 2 | 完成行 | `completed 35 of 35 run(s) in cifar10; 0 still pending` | 通过 |
| 3 | manifest / 摘要 / 视图 | 35；摘要单值且等于冻结值；`ts-compatible` 35；`calibration_applied` True 35 | 通过 |
| 4 | checkpoint 终态 | **38** | 通过（偏差说明见 §7.4） |
| 5 | 回收账目自洽 | 35 行全部满足 `reclaimed + on_disk == 200`；磁盘数合计 38 | 通过 |
| 6 | 重新 dry-run | `planned 35 / completed 35 / pending 0`，退出码 `0` | 通过（命令更正见 §7.5） |
| 7 | `ConvergenceWarning` | **40**（20 个 SAR run × 2.0） | 通过 |
| 8 | 备份与取回 | 见 §7.6 | 通过 |
| 9 | B 层策略披露 | 见 §7.7 | 文档条目 |
| 10 | 分段续跑闭环 | **未使用、未验证**（本批一趟跑完） | 文档条目 |

### 7.2 清单层

**35 个 run 全部成功**，失败记录 0（`failure_index.txt` 为空）。

| 维度 | 分布 |
|---|---|
| 方法 | `nnpu` 35 |
| 机制 | `scar` 15（`c_0.1` / `c_0.3` / `c_0.5` 各 5）；`sar_lbe_a` 10；`sar_lbe_b` 10 |
| 视图 | `ts-compatible` 35 |
| `calibration_applied` | True 35（与视图严格对应） |
| 选择产物 | PA + OA 15；仅 OA 20 |
| `protocol_deviation` | 全空 35 |
| `formal_eligible` | True 35 |
| `formal_blockers` | 空 |

选择产物的分布与协议规则逐一对上：含 PA + OA 的 15 个正是 SCAR 全部；仅 OA 的 20 个是
`sar_lbe_a` 与 `sar_lbe_b`（SAR 下 PA 仅作诊断日志，协议 §2.3）。

**视图统一为 `ts-compatible` 是单方法批次的必然结果**：`nnpu` 的
`native_sampling_assumption` 为 ts、`calibration_applied=True`，而本批不含任何 os 原生方法。
B1 / B2 / B3a 那种 `ts` / `os` 混合分布在这里不会出现，故 `calibration_applied` 无 False 分量。

### 7.3 门禁层

`scripts/aggregate_survey_runs.py` 在本地 manifests 树上运行（`--json` 报告）：

| 项 | 值 |
|---|---|
| 组 | **3** |
| 单元 | **35** |
| `comparable` | 35 |
| `blocked` | 0 |
| `refused` | 0 |
| `formal_ready` | **True** |

组的构成——同一 `comparability_group = cifar10/native_cnn/native_cnn`，按 mechanism 分榜：

| 机制 | 视图 | 单元 | 方法 |
|---|---|---|---|
| `scar` | ts | 15 | `nnpu` |
| `sar_lbe_a` | ts | 10 | `nnpu` |
| `sar_lbe_b` | ts | 10 | `nnpu` |

**B4 是五批中第一个 `formal_ready = True` 的批次**。B1 / B2 / B3a / B3b 因含 5 个 `pn_oracle`
run 的 `protocol_deviation: ['c_grid']` 而为 False；B4 不含 oracle，该偏差不存在。

### 7.4 制品状态：checkpoint 终态 38

第 4 项实测 **38**，低于本文件跑前估计的「约 42–43」。第 5 项逐 run 对账显示
`reclaimed + on_disk == refs` 对全部 35 行成立，故 **38 是正确终态，原估计是粗估**。

构成由保留规则决定——`runner.py:825-851` 的 `keep` 是**选中路径的集合**，PA 与 OA 选中同一
epoch 时路径相同、自动合并为一个文件：

| 组 | run 数 | 每 run 保留 | 小计 |
|---|---:|---:|---:|
| `sar_lbe_a` / `sar_lbe_b`（仅 OA 选择） | 20 | 1 | 20 |
| SCAR（PA + OA 合流） | 12 | 1 | 12 |
| SCAR（PA + OA 分叉） | 3 | 2 | 6 |
| **合计** | **35** | | **38** |

原估计相当于假设约 7–8 个 SCAR run 出现分叉，实测 3 个。结果树 `du -sb` 实测
**1,708,451,430 B ≈ 1.59 GiB**（回收后；未回收时该批为 291.8 GiB 量级）。

### 7.5 验收清单的两处更正

本文件的验收清单有两处缺口，执行时已按手册处理：

1. **缺 `PYTHONPATH`**。运行环境按 `uv sync --no-install-project` 建立（手册 03 §5），
   `pu_toolbox` 只经 `PYTHONPATH` 可见；手册 03 §7 与 13 §3 都要求
   `export PYTHONPATH=/root/autodl-tmp/pu-learning-toolbox`。故本节「不依赖环境变量」的说法
   对 `PYTHONPATH` 不成立——它对仓库路径类变量成立，但解释器路径与 `PYTHONPATH` 是前置条件。
2. **§10 证据采集未列入**。手册 13 §12 的备份会把证据目录（脚本中写作 `EVIDENCE`）整个拷入
   备份目录，而本批该目录当时只有 probe 与磁盘轨迹三项，缺手册 13 §10 产出的
   `failure_index.txt`、`artifact_index.txt`、`checkpoint_inventory.tsv`、`result_size.txt`。
   执行时已按 §10 补做，四项随备份留存（清单 38 行，与 §7.4 的终态一致）。

### 7.6 归档与取回

| 载体 | 内容 | 摘要 |
|---|---|---|
| 数据盘暂存 | `/root/autodl-tmp/pu-survey-backup-stage/B4_cifar10_native_cnn/` | — |
| 完整归档 | `B4_cifar10_native_cnn_20261003_092708.tar.gz`，**1,577,665,039 B** | `0f09ec35cd06bff1dc9b0f71d343d0c2caed5a53b4b580bb1ba58a7a886ea3f4` |
| 网盘副本 | `/root/autodl-fs/pu-survey-backups/pu-survey-backup-stage/B4_cifar10_native_cnn/` | 两侧 `sha256sum` 直接比对，一致 |
| 本机副本 | `F:\Temp\lab\P2.1\autodl_batch_backups\B4_cifar10_native_cnn\` | 见下表 |

本机取回三个文件，**源端与本机摘要逐字一致**：

| 文件 | sha256 |
|---|---|
| `B4_manifests.tar.gz` | `68054c37982959bc9d7366fa7817c19bd9ee07c2537f44167472301b2c620e71` |
| `B4_evidence.tar.gz` | `4c6c87e6ce822f9129633518a77c156f96a89ef8daee0d581bc11f5d5da4c047` |
| `B4_run.log` | `484037e2ca62b7daa12a89fbe797cfaafb0457d3eeec8035192953c25c0d071a` |

**取回范围**：本机只保存 manifests 树（35 份 `manifest.json` + 35 份 `method_ledger_entry.json`）、
evidence 包与 `B4_run.log`，**不保存 checkpoint**；完整权重仅存于网盘副本。本机 manifests 树已复核：
摘要单值且等于冻结值、视图全 `ts-compatible`、`formal_eligible` 全 True。

### 7.7 B 层策略披露（第 9 项）

按手册 16 §4.2，B4 采用**每 run selected checkpoint** 方案（D25 回收的产物）：保留被 `selection`
指向的 38 个权重，**放弃的是未选 epoch 的后续复核能力**——不能用新准则在旧轨迹上重选，也不能
重新生成某个未选 epoch 的预测。选中权重的复核能力完整保留（重算阈值、查看 selected epoch、
生成新预测）。

**与 B1 的关键差异**：B4 的 manifest **带 `reclaimed` 标记**。每个 run 的 200 个引用中 199（或 198）
个为 `reclaimed=true`，盘上实存的就是 `reclaimed=false` 的那 1–2 个。因此本批**可以**直接用该字段
判断文件存在性，不必像 B1 那样退回 sha256 校验或外部记录。

### 7.8 分段续跑闭环（第 10 项）

**未使用、未验证**：本批一趟跑完 35/35，未进入分段流程（手册 13 §14）。按手册 16 §10 的检查项，
该项保持未勾选，不因批次完成而视为已验证。

### 7.9 复现入口

验收命令见手册 13 §9（完成判定）、§10（manifest 与证据采集）、§12（备份）；第 6 项的
dry-run 命令需按 §7.5 第 1 条补 `PYTHONPATH`。门禁层可由
`uv run python scripts/aggregate_survey_runs.py <B4 manifests 根>` 复现（加 `--json` 取结构化报告）。
