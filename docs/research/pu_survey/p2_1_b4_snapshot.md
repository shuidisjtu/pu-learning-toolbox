# B4 批次快照（P2.1）· 草稿

状态日期：2026-10-01。**本文件是草稿**：B4 于 2026-10-01 09:34 起跑，计划耗时约 40 小时。
前置门禁、技术探针与回收实测已有实测值（§1–§4）；清单层与门禁层的统计待跑完填入（§7）。

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

## 5. 已知现象（预期）

`ConvergenceWarning`（SAR 标注生成器的 `LogisticRegression(max_iter=100)`）：B3a 为
120 次 / 60 个 SAR run、B3b 为 80 次 / 40 个，两批均 **2.0 次/SAR run**。B4 有 20 个 SAR run，
预期 **20–40 次**，实测计数待跑完填入。该警告不影响任何 run 的选模结果，**不修改**
（`max_iter` 属冻结实现，改动即协议变更）。

## 6. 本快照不含

- 方法排名、跨方法对比与数值裁决（P2.2）；
- 与参考文献的交叉验证结论（P2.2）；
- 期望单元完整性格网（口径未定，见 D13 遗留项）。

## 7. 待跑完填入（含验收命令）

以下命令一律用**字面路径**，不依赖环境变量（B4 跑批终端之外的新终端没有那些变量）。

1. **真实退出码**
   ```
   cat /root/autodl-tmp/pu-survey-logs/B4_cifar10_native_cnn/B4_exit_code.txt
   ```
   预期 `0`。

2. **完成行**
   ```
   tail -n 3 /root/autodl-tmp/pu-survey-logs/B4_cifar10_native_cnn/B4_run.log
   ```
   预期含 `completed 35 of 35 run(s)` 与 `0 still pending`。

3. **manifest 计数、协议摘要唯一性、视图分布**
   ```
   /root/p21-env/bin/python -c "import json,glob,collections; ms=sorted(glob.glob('/root/autodl-tmp/pu-survey-results/B4_cifar10_native_cnn/**/manifest.json',recursive=True)); print('manifests',len(ms)); print('protocol',set(json.load(open(p)).get('protocol_sha256') for p in ms)); print('views',collections.Counter(json.load(open(p)).get('run_view') for p in ms)); print('calib',collections.Counter(json.load(open(p)).get('calibration_applied') for p in ms))"
   ```
   预期：`manifests 35`；协议摘要为单值且等于冻结值；视图为 `{'ts-compatible': 35}`。

4. **checkpoint 终态**（回收后应只剩选中的引用）
   ```
   find /root/autodl-tmp/pu-survey-results/B4_cifar10_native_cnn -name "epoch_*.pt" | wc -l
   ```
   预期 **约 42–43**（35 个 run 各自选中的 PA/OA 权重），**不是 7000**。

5. **回收账目自洽**（逐个 manifest 核对 `reclaimed_true + files_on_disk == refs`）
   ```
   /root/p21-env/bin/python -c "import json,glob,os; [print(p, (lambda cps: (len(cps), sum(1 for c in cps if c.get('reclaimed')), len(glob.glob(os.path.dirname(p)+'/**/epoch_*.pt',recursive=True))))([c for r in json.load(open(p)).get('candidate_runs',[]) for c in r.get('epoch_checkpoints',[])])) for p in sorted(glob.glob('/root/autodl-tmp/pu-survey-results/B4_cifar10_native_cnn/**/manifest.json',recursive=True))]"
   ```

6. **重新 dry-run**（§9 完成判定第三条）。**用独立的 `--plan-json`**，避免覆盖 §6 用过的
   跑前快照：
   ```
   cd /root/autodl-tmp/pu-learning-toolbox && /root/p21-env/bin/python scripts/run_survey_pilot.py --splits /root/autodl-tmp/pu-survey-data/splits --results /root/autodl-tmp/pu-survey-results/B4_cifar10_native_cnn --protocol survey-v1.2 --datasets cifar10 --methods nnpu --training-paths native_cnn --device cuda --reclaim-unselected-checkpoints --dry-run --plan-json /root/autodl-tmp/pu-survey-plans/B4_cifar10_native_cnn/B4_recheck_plan.json
   ```
   预期 `planned 35 / completed 35 / pending 0`，退出码 0。

7. **`ConvergenceWarning` 计数**
   ```
   grep -c ConvergenceWarning /root/autodl-tmp/pu-survey-logs/B4_cifar10_native_cnn/B4_run.log
   ```
   与 20 个 SAR run 对账。

8. **§12 备份**：按手册 12 节打包、生成 sha256、下载到
   `F:\Temp\lab\P2.1\autodl_batch_backups\B4_cifar10_native_cnn`，并复制到
   `/root/autodl-fs/pu-survey-backups/`。

9. **B 层策略披露**：按 16 号 §4.2，B4 采用**每 run selected checkpoint**（即 D25 回收的产物），
   须在本快照写明放弃的是未选 epoch 的后续复核能力。

10. **分段续跑闭环**：本批一趟跑完，未进入分段流程，按手册 13 §14 记为「未使用、未验证」。
