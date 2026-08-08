# Assigned Methods Benchmarks

本目录覆盖 HENG958 负责列表中的前五篇：Class-Prior Estimation、ReCPE、
Dist-PU、PUSB 和 LBE。

## 当前可运行实验

`clean_room_multiseed.json` 使用可控合成数据和当前 toolbox 实现，运行五个 seed：

```bash
python -m benchmarks.assigned_methods.run \
  --config benchmarks/assigned_methods/configs/clean_room_multiseed.json \
  --output benchmarks/assigned_methods/results/clean_room_multiseed
```

只运行部分方法：

```bash
python -m benchmarks.assigned_methods.run \
  --config benchmarks/assigned_methods/configs/clean_room_multiseed.json \
  --output /tmp/pu-benchmark \
  --methods class_prior_estimation recpe
```

运行配对的 SCAR/SAR 对比：

```bash
python -m benchmarks.assigned_methods.run \
  --config benchmarks/assigned_methods/configs/scar_sar_comparison.json \
  --output benchmarks/assigned_methods/results/scar_sar_comparison
```

该配置在相同 seed 和目标平均标记率下展开 `scar`、`linear`、`nonlinear`，报告分类指标、
propensity 诊断，以及相对于已知 Bayes posterior 的 Spearman、Kendall 和 pairwise
ranking accuracy。

每次运行生成：

- `trials.csv`：逐方法、逐 seed 的原始结果；
- `summary.csv`：均值和样本标准差；
- `resolved_config.json`：实际配置；
- `run_manifest.json`：配置哈希、代码 commit、环境版本和结论边界。

## 官方配置

`configs/official/` 是从锁定上游源码提取的配置证据，不由当前 runner 直接执行。
不可变来源记录在 `configs/official_sources.lock.json`。其中 CPE 的作者软件页面在核对时
不可访问，因此状态为 `partial`；其余方法锁定到 Git commit 或归档 SHA-256。

官方配置的 `locked_not_executed` 只表示参数已经核对，不表示论文实验已经运行。完整
paper-like 复现仍需要对应历史环境、数据、GPU，以及 ReCPE 的部分 MATLAB baseline。

在准备源码、数据或历史环境前，可先执行只读预检：

```bash
python -m benchmarks.assigned_methods.preflight_paper \
  --config-dir benchmarks/assigned_methods/configs/official \
  --output benchmarks/assigned_methods/results/official_preflight/current_node.json
```

迁移到实验节点后，以 `METHOD=PATH` 显式提供资源；预检会验证目录和官方入口文件：

```bash
python -m benchmarks.assigned_methods.preflight_paper \
  --config-dir benchmarks/assigned_methods/configs/official \
  --output /tmp/assigned-official-preflight.json \
  --source-root recpe=/path/to/recpe \
  --data-root recpe=/path/to/recpe-data
```

报告分别给出 `ready_for_official_execution` 和 `ready_for_toolbox_replication`。前者只检查
锁定官方代码所需的源码、数据、CUDA、MATLAB 和历史版本；后者还加入 clean-room toolbox
与论文算法/实验协议之间的实现差距，不能用“官方代码可启动”替代“toolbox 已精确复现”。

## PUSB 官方仓库 IJCNN1 扩展

PUSB 已提供独立的 official-aligned RBF 适配器和 IJCNN1 runner。IJCNN1 是发布仓库入口
的默认数据集，但不在论文 Table 2 的六个数据集中，因此这些运行属于
`official_repo_extension`，不是 `paper_protocol`。先运行缩小网格 smoke：

```bash
python -m benchmarks.assigned_methods.pusb_official_data \
  --config benchmarks/assigned_methods/configs/pusb_official_data_smoke.json \
  --data-root /path/to/pusb-data \
  --output benchmarks/assigned_methods/results/pusb_official_data_smoke
```

runner 会核验解压后 IJCNN1 的 SHA-256 和 `(49990, 22)` 形状，复刻官方 selected-positive
抽样，并保存 trial、summary、解析后配置和 provenance manifest。smoke 使用 30 个 RBF 基、
3 折 CV 和缩小网格，因此强制 `paper_claim=false`。

IJCNN1 可行子集的完整网格分批配置为：

```bash
python -m benchmarks.assigned_methods.pusb_official_data \
  --config benchmarks/assigned_methods/configs/pusb_official_data_feasible_multiseed.json \
  --data-root /path/to/pusb-data \
  --output benchmarks/assigned_methods/results/pusb_official_data_feasible_multiseed
```

长任务会逐 trial 原子写入 `trials.csv`。中断后使用同一命令并追加 `--resume`；runner 会
核验 `resolved_config.json`，拒绝使用不同配置续写同一结果目录。

现有证据确认三项差异：源码的正则目标与梯度相差系数 2；仓库 README 声称入口复现
Table 2，但入口默认 `ijcnn1`，而论文 Table 2 使用 mushrooms、shuttle、pageblocks、usps、
connect-4 和 spambase；IJCNN1 在 seed 2018 的 3,000 条 holdout 中只有 315 个正例，只能
构造 `pi=0.2` 的 1,000 条测试集。runner 不会静默改变该仓库扩展的采样协议。

## 已执行结果

`results/clean_room_multiseed/` 已在 2026-07-27 使用 seed `0..4` 完成 25 个 trial。
关键聚合结果如下：

| 方法 | 指标 | 均值 | 标准差 |
|---|---|---:|---:|
| Class-Prior Estimation | prior MAE | 0.0380 | 0.0192 |
| ReCPE | prior MAE | 0.2715 | 0.0227 |
| Dist-PU | ROC-AUC | 0.9595 | 0.0856 |
| PUSB baseline | ROC-AUC | 1.0000 | 0.0001 |
| LBE linear EM | ROC-AUC | 0.9887 | 0.0108 |

ReCPE 在该设置中的低估是当前默认 density-ratio CPE 后端的实际结果，不应删除或解释为
论文 ReCPE 的表现。完整指标和运行环境分别见 `summary.csv` 与 `run_manifest.json`。

`results/scar_sar_comparison/` 另外完成了 2 个 bias-aware 方法、3 种机制和 10 个 seed，
共 60 个配对 trial：

| 方法 | 机制 | ROC-AUC | Pairwise ranking accuracy |
|---|---|---:|---:|
| PUSB baseline | SCAR | 0.9998 | 0.9148 |
| PUSB baseline | linear SAR | 1.0000 | 0.9590 |
| PUSB baseline | nonlinear SAR | 1.0000 | 0.9537 |
| LBE linear EM | SCAR | 0.7284 | 0.6283 |
| LBE linear EM | linear SAR | 0.9968 | 0.8694 |
| LBE linear EM | nonlinear SAR | 0.9990 | 0.9061 |

这组高斯数据可分性较强，接近 1 的 AUC 不能外推为真实数据表现；结果的主要用途是验证
机制展开、配对 seed、排序指标和 propensity 诊断链路。

`results/pusb_official_data_smoke/` 已使用官方 IJCNN1、seed 2018、`pi=0.2`、400 P、
800 U 和 1,000 条测试样本完成 1 次端到端运行。缩小搜索选中 `sigma=1.0`、
`lambda=0.01`；分位数 accuracy 为 `0.7470`、balanced accuracy 为 `0.6038`、
ROC-AUC 为 `0.6664`。缩小 uLSIF 对照 accuracy 为 `0.7270`、ROC-AUC 为 `0.6640`。
该结果只证明执行链路可用，不是论文表格复现。

`results/pusb_official_data_feasible_multiseed/` 已完成 IJCNN1 仓库扩展的
3 seeds × 3 U sizes 共 9 个 trial，
使用 300 个基、5 折、完整 PUSB 9×8 网格及 `densratio 0.3.0` 默认 100 kernels/13×13
搜索。全部 CV 候选和最终重训均收敛，三 seed 均选中 `sigma=1.0`、`lambda=0.001`：

| U | PUSB accuracy | PUSB ROC-AUC | uLSIF accuracy | uLSIF ROC-AUC |
|---:|---:|---:|---:|---:|
| 800 | 0.7730 ± 0.0151 | 0.7061 ± 0.0093 | 0.7543 ± 0.0050 | 0.6519 ± 0.0325 |
| 1600 | 0.7683 ± 0.0170 | 0.6982 ± 0.0280 | 0.7543 ± 0.0050 | 0.6519 ± 0.0326 |
| 3200 | 0.7657 ± 0.0114 | 0.6983 ± 0.0210 | 0.7543 ± 0.0061 | 0.6520 ± 0.0326 |

这些结果验证仓库扩展的完整计算链路，但数据集不属于论文 Table 2，不能与论文表格直接
对照。论文六数据集现已完成数据锁与采样审计；下一阶段需先确认不可行单元的报告政策，
再执行可恢复的 `4 priors × 3 U sizes × 100 repetitions` 训练任务。

### PUSB Table 2 数据锁与采样审计

论文 Table 2 的 mushrooms、shuttle、pageblocks、usps、connect-4、spambase
已在 `configs/pusb_table2_datasets.json` 中锁定来源、目标文件 SHA-256、形状、标签映射
和类别计数。原始数据保存在仓库外，统一加载器会复现官方逐特征最大值归一化：

```bash
python -m benchmarks.assigned_methods.pusb_table2_data \
  --data-root /data2/user/zihenglin/official-data/pusb-table2 \
  --output benchmarks/assigned_methods/results/pusb_table2_data_audit/current_node.json
```

审计严格复现官方 `U -> prior -> repetition` 循环及从 2018 连续递增的 seed。72 个
`dataset × U × prior` 单元中，只有 45 个在全部 100 次重复中都能构造足量样本：USPS
与 connect-4 为 12/12，mushrooms 为 11/12，shuttle 为 6/12，pageblocks 为 1/12，
spambase 为 3/12。官方脚本不会检查无放回切片长度，其余单元会静默得到少于声明值的
测试集或未标记集；严格论文协议不得把这些运行标为相应的 `1000 test` 或 `U` 规模。

## 结论边界

当前结果必须按实际 fidelity 分别标为 `clean_room` 或 `official_repo_extension`：

- Dist-PU 使用 toolbox 全量 MLP，不是官方图像 backbone/mini-batch 两阶段训练；
- PUSB 的 IJCNN1 kernel 结果是官方仓库扩展，不是论文 Table 2；
- LBE 使用线性交替 Logistic Regression，不是官方 MLP + Adam；
- CPE 的 penL1 解析解已对齐论文；尚缺逐 `theta` CV 的精确实现证据和 MNIST/PCA 执行层；
- ReCPE 尚缺官方 FCNet 和全部 CPE baseline。
