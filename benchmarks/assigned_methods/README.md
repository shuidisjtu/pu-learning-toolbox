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

## 结论边界

当前结果必须标为 `clean_room`：

- Dist-PU 使用 toolbox 全量 MLP，不是官方图像 backbone/mini-batch 两阶段训练；
- PUSB 使用来源 Logistic Regression，不是官方 kernel PUSB；
- LBE 使用线性交替 Logistic Regression，不是官方 MLP + Adam；
- CPE 的 penL1 解析解已对齐论文；尚缺逐 `theta` CV 的精确实现证据和 MNIST/PCA 执行层；
- ReCPE 尚缺官方 FCNet 和全部 CPE baseline。
