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

## 已执行结果

`results/clean_room_multiseed/` 已在 2026-07-27 使用 seed `0..4` 完成 25 个 trial。
关键聚合结果如下：

| 方法 | 指标 | 均值 | 标准差 |
|---|---|---:|---:|
| Class-Prior Estimation | prior MAE | 0.0380 | 0.0192 |
| ReCPE | prior MAE | 0.2715 | 0.0227 |
| Dist-PU | ROC-AUC | 0.9595 | 0.0856 |
| PUSB baseline | ROC-AUC | 0.9128 | 0.0097 |
| LBE linear EM | ROC-AUC | 0.8762 | 0.0207 |

ReCPE 在该设置中的低估是当前默认 density-ratio CPE 后端的实际结果，不应删除或解释为
论文 ReCPE 的表现。完整指标和运行环境分别见 `summary.csv` 与 `run_manifest.json`。

## 结论边界

当前结果必须标为 `clean_room`：

- Dist-PU 使用 toolbox 全量 MLP，不是官方图像 backbone/mini-batch 两阶段训练；
- PUSB 使用来源 Logistic Regression，不是官方 kernel PUSB；
- LBE 使用线性交替 Logistic Regression，不是官方 MLP + Adam；
- CPE 尚缺 L1-QP 和论文 CV；
- ReCPE 尚缺官方 FCNet 和全部 CPE baseline。
