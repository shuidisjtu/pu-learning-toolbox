# Deep PU Benchmarks

本目录覆盖 InfoMax PU、Weighted Contrastive PU 和 DGPU。所有输出明确分为两层：

- `clean_room`：可立即执行的合成 case-control 多 seed 实验；
- `paper_like`：从论文锁定的视觉/文本配置，当前状态为 `locked_not_executed`。

## 运行 clean-room benchmark

```bash
python -m benchmarks.deep_pu.run \
  --config benchmarks/deep_pu/configs/clean_room_multiseed.json \
  --output benchmarks/deep_pu/results/clean_room_multiseed
```

只运行一个或两个方法：

```bash
python -m benchmarks.deep_pu.run \
  --config benchmarks/deep_pu/configs/clean_room_multiseed.json \
  --output /tmp/deep-pu-benchmark \
  --methods infomax_pu weighted_contrastive_pu
```

每次运行生成逐 seed `trials.csv`、均值/样本标准差 `summary.csv`、实际配置和包含代码、
环境及哈希的 `run_manifest.json`。manifest 永远设置 `paper_claim=false`。

## 方法适配

- InfoMax PU：toolbox PURL MLP + 线性 nnPU head；
- WConPU：表格 MLP、默认噪声 augmentation、prototype 和 momentum queue；
- DGPU：`GaussianConditionalGenerator` 只实现条件 generator 协议，用于验证多轮编排。

这三项都不是论文图像结果。`configs/official/` 锁定论文的数据集、网络、增强、epoch、
重复次数和外部依赖；`official_sources.lock.json` 记录 DOI/作者 PDF 及“未发现官方代码”。

## 已执行结果

`results/clean_room_multiseed/` 已使用 seed `0,1,2` 完成 9 个 trial。聚合结果：

| 方法 | ROC-AUC | Accuracy | Balanced accuracy | Bayes posterior Spearman |
|---|---:|---:|---:|---:|
| InfoMax PU | 0.5771 ± 0.3274 | 0.4956 ± 0.1099 | 0.5122 ± 0.1400 | 0.1315 ± 0.5748 |
| WConPU | 0.9976 ± 0.0012 | 0.7478 ± 0.3106 | 0.7946 ± 0.2462 | 0.9158 ± 0.0707 |
| DGPU Gaussian | 0.9153 ± 0.0661 | 0.8011 ± 0.1434 | 0.8164 ± 0.1205 | 0.6860 ± 0.1589 |

WConPU 的 AUC 稳定但 threshold accuracy 波动较大，说明排序与默认决策阈值不能混为一谈。
InfoMax PU 的短周期 PURL/nnPU 结果高度依赖初始化；这是当前快速配置的真实结果，不代表
论文 200/300 epoch 网络。完整逐 seed 数据、环境和限制分别见 `trials.csv`、`summary.csv`
和 `run_manifest.json`。

## 结论边界

clean-room 结果只回答以下问题：当前 native 接口能否在相同数据和 seed 下稳定执行，输出
是否有限，跨 seed 波动如何，方法特有状态是否完整。它不回答论文视觉 backbone 是否达到
原文精度，也不验证 DGPU 的扩散图像质量。
