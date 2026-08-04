# Deep PU Benchmarks

本目录覆盖 InfoMax PU、Weighted Contrastive PU 和 DGPU。所有输出明确分为三层：

- `clean_room`：可立即执行的合成 case-control 多 seed 实验；
- `official_data`：使用公开官方数据验证下载、PU split、训练、resume 和 provenance；
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

## 运行 official-data smoke

公开数据保存在仓库外，避免误提交大文件。首次运行会下载 Fashion-MNIST：

```bash
python -m benchmarks.deep_pu.run_official_data \
  --config benchmarks/deep_pu/configs/official_data_smoke_fashion_mnist.json \
  --output benchmarks/deep_pu/results/official_data_smoke_fashion_mnist \
  --data-root /tmp/pu-toolbox-data \
  --download \
  --resume
```

下载完成后可去掉 `--download`。`--resume` 只跳过配置一致且已完成的
`(method, seed)`；配置变化会直接报错，防止不同实验混写。runner 支持 MNIST、
Fashion-MNIST、CIFAR-10 和 20 Newsgroups。视觉数据在当前 native 二维接口中使用
flattened pixels，因此只能标记为 smoke。

每次执行保存 `preflight.json`、逐 seed `trials.csv`、`summary.csv`、
`resolved_config.json` 和 `run_manifest.json`。manifest 包含原始数据 SHA-256、split 哈希、
配置/runner 哈希、Git 状态及依赖版本。

## 审计完整论文配置

在训练前检查 GPU、EDM backend、授权数据和未接入模块：

```bash
python -m benchmarks.deep_pu.preflight_paper \
  --config-dir benchmarks/deep_pu/configs/official \
  --output benchmarks/deep_pu/results/official_preflight/current_node.json
```

迁移到 GPU 节点后，可用 `--edm-backend <import-path>` 声明 DGPU backend，并用
`--accept-dataset "CelebA"` 或 `--accept-dataset "Alzheimer MRI"` 明确确认授权数据。
这些参数只消除资源阻塞，不会自动消除配置中记录的实现差距。

InfoMax PU 的论文网络协议可先做只读 preflight：

```bash
python -m benchmarks.deep_pu.run_official_data \
  --config benchmarks/deep_pu/configs/official_data_infomax_fashion_protocol.json \
  --output benchmarks/deep_pu/results/infomax_fashion_protocol_preflight \
  --data-root /tmp/pu-toolbox-data \
  --preflight-only
```

该配置锁定 `d-60-20-1` PURL、全隐藏层 BN/ReLU、gradient noise `0.01`、
`m-300-300-300-1` nnPU head、Adam、200 epoch 和 20 seeds。论文未公开图像类别分组
编号和 batch size。runner 已接入互斥的 `50 P + 200 U` validation split 与 KM1/KM2
class-prior estimator；由于论文没有说明 KM 变体，配置暂锁 KM1。未公开细节和未执行的
20-seed 全量实验意味着结果仍须保持 `paper_claim=false`。

WConPU 的 CIFAR-10 视觉协议可做只读 preflight：

```bash
python -m benchmarks.deep_pu.run_official_data \
  --config benchmarks/deep_pu/configs/official_data_wconpu_cifar10_protocol.json \
  --output benchmarks/deep_pu/results/wconpu_cifar10_protocol_preflight \
  --data-root /tmp/pu-toolbox-data \
  --preflight-only
```

该链路支持 NCHW、clean-room 13-layer CNN、ResNet-18/50、SimAugment、RandAugment 和
cosine annealing。runner 会先隔离 clean 10% validation，再从剩余样本构造互斥的
`1000 P + 44000 U` 训练集；每个 seed 对两项 loss weight 执行 `4 x 4` grid search，候选
写入 `model_selection.csv`，最优参数从头 refit，且候选级支持断点续跑。论文未公开 CNN
逐层结构、增强参数及 validation 指标，原文 `nP/nU` 计数也存在重叠语义歧义；当前 accuracy
选择指标属于显式暂定协议，因此仍为 `paper_claim=false`。

## 方法适配

- InfoMax PU：toolbox PURL MLP + 线性 nnPU head；
- WConPU：表格 MLP 或 NCHW 视觉 encoder、双增强、prototype 和 momentum queue；
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

Fashion-MNIST official-data smoke 已完成 seed `0,1,2`。四个官方压缩文件通过 MD5，
逐文件 SHA-256 记录在 manifest 中；InfoMax PU 的 ROC-AUC 为
`0.4420 ± 0.0874`，balanced accuracy 为 `0.4622 ± 0.0588`。该结果使用 400 个训练样本、
500 个测试样本、flattened pixels 和 5 epoch，只证明真实数据执行链路，不用于性能比较。

当前节点的完整配置审计结果为 `all_ready=false`：无可用 CUDA；WConPU 仍缺论文未公开的
CNN/增强细节及 validation 指标，并且尚未执行长周期实验；DGPU 尚缺条件 EDM backend；CelebA 与 Alzheimer
MRI 访问未确认；InfoMax 仍有未公开协议字段。详见
`results/official_preflight/current_node.json`。

## 结论边界

clean-room 结果只回答以下问题：当前 native 接口能否在相同数据和 seed 下稳定执行，输出
是否有限，跨 seed 波动如何，方法特有状态是否完整。它不回答论文视觉 backbone 是否达到
原文精度，也不验证 DGPU 的扩散图像质量。

同样，`official_data` 只比 clean-room 多验证真实数据来源和 PU split。只有消除 official
配置的全部 blocker、执行完整重复次数并独立核对论文表格后，才能把结果升级为论文复现。
