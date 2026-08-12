# InfoMax PU Fashion-MNIST Prior-Matrix Report

## 执行结论

- 状态：`pi=0.3/0.5/0.7` 各 20 seeds，共 60/60 trials 完成。
- 完整性：每组 seed 为 `0..19`，无重复、无缺失、无非有限分类指标。
- 数据：官方 Fashion-MNIST 文件；压缩文件 SHA-256 和逐 seed split hash 见各目录
  `run_manifest.json`。
- 每次划分：1000 个标记正例、2000 个受控先验 U、50 P + 200 U 验证集、10000 test。
- 网络：PURL `d-60-20-1` 200 epoch；下游 `m-300-300-300-1` nnPU 200 epoch。
- 声明：所有结果均为 `paper_claim=false`。

## 聚合结果

均值后的 `±` 为 20 seeds 的样本标准差。

| U 类先验 | Accuracy | Balanced accuracy | ROC-AUC | AUC 中位数 | KM1 估计 | prior MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.6369 ± 0.0258 | 0.6369 ± 0.0258 | 0.8547 ± 0.0819 | 0.8879 | 0.8747 | 0.5747 |
| 0.5 | 0.6394 ± 0.1115 | 0.6394 ± 0.1115 | 0.6313 ± 0.3077 | 0.8363 | 0.8748 | 0.3748 |
| 0.7 | 0.5707 ± 0.1200 | 0.5707 ± 0.1200 | 0.3340 ± 0.2873 | 0.1714 | 0.8748 | 0.1748 |

ROC-AUC 的 95% Student-t 置信区间分别为：

- `pi=0.3`：`[0.8164, 0.8931]`；
- `pi=0.5`：`[0.4873, 0.7753]`；
- `pi=0.7`：`[0.1996, 0.4685]`。

## 稳定性与失配

| U 类先验 | AUC < 0.5 | AUC >= 0.8 |
|---:|---:|---:|
| 0.3 | 0/20 | 15/20 |
| 0.5 | 7/20 | 11/20 |
| 0.7 | 15/20 | 3/20 |

U 集合隐藏正率由 runner 无放回受控采样，三个目录中分别严格为 `0.3`、`0.5` 和
`0.7`。KM1 估计却在三组均饱和到约 `0.875`。当前证据表明 prior estimator 与暂定
图像协议严重失配，并与高先验组的大量反向排序同时出现；不能删除失败 seed，也不能使用
测试真值选择初始化。

## 测试集语义

三个配置均在完整 Fashion-MNIST canonical test split 上评估。暂定正类
`[0,1,2,3,4]` 占测试集一半，因此 `test_positive_rate=0.5`。配置中的
`class_prior` 和 `unlabeled_class_prior` 控制训练 U 分布，不代表测试集部署先验。
ROC-AUC 和 balanced accuracy 可用于跨先验比较；普通 accuracy 不应解释为 0.3/0.7
部署分布下的总体准确率。

## 论文一致性边界

已锁定论文公布的网络深度、BN/ReLU、gradient noise、优化器、样本数、epoch、三种先验
和重复次数。以下字段仍未公开，当前使用显式临时选择：

- 图像正类为 Fashion-MNIST `[0,1,2,3,4]`；
- mini-batch size 为 `256`；
- kernel-mean prior estimator 使用 KM1；
- 测试集使用 canonical split，而非按训练 U 先验重新采样；
- 像素缩放与标准化采用 runner 明示协议。

因此这是完整执行的暂定 `paper_protocol` 矩阵，不是官方论文数值复现。
