# P1.4 接收端数据复核（2026-10-04）

结论：本服务器已收到三数据集 × 5 seed 的真实 split；**取件完整性与数据契约技术复核通过**。
这更新的是接收端事实，不代替 HENG958 / shuidisjtu 的本人签署，也不证明 AutoDL 的 645 次训练结果。

## 1. 接收与验收

接收目录为 `data/splits`；原始三个 tar 位于
`data/incoming/p1.4-splits/P1.4仅划分和预处理后的数据产物`。
按照 [冻结索引](../data/split_artifacts_index.json)，逐文件及原始归档 sha256 校验通过：
**15 个 split、75 个文件、3 个归档**，缺失及摘要差异均为 0。
可归档的逐 split 读数见 [本轮证据](../data/pilot_review_20261004.json)。

| 复核项 | 结果 |
|---|---|
| 四角色及索引隔离 | 15/15 通过 `validate_bundle`；train / pu_val / clean_val / test 无交叠 |
| manifest 与数组 | 所有角色长度、索引次序、正例率、population/train prior 一致；标签合法、特征有限 |
| Spambase | 57 维；train-only 标准化均值/标准差偏差均小于 `1e-6` |
| IMDB | 384 维；四角色 L2 norm 误差不超过 `3e-7`；5 seed 的官方 test 文件摘要一致 |
| CIFAR-10 | uint8 NCHW；从 train 重新计算通道统计，与 manifest 差异最大约 `7.3e-12`；容差为 float32 epsilon |
| 真实脚本可执行性 | Spambase / IMDB 直接 oracle smoke；CIFAR 经冻结微型 CNN adapter 后 oracle smoke，均退出 0 |

CIFAR smoke 只验证四角色经过 adapter 的工程链路，**不使用正式 ResNet-18、不算正式 pilot 或 GPU 复现**。
三个 smoke 均只训练 1 次迭代，收敛警告预期；不据其 accuracy / AUC 作方法学结论。
原始 4D 图像不能直接交给二维 oracle；不带 encoder 的 nnPU 也不能代替正式 native CNN 装配。

## 2. 可重复检查

从仓库根目录执行（只读数据、不训练、不删除）：

```bash
PYTHONPATH=. python scripts/survey_splits_archive.py verify \
  --root data/splits \
  --index docs/research/pu_survey/data/split_artifacts_index.json \
  --archive-dir 'data/incoming/p1.4-splits/P1.4仅划分和预处理后的数据产物'
PYTHONPATH=. python scripts/review_survey_split_receipt.py \
  --archive-dir 'data/incoming/p1.4-splits/P1.4仅划分和预处理后的数据产物'
```

第二条输出逐角色读数 JSON；不要使用 `python -O`（检查中包含 assertions）。
本轮以 `bef43f5133fa4a6385b415effc2fbdc012323428` 为代码基线，工作树保留此前未提交算法修改，
不是 AutoDL frozen-lock 环境。完整性依据原始索引逐字节校验，不由当前代码版本替代。

## 3. 历史文档口径

P2.0a 交付/复核包与 HENG958 早期复核中的「P1.4 待重建 / 未收到」描述是当时记录，原签署保留。
**当前取件状态以本页为准**；执行计划、开发流程清单及索引入口已同步。
人工签署仍待实际复核人填写，不能把本轮代理技术检查写成其本人已签署。
