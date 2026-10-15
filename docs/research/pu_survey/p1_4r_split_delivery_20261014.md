# P1.4-R 四数据集划分交付记录（2026-10-14）

## 状态与边界

已完成本地生成、打包与自动复核；尚未完成接收方独立人工复核。本记录不代表 HENG958 签收，不代表提交、推送或 PR 已完成。旧 Pilot/ADNI 产物未覆盖。

- 数据集：`mnist`、`fashion_mnist`、`twenty_newsgroups`、`connect_4`。
- seeds：`0,1,2,3,4`；20 个 split、80 个角色 NPZ、20 个 manifest。
- 生成时的本地输出目录 `data/splits_p1_4r_20261014` 已在打包后清理，不再保留。
- 本地归档（交付物）：`data/archive/p1_4r_20261014`，含四个 tar、`review_report.json` 与 `split_artifacts_index.json`；数据文件不随仓库文档分发。
- 仓库证据：[文件与归档索引](data/p1_4r_split_artifacts_index_20261014.json)、[自动复核报告](data/p1_4r_split_review_20261014.json)。

## 每个 seed 的角色数量

顺序为 train / pu_val / clean_val / test。

| 数据集 | 数量 |
| --- | --- |
| MNIST | 54000 / 3000 / 3000 / 10000 |
| Fashion-MNIST | 54000 / 3000 / 3000 / 10000 |
| 20News | 10182 / 566 / 566 / 7532 |
| Connect-4 | 48640 / 2702 / 2703 / 13512 |

MNIST、Fashion-MNIST、20News 各自五个 seed 的 test 文件摘要完全相同；Connect-4 使用派生 test，五个 seed 的 test 摘要各不相同。官方 test 不参与训练源总体先验重组。

## 归档 SHA256

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| `mnist-splits.tar` | 62842880 | `c237ac9596a7841dad1adcddf44e7974fe6bc5b02efd333fb7cce57d5f77e79c` |
| `fashion_mnist-splits.tar` | 159723520 | `869f95795f0a36e87c37db00bb46e0b5650460d314e954fd6e9adc729070be86` |
| `twenty_newsgroups-splits.tar` | 155432960 | `e6ac92e3b5a51ed074691f77e1c99a5309b085e0e51c93833c0fe81011415a8e` |
| `connect_4-splits.tar` | 8519680 | `e720f6a4e7e5f64e182f553d73fe15ac5efc13a65dcd353f18cef638aeead5cd` |

## 自动复核与来源限制

复核报告 `integrity.ok=true`，`problems=[]`；20 个 split、100 个文件与索引一致。复核涵盖归档摘要、角色索引、标签与总体先验、图像 uint8/固定 NCHW 规格及 train-only 通道统计、文本 384 维 finite/unit norm、表格 126 维 train-only 标准化与常量列处理。

原始收据缺少下载时间，不补造时间：manifest 保留完整 `raw_handoff.receipt`、原收据字节 SHA256，并单独记录实际读取文件的输入身份。归档摘要不能代替实际输入摘要，原收据摘要也不代表已知下载日期。

生成使用 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 和固定 SBERT revision 的本地缓存。生成、打包、只读复核任务均退出 0。整合后相关实验测试退出 0；此前全仓非慢速、非 E2E 基线退出 0，但该基线早于最终读取器整合，不将其表述为最终版本全仓测试。

## 接收方复核

将四个 tar 经双方约定的带外渠道交付，接收方应先与仓库索引比较摘要，安全解包后执行：

```bash
uv run python scripts/review_survey_split_receipt.py \
  --splits <unpacked-root> \
  --index docs/research/pu_survey/data/p1_4r_split_artifacts_index_20261014.json \
  --archive-dir <archive-directory> \
  --datasets mnist,fashion_mnist,twenty_newsgroups,connect_4 \
  --seeds 0,1,2,3,4
```

独立复核结论应由接收方另行记录，本地自动报告不替代该结论。

## 后续独立重建（2026-10-15）

本执行端未收到上述四个原tar；已另行启动隔离重建，三数据集15个split语义检查通过，
20News编码仍在进行。部分对账、实际环境和未关闭事项见
[独立复核进度](reviews/p1_4r_independent_rebuild_20261015.md)。
这是新增证据，不覆盖本交付的索引、归档摘要或原自动报告，也不代表原tar接收验收。
