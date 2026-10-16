# P1.4-R 四数据集完整独立重建技术复核

后续原交付已收到，当前状态见[五数据集接收验收](p1_4r_receiving_acceptance_20261015.md)。
本页及机器收据保留为接收前历史证据，不再把原tar/manifest未收到列为当前缺口。

## 结论与边界

20News原CPU编码进程已正常退出0；四数据集五seed、20个split/100个文件已全部生成。
四个新归档与**新独立索引**完整性检查通过，全部20个split语义契约通过。
这不等于收到或验收10-14来源方原tar，也不替代HENG958的独立人工复核。

机器证据：[汇总收据](../data/p1_4r_rebuild_review_20261015.json)、
[新索引](../data/p1_4r_rebuild_artifacts_index_20261015.json)、
[完整语义报告](../data/p1_4r_rebuild_semantic_review_20261015.json)、
[对原索引的完整诊断](../data/p1_4r_rebuild_original_diagnostic_20261015.json)。
[早期部分复核](p1_4r_independent_rebuild_20261015.md)及其收据保留为历史截点。

## 全量检查结果

| 项 | 观测 |
|---|---|
| 新独立产物完整性 | 20 split、100文件、4归档，integrity.ok=true，problems=[] |
| 角色/索引/标签/先验 | 全20个split通过既有语义检查，四角色隔离 |
| MNIST/F-MNIST | uint8、固定NCHW、train-only通道统计通过 |
| Connect-4 | 126维float32、train-only标准化及常量列检查通过 |
| 20News | 384维float32、finite、unit norm通过，不能由此证明与来源方embedding数值相同 |
| 测试文件身份跨seed | 三个官方test各1个摘要；Connect-4派生test为5个不同摘要 |
| 原参考角色索引 | 20/20一致，完整分母保留，无缺失split |
| 原参考NPZ | 60个仅ZIP creator-system3→0可命中原SHA；其余20个文本NPZ仍different |
| 原参考manifest | 20/20不同；原manifest字节未接收，字段等价性未核实 |
| 原严格核验 | false，诊断退出2；original_archive_receipt_verified=false、formal_acceptance=false |

文本NPZ与原参考大小/摘要不同，无法证明仅ZIP平台标记不同；没有来源方原数组，
不推断是可接受浮点误差，不计算虚构的最大误差，也不据unit norm宣布数值复现。
其余60个的窄哈希证据仅说明压缩payload等其它字节一致，不独立验证来源方方法学正确。

## 新归档身份与操作

新归档位于本机 `data/review-p1_4r-20261015-GvAtLL/archive`，不随Git分发。
文件大小/完整SHA256见新索引；它们与10-14归档是不同交付物，不能覆盖原索引。
接收者只有拿到这些新tar及对应新索引后，才能执行：

```bash
python scripts/review_survey_split_receipt.py \
  --splits <new-rebuild-unpacked-root> \
  --index docs/research/pu_survey/data/p1_4r_rebuild_artifacts_index_20261015.json \
  --archive-dir <new-rebuild-archive-directory> \
  --datasets mnist,fashion_mnist,twenty_newsgroups,connect_4 \
  --seeds 0,1,2,3,4
```

这只核验新独立交付。对来源方原tar，仍使用10-14原索引与原接收流程，不交换两者身份。
此次生成/打包/语义脚本均退出0；原参考诊断退出2是如实拒收，不是改门禁后掩盖失败。

SBERT固定revision、权重/缓存/文本摘要、实际环境及脚本身份已在新收据登记。
仅做离线CPU编码，无GPU训练；临时text overlay不修改系统环境，且不能冒称正式frozen-lock。
语义审计复用现有工具，不是raw逐行回放；该额外检查草稿已暂存，按用户最新要求优先处理Issue #41。
没有修改冻结协议/矩阵、645个历史run或旧Pilot/ADNI数据，也没有公开上传raw或split。

## 尚待完成

获取来源方四个原tar/manifest，核验原身份并对账文本数组与manifest字段；
由HENG958本人记录独立结论。正式新实验的协议/配方、资源和环境批准另行进行，
本次技术检查不解除这些前置。
