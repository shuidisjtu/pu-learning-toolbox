# P1.4-R 五数据集接收与技术验收（2026-10-15）

## 当前结论

**技术验收通过，保留已登记限制；HENG958独立人工复核待办。**
本执行端已实际接收四新增数据集和ADNI替代集的原交付：5归档、25split、100个角色NPZ、
25个manifest，共125文件；全部与**交付前已入库的原索引**一致，不是只核对随包自带的摘要。
本记录更新当前取件/技术状态，不代填负责人或合作者签署，不批准主榜训练。

证据：[汇总收据](../data/p1_4r_receiving_acceptance_20261015.json)、
[接收完整性](../data/p1_4r_received_integrity_20261015.json)、
[四数据集语义](../data/p1_4r_received_four_review_20261015.json)、
[替代集语义](../data/p1_4r_received_adni_review_20261015.json)、
[源数据/重建对账](../data/p1_4r_received_source_comparison_20261015.json)、
[替代集原始像素回放](../data/p1_4r_received_adni_source_20261015.json)。
三份本轮验证job的源码和摘要作为[可审阅证据](../data/p1_4r_received_verification_jobs_20261015.json)留档；
它们是本轮代理编写的验证代码，不是收到的合作者结果生成脚本，也不引入新公共API。

## 收件、安全性与身份

- `p1_4r_20261004.zip`最终稳定文件为356877857字节，SHA256为
  `1c2cbbe1716aaa02fa75a38b84fedf830b6ea7670c568a67ad146b972218c916`。
  文件名不是内容日期：内部目录为 `p1_4r_20261014/`，含四原tar、原review report和原索引。
  首次观测时仍在写入、缺ZIP末尾目录；只使用随后稳定且CRC通过的完整文件验收，早期不稳定摘要不作为证据。
- `adni-splits.tar`为292147200字节，摘要
  `ae1a3fefe30d35aa3ef5822bee9d26d9ac110d2dbd7244b4ddcd5bea9a48414e`，与10-12仓库原索引一致。
- 两个收到的索引JSON与原仓库索引对象相同，文件字节差异均可由内存CRLF→LF完全解释；
  仍保留各自文件SHA，不改写原索引。tar/文件验证始终采用仓库原索引。
- ZIP成员固定白名单、拒绝加密/特殊文件并检查CRC；tar须完全匹配索引，拒绝重复、额外、
  缺失、链接、特殊/sparse成员或大小不符，解包文件逐一核对摘要。
- 隔离目录为本机 `data/review-p1_4r-received-20261015-rndXxi`，四集与替代集分别置于
  `splits_four`、`splits_adni`；原始上传件、Pilot、旧重建和所有历史结果均未覆盖。

## 验收范围

| 数据集 | 每seed train / pu_val / clean_val / test | 关键核验 |
|---|---|---|
| MNIST | 54000 / 3000 / 3000 / 10000 | 偶数为正；uint8 NCHW；官方test边界与train-only统计 |
| Fashion-MNIST | 54000 / 3000 / 3000 / 10000 | 正类0/2/3/4/6；uint8 NCHW；官方test与train-only统计 |
| 20News | 10182 / 566 / 566 / 7532 | 20→7映射、0–3正类、固定SBERT revision；384维finite/unit norm |
| Connect-4 | 48640 / 2702 / 2703 / 13512 | Win正类；b/o/x固定126维独热；只用train拟合z-score |
| ADNI替代集 | 4608 / 256 / 256 / 1280 | NonDemented正类；6400张；灰度128×128；train-only统计 |

25个split均通过角色索引/标签/先验、四角色隔离和完整源行覆盖检查。
四集raw reader input身份与manifest一致；图像/表格实际特征及标签逐行回放通过。
20News语料摘要、索引和标签通过；从收到的全部向量重建规范排序缓存，命中原manifest
缓存SHA `4306b7da57d36da6d3be95d7a5d78e415dc633c6bf14b2206fc8f04de44b88c6`，
且各seed均指向同一向量表。该核查不是另一次独立SBERT复现。

ADNI替代集四个原ZIP按已登记commit重新只读取得，大小/SHA与10-12收据全部相同。
使用仓库灰度/LANCZOS读取器回放后，5seed全部像素和源标签逐行一致；缩放后精确重复图像为0。
读取器/变换仍复用项目实现，不声称独立推导另一套图像处理。

## 此前重建差异已如何闭合

前三数据集60个角色文件的数组完全一致；文件SHA差异已有ZIP creator-system字节证据。
15个manifest在内存统一CRLF→LF后与本地重建逐字节一致，没有忽略字段或改磁盘文件。

20News的20个角色文件索引和标签一致、向量数值不完全一致；5个manifest解析后仅
`preprocessing.cache_sha256`不同。实测最大元素差 `3.390014171600342e-7`，
最大逐角色RMSE `4.568318472052615e-8`。这是观测，不据此新增“可接受”阈值；
unit-norm容差与向量元素差是不同量，不能混用。
不臆断差异来自哪一项环境因素，不宣布两个embedding制品身份等价。
后续使用原接收产物及其原索引，不能把本地重建向量或manifest混进去。

## 可复验入口与剩余限制

```bash
python scripts/review_survey_split_receipt.py \
  --splits data/review-p1_4r-received-20261015-rndXxi/splits_four \
  --index docs/research/pu_survey/data/p1_4r_split_artifacts_index_20261014.json \
  --archive-dir data/review-p1_4r-received-20261015-rndXxi/archives_four \
  --datasets mnist,fashion_mnist,twenty_newsgroups,connect_4 --seeds 0,1,2,3,4
python scripts/review_survey_split_receipt.py \
  --splits data/review-p1_4r-received-20261015-rndXxi/splits_adni \
  --index docs/research/pu_survey/data/adni_split_artifacts_index_20261012.json \
  --archive-dir data/review-p1_4r-received-20261015-rndXxi/archives_adni \
  --datasets adni --seeds 0,1,2,3,4
```

上述两项实际数据审计均退出0；预处理、归档及接收检查相关回归162项全部通过。
本轮没有训练模型、占GPU、执行上传的Python脚本或公开上传raw/split。
仍须保留：

1. HENG958本人独立复核/签署，不能用代理技术报告代替。
2. 替代集不是ADNI官方数据；缺subject id，不能排除同一人的切片跨角色，也不能用无精确重复代替该判断。
3. 来源许可/再分发不确定性仍按原收据保留；本次接收不新授公开发布权限。
4. 新数据主榜的正式协议、配方、资源及frozen-lock执行另行批准；旧冻结协议/比较矩阵不改。

下一项不依赖原训练任务的工作，是核对新收到的P2.1/P2.2修正包：先验包内身份、
再用仓库审计器重做Excel对账；没有收到的原结果树/manifest/权重仍须明确缺失。
