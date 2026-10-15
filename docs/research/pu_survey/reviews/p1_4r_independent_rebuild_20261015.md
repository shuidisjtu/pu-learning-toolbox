# P1.4-R 独立重建复核进度（2026-10-15）

本记录是部分技术复核的时间截点，不替代10-14来源方交付，也不代填 HENG958 签署。
完整范围仍为四数据集×五seed＝20个split/100个文件，未缩减验收范围。
机器摘要与证据文件身份见[进度收据](../data/p1_4r_rebuild_progress_20261015.json)。

## 已执行

从已核对的四数据集 raw handoff，在独立目录重建，原始输入、Pilot 和 ADNI 产物均未覆盖。
MNIST、Fashion-MNIST、Connect-4 共15个split/75个文件已经生成；逐split重新运行
`review_survey_split_receipt.audit_split`，角色隔离、索引/manifest、标签/先验、
图像uint8/NCHW及train-only通道统计、表格126维train-only统计均通过。
这是语义契约检查，不是原始输入逐行校验，也不声称收到来源方归档。

与10-14仓库原索引对账：15个角色索引摘要一致；60个NPZ只有在内存哈希流将
ZIP creator-system 3→0后才命中原文件SHA256，原文件身份仍不匹配。
15个manifest不同，没有原manifest字节，不能把差异全部归因为格式、收据换行或浮点数。
完整参考中20News的五个split尚未生成，仍登记5个manifest/20个NPZ缺失。
[只读诊断](../delivery/split_rebuild_diagnostic_20261015.md)退出2，严格核验仍失败；没有重写制品或放宽门禁。

20News固定SBERT权重已下载并核对SHA256，编码时启用离线模式、隐藏GPU、限制CPU线程；
本记录截点确认原编码进程仍存活，尚无完整文本制品，不因日志暂时无输出重启。
文本依赖位于独立临时overlay，未修补系统conda环境；实际版本逐项登记在收据。
CPU torch 与正式冻结环境不等同，本次不能标为 frozen-lock 正式实验。

## 复现入口与剩余动作

生成入口为 `scripts/prepare_survey_splits.py`，输入根目录
`data/outgoing/raw-handoff-20261013/payload/data/raw`，输出根目录
`data/review-p1_4r-20261015-GvAtLL/splits`，seeds为0–4。
SBERT revision、权重摘要、脚本摘要、隔离环境及部分报告摘要已在收据绑定。
机器路径用于定位本次证据，不是可跨机器复用的安装路径。

1. 等原20News进程结束并检查五seed制品，再建立**新的**四数据集归档/索引。
2. 用 `scripts/review_survey_split_receipt.py` 绑定新索引和新归档复核全部20个split；
   该检查只能证明新独立产物的技术一致性，不能验证未收到的原tar。
3. 对完整原索引执行只读重建诊断，任何未证明的文本差异仍为待核，不能用unit norm替代数值对账。
4. 获取原tar/manifest后完成来源交付身份与逐字段差异复核，由HENG958记录独立结论。

不改冻结协议、比较矩阵或645个历史run，不跑正式PU训练；没有上传raw或新split到GitHub。
数据分享方式与公开再分发条件仍另行确认，不属于本技术复核的放行权限。
