# 独立重建与原收据的差异诊断（2026-10-15）

后续收到原交付后的数组/manifest对账见[接收验收](../reviews/p1_4r_receiving_acceptance_20261015.md)。
本工具边界和以下开发阶段观测保留，不用后续验收结果倒改原严格诊断。

这是 P1.4-R 独立技术复核工具，不是数据接收签署、原归档验收或协议批准。
本次只新增旁路诊断，**没有放宽 `split_archive.verify` 或任何既有门禁**。

## 为什么不能只看重新生成的 NPZ 文件哈希

NumPy NPZ 是 ZIP 容器。相同压缩数据可以因 central-directory 的 creator-system
字节不同而产生不同文件 SHA256（0 为 DOS/Windows 家族，3 为 Unix）。
重建时应先保留原严格文件/索引检查，再查是否仅这个字节不同；
不能把 ZIP 重新打包、重新保存数组后“相似”当作字节一致。

新增 `scripts/diagnose_survey_split_rebuild.py`：

```bash
python scripts/diagnose_survey_split_rebuild.py \
  --root <independently-rebuilt-splits> \
  --index docs/research/pu_survey/data/p1_4r_split_artifacts_index_20261014.json
```

输出完整 JSON，所有 reference 数据集/seed 均保留，不通过删除缺失的数据集缩小分母。
它先运行现有严格 verifier，再逐角色报告：

| 状态 | 含义 |
|---|---|
| byte_identical | 实际文件大小与 SHA256 完全匹配 |
| zip_creator_system_only | 只在内存哈希流替换 creator-system 字节后命中**原 SHA256**，其余所有文件字节未改 |
| different | 未能证明仅上述字节不同，不推断为“浮点误差可接受” |
| missing / unreadable | 原索引要求的制品缺失或无法读取，保留为未完成 |

对 ZIP 做 CRC 检查、固定三成员检查、加密/不支持 creator 拒绝及未压缩512MiB上限；
哈希采用流式稀疏字节替换，不改磁盘数据，不解包到任意路径，不加载 pickle。
manifest 不做忽略字段的归一化；原 manifest 缺失或不同仍如实登记。
role indices 从 NPZ 重算，再与原 reference 的摘要比较，不只信本机新 manifest。
路径、文件清单与符号链接有前置拒绝，原 verifier 的幽灵/额外文件发现不被覆盖。

## 必须区分的三层证据

1. **严格原文件身份**：`strict_unpacked_reference_index.ok` 来自旧 verifier。
   NPZ 平台诊断不会把 false 改成 true；CLI 原索引不匹配仍返回2，运行错误返回1。
2. **独立重建诊断**：creator-only 命中证明原文件与本机文件的全部其它字节相同，
   是压缩 payload 等价的窄证据，不验证来源方当初的数据语义是否正确。
3. **原始归档/人工接收**：本工具不验证未收到的 producer tar，不代填人类签署；
   `original_archive_receipt_verified=false`、`formal_acceptance=false` 始终明确。
   本机新生成的归档只能绑定本机的新索引，不能覆盖原来发布的索引/归档身份。

2026-10-15 开发阶段实际观测：在隔离目录重建 MNIST、Fashion-MNIST、Connect-4，
15个 split 的角色索引摘要匹配；60个角色 NPZ 原 SHA 不匹配，但仅 creator3→0 后
全部命中原 SHA。原 manifest 15个不同；原参考仍包含尚未生成的20News五个split/20个NPZ，
工具如实报告 missing/not_verified，严格结果和 formal_acceptance 都为 false。
这是工具开发时的部分观测，**不是四数据集已经验收**。

测试覆盖双方向哈希复核/不写文件/CLI不放行、真实数组变更、其它 ZIP 元数据变更、
损坏 deflate、诊断上限、整组缺失仍保留分母、路径/清单/size拒绝、symlink和幽灵split。
后续完整独立重建、语义复核及原归档接收须另行记录，不从本工具测试推出。

## 本次工具验证

新增单测14项通过；全仓 `not slow and not e2e` 测试3837项通过、59项跳过，
无失败或错误（3896项，263.428秒）。这是 CPU 开发环境的代码回归结果，
不是四数据集验收或正式 GPU 复现。格式、测试质量、文档链接、项目元数据、
API覆盖、技能同步、公式、基线默认配置、注释、目录结构/依赖层和八profile准入草稿
门禁通过；没有增加豁免或启用尚未冻结的正式 recipe registry。
