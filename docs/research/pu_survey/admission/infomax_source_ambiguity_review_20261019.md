# InfoMax-PU 论文未公开配置核对（2026-10-19）

## 范围与结论

依据 Sakai, Niu, Sugiyama 的论文全文（arXiv:1710.05359v4，共 26 页），核对开发清单中列出的 MNIST/Fashion-MNIST 类别分组、mini-batch size 和 KM 变体。结论是：三者均未由论文完整指定；当前 benchmark 配置采用的具体值只能作为明确披露的工程选择，不能据此宣称逐字复现作者设置。

| 待核项 | 论文证据 | 核对结论 | 当前配置及边界 |
|---|---|---|---|
| MNIST/Fashion-MNIST 正负类别 ID | §5.2 仅称将全部类别分为两组以形成二分类任务 | 未报告具体类别 ID；在全文中未找到能唯一恢复分组的其他说明 | Fashion-MNIST `[0,1,2,3,4]` 是临时分组选择，`positive_definition` 明确为 provisional；不可写成作者标签映射 |
| 表示训练 batch size | §5.2 说明以四个 mini-batch 更新密度比头、再以一个 mini-batch 更新表示网络 | 报告了更新节奏，没有数值 batch size | 当前值 256 是工程实现选择；更新比例 4:1 与绝对 batch 大小是不同事实 |
| 类先验 KM 变体 | §5.2 称使用 Ramaswamy et al. (2016) 的 kernel mean embedding (KM) 方法 | 论文没有命名 KM1 或 KM2 | 配置显式选择 KM1，保留为 provisional；不能将其归因为论文指定 |

上述结论来自论文完整可检索正文及附录，而非仅依据未找到某个关键词：原文 §5.2（arXiv PDF 第 13 页附近）明确给出网络、优化器、4:1 更新、KM 引用、MNIST/Fashion-MNIST 分组概述及样本数，但在类别分组、mini-batch 数值和 KM 子变体处没有进一步定义。Algorithm 1（§4）只写交替更新，也没有给出 batch 数值。来源：[论文全文](https://arxiv.org/pdf/1710.05359)。

## 与仓库配置的对应

- `benchmarks/deep_pu/configs/official_data_infomax_fashion_protocol_pi03.json`、`pi05.json`、`pi07.json` 将类别 `[0,1,2,3,4]`、表示/分类 batch size 256 和 KM1 写为显式值，并逐项保留 known gaps。
- `benchmarks/deep_pu/configs/official/infomax_pu.json` 也将 KM 记为论文来源、变体记为论文未指定。
- `docs/research/method_cards/InfoMax-PU.md` 已将这些值标为临时工程选择；本记录补充了逐项来源位置与核对结论。
- 此核查不修改训练算法、已有 benchmark 产物、冻结协议或正式准入状态。现存 Fashion-MNIST 结果的 `paper_claim=false` 必须保持；若未来要声称论文数值复现，需先有依据锁定标签映射、batch size 与 KM 实现，再批准正式 recipe 并执行对应 benchmark。

## 验收范围

本记录关闭的是“论文是否给出三项细节”的来源审计，不替代 PURL 全算法复现审查、KM 作者实现复核或正式数据/资源验收。现有配置测试应持续断言这些差异仍显式登记，防止临时值在维护中变成未标注的隐式论文事实。
