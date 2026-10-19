# Self-PU 官方来源版本核对（2026-10-19）

## 核对范围

只核对官方来源身份、许可声明和 immutable revision 是否进入仓库来源锁；不复核训练算法、
不重放作者代码，也不替代 Self-PU 方法学复核或正式准入。

## 证据与结论

- 官方论文摘要指向作者代码 `TAMU-VITA/Self-PU`；该旧组织路径目前对应规范仓库
  [`VITA-Group/Self-PU`](https://github.com/VITA-Group/Self-PU)。官方仓库 README 自述为 ICML 2020
  Self-PU，且 GitHub 页面标出 MIT license。
- 2026-10-19 只读页面确认当前展示分支为 `master`；随后 `git ls-remote` 对公开仓库只读查询
  `refs/heads/master`，返回 commit `a0e332ae4f8110e2490d597876e36bf837e1060f`。该 SHA 写入
  `benchmarks/assigned_methods/configs/official_sources.lock.json` 和方法台账。未取得 commit
  日期或保存上游文件 hash，因此不扩张为文件级内容校验。
- 该锁固定 2026-10-19 观察到的官方源码快照；台账 `official_exact` 只表示作者来源身份已确认，
  不表示工具箱 clean-room 实现与上游逐行相同。后续行为复核应引用该 SHA 下的具体文件，而非
  浮动 `master`。

## 实验与实现边界

本工具箱是 clean-room 实现。现有 Self-PU 消融、TS 角色适配、`untrusted_u` / `paper_population`
一致性域审计以及 clean-support 探索均保持原样；本记录不改变算法、默认配置、来源状态枚举、
冻结 Pilot、历史结果或 clean-support 标签预算。官方来源确认和 MIT 声明不能推出数值复现或
对第三方数据制品的再分发授权。

## 来源

- [官方仓库](https://github.com/VITA-Group/Self-PU)；官方 README 包含方法身份、训练入口与 MIT 标记。
- [arXiv 论文摘要](https://arxiv.org/abs/2006.11280)；摘要中的作者代码链接为 TAMU-VITA 旧组织路径。
- 本地核验记录：`benchmarks/assigned_methods/configs/official_sources.lock.json`。

## 后续行为核查

本记录仅限来源身份与版本 pin；2026-10-19 后续对固定 SHA 的训练数据流与 meta 目标另作
[源码行为复核](selfpu_source_behavior_review_20261019.md)。该复核发现上游训练入口把 `split=test`
数据用于熵 meta 更新和逐 epoch test 准确率选模；不据此宣称论文使用同一协议，也不把该行为引入工具箱。
