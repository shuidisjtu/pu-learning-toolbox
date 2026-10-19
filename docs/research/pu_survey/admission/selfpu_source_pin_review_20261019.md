# Self-PU 官方来源版本核对（2026-10-19）

## 核对范围

只核对官方来源身份、许可声明和 immutable revision 是否进入仓库来源锁；不复核训练算法、
不重放作者代码，也不替代 Self-PU 方法学复核或正式准入。

## 证据与结论

- 官方论文摘要指向作者代码 `TAMU-VITA/Self-PU`；该旧组织路径目前对应规范仓库
  [`VITA-Group/Self-PU`](https://github.com/VITA-Group/Self-PU)。官方仓库 README 自述为 ICML 2020
  Self-PU，且 GitHub 页面标出 MIT license。
- 2026-10-19 只读查看仓库页面可确认其当前展示分支为 `master`。本轮无法读取提交历史的
  immutable SHA：浏览器未能取回 commits 页，终端也无法建立到 GitHub 的只读连接；没有据此
  臆造 commit、日期或文件摘要。
- `benchmarks/assigned_methods/configs/official_sources.lock.json` 因而将 `self_pu` 登记为
  `status=partial`、`commit=null`。台账 `official_exact` 只表示作者仓库来源已识别，不表示
  某一源码快照已锁定。需要版本化源码审计时，应先取得 commit SHA，再对照论文和方法卡中的
  关键文件/训练入口。

## 实验与实现边界

本工具箱是 clean-room 实现。现有 Self-PU 消融、TS 角色适配、`untrusted_u` / `paper_population`
一致性域审计以及 clean-support 探索均保持原样；本记录不改变算法、默认配置、来源状态枚举、
冻结 Pilot、历史结果或 clean-support 标签预算。官方来源确认和 MIT 声明不能推出数值复现或
对第三方数据制品的再分发授权。

## 来源

- [官方仓库](https://github.com/VITA-Group/Self-PU)；官方 README 包含方法身份、训练入口与 MIT 标记。
- [arXiv 论文摘要](https://arxiv.org/abs/2006.11280)；摘要中的作者代码链接为 TAMU-VITA 旧组织路径。
- 本地核验记录：`benchmarks/assigned_methods/configs/official_sources.lock.json`。
