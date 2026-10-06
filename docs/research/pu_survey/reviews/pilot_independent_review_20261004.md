# Pilot 独立技术复核（2026-10-04）

> **同日后续收件**：此前待取的 Excel 已收到且摘要一致；645-run → 183-row 独立数值对账通过，见 [Excel 验收补记](../delivery/p21_workbook_review_20261004.md)。本文「未收到表格」属于本轮前半段记录，原始 manifest / 日志 / 权重仍未取得。

范围：按 [执行计划](../survey_execution_plan.md) 复核 P2.0b、P2.0c、P2.0e、P2.1、P2.2；
暂停新增算法集成。P1.4 接收验收见 [数据复核](../delivery/p1_4_review.md)。
复核主体是 Codex 代理技术检查，**不是两名实施主体的本人签署**。

## 1. 分项结论与边界

| 项 | 本轮已核 | 结论 / 未决项 |
|---|---|---|
| P2.0b | 能力声明、标签门禁、oracle 拒绝路径、真实四角色数据契约 | 技术回归确认；合作者签署未代填 |
| P2.0c | 当前 v3 矩阵 54 锚点 / 194 映射、来源原表、归属与代码矛盾 | 见 §2；PUSB 单位及协议完整性仍不能全部接受，矩阵保持 pending |
| P2.0e | 五方法 TS view、校准接线、训练视图路由和契约测试 | 技术回归确认；不据此签署五方法方法学等价，也不把 Self-PU 消融说成完整版 |
| P2.1 | 真实数据可用；五批 dry-run 重建 215/215/110/70/35，总 645 | 计划覆盖确认；正式执行结果仍须取源端制品核验 |
| P2.2 | 审计/汇总/对照工具回归；快照与交接文档交叉检查 | 工具确认并修正文档矛盾；未独立复算正式 645 行，不冒报验收通过 |

核心回归 **420 passed / 0 failures / 0 errors**：P2.0 与对照 205 项，审计/汇总/交付 215 项。
新增接收端脚本与文档检查器测试另有 **18 passed**，本轮合计 438 passed；文档一致性检查及差异空白检查通过。
后者 9 条磁盘容量警告来自测试 fixture 的估算口径，不是实机磁盘审计失败。
模块、逐 split 读数、五批计划身份及已下载锁定源码摘要见
[机器可读证据](../data/pilot_review_20261004.json)。在证据列出的模块上用 `pytest -q` 可重跑。
本轮计划读取当前工作树，completed=0 表示指定的空结果树没有制品，**不是否认 AutoDL 已完成**。
这些计划不能替代源端参考 plan JSON；原执行与重放代码身份必须分别记录。

## 2. P2.0c 原始来源复核（三档结论）

对象为当前 `survey_comparison_v3.json`，绑定 `survey-v1.2`；不覆盖历史 v1/v2 复核记录。

### 2.1 nnPU 原论文：无法判定（完整断言），正文确认

[Kiryo et al. 2017 正文](https://papers.nips.cc/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf)
的 CIFAR-10 证据为 Figure 2(d) / 3(d) 风险曲线，未找到可转录的 accuracy/error 数值。
本轮没有取得并完整检查独立补充材料，因此不能将「正文、附录、所有补充均没有」签为确认。
该来源继续不贡献数值锚点；Self-PU 表中的 nnPU 数值不得归为 Kiryo 原论文产出。

### 2.2 Dist-PU：确认（原表读数）

[Zhao et al. 2022](https://arxiv.org/pdf/2212.02801) Table 2，印刷页 6：
Dist-PU `91.88 ± 0.52`、nnPU `88.89 ± 0.45`，ACC %，5 seeds。
Table 1 / §4 记录 CIFAR P=1000、U=50000、test=10000、先验 0.4、车辆正类 `{0,1,8,9}`。
ImageNet 通道均值/标准差不同于我方 train-only 统计；论文 13-layer CNN 也不能与 adapter 等价。
确认转录不等于确认本项目训练协议可直接数值对照。

### 2.3 Self-PU：确认读数与第三方归属；完整版 PA 资格受限

[Chen et al. 2020](https://proceedings.mlr.press/v119/chen20b/chen20b.pdf)
Table 7，印刷页 9：Self-PU `89.68 ± 0.22`、nnPU `88.60 ± 0.40`、uPU `88.00 ± 0.62`；
13-layer CNN、5 runs、P=1000。作者引用 Kiryo 官方仓库复现 baseline，
nnPU/uPU 的 `third_party_reproduction` 归属确认。
完整版的 validation meta-reweighting 需要干净验证标签，不能冒充纯 PA 训练；
我方 `without_clean_validation_meta_reweighting` 是已披露消融，不能声称复现完整 Self-PU 原表。

### 2.4 PUSB：确认指标与均值；无法判定离散度单位

[Kato et al. 2019](https://openreview.net/references/pdf?id=ryNH0z6BE)
Table 2，印刷页 9，Spambase π=0.4：表头 **error rate**，PUSB `18.9 (.018)`，
100 次重复；accuracy 若转换为 `100 - error`，均值为 81.1%。
本轮未找到对括号单位的明确解释，**1.8 个百分点只是假定括号为 fraction 的读法**。
不能将该假定签为确认或用其改变池化阈值；该锚点保持 `pending_review`。
扫的是 π，不是标记频率 c；π=0.8 的 PU 退化值不能充作方法优势证据。

### 2.5 两篇 benchmark：确认转录；协议完整性尚未通过

[PU-Bench 正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/6277b8eb4230b35d4fabd576aa99ee07-Paper-Conference.pdf)
Table 1，印刷页 6：我方登记的 6 方法 × 3 数据集共 18 个均值及离散度与原表一致。
正文/附录记录 cc/SCAR、c=0.1、10 seeds；选择准则的论文/代码矛盾仍须明确保留。

[PUBench v2](https://arxiv.org/pdf/2509.24228v2) Table 1/2，印刷页 8/9：
5 方法 × 2 case × PA/PAUC/OA 共 30 个均值及离散度与原表一致，case 类别映射不同于本项目。
锁定 commit `a9a62b05b0f222c72aff4df8992307376c78d682`：

- [core/networks.py:42](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/core/networks.py#L42)
  调用 `resnet(depth=32)`，与论文 ResNet-34 矛盾确认。
- [collect_results.py:21–29](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/collect_results.py#L21)
  写明 standard error，计算 `100*np.std(../data/sqrt(n))`，`ddof=0`。以代码为准时已是 SE，不可再除一次 sqrt(n)。
  **不能仅靠代码证明印刷表确由这一函数生成**；缺原日志/checkpoint 时保留产出来源的不确定性。
- [train.py:103–104](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/train.py#L103)
  从 P、U 各切 10% validation，确认。
- [data/datasets.py:19–20](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/data/datasets.py#L19)
  固定 CIFAR mean `(0.4914,0.4822,0.4465)`、std `(0.2471,0.2435,0.2616)`；
  :72–76 使用 `TransformFixMatch`，:415–439 返回 weak/strong 双视图。之前对照表中的归一化「—」应补充，
  但具体算法实际消费哪些视图还须沿训练代码核实，不能泛称所有方法都使用强增强。
- 论文 Appendix C/D（印刷页 19）记录 3 次随机划分、10 组随机超参数、20,000 iterations，
  与我方预注册单候选/epoch 预算不同。除 backbone 外，搜索与增强口径亦需显式披露。

**推翻**「协议差异已完整登记」这一整体断言：上述固定归一化、增强消费与搜索预算需补充取证/登记。
不要事后改阈值、锚点或把定性映射升级 numeric；若需改机器矩阵，按既有规则重发版本并逐映射复核。
本轮仅记录修订请求，v3 不静默改写，36 个待审锚点 / 7 条待审映射、4 条代码矛盾的待审状态不解除。
当前 194 映射仍为 111 无直接锚点 / 54 PA 阻断 / 22 量级趋势 / 4 不可运行 / 3 背景，numeric=0。

## 3. P2.2 文档矛盾修正

**推翻**「B4 是五批中唯一 formal_ready=True」：
[B3a+B3b 快照](../delivery/p2_1_b3ab_snapshot.md) §1/2 记录 B3b 70/70 eligible、偏差为空、无阻断；
B4 同样无 oracle。按单批边界应为 **B3b 与 B4** 无阻断；
B1/B2/B3a 因 oracle c_grid 偏差为 False，**B3a+B3b 合并分析组仍为 False**（§3）。
执行计划、P2.2 交付与复核包已统一这个粒度，绝不将合并分析组改为 True。
这是快照/规则交叉复核；原始报告尚未取得，最终原始 `formal_ready` 字段仍须与源端报告对账。

## 4. 独立验收剩余输入及执行顺序

本服务器没有定位到 XLSX，也没有 AutoDL SSH host 配置；不猜账号或连接凭据。
按 [P2.2 制品索引](../data/p2_2_artifacts_index.json)，需要：

1. `P2.1_results_645runs_20261003.xlsx`：126370 bytes，sha256
   `cafedc98d8983764625d7b32e62301aa8de87e7da9d772e185f6a135df9c1244`。
2. 五批 manifests、逐 epoch selection、白名单 config、`01_audit/02_summary/03_comparison/04_evidence`，
   及五份源端 reference plan JSON、日志/退出码记录。
3. checkpoint 位置/路径映射及归档校验资料、Self-PU 消融证据、快照附件，分别补 A10/A11/A16/A14。

先按索引核制品身份，再在原分析 commit `40826a231ff4871b77c461b2b4d9774e5eb9d35d` 的独立环境重跑三个入口，
核 645 manifest、48 groups、183 rows（180 formal / 3 partial）、0 refused/not_reproducible、numeric=0，
逐行与 XLSX 对账，最后填 P2.2 C/O 复核及签署表。
当前仓库公布的这些统计是**产出方记录**，不是本轮独立复算值。
17 条原审计中的 7 not_run / 1 not_applicable 不因本轮工具测试通过而变成 pass。
P1.4 收件不等于收到正式结果树；签署与 PA 判读的阻断位保持不变。
