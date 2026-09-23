# P2.0c 复核包

复核对象：`pu_toolbox/experiment/survey_comparison_v2.json`（`survey-comparison-v2`，摘要
`764ef03a…5b4095b`）。交付说明见 [交付文档](p2_0c_delivery.md)；矩阵绑定的执行协议为 `survey-v1.2`。

| 时点 | comparison 摘要 | 说明 |
|---|---|---|
| C0 | `060c6cd9…b6335bc` | 合并入 main 时，全部锚点与映射 `pending_review` |
| C0′ | `412d5f4f…2f9e1a0` | shuidisjtu 自审落库后；保留为历史基线 |
| C0″ | `90c389c8…58d6563` | 独立技术审计修订后；36 个锚点与 7 条行级映射退回待审，未签署 |

绑定的执行协议摘要 `S0 = b5b6b5f4…ed2ff20` 在本轮审查范围内未变——**移除 `survey_protocol_v1.json`
的 P2.0c 阻断项属于签署动作**，不在自审范围内。

> **2026-09-20 补记**：PA 正式准则实现（PR #65）改写了 `pa_criterion` 与 decisions，
> 协议摘要变为 `S0′ = c15b0c9e…eaff529`，`survey_comparison_v1.json` 的绑定已同步重绑。
> **54 条 `blocked_pending_pa_criterion` 未动**，`review_status` 仍为 `pending_collaborator_review`——
> 实现准则不等于裁决这些单元。签署流程请以 `S0′` 为基线。

> **2026-09-23 补记**：审计 nnPU/Spambase 锚点时发现 v1 的两处登记缺陷——错挂在全部数据集行上的
> CIFAR 预处理差异、以及缺失的训练视图维度。已按预注册规则「确需修订须记录理由并重发预注册
> 版本」重发为 `survey-comparison-v2.json`（摘要 `764ef03a…5b4095b`）。**锚点数值、判定规则与
> 资格判定均未变动**，仅修正 15 个 PU-Bench 映射的 `protocol_differences`；`v1` 原样留档。
> 修订理由与重放证据见 [执行计划](survey_execution_plan.md) 决策 D10。**本轮复核与签署的对象为 v2。**

## 1. shuidisjtu 自审

### 1.1 署名口径

取证底稿的状态列写「**本人回查原始来源并记录表号页码**」。复核人确认该措辞成立，即表中所记的
表号、页码与源码位置为其亲手核对，签署记录按此措辞引用，不作分层改口径。

### 1.2 三处判读（均为保持现状）

| 项 | 判定 | 含义 |
|---|---|---|
| `pusb_kernel` 与某 benchmark 对应行的身份 | 保持 `magnitude_and_trend` | 承认二者同属一个方法家族但不构成直接等价，只比量级与趋势，不做数值裁决 |
| 两处 PUSB 行级异常 | 保留为已知疑点 | 不判为排版错误、不剔除相关锚点；异常本身留在记录里供后来者复查 |
| `18.9 (.018)` 的离散度读法 | 当时按分数读作 1.8 个百分点 | 属 C0′ 自审假设；现行该行并不参与数值裁决，且单位未获独立证实，见 §2.1 |

第一项的备选是降为「无直接对照」，第二项是剔除锚点，第三项是改按百分数读。三项均取现状，
因此**自审不涉及任何数值、分类或映射的修改**，落库只写复核状态字段。

### 1.3 抽样数值复核

已抽样核对：唯一带指标转换的条目（原论文的 error rate→accuracy）、归属层存疑的第三方复现条目、
另一原论文的两个 CIFAR 数值、以及 benchmark 标准误判读与显著性上标丢弃两处转录处理。
抽样结论与登记一致。

### 1.4 覆盖事实与矛盾记录

`no_direct_anchor` 111 条、`blocked_pending_pa_criterion` 54 条，共 165 条不逐条复核；另有
`non_runnable` 4 条，三类合计 169 条。无直接锚点/PA 阻断的分类由三条来源覆盖事实支撑：
只有 PU-Bench 有标签频率轴且仅一档；PUBench 用固定正例率而非标记频率；四篇原论文
给的是真值类先验、全部没有该轴。非可运行行则由执行矩阵和覆盖测试单独证明。

四条论文/代码矛盾维持 `resolution=code`；其中 backbone 深度一条若被推翻，受影响锚点的
协议可信度与阈值公式须同步修改，该连带关系已记在交付文档。

### 1.5 落库结果

54 个锚点与 194 条映射 `review_state: pending_review → accepted`，`verified_by: ["shuidisjtu"]`。
`review_status` 保持 `pending_collaborator_review`，`formal_blockers` 保持 `["collaborator_review"]`。

> 本段记录 C0′ 的历史状态；技术审计后的待审状态和新增门禁见 §2。

## 2. 独立复核交接（HENG958 正式确认待办）

范围是深度方法数值与 CIFAR 训练协议（backbone、通道归一化、验证集比例、类别映射、重复数、选模协议）。

**逐项复核材料见 [issue #57](https://github.com/shuidisjtu/pu-learning-toolbox/issues/57)**：每项给出待核
断言、出处、登记值、推翻条件与可核位置（仓库 + 锁定 commit + 文件行号），并写明不需要复核的部分与
结论格式。清单只在 issue 维护——仓库内留第二份副本只会漂移，而漂移的必然是这一份；本文件记录的是
复核结论与它对矩阵的影响。

判定为「推翻」或「无法判定」的条目，其锚点须回到 `pending_review`，不得随签署一并接受。

### 2.1 来源技术审计结论（2026-09-18；不代替 HENG958 签署）

以下为对锁定来源的辅助核查结论，供合作者在 issue #57 复核。仅审来源登记与协议差异，
不验收算法实现，也不产生 pilot 数值。

| issue 项 | 判定 | 证据与矩阵处理 |
|---|---|---|
| 1 nnPU | 确认 | [原论文](https://papers.nips.cc/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf) CIFAR-10 仅有 Figure 2(d)/3(d) 风险曲线；[补充材料](https://papers.nips.cc/paper_files/paper/2017/file/7cce53cf90577442771720a370c3c723-Supplemental.zip) 未找到可登记的 CIFAR accuracy/error 数值。维持无原论文锚点。 |
| 2 Dist-PU | 确认 | [论文 Table 1–2](https://arxiv.org/pdf/2212.02801) 的 CIFAR ACC 为 Dist-PU `91.88 (0.52)`、nnPU `88.89 (0.45)`；5 seeds、1000 标记正例、先验 0.4、正类 `{0,1,8,9}`，并用 ImageNet 通道统计。数值保留，行级协议差异须按来源改写。 |
| 3 Self-PU | 推翻协议口径完整性 | [论文 Table 7、脚注与式 (5)](https://proceedings.mlr.press/v119/chen20b/chen20b.pdf) 支持 Self-PU `89.68 (0.22)`、nnPU `88.60 (0.40)`、uPU `88.00 (0.62)`，13-layer CNN、5 runs；nnPU/uPU 为作者调用 Kiryo 官方代码的第三方复现，并非 nnPU 原论文数值。但完整自校准使用真实标签验证批次，是 PA-ineligible。本项目 `require_validation=false` 为 no-meta 消融，行级映射须显式区分，不得冒充论文完整方法。 |
| 4 PUSB | 离散度单位无法判定 | [论文 Table 2](https://openreview.net/references/pdf?id=ryNH0z6BE) 确认为 error rate `18.9 (.018)`；原文未明示括号单位，`1.8` 个百分点只能暂作解释，锚点须待审。该行当前为 `magnitude_and_trend`，不进入数值裁决；issue 所述“直接改变判定阈值”不适用于当前矩阵。 |
| 5 CIFAR 协议 | 推翻登记完整性 | [PUBench 论文](https://arxiv.org/pdf/2509.24228)写 ResNet-34/标准差，锁定[网络代码](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/core/networks.py#L42)为 ResNet-32、[汇总代码](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/collect_results.py#L21)为标准误，这两项已正确登记；但[固定 CIFAR 通道统计/训练变换](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/data/datasets.py#L19)与本项目 train-only、无增强口径不同，且若干行级映射误写“来源扫描类先验”。按来源补全后重审。 |

第 5 项其余协议轴按锁定代码再核：[PU-Bench 的 CIFAR 输入](https://github.com/XiXiphus/PU-Bench/blob/2d95a19eefd72e66ff30128ec2f65e1d1d4cc077/data/CIFAR10_PU.py#L76)
仅 `/255`，[配置](https://github.com/XiXiphus/PU-Bench/blob/2d95a19eefd72e66ff30128ec2f65e1d1d4cc077/config/datasets_vary_c/param_sweep_cifar10.yaml#L12)
用 1% 验证比例与 `{0,1,8,9}` 正类，[选模代码](https://github.com/XiXiphus/PU-Bench/blob/2d95a19eefd72e66ff30128ec2f65e1d1d4cc077/config/methods/pn.yaml#L16)
用代理准确率；PUBench [训练入口](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/train.py#L103)
从 P/U 各留 10%，[类别映射](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/lib/misc.py#L304)
的 Case 1 `{0,1,2,8,9}`、Case 2 `{2,3,5,7,9}` 均不同于本项目。PUBench 结果为 3 次随机划分、
PA/PAUC/OA 分别选模；与本项目 ResNet-18、train-only 通道统计、10–20% 验证比例、PA/OA 双协议均不能直接等同。
上述差异已写入对应行级映射；论文与代码两条矛盾维持 `resolution=code`，但无法证明论文表格当年究竟使用
哪一版训练代码或误差棒，故相应锚点仍保持 `protocol_provenance=uncertain`。

### 2.2 P0 修订边界

- C0′ 的 54/194 `accepted` 是提交者自审的历史记录，不是合作者签署。对受上述第 3–5 项影响的
  锚点/行级映射回退 `pending_review`，保留原 `verified_by` 以记录曾由谁自审；不改论文读数、
  `eligibility` 或已冻结的数值判定规则。C0″ 现为锚点 18 accepted / 36 pending、映射
  187 accepted / 7 pending；退回范围为 PUBench CIFAR 30 条、Dist-PU 2 条、Self-PU 3 条、
  PUSB 1 条。摘要另记，不复用 C0′；36 条并非读数全部被推翻，而是相关协议或单位仍须复核。
- 四条 `contradictions` 目前也为 `pending_review`。正式 `review_status=accepted` 必须要求它们
  全部通过复核且无 `formal_blockers`；工程门禁与测试在 P0 补齐。PUSB 单位若最终无法查明，
  不得把暂定离散度签为确定值。
- HENG958 在 issue #57 给出真实复核意见之前，本节仅是技术审计记录；§3 的 S1/C1/C2
  双人原子签署步骤不执行。P2.0b 与 R9 可并行推进，正式 P2.1 仍受全部阶段 A 门禁约束。

## 3. 签署记录（未开始）

双人复核完成后按固定顺序原子落地：执行协议仅移除 P2.0c 阻断项得到 `S1`；comparison 改绑 `S1`
仍保持未接受得到 `C1`（双人最终复核对象）；comparison 置 `accepted` 并仅移除自身阻断项得到 `C2`；
`S1` 与 `C2` 在同一个 PR 内落地，不提交中间悬挂状态。签署记录须含交付 commit、验证 HEAD、
comparison version/digest、来源复核者与保留边界。
