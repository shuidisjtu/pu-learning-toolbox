# P2.0c 复核包

复核对象：`pu_toolbox/experiment/survey_comparison_v1.json`（`survey-comparison-v1`）。
交付说明见 [交付文档](p2_0c_delivery.md)；矩阵绑定的执行协议为 `survey-v1.2`。

| 时点 | comparison 摘要 | 说明 |
|---|---|---|
| C0 | `060c6cd9…b6335bc` | 合并入 main 时，全部锚点与映射 `pending_review` |
| C0′ | `412d5f4f…2f9e1a0` | shuidisjtu 自审落库后（本次） |

绑定的执行协议摘要 `S0 = b5b6b5f4…ed2ff20` 全程未变——**移除 `survey_protocol_v1.json` 的 P2.0c
阻断项属于签署动作**，不在自审范围内。

## 1. shuidisjtu 自审

### 1.1 署名口径

取证底稿的状态列写「**本人回查原始来源并记录表号页码**」。复核人确认该措辞成立，即表中所记的
表号、页码与源码位置为其亲手核对，签署记录按此措辞引用，不作分层改口径。

### 1.2 三处判读（均为保持现状）

| 项 | 判定 | 含义 |
|---|---|---|
| `pusb_kernel` 与某 benchmark 对应行的身份 | 保持 `magnitude_and_trend` | 承认二者同属一个方法家族但不构成直接等价，只比量级与趋势，不做数值裁决 |
| 两处 PUSB 行级异常 | 保留为已知疑点 | 不判为排版错误、不剔除相关锚点；异常本身留在记录里供后来者复查 |
| `18.9 (.018)` 的离散度读法 | 按分数读作 1.8 个百分点 | 该值直接进入池化标准误，此读数即阈值依据 |

第一项的备选是降为「无直接对照」，第二项是剔除锚点，第三项是改按百分数读。三项均取现状，
因此**自审不涉及任何数值、分类或映射的修改**，落库只写复核状态字段。

### 1.3 抽样数值复核

已抽样核对：唯一带指标转换的条目（原论文的 error rate→accuracy）、归属层存疑的第三方复现条目、
另一原论文的两个 CIFAR 数值、以及 benchmark 标准误判读与显著性上标丢弃两处转录处理。
抽样结论与登记一致。

### 1.4 覆盖事实与矛盾记录

无锚点的 169 条映射不逐条复核，其分类由三条来源覆盖事实支撑：只有其中一个 benchmark 具备标签频率轴
且仅一档；另一个用固定正例率而非标记频率；四篇原论文给的是真值类先验、全部没有该轴。三条成立，
则 169 条成立。

四条论文/代码矛盾维持 `resolution=code`；其中 backbone 深度一条若被推翻，受影响锚点的
协议可信度与阈值公式须同步修改，该连带关系已记在交付文档。

### 1.5 落库结果

54 个锚点与 194 条映射 `review_state: pending_review → accepted`，`verified_by: ["shuidisjtu"]`。
`review_status` 保持 `pending_collaborator_review`，`formal_blockers` 保持 `["collaborator_review"]`。

> 状态约束说明：`review_status=accepted` 会强制每个锚点与映射都必须 accepted，
> 因此「先自审、后合作者复核」是允许的；反向不成立。

## 2. HENG958 独立复核（未开始）

范围：nnPU、Dist-PU、Self-PU 的数值，以及 CIFAR backbone / 训练协议差异。
判定为「推翻」或「无法判定」的条目，其锚点须回到 `pending_review`，不得随签署一并接受。

## 3. 签署记录（未开始）

双人复核完成后按固定顺序原子落地：执行协议仅移除 P2.0c 阻断项得到 `S1`；comparison 改绑 `S1`
仍保持未接受得到 `C1`（双人最终复核对象）；comparison 置 `accepted` 并仅移除自身阻断项得到 `C2`；
`S1` 与 `C2` 在同一个 PR 内落地，不提交中间悬挂状态。签署记录须含交付 commit、验证 HEAD、
comparison version/digest、来源复核者与保留边界。
