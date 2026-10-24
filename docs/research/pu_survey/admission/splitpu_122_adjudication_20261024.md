# Split-PU #122 三项裁决（2026-10-24）

裁决人：项目负责人；日期 2026-10-24。记录见 v3 单元翻牌清单「已裁决事项 1」，
采纳 [20261022 复核](../reviews/heng958_task_review_20261022.md) §1.3 的三条建议。
本记录只固化已经做出的决定，不改冻结协议、不改台账、不改默认训练目标。

| 项 | 裁决 | 落点 |
|---|---|---|
| JSD 方向 | **保持锁定源码 `teacher=0.7` 与现有归一化分母**；论文 Eq. (6) 的 0.3 方向作为已登记差异，不另立变体 | `splitpu_js_loss` 现状、[方法卡](../../method_cards/Split-PU.md)、比较矩阵差异登记 |
| TS batch 比例 | **登记为 benchmark 近似**：均衡 mini-batch 的 `U ∪ P` 不精确恢复总体 π，本轮不按 π 加权 | 协议 §2.3「共性近似」、方法卡 |
| 台账指针 | **授权修正，但在下一次版本化台账更新时执行**；当前冻结台账不动 | `method_ledger.json` 的 `prior_semantics` 指针（现写 `split_pu.py:223-231`，实际先验赋值在 `:221`） |

## 为什么保持源码 0.7

只把 `js_teacher_weight` 改成 0.3 会改变归一化：当前 helper 的分母是 `-(1-w)log(1-w)`，
单独改权重会引入约 **1.4466 倍**缩放，不能称为"论文式修复"。要真正对齐论文需要同时交换
target/pred 并保留分母，并联动实现、测试、方法卡与版本登记——属于另一条独立变更，
不在本轮范围。证据见 [2026-10-18 表征](splitpu_122_characterization_20261018.md) §3。

## 后果

- Split-PU 的两个 `native_2d` 单元翻牌：v3 中 `spambase`、`imdb` 已 `runnable`，
  `cifar10/cnn_feature_adapter` 行同批翻牌。
- **#122 可以关闭**；`#88` 的残余项里不再有 JSD 方向，只剩 LaGAM support 正式路径与共通配方前置。

## 尚未落到冻结文件的部分（不静默改 v3.0）

- `survey_protocol_v3.json` 的 `method_profiles.split_pu.configuration_review.findings` 与
  `configuration_audit.method_findings.split_pu` 仍写着
  "provisional while issue #122 ... remain unresolved"。这两处随下一次版本化协议更新修正，
  与同一文件里 `formal_blockers`/`draft_scope` 的陈旧措辞一并处理（见
  [执行计划 §1.4.1](../survey_execution_plan.md)）。
- 台账指针按上表推迟；在版本化更新前，引用台账时应以本记录的 `:221` 为准。
