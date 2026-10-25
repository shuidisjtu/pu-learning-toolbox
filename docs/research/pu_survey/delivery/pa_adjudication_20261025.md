# 54 条 PA 映射裁定（2026-10-25，D41）

> **2026-10-25 更正**：本记录的第一版把 18 个 `proxy_acc` 锚点从 OA 单元迁到 PA 单元。
> 读完两篇参考文献 PDF 后判定该方向**错误**，已按下文重写。错因见 §4。矩阵侧尚未落地，
> 因此没有产生任何需要回滚的制品，只需修正裁定本身。

D38 记的是「PA 准则不再是外部依赖，剩下的是我方重新裁定这 54 条映射」。本记录给出裁定结果、
依据，以及为什么它落在 comparison v5 而不是现在改矩阵。

## 1. 裁定结论

54 条 `blocked_pending_pa_criterion` = 6 方法 × 3 数据集 × 3 个 `c`，**全部是 `pa` 选模单元**：

|  | 条数 | 新 eligibility | 锚点 |
|---|---:|---|---|
| 全部 PA 单元（c=0.1/0.3/0.5 与 SAR 行） | 54 | `no_direct_anchor` | 无 |

`blocked_pending_pa_criterion` 在 v5 里归零。**18 条 c=0.1 的 OA 单元维持现状**
（`magnitude_and_trend` + 各自锚点），不做迁移。

## 2. 依据：两篇参考文献怎么选模

P4.2a 的前置是核对两篇基准论文是否真的报了这四个方法。核对结果见
[p42a_paper_check_20261025](p42a_paper_check_20261025.md)，与本次裁定相关的有两条：

- **Chen et al.（PU-Bench）Table 1** 确实报了 `pulda` / `robust_pu` / `holistic_pu` / `lagam`，
  8 数据集、`c=0.1`、SCAR、accuracy mean±std。但它的选模是
  **「validation-set macro-F1 最高的 checkpoint」**（正文 §3 与 §D.2.1 两处都这么写），
  而该验证集是**带真标签**的（Table B.2 把 `Validation` 列在 PN 划分统计里；Wang et al. 附录 A
  直接指出「they assumed that the validation set had true labels, which is unrealistic」）。
- **Wang et al.** 的 17 个方法里只有 `Robust-PU` / `Holistic-PU`，没有 `pulda` / `lagam`；
  数据集是 CIFAR-10 / ImageNette / USPS / Letter，且**按 PA / PAUC / OA 三种准则分别报数**。

结论：**Chen 的数字来自真标签选模**，与我们的 **OA**（clean_val 真标签选模）同类，
与 **PA**（PU 视图 proxy accuracy）不同类。所以：

- Chen 的锚点归属 OA 单元是对的（v4 现状就是如此），PA 单元拿不到任何 Chen 锚点；
- Wang 确实有 PA 口径的数字，但只有两个方法、且其 CIFAR-10 正类子集（Case 1 `{0,1,2,8,9}` /
  Case 2 `{2,3,5,7,9}`，π=0.3、20k 子集）与我们（`{0,1,8,9}`、π=0.4、45k）都不同——
  v4 已经把它们用在 `row_*_cifar10` 行级映射（`magnitude_and_trend`），这是它们该在的位置，
  不能升格为单元级锚点。

## 3. 附带发现：18 个 Chen 锚点的协议标签是错的

v4 里 18 个 `pu_bench_table1_*_c01_accuracy` 的 `selection_protocol` 写作 `proxy_acc`。
按 Chen 正文与 Wang 附录 A，来源实际是**真标签验证集的 macro-F1**。这不是裁定问题，是
**锚点元数据错误**：它让「我们的 OA 单元 vs 人家的 proxy-acc 数字」看起来像同类比较。

处理：在 v5 里把这 18 条的 `selection_protocol` 改成实情（真标签验证 + macro-F1 准则），
并把「选模准则 macro-F1 vs 我们的 accuracy」「验证集 1%（Nv=37/250/500）vs 10–20%」写进对应
映射的 `protocol_differences`。`magnitude_and_trend` 的判定不变——差异只会更清楚，不会更小。

## 4. 错因

第一版只读了 v4 的元数据：看到锚点标的 `proxy_acc`、又看到它挂在 OA 单元上，就认定「锚点挂错单元」。
实际相反——**挂对了单元，标错了协议**。教训：锚点的 `selection_protocol` 是二手记录，
涉及 eligibility 归属时必须回原文核实，不能拿矩阵自己的字段当证据链的起点。

## 5. 为什么不现在改矩阵

- **v5 现在绑不了 v3**：`validate_comparison_coverage(survey=v3, comparison=…)` 按 v3 的
  `execution_units` 枚举全部单元，包含 `self_pu_full_meta` 与 7 个 P4.2 缺口 profile；这些条目
  还没有映射，覆盖率门禁会直接失败。v5 必须与 P4.2a/b 一起出。
- **原地改 v4 会追溯改写历史**：645-run 的裁决是按当时的矩阵做的，改 v4 等于让既有结论变质，
  违反「历史制品不改写」。裁定规则先记录（D41），应用留给 v5。

## 6. 同时做的：v3 静态阻断位按现状清理

`formal_blockers` 三条 → 两条：

- **删掉** `per_epoch_independent_PA_OA_checkpoint_selection`。准则本体 2026-09-20 已实现；
  而且这个标识符是**动态阻断位**：`runner._checkpoint_context` 按每次运行的真实 checkpoint 轨迹
  加减它（轨迹完整就摘掉，不完整就加上）。静态列表里再留一条会让 runner 永远摘不掉，
  所以正确处理是删掉静态那条、保留动态判定，而不是把它改写成一句话。
- **改写**「Formal admission still awaits the P4.1-B budget freeze and the versioned profile-to-
  comparison mapping」→ 现状：P4.1-B 冻结与 profile 映射均已于 2026-10-25 完成（D36/D40），
  剩下 comparison v5 与多 seed 批次。
- 「CVIR/KLDCE/…/GradPU/LaGAM·PULNS 的 cifar10 行仍缺预算或 runner」这条保持不变，仍然成立。

没有任何 `formal_eligible` 因此翻转为 true：v3 的 profile 身份协议每次运行仍会由
`profile_identity_extension_engineering_only` 挡住。

## 7. 摘要连带刷新（协议就地改，版本仍 `survey-v3.0`）

| 绑定 | 旧 → 新 |
|---|---|
| v3 规范摘要（registry / v0.2 草稿 / profile map 的 `protocol_basis`） | `44128fcb…` → `50a96eb1…` |
| v3 原始字节摘要（clean-support 清单 `protocol_policy_sha256`） | `57abbcb1…` → `b3ea7e5d…` |
| clean-support 清单 `receipt_sha256` | `79e92d10…` → `1577b39e…` |

25 份 clean-support 制品未重建，其自身 `support_manifest.json` 的收据也未改。

## 8. 验证

- `check_survey_recipe_registry.py --require-present --require-locked` 通过（registry sha `d79d48ed…`）；
- v0.2 草稿门禁通过（`a4eb856a…`）；`check_survey_profile_map.py --require-present` 通过
  （map sha `a33bacfe…`，9 mapped / 14 not_in_matrix / 1 deviation）；
- `tests/unit/experiment` 与 `tests/integration/test_survey_recipe_binding_v3.py` 全绿；
- 已生成工作簿的 `formal_blockers` 列是**运行当时**（10-25 早于本次清理）的文本，属历史记录，不改写。

## 9. 未完成

- 54 条裁定在 comparison v5 里应用；v5 同时绑 v3 并收录 P4.2a/b 条目（Chen 的 4 方法锚点挂 OA 单元）。
- 18 个 Chen 锚点的 `selection_protocol` 元数据修正，同批做。

## 10. 后续清理（2026-10-25 晚）：删除过期表述

v5 接管运行链路后，协议 `selection_spec` 里那句「54 条 `blocked_pending_pa_criterion` 映射保持 blocked
直到 P2.0c 裁定」已与实际矩阵相反（v5 中它们全是 `no_direct_anchor`），删除；同一笔把
`formal_blockers` 里「仍在等 comparison v5」改写为「等 multi-seed 批次」，其余表述不动。

| 绑定 | 旧 → 新 |
|---|---|
| v3 规范摘要（registry / v0.2 草稿 / profile map 的 `bound_survey_protocol`） | `50a96eb1…` → `d70bbf4c…` |
| v3 原始字节摘要（clean-support 清单 `protocol_policy_sha256`） | `b3ea7e5d…` → `ef8c2eb4…` |
| clean-support 清单 `receipt_sha256` | `1577b39e…` → `00f093c9…` |
| comparison v5 的 `bound_survey_protocol` | 随规范摘要刷新，v5 摘要 → `ad8e78c7…` |
| profile map v2 的 `bound_comparison` | 随 v5 摘要刷新，map 摘要 → `01aa9d92…` |

v1/v2 协议（历史绑定）不含这次清理，其摘要不变；已产出的 manifest 与工作簿保留运行当时的
`formal_blockers` 文本，不改写。25 份 clean-support 制品未重建。
