# 54 条 PA 映射裁定（2026-10-25，D41）

D38 记的是「PA 准则不再是外部依赖，剩下的是我方重新裁定这 54 条映射」。本记录给出裁定结果、
依据，以及为什么它落在 comparison v5 而不是现在改矩阵。

## 裁定结论

54 条 `blocked_pending_pa_criterion` = 6 方法 × 3 数据集 × 3 个 `c`，**全部是 `pa` 选模单元**：

|  | 条数 | 新 eligibility | 锚点 |
|---|---:|---|---|
| 有 `proxy_acc` 锚点的 c=0.1 PA 单元 | 18 | `magnitude_and_trend` | 从同格 OA 单元**迁入** |
| 其余 PA 单元（c=0.3/0.5 与 SAR 行） | 36 | `no_direct_anchor` | 无 |
| 让出锚点的 c=0.1 OA 单元 | 18 | `no_direct_anchor` | 迁出 |

`blocked_pending_pa_criterion` 在 v5 里归零。

## 依据：锚点挂错了单元

裁定前先查了 v4 的 28 个 `proxy_acc` / `pa` 锚点：

- 18 个是 c=0.1 的单元锚点（6 方法 × 3 数据集），来源是 PU-Bench Table 1，**用 proxy accuracy
  选出的数字**；这正是 R9 准则（`strategies.proxy_accuracy`，Wang et al. 2026 Def 1 的 OS 分支）
  的口径。
- 10 个是 cifar10 的 row-level 锚点，挂在 row-level 映射上，不受本次影响。

这 18 个锚点现在挂在我们的 **OA** 单元上（`unit_*_c0.1_oa`，`magnitude_and_trend`）。但 OA 用
clean 标签选模，与来源的 proxy accuracy 不是同一个准则——**能对上口径的是我们的 PA 单元**，
而 PA 单元反而是 `blocked` 且一个锚点都没有。所以这不是"给 PA 补个锚点"，是**把锚点挪到
准则对得上的那个单元上**。

`validate_comparison_coverage` 强制「一个锚点只能被一条映射引用」，因此迁移是唯一形式，
同一个已发表数字不会在 PA 和 OA 两侧各算一次。

## 为什么不是 `numeric`

来源与我们的口径仍有实质差异，任何一条都足以拒绝 `numeric`：

- backbone：基准是逐模态 CNN/MLP，我们是冻结 ResNet-18 特征 + MLP 头（cifar10）或 mlp128（2d）；
- 验证划分：基准 0.01，我们 10–20%；
- 候选池与预算：我们的是 v3 声明的阶段预算，不是基准的调参过程。

所以按 `magnitude_and_trend` 记录，与这些单元此前在 OA 侧得到的对待一致，不升级承诺。

## 为什么不现在改矩阵

- **v5 现在绑不了 v3**：`validate_comparison_coverage(survey=v3, comparison=…)` 按 v3 的
  `execution_units` 枚举全部单元，包含 `self_pu_full_meta` 与 7 个 P4.2 缺口 profile；这些条目
  还没有映射，覆盖率门禁会直接失败。v5 必须与 P4.2a/b 一起出。
- **原地改 v4 会追溯改写历史**：645-run 的裁决是按当时的矩阵做的，改 v4 等于让既有结论变质，
  违反「历史制品不改写」。所以裁定规则先记录（D41），应用留给 v5。

## 同时做的：v3 静态阻断位按现状清理

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

## 摘要连带刷新（协议就地改，版本仍 `survey-v3.0`）

| 绑定 | 旧 → 新 |
|---|---|
| v3 规范摘要（registry / v0.2 草稿 / profile map 的 `protocol_basis`） | `44128fcb…` → `50a96eb1…` |
| v3 原始字节摘要（clean-support 清单 `protocol_policy_sha256`） | `57abbcb1…` → `b3ea7e5d…` |
| clean-support 清单 `receipt_sha256` | `79e92d10…` → `1577b39e…` |

25 份 clean-support 制品未重建，其自身 `support_manifest.json` 的收据也未改；清单里写明
"policy 段落（c_tokens、support 规模、seeds、split 策略）未变，只是提示摘要在动"。

## 验证

- `check_survey_recipe_registry.py --require-present --require-locked` 通过（registry sha `d79d48ed…`）；
- v0.2 草稿门禁通过（`a4eb856a…`）；`check_survey_profile_map.py --require-present` 通过
  （map sha `a33bacfe…`，9 mapped / 14 not_in_matrix / 1 deviation）；
- `tests/unit/experiment` 与 `tests/integration/test_survey_recipe_binding_v3.py` 全绿；
- 已生成工作簿的 `formal_blockers` 列是**运行当时**（10-25 早于本次清理）的文本，属历史记录，不改写。

## 未完成

- 54 条裁定在 comparison v5 里应用；v5 同时绑 v3 并收录 P4.2a/b 条目。
- P4.2a 仍需先核对两篇基准论文表格是否真报那 4 个方法。
