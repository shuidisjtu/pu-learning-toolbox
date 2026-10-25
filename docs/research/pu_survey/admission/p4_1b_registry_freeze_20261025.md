# P4.1-B 配方注册表冻结（2026-10-25）

`pu_toolbox/experiment/survey_recipe_registry_v1.json`（`survey-recipes-v1.0`，schema 2.0）已落地，
绑定 `survey-v3.0`（协议摘要 `44128fcb…`）。CI 的生产注册表步骤从"待物化则跳过"变为真跑：
`--require-present --require-locked`。

## 冻结范围

范围由用户 2026-10-25 决定：**有 runnable 执行单元的 profile 锁 `locked`，其余标 `excluded`**。
按此规则，23 个 profile 中 **15 个锁、8 个排除**，覆盖 v3 的全部 40 个 runnable 单元。

| 状态 | profile |
|---|---|
| `locked`（15） | `upu`、`nnpu`、`dist_pu`、`pusb_kernel`、`lbe`、`self_pu`、`self_pu_full_meta`、`pulda`、`robust_pu`、`split_pu`、`pulns`、`lagam`、`genpu`、`holistic_pu`、`pn_oracle` |
| `excluded`（8） | `kldce`、`p3mix`、`puet`、`gradpu`、`pan`、`rp`、`cvir`、`vpu` |

排除不是"删掉"，而是带理由留在表里：每条 `exclusion_reason` 抄自协议里该方法的
`non_runnable_reason`（缺预算、缺 runner、待 P4.1-B 复核），所以打开注册表就能看到"为什么这个
方法没有配方"，而不必回协议里翻。

## 锁定加了什么

- 每个 locked profile 的每个 candidate 带 `resolved_snapshot`：把 `base_ref` 指向的协议段落展开成
  具体 `params`、`budget_declaration`、`model_family`、`inner_search_id`，并给 `algorithm`。
  快照是**算出来的**，不是写的；`validate_registry` 会重算并比对，所以手改一个参数就报
  `resolved_snapshot_mismatch`。
- `review.open_questions` 里"锁定前需方法负责人审阅并签署"这一条被移除，换成一条 `decision` 字段，
  指向本文档；`reviewers` 保持空数组。这是刻意的：项目决定不要求签署，用空数组冒充签署记录比
  不写更糟。
- 尚未锁的 profile 保留原来的开放问题，因为它仍然是 draft 语义上的未决项。

## 怎么复现

注册表是生成物，不是手写物。同一命令产出同一字节：

```bash
uv run python scripts/prepare_survey_recipe_registry.py --protocol survey-v3.0 --lifecycle locked
```

`tests/unit/experiment/test_survey_recipe_registry_locked.py` 把"提交的文件 == 重新生成的结果"
钉住，并逐条重算快照；`tests/unit/scripts/test_check_survey_recipe_registry.py` 钉住 gate 的失败路径。
CI 里两道：v0.2 草稿（schema 2.0 草稿）与生产注册表（`--require-locked`）。

## 它不证明什么

- **不是方法准入**。注册表只回答"这个 runnable 单元的配方是什么"；v3 的 40/64 不变，
  `formal_blockers` 未清，正式 GPU 预算与效果结论都不在这里。
- **不回溯绑定历史制品**。旧 pilot manifest 仍是 `legacy_unbound`，`review` 里那条开放问题保留：
  不给历史结果补绑。
- **不代表数值 parity**。GEN-PU 的 256 宽配方已对齐（见[宽度对齐记录](../delivery/genpu_width_alignment_20261025.md)），
  但全量资源与数值 parity 仍未做。

## 验收

| 项 | 结果 |
|---|---|
| `check_survey_recipe_registry.py --require-present --require-locked` | 通过，23 profile、0 警告，`sha256 6d4e7bce…` |
| `build_locked_registry(v3)` 与提交文件逐字段一致 | 通过 |
| 手改快照（`hidden_dim` 改回 128） | 被 `resolved_snapshot_mismatch` 拒绝 |
| draft 注册表直接运行时绑定 | 被拒（`must be locked before runtime binding`） |
