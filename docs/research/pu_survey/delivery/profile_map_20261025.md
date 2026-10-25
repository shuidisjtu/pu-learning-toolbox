# profile → 比较矩阵映射落地（2026-10-25）

新增 `pu_toolbox/experiment/survey_comparison_profile_map_v1.json`（`survey-profile-map-v1`），
把 v3 的 profile ID 接到 `survey_comparison_v4.json` 的既有条目上。此前 `survey_protocol.py`
对每一份 v3 manifest 硬写 `comparison_registration_status = "missing_versioned_profile_mapping"`，
读者分不清「条目还没写」和「本来就不在矩阵里」。

23 个 profile：**9 mapped / 14 not_in_matrix**（7 个 P4.2 缺口 + 7 个非 runnable），1 条显式偏差。

## 落地了什么

- 映射文件双绑定：`survey-v3.0` 协议摘要（`44128fcb…`）与 `survey-comparison-v4` 摘要（`7a20b127…`），
  任一漂移即拒；`map_version`、`profile_map_sha256` 随 manifest 记录，历史 manifest 不改写。
- 模块 `survey_profile_map.py`：`load` / `validate` / `resolve_profile_mapping` /
  `profile_registration_status`，以及 `profile_map_binds`。
- 运行侧接线：`build_execution_context` 记 `mapped_inherited` / `mapped_with_profile_deviations` /
  `not_in_matrix` 三态；`runner._run_comparison` 对 `mapped` profile 按继承条目解析每个结果单元
  （SCAR 一跑两个单元照旧分开），`not_in_matrix` 不产出比较块。
- 门禁 `scripts/check_survey_profile_map.py`（CI `--require-present`）：双绑定、profile 全函数覆盖、
  继承目标必须是矩阵里存在的 method、v4 的 method 必须有 profile 继承（防孤立条目）、偏差词表与轴闭合。

## 三个需要记住的取舍

**继承 + 显式偏差，不重键 194 条映射。** `self_pu` 与 `self_pu_full_meta` 是同一算法、两种监督契约，
v4 描述的是 no-meta 消融。按 profile 重键会让每条条目复制多份并各自漂移；继承一条、记录本 profile 的
偏离，才能把「同一行」和「同一个数」分开。语义边界：继承说明用哪条条目，不说明数值等于该条目的锚点。

**`not_in_matrix` 只陈述矩阵，不陈述可运行性。** `kldce` 被配方注册表标 `excluded`，却在 v4 里有
三条 non-runnable 标记行。按「不可运行」判成 `not_in_matrix` 会写出与矩阵相反的事实，并让这三行在
反查门禁里成为没有任何 profile 可达的孤立条目，所以记 `mapped` 并在理由里写明它不参与运行。
反过来 `lagam` 在 spambase 能跑却全矩阵无条目，这才是 P4.2 缺口。

**一个 profile 对不同锚点族的偏差可以相反。** `self_pu_full_meta` 的 clean-validation meta 就是论文
Table 7 的设计（对 `chen_table7` 无 meta 偏差），但与 PU-Bench 受控实现的 meta target（validation U）
相反，因此偏差带 `against: pu_bench_table1`。混写成一条「与基准有差异」会丢掉方向。

**窄化矩阵如实降级，不报错。** 测试夹具与派生草案矩阵带 profile 身份 schema 但不是映射冻结的那一份；
此时 manifest 仍记 `missing_versioned_profile_mapping`（这份矩阵确实没有版本化映射），而 `resolve`
路径对绑定不符仍直接报错。矩阵漂移由门禁在 CI 上按已提交文件重算摘要兜住。

## 验证

- `scripts/check_survey_profile_map.py --require-present`：
  `23 profile(s): 9 mapped / 14 not_in_matrix, 1 deviation(s)`，协议 basis `survey-v3.0`，
  矩阵 basis `survey-comparison-v4`，map sha256 `bc8c22ed…`。
- `self_pu_full_meta` 在 spambase/SCAR/0.3/OA 解析到 `unit_self_pu_spambase_scar_0.3_oa`
  （同一 `self_pu` 条目），偏差 1 条且带 `source: profile_map`。
- `tests/unit/experiment`（含新增 14 例 mapping 测试）与
  `tests/integration/test_survey_recipe_binding_v3.py`（真实 `nnpu`/`pusb_kernel` 主 CLI 链）全绿；
  格式/测试质量/注释/文档链接/API/元数据/结构/配方注册表门禁通过。

## 未完成

- 7 个 runnable profile 仍是 `not_in_matrix`：P4.2a 需先核对两篇基准论文表格是否真报 `pulda`/
  `robust_pu`/`holistic_pu`/`lagam`，P4.2b 的 `genpu`/`pulns`/`split_pu` 只做定性交代（D37）。
- 本文件**不改 `eligibility`**：54 条 PA 映射仍是 `blocked_pending_pa_criterion`（D38），
  mapping 落地不等于 PA 行可做数值裁决。
- 结果进入汇总矩阵仍要等 comparison v5 绑定 v3（D31 不变，v4 继续绑 `survey-v2.0`）。
