# GEN-PU 配方宽度对齐（2026-10-25）

一句话：v3 协议里 GEN-PU 的 `hidden_dim` 从构造器默认 **128** 对齐到冻结配方的 **256**，profile 标签
同步由 `mlp128` 改为 `mlp256`，让 profile、阶段预算、backbone 规格、`genpu_recipe.py` 说的是同一套架构。

## 为什么必须改

同一份配方曾经有三个数：

| 来源 | `hidden_dim` |
|---|---|
| v3 `method_profiles.genpu.params`、`budgets.genpu.staged_budget.parameter_values`、`backbone_specs.genpu_native_mlp` | 128 |
| 10-23 冻结配方（`genpu_recipe.py`，机器收据 `recipe.parameters`） | 256 |
| `GenPUClassifier` 构造器默认 | 128 |

关键点不是"两个数不一样"，而是 **128 这条链从来没有跑通过**：`GenPURecipeTrainer` 把 256 写进固定配方
白名单，128 会在 `validate_experiment` 被判 `GenPU fixed recipe parameters changed`。协议里的 128 是
构造器默认值被留在了冻结时，不是任何一次实际运行的参数。

因此保留 128 等于用一个不可能执行的数字去冒充冻结配方；而 recipe registry 只"指向协议参数"，GEN-PU
一旦锁进 P4.1-B，`resolved_snapshot` 会把 128 记成正式绑定。这是本次对齐的直接动因。

## 改了什么

| 位置 | 改动 |
|---|---|
| `method_profiles.genpu.params.hidden_dim` | 128 → 256 |
| `budgets.genpu.staged_budget.parameter_values.hidden_dim` | 128 → 256 |
| `backbone_specs.genpu_native_mlp.parameter_values.hidden_dim` | 128 → 256 |
| `method_profiles.genpu.model_family` | `mlp128` → `mlp256` |
| `method_profiles.genpu.configuration_status` | `provisional_constructor_defaults` → `owner_confirmed_frozen_2026-10-24` |
| `method_profiles.genpu.configuration_review.findings` | 追加第三条，记录这次对齐的理由与范围 |
| `docs/research/pu_survey/data/clean_support_materialization_20261024.json` | 刷新 `protocol_policy_sha256` 并加 `protocol_policy_sha256_note`，重算 `receipt_sha256` |

散文字段与 `configuration_status` 在代码里没有消费者（`model_family` 只进 registry 快照），所以这次是
纯数据改动，没有改执行代码。三点需要记住的口径：

- `execution_units` 的 `model_family` 保持 `genpu_native` / `genpu_native_cnn`。阶段预算校验器
  （`survey_staged_budget.validate_protocol_budgets`）强制它是 `算法名_native`，与 profile 层标签不同层。
- `mlp256` 不需要新增 `backbone_specs` 条目。profile 层 `model_family` 与 backbone 规格不是一一对应，
  已有先例：`holistic_pu` 的 profile 标签是 `mlp64`，没有同名 backbone 规格。
- 冻结协议就地改，版本仍是 `survey-v3.0`（既定冻结方式）。clean-support 制品本身没有重建：这次改的
  参数不参与 support 的 c、support size、seed 与 split 口径。

## 逐组件体积：64 MiB 上界在 256 下仍然成立

`checkpoint_bytes_per_component = 67,108,864`（64 MiB）是 10-24 冻结时按 **256 配方**的实测声明的，
不是按 128 算的。这次补测了两条从未测过的 native-2d 行：

| 行 | 输入维 | 逐组件实测 | 距 64 MiB |
|---|---:|---:|---:|
| spambase / `native_2d` | 57 | 326,593 B | 205.5× |
| imdb / `native_2d`（sbert384） | 384 | 661,441 B | 101.5× |
| cifar10 / `native_cnn` | — | 45,542,038 B | 1.47× |

cifar10 一行不是这次测的：10-23 收据的 `recipe.parameters.hidden_dim` 就是 256，45,542,038 B 是那批
三份快照的单份体积（三份合计 136,626,114 B）。所以 64 MiB 上界在原配方宽度下已被实测覆盖，本次对齐
没有把它撑破。

复现方式与全部字段见[机器收据](../data/genpu_width_measure_20261025.json)：真实验证路径是
`ExperimentRunner → GenPURecipeTrainer → 逐 epoch checkpoint`，CPU，合成四角色 SCAR 数组，
1 GAN + 3 PN 轮、batch 8、seed 0、c=0.5。这不是正式预算读数，也不是效果证据。

## 核过什么

- 装配：`assemble_model(v3, spambase/genpu)` 现在构造 `GenPUClassifier(hidden_dim=256)`，与
  `GenPURecipeTrainer` 的固定配方一致；`validate_protocol_budgets(v3)` 通过。
- 测试：`tests/unit/experiment` 与 `tests/contract` 全绿（见提交记录）。
- 收据：clean-support 清单的 `protocol_policy_sha256` 与协议文件逐字节一致，`receipt_sha256` 重算后
  通过 `tests/contract/test_clean_support_materialization_receipt.py`。

## 还没做

正式预算（100 GAN + 100 PN 轮、batch 64）下的容量预检、GPU 峰值实测、数值 parity 都不在本次范围。
第 3 段真实绑定运行前要重跑一次容量预检：本机 GPU 可用就在本机跑，否则列为待办（AutoDL 有空闲主机
时补跑，或写进队友下一份任务单）。
