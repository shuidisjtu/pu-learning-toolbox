# P4.1-B 配方注册表冻结（2026-10-25）

`pu_toolbox/experiment/survey_recipe_registry_v1.json`（`survey-recipes-v1.0`，schema 2.0）已落地，
绑定 `survey-v3.0`（协议摘要 `44128fcb…`）。CI 的生产注册表步骤从"待物化则跳过"变为真跑：
`--require-present --require-locked`。

## 冻结范围

### 2026-11-01 后续配方决定（不回写 v3.0）

用户明确确认：**沿用当前协议暂定参数，作为本轮项目配方；论文差异单独保留说明**。
适用此前 excluded 的 KLDCE、P3MIX、PUET、GradPU、PAN、RP、CVIR、VPU。
因此这些方法不再等待“是否采用当前参数”的确认；运行入口、跨数据集接线、资源预检和结果验收仍须逐项通过。
新配方使用独立版本，不修改已冻结 v3.0、旧注册表或历史运行身份，不补造合作者签署。
P3MIX 为协议的 base 变体；未列明的参数须展开当前实现默认值并明确记录，不根据已有成绩调参。
此决定不等于论文数值复现、SAR 有效性证明或正式结果准入；原有比较矩阵复核要求继续保留。

落地：`survey_protocol_v3_1.json`、`survey_recipe_registry_v1_1.json`、比较矩阵 v7 和映射 v4。
新版本 23 个 profile 有锁定配方，171 个单元具备运行接线；新 59 个单元共 1845 fit，旧结果不补绑。

| 处理 | 来源/原有行为 → 本轮选择与理由 | 验证范围 |
|---|---|---|
| P3MIX | 协议 base、源码默认 `role_balanced`；显式展开该默认值。此前 `merged_shuffle` 候选不混入 | CPU 二维/CNN 真训练、阶段与选模；非论文数值复现 |
| GradPU | 依照已登记例外用私有 CNN13/no-BN；构造 encoder 时绑定实验 seed，避免依赖外部 RNG | 相同 seed 参数相同、不同 seed 不同、外部 RNG 不变；CPU CNN 真训练 |
| PAN | 保留本项目 100 epoch、batch64、lr=1e-4、C/D 每步各更新一次，不宣称作者采样重放 | 完整 checkpoint 数与 C/D 更新计数；SCAR PA/OA、SAR OA |
| VPU | 保留协议 100 epoch/batch128/lr=3e-4，不改成作者 50/500/3e-5；`cpu_stream` 仅改变驻留，禁用源验证诊断选模 | 全训练 TS 边际、归一化模型 checkpoint、SCAR/SAR 小规模真训练 |
| CVIR | 总体 pi 绑定 TS 的完整训练风险池；warm-start 1 + CVIR 20 均可选。仍用论文加权目标，非作者 balanced-CE/SGD 配方；`cpu_stream` 不缩批 | 双阶段 21 个 checkpoint 的预算与实际计数；不向 estimator.fit 传错总体 pi |
| PUET/RP | 100 棵树 / 3-fold + refit 沿用当前参数；不用伪 epoch | 实际树数/嵌套拟合收据，SCAR/SAR CPU 真训练 |
| KLDCE | flip=1−实际 SCAR c，population pi 与选择规则不变；运行时绑定经过协议重算后才进入注册表校验 | CPU 真训练、计数绑定、QP/预测内存下界筛查；256 GiB 下界上限不等于峰值内存承诺 |

CNN 与五角色支持路径相关回归 125 项（123 通过、2 个 KLDCE SAR 显式跳过）；
另一组入口/注册表/历史 CVIR 回归 112 项（108 通过、4 跳过）。这些是工程回归，不是 1845 fit 已跑完。
新增矩阵/配方均通过注册表与比较映射校验；随后 PUET/RP 图像路径测试见下节。

### 2026-11-01 PUET/RP 图像适配测试（v3.2，不改运行中的 v3.1）

用户要求确定余下图像路径并开始测试。复用其他传统方法已有的固定随机 ResNet-18
512 维特征，而非原生 CNN：同数据集/split/seed 的 encoder、train-only 归一化和特征缓存
跨方法/c/机制共享；不预训练、不微调、不新增标签、不使用测试标签挑选表示。
SCAR/SAR 仍在原始数据空间生成，再按相同行索引用于特征；PUET 的 TS 校准和 RP 的 OS
语义不变。PUET 100 棵树，RP 3-fold OOF + 最后 refit，沿用 v3.1 参数。

来源差异：PUET 方法卡登记作者 CIFAR 用原图 3072 维；这里用固定随机 512 维特征，
理由是与现有传统算法图像路径一致，不据测试分数挑表示。RP 同样是显式新增图像包装，
不称作者图像实验重放。随机特征可能损失信息；不得与端到端 CNN 混排或用单 seed 宣称优越。

`survey-v3.2` / `survey-recipes-v1.2`、比较矩阵 v8 / 映射 v5 只增八个单元，旧 171 行
及配方不变。新增 280 fit（两方法 × 四数据集 × 五 seed × 七个 c/机制配置），24 个
方法/数据集/机制首轮完整配置测试通过后才放行各自余项；失败组挂起，不减树数/折数重试。
独立 CPU 队列一次一个 worker、两线程，不占额外 GPU；原有四卡批次继续运行。
校验包含真实 ResNet 特征与算法 fit、缓存/来源空间标签隔离、实际树/折计数和版本绑定；
完整数据结果、五 seed 数值与正式准入仍需验收，不补造来源或人工签署。

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

## 运行链绑定

本机 T600（CUDA 可用）用主 CLI 跑两个 v3 单元，都带
`--recipe-registry pu_toolbox/experiment/survey_recipe_registry_v1.json`：

| 方法 | 数据 | manifest 绑定 |
|---|---|---|
| `nnpu` | spambase `split_0`、c=0.5、seed0、π=0.3 | `recipe_registry_sha256 6d4e7bce…`、`nnpu/survey_shared_nnpu`、`inner_search_id null` |
| `pusb_kernel` | 同上 | `pusb_kernel/survey_shared_pusb_kernel`、`inner_search_id pusb_kernel_internal_cv_v1` |

绑定字段同时出现在 manifest 顶层、每条 `candidate_runs` 和 PA/OA 两个 `selection` 条目；输出目录名带
注册表摘要（`recipe_<sha256>`），两个注册表不会互相覆盖结果。负向检查三条：

- `--recipe-registry-sha256` 传错 → 退出 1，`recipe registry changed since preflight`；
- 把绑定 manifest 直接送进聚合、不给外部重算的期望 → `recipe-bound aggregation requires the v2
  batch audit/summary entry point`；
- 改 `resolved_params_sha256` → `validate_consumed_bindings` 拒绝。

集成测试 `tests/integration/test_survey_recipe_binding_v3.py` 把这条链固化为回归：用合成四角色数据
跑一次真实 CLI，断言 6 个绑定字段与两条 selection 一致、通过聚合 gate，并验证篡改与"无期望聚合"
都被拒。机器字段见[收据](../data/survey_recipe_binding_20261025.json)，运行目录留在 `tmp/` 不提交。

## 它不证明什么

- **不是方法准入**。注册表只回答"这个 runnable 单元的配方是什么"；v3 的 40/64 不变，
  `formal_blockers` 未清，正式 GPU 预算与效果结论都不在这里。
- **不回溯绑定历史制品**。旧 pilot manifest 仍是 `legacy_unbound`，`review` 里那条开放问题保留：
  不给历史结果补绑。
- **不代表数值 parity**。GEN-PU 的 256 宽配方已对齐（见宽度对齐记录），
  但全量资源与数值 parity 仍未做。

## 验收

| 项 | 结果 |
|---|---|
| `check_survey_recipe_registry.py --require-present --require-locked` | 通过，23 profile、0 警告，`sha256 6d4e7bce…` |
| `build_locked_registry(v3)` 与提交文件逐字段一致 | 通过 |
| 手改快照（`hidden_dim` 改回 128） | 被 `resolved_snapshot_mismatch` 拒绝 |
| draft 注册表直接运行时绑定 | 被拒（`must be locked before runtime binding`） |
| 本机 T600 真实 CLI 绑定（`nnpu`、`pusb_kernel`） | 6 个绑定字段与两条 selection 一致；错误摘要/篡改/无期望聚合均被拒 |
