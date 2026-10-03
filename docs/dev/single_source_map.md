# 单源地图：概念—真相源—消费者

> 本文档回答「某个概念在代码里的权威实现在哪、谁在读它、有没有第二份」。
> 模块分层与依赖方向见 [`architecture.md`](architecture.md)；治理批次与审计历史见
> [`architecture_principles.md`](architecture_principles.md) §5。
>
> 本表是**审计快照**（记于 2026-10-03，BASE `f1182ec`）。**发现重复不等于已收敛**——本批只记录，
> 迁移在后续批次。

| 概念 | 权威真相源 | 消费者 | 兼容入口 | 判据 |
|---|---|---|---|---|
| 标签语义 | `pu_toolbox/core/labels.py:92` `normalize_pu_labels` / `:145` `normalize_pnu_labels`；字面值在 `pu_toolbox/core/config.py:8`-`:10` | `core/validation.py`（`validate_pu_X_y` / `validate_pnu_X_y` 的前置归一）、`metrics/classification.py`、`preprocessing/`、`model_selection/split.py`、`diagnostics/`、`workflows/shift.py`、`estimators/deep/self_pu.py`、`estimators/risk/nnpu.py`、`estimators/risk/vpu.py` | 无 | 单一 |
| 设备 | `pu_toolbox/core/device.py:11` `resolve_device_name` / `:29` `resolve_device` | 14 个 torch 估计器（`estimators/deep/*`、`estimators/risk/{dist_pu,nnpu,pulda,vpu}.py`、`estimators/research/*`）、`workflows/pipeline.py:431`、`workflows/_reporting.py:91` | 无 | 单一 |
| 随机源 | `pu_toolbox/core/random.py:8` `check_random_state` | `experiment/strategies.py`、`preprocessing/pu_labeling.py`、`preprocessing/selection_bias.py` | 无 | 重复（可收敛） |
| JSON 安全转换 | `pu_toolbox/utils/serialization.py:38` `json_safe` | `diagnostics/{benchmark,domain_assumptions,report,shift,shift_monitor,uncertainty}.py`、`preprocessing/data_profiler.py`、`workflows/report.py` | 无 | 重复（可收敛） |
| 哈希 | `pu_toolbox/utils/serialization.py:32` `canonical_hash` | `diagnostics/benchmark.py`、`experiment/datasets.py`、`experiment/split_archive.py`、`experiment/strategies.py` | 无 | 重复（可收敛） |
| RBF 权重 | `pu_toolbox/utils/basis.py:34` `build_rbf_basis`（`:62` `rbf_weights` 建于其上） | `prior/pen_l1.py`、`prior/kernel_mean.py`、`estimators/risk/kldce.py`、`estimators/risk/pnu.py`、`estimators/risk/upu.py`、`utils/basis.py:101` `resolve_basis_fn` | `pu_toolbox/utils/__init__.py` 重导出 | 重复（不可合并，理由：见说明） |
| 类先验推导 | `pu_toolbox/estimators/risk/_class_prior.py:6` `solve_prior_from_positive_fraction` | `estimators/risk/kldce.py:935`、`estimators/risk/ldce.py:429` | 无 | 单一 |
| 训练视图 | `pu_toolbox/core/training_views.py:120` `build_training_view`；角色词表 `:42` `ViewRole` + `:48` `ROLES`；视图词表 `:38` `RunView` + `:47` `RUN_VIEWS`（运行时元组由 `get_args` 派生） | `estimators/risk/{vpu,upu,nnpu,dist_pu}.py`、`estimators/bias_aware/pusb_kernel.py`、`estimators/deep/self_pu.py`、`experiment/` 各消费者、`scripts/{run_survey_experiment,run_survey_pilot,prepare_survey_splits}.py`；另有三项词表与两处子集未并（不同概念，见说明） | `experiment/training_views.py:42` `TSOSBatchView`（legacy 边界适配器） | 单一（角色名与 os/ts 视图；未并项见说明） |
| 方法能力字段 | 估计器类属性（如 `estimators/risk/puet.py:85`-`:86`）经 `registry/registry.py:141` `_sync_class_metadata_to_registry` 覆盖 `registry/builtin_methods.py` 的字面量 | `registry/registry.py` `get_metadata`、`advisor/`、`cli/`、`workflows/pipeline.py`、`ui/` | 无 | 重复（可收敛） |

## 说明

### 随机源

**重复形态。** `core/random.py:8` 的 `check_random_state` 定义了唯一归一化入口（接受 `int` /
`RandomState` / `None`），但它只被 `experiment/strategies.py`、`preprocessing/pu_labeling.py`、
`preprocessing/selection_bias.py` 使用。`estimators/` 与 `prior/` 下有 **19** 处直接构造
`np.random.RandomState(`，绕过该 helper。模式 `np\.random\.RandomState\(` 命中 19 处，其中 18 处是
`np.random.RandomState(self.random_state)`（模式 `np\.random\.RandomState\(self\.random_state\)`），
另 1 处 `prior/pen_l1.py:27` 是硬编码 `np.random.RandomState(0)`。**两者差 1**：把「19 处
`self.random_state`」当作事实会多算一处。

**第二个入口点（不同生成器族）。** 另有 `diagnostics/domain_assumptions.py:338` 的
`np.random.default_rng(random_state)`，供 `_bootstrap_domain_uncertainty` 做重复抽样。它构造的是
`Generator` 而非 `RandomState`，不受 `check_random_state` 归一，也不计入上文 19 处；但按本图收敛
RNG 约定时这一点须一并纳入。

**为何现在不能直接合并。** 两种写法语义不等价，且实测（numpy 2.4.6）：内联式
`np.random.RandomState(RandomState(42))` 抛 `TypeError`（`Cannot cast scalar from dtype('O') to
dtype('int64')`），而 `check_random_state(RandomState(42))` 原样返回该实例并保留同一对象身份。
因此把内联式替换成 helper 是**放宽**（原本崩溃的调用变为可用），会改变那些估计器 `random_state`
参数的实际接受域，属于行为变更而非等价重构。

**收敛前提。** 先逐一确认各估计器 `random_state` 参数的契约是否承诺接受 `RandomState` 实例；
统一迁移到 `check_random_state` 后，补一条「传入 `RandomState` 实例」的回归测试。

### JSON 安全转换

**重复形态。** `utils/serialization.py:38` 的 `json_safe` 是报告载荷的通用归一化（NaN/Inf → None、
`np.generic` → `item`、`Path` → `str`、递归 dict/list），消费者在 `diagnostics/`、`workflows/`、
`preprocessing/data_profiler.py`。`experiment/` 层另有 3 份索引/标量序列化器：
`experiment/datasets.py:455` `_json_indices`（弱版，无校验）、`experiment/training_views.py:155`
`_json_indices` 与 `experiment/feature_adapter.py:364` `_json_scalars`（后两者逐行同构，仅形参名与
报错措辞不同）。模式 `grep -rn 'def _json_indices\|def _json_scalars\|def json_safe' pu_toolbox/`
命中 4 处。

**为何现在不能直接合并。** 语义不同：`json_safe` 永不抛错（NaN → None），而 experiment 的两个索引器
遇到非 JSON 标量或非有限值会抛 `ValueError`；`datasets` 的弱版两者都不做。把弱版换成强版会在现有
输入上新增异常路径。

**收敛前提。** 先确认 experiment 清单里的索引实际都是有限 JSON 标量；再把 `training_views` 与
`feature_adapter` 两份同构实现并为一处；最后才评估索引器是否要归并到 `json_safe` 一侧。

### 哈希

**重复形态。** `utils/serialization.py:32` `canonical_hash` 是权威（`json.dumps` 带 `sort_keys` 与
紧凑 `separators` 再取 sha256）。另有 5 份副本，模式
`grep -rn 'def _json_sha256\|def _array_sha256\|def canonical_hash' pu_toolbox/` 命中 6 处：

- `_json_sha256` 三份：`experiment/image.py:335`（与 `canonical_hash` 逐字节等价，实测摘要相同）、
  `experiment/feature_adapter.py:385`（多 `allow_nan=False`，有限输入下摘要相同）、
  `experiment/text.py:206`（多 `ensure_ascii=False` 与 utf-8 编码，**非 ASCII 载荷下摘要不同**，实测不等）。
- `_array_sha256` 两份：`experiment/feature_adapter.py:377` 与 `experiment/image.py:327`（逐行相同）。

**为何现在不能直接合并。** 三份 `_json_sha256` 对同一载荷可能给出不同摘要（`text` 的 UTF-8 变体、
`feature_adapter` 拒绝 NaN）。而实验层摘要被 `experiment/survey_execution.py` 与
`experiment/survey_protocol.py` 当作复算校验，直接合并会改变已落盘的清单/缓存摘要。

**收敛前提。** 要么接受一次摘要变更并同步重生成受影响清单，要么把差异显式化为参数
（`allow_nan`、`ensure_ascii`）并先证明现存制品在新实现下摘要不变。

### 训练视图

**已收敛（单一）。** 角色名与 os/ts 视图名的唯一声明在 `pu_toolbox/core/training_views.py`：
`:42` `ViewRole` 与 `:38` `RunView` 两个 `Literal` 别名是唯一来源，运行时元组 `:48` `ROLES` /
`:47` `RUN_VIEWS` 由 `typing.get_args` 从别名派生——改别名即同步改元组，两者不可能漂移；
`:120` `build_training_view` 是唯一构造器。消费端改为导入该单源、不再内联复制：
`estimators/` 的六个 os/ts 消费点（`risk/{vpu,upu,nnpu,dist_pu}.py`、`bias_aware/pusb_kernel.py`、
`deep/self_pu.py`）、`experiment/` 层各消费者、`scripts/run_survey_experiment.py`（`--os-or-ts` 的
choices 与拆分文件角色名）、`scripts/run_survey_pilot.py`（choices）、
`scripts/prepare_survey_splits.py`（写拆分与尺寸）。收敛判据：模式
`grep -rn 'not in {"os", "ts"}' pu_toolbox/estimators/` 无命中；模式
`grep -rn '"train", "pu_val", "clean_val", "test"' pu_toolbox/ scripts/` 仅命中
`core/training_views.py:42`（单源本身）。

**未并，且是不同概念（三项，非遗漏）。**

- `experiment/training_views.py:38` `SamplingAssumption = Literal["os", "ts", "both"]` 是**台账词表**：
  比运行视图多一个 `"both"`，表达「方法原生支持的假设」，属于 survey 政策。core 不认识台账（政策留在
  实验层），故不下沉。
- `experiment/training_views.py:171` `LEGAL_RUN_VIEWS = frozenset({"os-compatible", "ts-compatible"})`
  是 manifest 的**带后缀拼写**，与 core 的小写 `RunView` 是两套词表；两者由边界适配器在出口换算。
- `run_view` 的祖传拼写 `"TS-compatible"` / `"OS"`（`experiment/training_views.py` 出口）同样留在实验层：
  它们是历史制品里已落盘的字符串，改词表会改写既有 manifest。

**未并，且是不同概念（两处子集）。**

- `experiment/survey_execution.py:141` 与 `experiment/survey_protocol.py:596` 的 `("train", "pu_val")`
  是**带生成 PU 标签视图的分区**：标签视图只在 train/pu_val 上生成，`clean_val`/`test` 保留真实标签。
  前者取 adapted→source 的特征映射，后者逐角色核对生成元数据。它不是四角色的缺省子集，
  扩成 `ROLES` 会访问没有生成标签视图的角色。
- `scripts/prepare_survey_splits.py:277`（`prepare_text`）的 `("train", "pu_val", "clean_val")` 是
  「其行取自 train 文本池的角色」：这些角色的行按 `texts_train` 池编码，`test` 的行来自另一个池、在
  循环外单独追加。扩成 `ROLES` 会用 test 的行去索引 train 池（读错位），是 bug 而非重构。

### 方法能力字段

**重复形态。** 同一事实写两处：`registry/builtin_methods.py` 在 `AlgorithmMetadata(...)` 里写
`implementation_status=` / `source_status=` 字面量（真实字面量各 24 处），估计器类再声明同名类
属性（如 `estimators/risk/puet.py:85`-`:86`）。核对时 `grep -c 'source_status='` 得 24，而
`grep -c 'implementation_status='` 得 **25**——`register_all_builtin_methods` 的 docstring（`:534`）
另含一次该串，须减去这处文档提及才等于 24 条字面量。注册时 `bind_estimator_class` →
`registry/registry.py:141` `_sync_class_metadata_to_registry` 用 `_SYNC_FIELDS`（含这两字段）对类属性
做 `setattr`，覆盖字面量。

**实测（受控实验）。** 把 `_BUILTIN` 中 `pusb_kernel` 的字面量故意改成 `NOT_FOUND` 后注册，注册表
仍显示类属性值 `official_related`（EXP1）；再把类属性改成 `OFFICIAL_BUNDLE` 重新绑定，注册表随即
变为 `official_bundle`（EXP2）。按 `registry.py` 的同步判据复算：24 个已绑定方法里各有 23 个对这两
个字段触发覆盖；唯一字面量存活的是 `class_prior_estimation`（其 `ClassPriorEstimator` 未声明这两个
类属性）。

**为何现在不能直接合并。** 两处写在 24 个条目上取值恰好全部一致（实测 mismatch = 0），因此**当前
没有可观测的不一致**；真正的风险是潜在漂移——只改字面量而不改类属性时会被静默忽略。所以这不是
「两值冲突」，而是「两处声明同一事实、其中一处（23/24）是死写」。

**收敛前提。** 把类属性定为唯一真相源：`builtin_methods.py` 的字面量降级为仅供无绑定类的 `api_only`
条目使用的兜底并在注释中写明；补一条断言「已绑定方法的能力字段来自类声明」的测试。

### RBF 权重

**重复形态。** `utils/basis.py:34` `build_rbf_basis` 是共享实现，`prior/pen_l1.py`、
`prior/kernel_mean.py`、`estimators/risk/kldce.py`、`estimators/risk/pnu.py`、`estimators/risk/upu.py`
都走它（`kldce.py:439` 另用基于它的 `rbf_weights`）。`estimators/bias_aware/pusb_kernel.py:36`
`_squared_distances` 与 `:46` `_rbf_design` 是另一套等价数学，其源码注释自陈
「Formula identical to `utils.basis.build_rbf_basis`」。

**为何现在不能直接合并。** `pusb_kernel` 的 CV 网格要在多组 sigma 上复用同一份预算平方距离矩阵，
且 `_rbf_design` 额外追加一列截距；`build_rbf_basis` 每次从 X 与 centers 重算距离且不含截距。直接
合并会重复计算距离并改变设计矩阵形状。

**收敛前提。** 若要收敛，需把 `build_rbf_basis` 拆出「距离预计算 → exp → 可选截距」的分层 API，
并先做基准确认不引入额外开销。
