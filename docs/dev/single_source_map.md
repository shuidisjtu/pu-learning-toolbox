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
| 训练视图 | `pu_toolbox/core/training_views.py:117` `build_training_view`；角色词表 `:42` `ViewRole` / `:45` `_LEGAL_ROLES` | `estimators/risk/vpu.py:152`、`experiment/training_views.py:85`；角色元组另有 9 处内联复制、2 处集合字面量 | `experiment/training_views.py:42` `TSOSBatchView`（legacy 边界适配器） | 重复（可收敛） |
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

**重复形态。** `core/training_views.py:117` `build_training_view` 是唯一构造器，角色词表在 `:42`
`ViewRole`（Literal）与 `:45` `_LEGAL_ROLES`（元组）。但同一四元角色在 experiment/ 层被反复内联：
模式 `grep -rn '("train", "pu_val", "clean_val", "test")' pu_toolbox/` 命中 11 处（含核心 1 处元组与
`survey_protocol.py:26` 这第二份模块级定义 `ROLES`），其余为 `bundle.py`×2、`datasets.py`×2、
`feature_adapter.py`×4、`runner.py`×1 的内联字面元组；另有 2 处集合字面量
（`experiment/image.py:323`、`experiment/training_views.py:145`）与 2 处类型别名
（`core/training_views.py:42`、`experiment/image.py:14`）。`pu_val` 在 `experiment/` 之外出现 34 次。

**为何现在不能直接合并。** 角色名被当纯字符串键分散使用；`core` 不能反向依赖 `experiment`，而
`experiment/survey_protocol.py:26` 的 `ROLES` 是跨模块导入点；`Literal[...]` 类型别名也无法由运行时
元组推导。一次替换触及面过广。

**收敛前提。** 先在 `core` 暴露一个公开元组作为唯一定义，让各内联字面量改为导入；`Literal` 若要由
元组派生，需先确认 mypy/typing 的可行方案。

### 方法能力字段

**重复形态。** 同一事实写两处：`registry/builtin_methods.py` 在 `AlgorithmMetadata(...)` 里写
`implementation_status=` / `source_status=` 字面量（模式 `grep -c` 各 24 处），估计器类再声明同名类
属性（如 `estimators/risk/puet.py:85`-`:86`）。注册时 `bind_estimator_class` →
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
