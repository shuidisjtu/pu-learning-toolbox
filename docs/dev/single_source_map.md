# 单源地图：概念—真相源—消费者

> 本文档回答「某个概念在代码里的权威实现在哪、谁在读它、有没有第二份」。
> 模块分层与依赖方向见 [`architecture.md`](architecture.md)；治理批次与审计历史见
> [`architecture_principles.md`](architecture_principles.md) §5。
>
> 本表以 2026-10-03、BASE `f1182ec` 的审计为基线。**各行随其收敛落地就地更新**：未更新行，
> 其行号仍指 `f1182ec` 基线；已更新行的行号按**其权威源当前形态**重取，并标注能复现这些行号
> 的那一笔提交——它通常是本行的收敛提交（如「训练视图」，锚 `d5ebd57`）；若权威源在本行收敛
> 之前就已定形、此后未再改动，则锚定形它的那一笔（如「JSON 安全转换」，锚 `6a7c256`，
> `json_scalars` 即在那笔落地）。当前「训练视图」「JSON 安全转换」「哈希」三行已更新（「哈希」同属后
> 一类：权威源行号按 `9fe2004` 重取，其严格版 `strict_canonical_hash` 在那笔落地）。**发现重复不等于
> 已收敛**——其余各行只记录基线现状，迁移在后续批次；本表整体并非当前快照。

| 概念 | 权威真相源 | 消费者 | 兼容入口 | 判据 |
|---|---|---|---|---|
| 标签语义 | `pu_toolbox/core/labels.py:92` `normalize_pu_labels` / `:145` `normalize_pnu_labels`；字面值在 `pu_toolbox/core/config.py:8`-`:10` | `core/validation.py`（`validate_pu_X_y` / `validate_pnu_X_y` 的前置归一）、`metrics/classification.py`、`preprocessing/`、`model_selection/split.py`、`diagnostics/`、`workflows/shift.py`、`estimators/deep/self_pu.py`、`estimators/risk/nnpu.py`、`estimators/risk/vpu.py` | 无 | 单一 |
| 设备 | `pu_toolbox/core/device.py:11` `resolve_device_name` / `:29` `resolve_device` | 14 个 torch 估计器（`estimators/deep/*`、`estimators/risk/{dist_pu,nnpu,pulda,vpu}.py`、`estimators/research/*`）、`workflows/pipeline.py:431`、`workflows/_reporting.py:91` | 无 | 单一 |
| 随机源 | `pu_toolbox/core/random.py:8` `check_random_state` | `experiment/strategies.py`、`preprocessing/pu_labeling.py`、`preprocessing/selection_bias.py` | 无 | 重复（可收敛） |
| JSON 安全转换 | `pu_toolbox/utils/serialization.py:39` `json_safe`（宽容，报告载荷）与 `:54` `json_scalars`（严格，清单载荷）——**两者不是同一概念，不合并** | `json_safe`：`diagnostics/{benchmark,domain_assumptions,report,shift,shift_monitor,uncertainty}.py`、`preprocessing/data_profiler.py`、`workflows/report.py`；`json_scalars`：`experiment/{training_views,feature_adapter,datasets,survey_execution,survey_protocol}.py` | 无 | 单一 |
| 哈希 | `pu_toolbox/utils/serialization.py:34` `canonical_hash`（宽容，`allow_nan` 默认——报告与清单载荷）与 `:40` `strict_canonical_hash`（严格，`allow_nan=False`——制品身份）——**两者不是同一概念，不合并** | `canonical_hash`：`diagnostics/benchmark.py`、`experiment/{datasets,split_archive,strategies,image,training_views}.py`；`strict_canonical_hash`：`experiment/{feature_adapter,survey_protocol,survey_comparison}.py` | `survey_protocol.digest`、`survey_comparison.comparison_digest`（**只改委托、不改名**的公开入口） | 单一 |
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

**已收敛（单一）。** 权威源是 `utils/serialization.py` 的**一对**函数，二者**不是同一概念，不合并**：

- `:39` `json_safe`——**宽容**（NaN/Inf → None、`np.generic` → `item`、`Path` → `str`、递归
  dict/list），用于报告载荷：报告要么写得出来，要么不成其为报告，故它永不抛错。消费者在
  `diagnostics/`、`workflows/`、`preprocessing/data_profiler.py`。
- `:54` `json_scalars`——**严格**（非 JSON 标量或非有限浮点一律 `ValueError`），用于清单载荷
  （索引列表），因为这些列表会被摘要进 survey 制品：被静默强转的一个元素会在无人选择的情况下移动
  摘要。`name` 参数把报错归因到拒绝它的调用方。

**本批收敛了 8 处调用表达式。** `experiment/training_views.py` 调 3、`experiment/feature_adapter.py`
调 2、`experiment/datasets.py` 调 1，外加**审计漏记的 2 处内联**（下段）。三个旧定义已删：
`training_views._json_indices`、`feature_adapter._json_scalars`、`datasets._json_indices`。收敛判据：
模式 `grep -rn 'def _json_indices\|def _json_scalars\|def json_safe' pu_toolbox/` 命中 1 处——只剩
`serialization.py` 的 `def json_safe`（`__pycache__` 里的旧字节码命中不算）。

**审计漏记的 2 处内联（本批已收）。** 基线审计的 grep 口径是 `def _json_*`，**数不到内联写法**，
故少记两处：

- `experiment/survey_execution.py` 里 `cached_adapter` 的适配器**缓存键** `"indices"`；
- `experiment/survey_protocol.py` 里 `runner_protocol_context` 的表征 **`split_sha256`**。

这两处正是风险所在：`split_sha256` 这一个字段曾被**两套实现各算一次**（`feature_adapter` 走严格版、
`survey_protocol` 走弱版），今天相等靠的是输入恰好老实、而非共同契约。故真实清单是
5 文件 / 3 定义 / **8 处调用表达式**。

**为何不并入 `json_safe`。** 语义相反：`json_safe` 遇非有限值返回 `None`，`json_scalars` 抛
`ValueError`。把严格版换成 `json_safe`，会让「被哈希的列表里出现 NaN」从报错变成静默改写摘要——
正是本批要消除的失败模式。

**未并，且是不同概念（一处）。** `experiment/bundle.py:74` 的
`np.asarray(..., dtype=object).tolist()` 也不属本族：它在 `dtype=object` 上做**重叠判定**，需收
任意可哈希对象而非 JSON 标量，与 `json_scalars` 的严格 JSON 标量契约无关。

### 哈希

**已收敛（单一）。** 权威源是 `utils/serialization.py` 的**一对**函数，二者**不是同一概念，不合并**，
与上文「JSON 安全转换」的 `json_safe` / `json_scalars` 对偶同构（行号锚 `9fe2004`，严格版即在那笔落地）：

- `:34` `canonical_hash`——**宽容**（`allow_nan` 默认），用于**报告与清单载荷**：报告要么写得出去，
  写 `NaN` 是想要的，故它不拒绝非有限浮点。
- `:40` `strict_canonical_hash`——**严格**（`allow_nan=False`），用于**制品身份摘要**（`split_sha256`、
  `cache_key`、`protocol_sha256` 这类字段）：写不出去的数要先拒绝，否则摘要会在无人选择的情况下改变。

两者只差 `allow_nan`，在**非有限输入上分道扬镳**：宽容版写 `NaN`/`Infinity`（非严格 JSON），严格版抛
`ValueError`。并成带开关的一个名字，就是**把两个契约塞进一个名字**——同 E2 对 `json_safe` /
`json_scalars` 的裁定。

**本批收敛 6 个实现体（3 个 G1 + 3 个 G2）到这两个具名函数。**

- G1（严格，3 个）：`experiment/feature_adapter.py` 的 `_json_sha256`（已删）、
  `experiment/survey_protocol.py` 的 `digest`、`experiment/survey_comparison.py` 的
  `comparison_digest`——后两者**保留函数名**，函数体改为委托。
- G2（宽容，3 个）：`experiment/image.py` 的 `_json_sha256`（已删）、`experiment/training_views.py`
  的 `indices_sha256`（内联，已改调），并入本行权威源的 `canonical_hash` 本体（**一字未改**）。

**委托（保留名字、只改转发）。** 两个公开名 `survey_protocol.digest` 与
`survey_comparison.comparison_digest` **只改委托、不改名**：`scripts/` 三个脚本
（`audit_survey_batches`、`compare_survey_results`、`summarize_survey_results`）与 `tests/` 四个文件
直接 import 后者；另有 `experiment/survey_recipe_registry.py` 的 `canonical_digest`（`:264`）与
`resolved_params_digest`（`:274`）两处，以及 `experiment/survey_execution.py:201` 的内联
`digest(...)`（`cache_key`）。私有包装 `survey_comparison._survey_digest` 已删除，其唯一调用点改指
`comparison_digest`——收敛后「与 `survey_protocol.digest` 相同」由结构保证，不再靠 docstring 自述。

**G1/G2 的全部调用方。**

- G1（`strict_canonical_hash`，直接或经上列公开名）：`experiment/feature_adapter.py`（4 处：
  `encoder_fit_indices_sha256`、`representation_sha256`、`split_sha256`、`fairness_sha256`）、
  `experiment/survey_protocol.py`（`digest` 本体、`load_protocol` 末行、`runner_protocol_context` 的
  `protocol_sha256` / `split_sha256`）、`experiment/survey_comparison.py`（`comparison_digest` /
  `_validate_binding` / `comparison_context` / `build_comparison_report`）、
  `experiment/survey_recipe_registry.py`、`experiment/survey_execution.py`。
- G2（`canonical_hash` 直接调用）：`diagnostics/benchmark.py`、`experiment/datasets.py`、
  `experiment/split_archive.py`、`experiment/strategies.py`，以及本批新增的 `experiment/image.py`
  与 `experiment/training_views.py`。

**知情保留（第三种语义，一处）。** `experiment/text.py:206` 的 `_json_sha256` **一字未动**：它多一个
`ensure_ascii=False`（并以 utf-8 编码），在**含非 ASCII 码点**的语料上摘要与上两者都不同，且其产物
`cache_key` **直接是缓存文件名**——改它会让已记录的 `texts_sha256` 与本地文本缓存失效，属破坏性兼容
变更。也不为此加 `ensure_ascii` 开关：把第三个语义并成一个带开关的名字，与上文两条对偶的裁决同理。

**未并（另一族，非重复）。** `_array_sha256` 两份（`experiment/feature_adapter.py` 与
`experiment/image.py`）哈希的是**数组 dtype/shape/bytes**，不是 JSON 规范形式，不属本族，本批不动。

**收敛判据。** 模式 `grep -rn 'def _json_sha256\|def _survey_digest\|def canonical_hash'
pu_toolbox/ --include=*.py` 命中 2 处：`utils/serialization.py` 的 `def canonical_hash`（单源本体）
与 `experiment/text.py:206` 的 `def _json_sha256`（上列知情保留）。被删的两处私有体
（`feature_adapter` / `image.py` 的 `_json_sha256`）与 `_survey_digest` 必须零命中；该模式**不匹配**
`def strict_canonical_hash`（另一字符串）。

### 训练视图

**已收敛（单一）。** 角色名与 os/ts 视图名的唯一声明在 `pu_toolbox/core/training_views.py`：
`:42` `ViewRole` 与 `:38` `RunView` 两个 `Literal` 别名是唯一来源，运行时元组 `:48` `ROLES` /
`:47` `RUN_VIEWS` 由 `typing.get_args` 从别名派生——改别名即同步改元组，两者不可能漂移；
`:120` `build_training_view` 是唯一构造器。消费端改为导入该单源、不再内联复制：
`estimators/` 的六个 os/ts 消费点（`risk/{vpu,upu,nnpu,dist_pu}.py`、`bias_aware/pusb_kernel.py`、
`deep/self_pu.py`）、`experiment/` 层各消费者、`scripts/run_survey_experiment.py`（`--os-or-ts` 的
choices 与拆分文件角色名）、`scripts/run_survey_pilot.py`（choices）、
`scripts/prepare_survey_splits.py`（写拆分与尺寸）。收敛判据：模式
`grep -rn 'not in {"os", "ts"}' pu_toolbox/ scripts/` 无命中；模式
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
