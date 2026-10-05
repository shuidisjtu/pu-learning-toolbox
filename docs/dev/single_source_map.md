# 单源地图：概念—真相源—消费者

> 本文档回答「某个概念在代码里的权威实现在哪、谁在读它、有没有第二份」。
> 模块分层与依赖方向见 [`architecture.md`](architecture.md)；治理批次与审计历史见
> [`architecture_principles.md`](architecture_principles.md) §5。
>
> 各行随其收敛落地就地更新，行号按权威源当前形态重取。**本表整体并非当前快照**——未更新行的
> 行号可能已过期，发现重复不等于已收敛。

| 概念 | 权威真相源 | 消费者 | 兼容入口 | 判据 |
|---|---|---|---|---|
| 标签语义 | `pu_toolbox/core/labels.py:92` `normalize_pu_labels` / `:145` `normalize_pnu_labels`；字面值在 `pu_toolbox/core/config.py:8`-`:10` | `core/validation.py`（`validate_pu_X_y` / `validate_pnu_X_y` 的前置归一）、`metrics/classification.py`、`preprocessing/`、`model_selection/split.py`、`diagnostics/`、`workflows/shift.py`、`estimators/deep/self_pu.py`、`estimators/risk/nnpu.py`、`estimators/risk/vpu.py` | 无 | 单一 |
| 设备 | `pu_toolbox/core/device.py:11` `resolve_device_name` / `:29` `resolve_device` | 14 个 torch 估计器（`estimators/deep/*`、`estimators/risk/{dist_pu,nnpu,pulda,vpu}.py`、`estimators/research/*`）、`workflows/pipeline.py:431`、`workflows/_reporting.py:91` | 无 | 单一 |
| 随机源 | `pu_toolbox/core/random.py:8` `check_random_state` | `experiment/strategies.py`、`preprocessing/pu_labeling.py`、`preprocessing/selection_bias.py` | 无 | 重复（不可合并，理由：见说明） |
| JSON 安全转换 | `pu_toolbox/utils/serialization.py:92` `json_safe`（宽容，报告载荷）与 `:107` `json_scalars`（严格，清单载荷）——**两者不是同一概念，不合并** | `json_safe`：`diagnostics/{benchmark,domain_assumptions,report,shift,shift_monitor,uncertainty}.py`、`preprocessing/data_profiler.py`、`workflows/report.py`；`json_scalars`：`experiment/{training_views,feature_adapter,datasets,survey_execution,survey_protocol}.py` | 无 | 单一 |
| 哈希 | `pu_toolbox/utils/serialization.py` 的**一组三族**：`:41` `canonical_hash`（宽容，`allow_nan` 默认——报告与清单载荷）与 `:47` `strict_canonical_hash`（严格，`allow_nan=False`——制品身份）为一族，**两者不是同一概念，不合并**；`:67` `array_hash`（数组 dtype/shape/字节）与 `:83` `file_hash`（文件字节，流式）各为一族 | `canonical_hash`：`diagnostics/benchmark.py`、`experiment/{datasets,split_archive,strategies,image,training_views}.py`；`strict_canonical_hash`：`experiment/{feature_adapter,survey_protocol,survey_comparison}.py`；`array_hash`：`experiment/{feature_adapter,image,survey_execution}.py`，并经公开名 `survey_protocol.array_digest` 供 `experiment/survey_execution.py` 与 `experiment/survey_protocol.py` 消费；`file_hash`：`experiment/{checkpoints,text}.py`，并经公开名 `split_archive.file_sha256` 供 `experiment/split_archive.py` 与 `tests/` 消费 | `survey_protocol.digest`、`survey_comparison.comparison_digest`、`survey_protocol.array_digest`、`split_archive.file_sha256`（**只改委托、不改名**的公开入口） | 单一 |
| RBF 权重 | `pu_toolbox/utils/basis.py:34` `build_rbf_basis`（`:62` `rbf_weights` 建于其上） | `prior/pen_l1.py`、`prior/kernel_mean.py`、`estimators/risk/kldce.py`、`estimators/risk/pnu.py`、`estimators/risk/upu.py`、`utils/basis.py:101` `resolve_basis_fn` | `pu_toolbox/utils/__init__.py` 重导出 | 重复（不可合并，理由：见说明） |
| 类先验推导 | `pu_toolbox/estimators/risk/_class_prior.py:6` `solve_prior_from_positive_fraction` | `estimators/risk/kldce.py:935`、`estimators/risk/ldce.py:429` | 无 | 单一 |
| 训练视图 | `pu_toolbox/core/training_views.py:120` `build_training_view`；角色词表 `:42` `ViewRole` + `:48` `ROLES`；视图词表 `:38` `RunView` + `:47` `RUN_VIEWS`（运行时元组由 `get_args` 派生） | `estimators/risk/{vpu,upu,nnpu,dist_pu}.py`、`estimators/bias_aware/pusb_kernel.py`、`estimators/deep/self_pu.py`、`experiment/` 各消费者、`scripts/{run_survey_experiment,run_survey_pilot,prepare_survey_splits}.py`；另有三项词表与两处子集未并（不同概念，见说明） | `experiment/training_views.py:42` `TSOSBatchView`（legacy 边界适配器） | 单一（角色名与 os/ts 视图；未并项见说明） |
| 方法能力字段 | 估计器类属性（如 `estimators/risk/puet.py:85`-`:86`），经 `registry/registry.py:141` `_sync_class_metadata_to_registry` 同步；**类未声明的字段以条目字面量为源**（`class_prior_estimation` 的 8 个字段、`pusb`/`lbe` 的 `requires_class_prior` 即如此） | `registry/registry.py` `get_metadata`、`advisor/`、`cli/`、`workflows/pipeline.py`、`ui/` | 无 | 单一（逐字段判定；`api_only` 边界见说明） |

## 说明

### 随机源

**判定：重复（不可合并）。** 两个实现不是同一概念的两个副本，而是**两类消费者各持一个契约**：
`preprocessing/` 声明宽契约 `int | np.random.RandomState | None`（接受实例并原样返回以延续随机流，
是承重用法），估计器与先验层统一声明 `int | None` 且无一处传实例。把内联式换成 helper 是
**双向变更**：实例由 `TypeError` 变可用（放宽），array-like 整数（如 `np.array(42)`）由可用变
`TypeError`（收紧）；对 3 个直传 `torch.manual_seed` 的类放宽不可达，改播种会改变随机轨迹。
`diagnostics/domain_assumptions.py` 的 `Generator` 族另属一族，不受 `check_random_state` 归一。

### JSON 安全转换

**已收敛（单一）。** 权威源是 `utils/serialization.py` 的**一对**函数，二者**不是同一概念，不合并**：

- `:56` `json_safe`——**宽容**（NaN/Inf → None、`np.generic` → `item`、`Path` → `str`、递归
  dict/list），用于报告载荷，消费者在 `diagnostics/`、`workflows/`、`preprocessing/data_profiler.py`。
- `:71` `json_scalars`——**严格**（非 JSON 标量或非有限浮点一律 `ValueError`），用于清单载荷
  （索引列表）。`name` 参数把报错归因到拒绝它的调用方。

语义相反：`json_safe` 遇非有限值返回 `None`，`json_scalars` 抛 `ValueError`，故不合并。收敛判据：
模式 `grep -rn 'def _json_indices\|def _json_scalars\|def json_safe' pu_toolbox/` 命中 1 处——只剩
`serialization.py` 的 `def json_safe`。`experiment/bundle.py:74` 的
`np.asarray(..., dtype=object).tolist()` 是不同概念（收任意可哈希对象），不属本族。

### 哈希

**已收敛（单一）。** 权威源是 `utils/serialization.py` 的**一组三族**，三族**不是同一概念，不合并**。
JSON 对与上文「JSON 安全转换」的 `json_safe` / `json_scalars` 对偶同构：

- `:41` `canonical_hash`——**宽容**（`allow_nan` 默认），用于**报告与清单载荷**：报告要么写得出去，
  写 `NaN` 是想要的，故它不拒绝非有限浮点。
- `:47` `strict_canonical_hash`——**严格**（`allow_nan=False`），用于**制品身份摘要**（`split_sha256`、
  `cache_key`、`protocol_sha256` 这类字段）：写不出去的数要先拒绝，否则摘要会在无人选择的情况下改变。
- `:67` `array_hash`——数组的 dtype / shape / 字节摘要，供 `feature_sha256`、`train_data_sha256`
  这类**制品身份**使用。
- `:83` `file_hash`——文件字节的流式摘要（1 MiB 分块，`_FILE_CHUNK_BYTES`），供 `cache_sha256`、
  `manifest_sha256` 使用。

JSON 对只差 `allow_nan`，在**非有限输入上分道扬镳**：宽容版写 `NaN`/`Infinity`（非严格 JSON），严格版抛
`ValueError`，故不合并（同 `json_safe` / `json_scalars` 的裁定）。

**知情保留（第三种语义，一处）。** `experiment/text.py:208` 的 `_json_sha256` 多一个
`ensure_ascii=False`，其产物 `cache_key` **直接是缓存文件名**，改它会让已记录的 `texts_sha256` 失效。
公开名 `survey_protocol.digest` / `comparison_digest` / `array_digest` / `split_archive.file_sha256`
**只改委托、不改名**。

**未并（另一族，登记为开口）。** `benchmarks/` 下 **13 处** `hashlib.sha256(` 仍是各自为政，
产物是**基准配置身份**（`runner_sha256` / `dataset_sha256`），收敛会改写已记录值。
`pu_toolbox/ui/app.py:46` 的 `sha256(config_bytes)` 同样未收敛：入参是 Streamlit 上传文件的
**内存字节块**而非文件路径，与 `file_hash`（收路径）签名不同族。

**收敛判据。** JSON 对：模式 `grep -rn 'def _json_sha256\|def _survey_digest\|def canonical_hash'
pu_toolbox/ --include=*.py` 命中 2 处：`utils/serialization.py` 的 `def canonical_hash`（单源本体）
与 `experiment/text.py:208` 的 `def _json_sha256`（上列知情保留）。其余旧定义
（`feature_adapter` / `image.py` 的 `_json_sha256`）与 `_survey_digest` 必须零命中；该模式**不匹配**
`def strict_canonical_hash`（另一字符串）。二进制两组：模式 `grep -rn 'def _array_sha256\|def _file_digest
\|def _file_sha256\|def array_digest\|def file_sha256' pu_toolbox/ --include=*.py` **恰好命中 2 处**——
`experiment/survey_protocol.py` 的 `def array_digest` 与 `experiment/split_archive.py` 的
`def file_sha256`（两个公开委托体）；`feature_adapter` / `image` 的 `_array_sha256`、`checkpoints` 的
`_file_digest`、`text` 的 `_file_sha256` 必须零命中，而 `utils/serialization.py` 的
`def array_hash` / `def file_hash` **不匹配**该模式（不同字符串）。

### 训练视图

**已收敛（单一）。** 角色名与 os/ts 视图名的唯一声明在 `pu_toolbox/core/training_views.py`：
`:42` `ViewRole` 与 `:38` `RunView` 两个 `Literal` 别名是唯一来源，运行时元组 `:48` `ROLES` /
`:47` `RUN_VIEWS` 由 `typing.get_args` 从别名派生——改别名即同步改元组，两者不可能漂移；
`:120` `build_training_view` 是唯一构造器。收敛判据：模式
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

**已收敛（单一）。** 能力字段的权威源是估计器类属性：注册时 `bind_estimator_class` →
`registry/registry.py:141` `_sync_class_metadata_to_registry` 按 `_SYNC_FIELDS`（**13** 个成员）
对类**自身**声明的字段做 `setattr`。分两半：

- **8 个条目字段**：`family` / `assumption` / `scenario` / `requires_class_prior` /
  `implementation_status` / `source_status` / `backend` / `maturity`。
- **5 个仅类/默认值字段**：`native_architectures` / `input_ndims` / `encoder_parameter` /
  `trains_encoder`（4 个架构能力字段）与 `label_semantics`，从不写字面量。

逐字段判定：**类已声明的字段不得在字面量里重复出现**，**类未声明的字段必须在字面量里显式声明**
（此时字面量是唯一源），由 `tests/test_builtin_methods.py` 的双向契约（clause 1 / clause 2）守卫。
边界：clause 1 无条件禁止任何条目（含未来 `api_only`）写那 5 个仅类字段，`api_only` 条目因此没有
声明途径——今日 `api_only` 为 0，规则空转。另 `experiment/method_ledger.json` 的
`implementation_status` 与 `class` 是**未受门的第三份**，漂移不会使任何测试变红。

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
