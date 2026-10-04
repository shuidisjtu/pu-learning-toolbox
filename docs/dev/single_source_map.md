# 单源地图：概念—真相源—消费者

> 本文档回答「某个概念在代码里的权威实现在哪、谁在读它、有没有第二份」。
> 模块分层与依赖方向见 [`architecture.md`](architecture.md)；治理批次与审计历史见
> [`architecture_principles.md`](architecture_principles.md) §5。
>
> 本表以 2026-10-03、BASE `f1182ec` 的审计为基线。**各行随其收敛落地就地更新**：未更新行，
> 其行号仍指 `f1182ec` 基线；已更新行的行号按**其权威源当前形态**重取，并标注能复现这些行号
> 的那一笔提交——它通常是本行的收敛提交（如「训练视图」，锚 `d5ebd57`）；若权威源在被重取之前
> 已由更早一笔定形、此后未再改动，则锚定形它的那一笔。当前「训练视图」「JSON 安全转换」
> 「哈希」三行已更新：后两行的权威源同在 `pu_toolbox/utils/serialization.py`，均按 `0f40217`
> 重取（该笔把 `array_hash` / `file_hash` 并入该模块，令 JSON 对与 `json_safe` / `json_scalars`
> 一并下移；`strict_canonical_hash` 则在更早的 `9fe2004` 落地）。另有「方法能力字段」行在批次 E4
> 收敛为单一（权威源**逐字段**判定：类声明了就以类为源，未声明则以条目字面量为源；行号锚
> `f54c0c6`，该笔最后动过本行引用的两个文件）。**发现重复不等于
> 已收敛**——其余各行只记录基线现状，迁移在后续批次；本表整体并非当前快照。

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

**判定：重复（不可合并）。** 批次 D 曾判「重复（可收敛）」并给出前提；批次 E5 复核后改判，
理由见下。两个实现不是同一概念的两个副本，而是**两类消费者各持一个契约**。

**两个契约（逐字）。** `core/random.py:8` 的 `check_random_state` 声明
`seed: int | np.random.RandomState | None`。它的三个消费模块里，**只有 `preprocessing/` 声明宽契约**：
`preprocessing/pu_labeling.py:71`、`:136`、`:197` 等声明
`random_state: int | np.random.RandomState | None`，docstring 亦写作
「int or np.random.RandomState or None」；`experiment/` 侧只**消费**该 helper，
其 `experiment/protocols.py:42` 的 `seed` 声明为 `int | None`。估计器与先验层中**带内联
`np.random.RandomState(self.random_state)` 构造点的 18 个类**则统一声明
`random_state: int | None`（该层声明此参数的类不止 18 个——`dist_pu`、`infomax_pu`、`research/*`
等并无该构造点，另有一处工厂函数参数；`docs/user/reference/api.md` 凡给出类型处均为
`` `int \| None` ``，仅 `InfoMaxPUClassifier`、`WeightedContrastivePUClassifier`、`DGPUClassifier`
三条只给签名、无类型列），且**无一处传入实例**。

**为何不能合并。** 把内联式换成 helper 不是「放宽」而是**双向变更**（numpy 2.4.6 实测）：

| 候选 | 内联 `np.random.RandomState(v)` | `check_random_state(v)` | 方向 |
|---|---|---|---|
| `RandomState` 实例 | `TypeError` | 通过，且原样返回同一对象 | 放宽 |
| `np.array(42)`（0-d） | 通过 | `TypeError` | 收紧 |
| `[42]` | 通过 | `TypeError` | 收紧 |
| `np.array([42])` | 通过 | `TypeError` | 收紧 |

其余候选（`int` / `np.int64` / `np.uint32` / `bool` / `None` / `Generator` / `str` / `float` /
越界与负数）两侧同结果。若照「放宽」的旧记述办事，会漏掉收紧面。

**放宽对 3 个类不可达。** 10 个 torch 类里 7 个从各自构造的 rng 派生种子
（七处均为 `torch.manual_seed(...)`，入参取自各自构造的 rng 的 `randint(0, 2**31)`——
其中 6 处带 `int()`，`nnpu` 一处不带，语义等价：`nnpu`、`vpu`、`pulda`、`lagam`、
`split_pu`、`grad_pu`、`robust_pu`），迁移不改变轨迹；另 3 个把属性直传
（`dgpu:153`、`self_pu:579`、`weighted_contrastive_pu:167` 的
`torch.manual_seed(self.random_state)`），实测对实例抛 `TypeError`，`dgpu:275` 为
`self.random_state + 2 * round_index`、`:285` 为 `self.random_state + 2 * round_index + 1`
的算术约束。对它们换 helper 只是把报错从 numpy 行
搬到 torch 行，且在全局状态已被改动之后。若为「彻底放宽」而把直传改为派生，torch 种子会由
`N` 变成派生值，**改变这三类的随机轨迹**——越界。

**实例传递在外层是承重用法。** `pu_labeling.py:408` 把 `:400` 建好的 `rng` 传给
`make_scar_labels`，`selection_bias.py:362` 同样链式传递，目的是让下游**延续同一条随机流**；
测试侧 `tests/conftest.py:20` 的 `rng` fixture 经 `tests/helpers.py:38`、`:56`、
`tests/integration/test_run.py:265`、`tests/unit/estimators/test_elkan_noto.py` 13 处传入。
即「接受实例」有真实消费者，不是遗留兼容。

**helper 调用点共 10 处**：`experiment/strategies.py` 3 处
（`:92`/`:154`/`:191`，其 `seed` 由 `experiment/protocols.py:42` 声明为 `int | None`）、
`preprocessing/pu_labeling.py` 5 处（`:107`/`:179`/`:300`/`:350`/`:400`）、
`preprocessing/selection_bias.py` 2 处（`:246`/`:345`）。

**第 19 处不是绕过。** `prior/pen_l1.py:27` 的 `np.random.RandomState(0)` 位于模块级
`_median_pairwise_distance`，而 `ClassPriorEstimator` **没有 `random_state` 参数**；
它是固定种子的确定性行子采样，不是「绕过归一化的参数」。改写为 `check_random_state(0)`
行为零改变，属纯装饰。

**第二个生成器族（本次不动）。** `diagnostics/domain_assumptions.py:338` 的
`np.random.default_rng(random_state)` 构造的是 `Generator` 而非 `RandomState`，不受
`check_random_state` 归一；其公开入口 `analyze_domain_assumptions` 只声明
`random_state: int | None`，改动会改变 bootstrap 置信区间，故须单独立项。

**登记项。** `check_random_state` 此前**零直接测试**（批次 E5 已补
`tests/unit/core/test_random.py`）、未从 `core/__init__.py` 或 `pu_toolbox/__init__.py` 导出、
未出现在 `api.md`，也不在 `CONTRIBUTING.md` §5.1 的「必须复用」表内——**这是有意的**：
把它写进 §5.1 会误导贡献者在估计器层复用它。

**若将来要收敛。** 前提是同时解决：① 目标层的 `random_state` 契约改为
`int | RandomState | None` 并接受 array-like 整数的收紧；② 对 3 个直传 torch 类给出
不改变轨迹的方案（或明确其放宽不可达）；③ `dgpu` 的算术点单独裁定。

### JSON 安全转换

**已收敛（单一）。** 权威源是 `utils/serialization.py` 的**一对**函数，二者**不是同一概念，不合并**：

- `:56` `json_safe`——**宽容**（NaN/Inf → None、`np.generic` → `item`、`Path` → `str`、递归
  dict/list），用于报告载荷。它不抛错是指**常规载荷**：自引用结构会 `RecursionError`，
  未识别类型（`np.ndarray`、`set`、`bytes`）会**原样返回**，故它不保证输出可序列化。
  消费者在 `diagnostics/`、`workflows/`、`preprocessing/data_profiler.py`。
- `:71` `json_scalars`——**严格**（非 JSON 标量或非有限浮点一律 `ValueError`），用于清单载荷
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

**已收敛（单一）。** 权威源是 `utils/serialization.py` 的**一组三族**，三族**不是同一概念，不合并**。
JSON 对与上文「JSON 安全转换」的 `json_safe` / `json_scalars` 对偶同构（行号锚 `0f40217`，
`strict_canonical_hash` 落地于更早的 `9fe2004`）：

- `:41` `canonical_hash`——**宽容**（`allow_nan` 默认），用于**报告与清单载荷**：报告要么写得出去，
  写 `NaN` 是想要的，故它不拒绝非有限浮点。
- `:47` `strict_canonical_hash`——**严格**（`allow_nan=False`），用于**制品身份摘要**（`split_sha256`、
  `cache_key`、`protocol_sha256` 这类字段）：写不出去的数要先拒绝，否则摘要会在无人选择的情况下改变。
- `:67` `array_hash`——数组的 dtype / shape / 字节摘要，供 `feature_sha256`、`train_data_sha256`
  这类**制品身份**使用。
- `:83` `file_hash`——文件字节的流式摘要（1 MiB 分块，`_FILE_CHUNK_BYTES`），供 `cache_sha256`、
  `manifest_sha256` 使用。

JSON 对只差 `allow_nan`，在**非有限输入上分道扬镳**：宽容版写 `NaN`/`Infinity`（非严格 JSON），严格版抛
`ValueError`。并成带开关的一个名字，就是**把两个契约塞进一个名字**——同 E2 对 `json_safe` /
`json_scalars` 的裁定。

**E3a 收敛 6 个实现体（3 个 G1 + 3 个 G2）到 JSON 对。**

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

**G1/G2 的全部 `pu_toolbox/` 内调用方。**

- G1（`strict_canonical_hash`，直接或经上列公开名）：`experiment/feature_adapter.py`（4 处：
  `encoder_fit_indices_sha256`、`representation_sha256`、`split_sha256`、`fairness_sha256`）、
  `experiment/survey_protocol.py`（`digest` 本体、`load_protocol` 末行、`runner_protocol_context` 的
  `protocol_sha256` / `split_sha256`）、`experiment/survey_comparison.py`（`comparison_digest` /
  `_validate_binding` / `comparison_context` / `build_comparison_report`）、
  `experiment/survey_recipe_registry.py`、`experiment/survey_execution.py`。
- G2（`canonical_hash` 直接调用）：`diagnostics/benchmark.py`、`experiment/datasets.py`、
  `experiment/split_archive.py`、`experiment/strategies.py`，以及 E3a 新增的 `experiment/image.py`
  与 `experiment/training_views.py`。

`canonical_hash` 另有 `benchmarks/` 侧消费者：它们经 `benchmarks/_common.py` 的兼容 re-export
（该模块重新导出它并列入自身 `__all__`）使用，不在上列 `pu_toolbox/` 口径内。

**知情保留（第三种语义，一处）。** `experiment/text.py:208` 的 `_json_sha256` **一字未动**：它多一个
`ensure_ascii=False`（并以 utf-8 编码），在**含非 ASCII 码点**的语料上摘要与上两者都不同，且其产物
`cache_key` **直接是缓存文件名**——改它会让已记录的 `texts_sha256` 与本地文本缓存失效，属破坏性兼容
变更。也不为此加 `ensure_ascii` 开关：把第三个语义并成一个带开关的名字，与上文两条对偶的裁决同理。
**勿与 File-1 的 `_file_sha256` 混淆**：后者是同文件里的**文件**摘要（E3b 已删、去向 `file_hash`），
而这条是 **JSON** 摘要，两者正交。

**E3b 把两组二进制内容摘要各收敛到一个 helper（`array_hash` / `file_hash`）。**

- **Arr-1（数组，3 个实现体）。** `array_hash`（`:67`）是数组摘要的唯一配方。
  `experiment/feature_adapter.py` 的 `_array_sha256`（调点现 `:161`）与 `experiment/image.py` 的
  `_array_sha256`（调点现 `:159`）**已删**，两处改调 `array_hash`；`experiment/survey_protocol.py:188`
  的公开名 `array_digest` **留名改委托**（`return array_hash(value)`）。`experiment/survey_execution.py`
  原有一行**函数内** `from .feature_adapter import _array_sha256`（`:219`）与其唯一调用点（现 `:226`）：
  函数内导入已删，调用改调模块级 `array_hash`（该模块本就模块级 import 同包，上提不引入环）。
- **0-d 分叉与裁定。** 三个数组实现**并非逐行同构**：`survey_protocol.array_digest` 先
  `np.ascontiguousarray(value)` 再从**返回值**读 shape，而该函数把 0-d 提升为 1-d（`ndmin=1`），
  shape 串由 `[]` 变成 `[1]`；两处 `_array_sha256` 则从**原数组**读 shape。裁定为**取原数组的
  shape**（即 `array_hash` 的语义），理由：其背后已落盘制品更多（`feature_sha256` ×4 +
  `train_data_sha256`）。该裁定**无任何冻结制品覆盖 0-d 输入**，守卫只有
  `tests/unit/utils/test_content_hashes.py` 的两条合成用例，覆盖两个面：
  `test_edge_array_hash_reads_the_shape_of_the_callers_array` 钉住 helper 的字面摘要，
  `test_param_the_public_entry_point_carries_the_same_zero_d_ruling` 钉住**公开入口**
  （并复现入口改前的配方 `4a95a9fb…d03` 作为阴性对照——只守 helper 时，把入口改回旧配方
  不会有任何测试变红）。
- **File-1（文件，3 个实现体）。** `file_hash`（`:83`）是文件字节摘要的唯一配方。
  `experiment/checkpoints.py` 的 `_file_digest`（调点现 `:113`、`:249`）与 `experiment/text.py` 的
  `_file_sha256`（调点现 `:91`、`:122`）**已删**，四处改调 `file_hash`；`experiment/split_archive.py:62`
  的公开名 `file_sha256` **留名改委托**（`return file_hash(path)`），其 5 个内部调用点
  （现 `:121`、`:130`、`:208`、`:271`、`:395`）引用模块全局名不变，故
  `tests/unit/experiment/test_split_archive.py:198` 对该全局名的 monkeypatch 仍生效。

**两个公开委托名及其消费者（E3b）。** `survey_protocol.array_digest`：`experiment/survey_execution.py:29`
模块级 import，调用点 `:153`、`:206`；另有 `experiment/survey_protocol.py:455` 内部一处。
`split_archive.file_sha256`：`experiment/split_archive.py` 内部 5 处（上列），
`tests/unit/experiment/test_split_archive.py`（import 并在 `:198` monkeypatch）。

**未并（另一族，登记为开口）。** `benchmarks/` 下 **13 处** `hashlib.sha256(`（横跨 **10 个模块**）
仍是各自为政：具名 helper 有 `assigned_methods/pusb_official_data.py:33` `_sha256` 与
`assigned_methods/runner.py:386` `_file_sha256`（均一次性 `read_bytes()`），与
`assigned_methods/pusb_table2_{aggregate,data,report}.py`、`deep_pu/official_data.py:560` `_sha256_file`
（1 MiB 流式）；其余为内联调用（`assigned_methods/preflight_paper.py:92`/`:126`、四个 `runner_sha256`
内联点、`deep_pu/official_data.py:327` 的数组字节一次性）。它们的产物是**基准配置身份**
（`runner_sha256` / `dataset_sha256`），收敛会改写已记录值，且不在本批两道治理 ratchet 的语料内，
故留待各自冻结与验收。`pu_toolbox/ui/app.py:46` 的 `sha256(config_bytes)` 同样未收敛：入参是
Streamlit 上传文件的**内存字节块**而非文件路径，与 `file_hash`（收路径）签名不同族。

**收敛判据。** JSON 对：模式 `grep -rn 'def _json_sha256\|def _survey_digest\|def canonical_hash'
pu_toolbox/ --include=*.py` 命中 2 处：`utils/serialization.py` 的 `def canonical_hash`（单源本体）
与 `experiment/text.py:208` 的 `def _json_sha256`（上列知情保留）。被删的两处私有体
（`feature_adapter` / `image.py` 的 `_json_sha256`）与 `_survey_digest` 必须零命中；该模式**不匹配**
`def strict_canonical_hash`（另一字符串）。E3b 两组：模式 `grep -rn 'def _array_sha256\|def _file_digest
\|def _file_sha256\|def array_digest\|def file_sha256' pu_toolbox/ --include=*.py` **恰好命中 2 处**——
`experiment/survey_protocol.py` 的 `def array_digest` 与 `experiment/split_archive.py` 的
`def file_sha256`（两个公开委托体）；`feature_adapter` / `image` 的 `_array_sha256`、`checkpoints` 的
`_file_digest`、`text` 的 `_file_sha256` 必须零命中，而 `utils/serialization.py` 的
`def array_hash` / `def file_hash` **不匹配**该模式（不同字符串）。

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

**已收敛（单一）。** 能力字段的权威源是估计器类属性：注册时 `bind_estimator_class` →
`registry/registry.py:141` `_sync_class_metadata_to_registry` 按 `_SYNC_FIELDS`
（`registry/registry.py:124`-`:138`）对类**自身**声明的字段做 `setattr`，写进注册表条目。
`_SYNC_FIELDS` 共 **13** 个成员，分两半：

- **8 个条目字段**：`family` / `assumption` / `scenario` / `requires_class_prior` /
  `implementation_status` / `source_status` / `backend` / `maturity`。它们曾被同时写在
  `registry/builtin_methods.py` 的 `AlgorithmMetadata(...)` 字面量里与类属性上，本批已收敛为单源。
- **5 个仅类/默认值字段**：`native_architectures` / `input_ndims` / `encoder_parameter` /
  `trains_encoder`（4 个架构能力字段）与 `label_semantics`。它们**从来不写字面量**，只存在于类属性
  与 dataclass 默认值中，且已由 `tests/contract/test_capability_declarations.py` 守卫。

**审计的两条错误记述（本批订正）。** 批次 D 审计行只点了 `implementation_status` / `source_status`
两个字段，把 8 个字的同步面**少记了 6 个**；它另有两处与事实不符：

1. 原文称「只改字面量而不改类属性时会被静默忽略」——**不成立**。
   `tests/test_builtin_methods.py` 的事故守卫自 `4d5eebe`（PR #12）起就在比对字面量与类值，
   这类漂移早已会被它抓住。
2. 原文的「收敛前提：补一条断言」——**那条断言早已存在**。若照做，是重写一条既有测试，
   而不是新增保护。

**真正的缺口在活写一侧。** 那条既有守卫对「类未声明的字段」直接跳过（源码里是 `continue`），
于是它守的是 182 处**死写**（写错也无害，反正会被类属性覆盖），跳过的是 10 处**活写**
（字面量即权威，写错有后果）。10 处活写里有 **7 处此前无任何守卫**，全部落在
`class_prior_estimation`：**漏写**会静默落回 dataclass 默认值（`family` → `CLASSIC_CALIBRATION`、
`source_status` → `UNKNOWN`、`scenario` / `assumption` → `[UNKNOWN]`），**写成另一个合法值**则会
静默取那个错值——后者更隐蔽，连默认值的痕迹都没有。
**另 3 处已各有守卫**（本批新增的钉子对它们构成冗余，不是新增保护）：
`class_prior_estimation.implementation_status` 由既有的
`test_basic_implementation_status_distribution`（断言每个注册方法必须是 NATIVE）泛覆盖；
`pusb` / `lbe` 的 `requires_class_prior` 由 `tests/contract/test_ledger_registry_consistency.py`
的 `test_prior_semantics_consistent_with_registry_class_prior` 守住（台账的 `prior_semantics`
与注册表该字段必须一致，改任一侧即变红）。故准确口径是
「`class_prior_estimation` 的 8 处活写里 **7 处**此前无守卫」——既不是「8 个字段此前无任何守卫」，
也不是「10 处里只有 1 处有守卫」。

**收敛动作（本批）。** 删除 **182** 处死写（= 23 条目 × 8 字段 − 2），保留 **10** 处活写：
`class_prior_estimation` 的全部 8 个，与 `pusb` / `lbe` 的 `requires_class_prior`（这两个方法的该
字段靠基类默认，而基类被同步判据刻意排除，故字面量是唯一源）。删除后
`registry/builtin_methods.py` 不再重复声明这 8 个字段，要看某法的取值请去其类属性。

**守卫现状（已从「事后发现一致」改为「禁止重复声明」）。** 事故守卫原地重写为**双向契约**
（`tests/test_builtin_methods.py` 的 `TestBuiltinRegistration.test_static_entries_do_not_redeclare_class_fields`）：

- clause 1：类已声明的字段**不得**在条目字面量里重复出现（死写回潮即变红）；
- clause 2：类未声明的字段**必须**在字面量里显式声明；`class_prior_estimation` 的 8 个值被逐个钉住
  （钉具体值，不是钉「等于自己」）。

守卫解析源码 AST 而非读 metadata 对象——删掉的 kwarg 仍会以 dataclass 默认值响应 `getattr`，
属性访问分不出「省略」与「声明」。需如实说明：`class_prior_estimation.implementation_status` 的这
条钉子与上文那条泛覆盖测试**构成冗余**，不是新增保护。

**已知边界（本批如实登记，未解决）。** clause 1 无条件禁止**任何**条目（含未来的 `api_only`）在
字面量里写那 5 个仅类/默认值字段。对已绑定条目这是正确的（类才是源），但 `api_only` 条目没有可
绑定的类，于是只能取 dataclass 默认值——它的架构能力与 `label_semantics` **没有**声明途径
（`AlgorithmMetadata` 本身接受这些 kwarg，是守卫的 clause 1 拒绝，不是无法表达）。
今日 24 个条目全部已绑定、`api_only` 为 **0**，故这条规则空转；一旦新增 `api_only` 条目，该限制与
「字面量是唯一源」的原则相冲突，届时需要重新裁定。

**行号锚。** 本批删除 182 行后 `registry/builtin_methods.py` 全文件行号位移，本节引它的行号按
`f5cf55d` 重取：`register_all_builtin_methods` 的 docstring 现 `:366`（原 `:534`）。取号依据：
`grep -n 'implementation_status=' pu_toolbox/registry/builtin_methods.py` 命中 `:79` 与 `:366`
两处，前者是 `class_prior_estimation` 的活写字面量，后者即该 docstring。
`registry/registry.py`（`:141`、`:124`-`:138`）本批未动，行号不受影响。

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
