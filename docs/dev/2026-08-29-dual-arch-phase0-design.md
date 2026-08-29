# 双架构能力契约（阶段 0）设计规范

- 日期：2026-08-29
- 状态：已与用户确认（brainstorming 两个部分均获批）
- 上游：[`dual_architecture_plan.md`](dual_architecture_plan.md)（已对照 v1.11.0 代码现状核实修订，commit 65e7ee9）
- 分支：`feature/dual-arch-capability-contract`

## 1. 背景与范围

上游计划分为 5 个阶段。本规范仅覆盖**阶段 0：能力契约和文档，不改变默认行为**。

**范围内**：

1. Registry 增加架构能力字段（4 个）；
2. Pipeline 的 CNN gate 增加元数据并行校验（与签名检查结论一致，漂移时报错）；
3. `list-methods` 展示能力信息（加 2 列）；
4. 统一 encoder 输出校验（新 helper，接入 infomax_pu / wconpu）；
5. 契约测试强制新算法声明能力字段；
6. 文档：`architecture.md` schema 节更新 + 新算法接入模板。

**范围外（明确不做，归属后续阶段）**：

- 阶段 1：`build_encoder` 公共导出、报告 provenance 字段、UI 元数据驱动、CV fold 隔离测试；
- 阶段 2：nnpu `encoder` 参数试点；
- 阶段 3：self_pu / dgpu（长期搁置）；
- 阶段 4：`adapter_architectures` 字段（YAGNI，待真正有适配算法时再加）；
- 逐 fold 独立随机初始化（可选改进，非本阶段）。

## 2. 已确认决策

| 决策点 | 结论 |
|---|---|
| 本轮范围 | 仅阶段 0 |
| 字段集合 | 4 字段：`native_architectures` / `input_ndims` / `encoder_parameter` / `trains_encoder`；`tabular_only` 派生；`adapter_architectures` 留待阶段 4 |
| 消费点 | Pipeline 校验 + `list-methods` 展示（Registry API 查询自动获得新字段） |
| 接入模板 | 文档 + 契约测试强制 |
| 字段权威来源 | 类属性权威 + `_SYNC_FIELDS` 同步进 Registry |

## 3. 数据模型

### 3.1 基类默认值（`core/base.py`，tabular 默认）

```python
native_architectures: frozenset[str] = frozenset()   # 原生架构路径 ⊆ {"mlp", "cnn"}
input_ndims: frozenset[int] = frozenset({2})         # 支持输入维度 ⊆ {2, 4}
encoder_parameter: str | None = None                 # 接收 encoder 的构造参数名
trains_encoder: bool = False                         # 是否端到端训练注入的 encoder
```

用 `frozenset` 而非 `set`：类属性在类间与注册表间共享，必须不可变。

### 3.2 各算法声明

仅深度算法需显式声明；传统算法继承默认值：

| 算法 | native_architectures | input_ndims | encoder_parameter | trains_encoder |
|---|---|---|---|---|
| infomax_pu、weighted_contrastive_pu | frozenset({"mlp","cnn"}) | frozenset({2,4}) | "encoder" | True |
| self_pu | frozenset({"mlp"}) | frozenset({2,4}) | None（`backbone` 是完整模型） | False |
| nnpu、dist_pu、dgpu | frozenset({"mlp"}) | frozenset({2}) | None | False |
| 传统 9 个（elkan_noto/upu/pnu/centroid_pu/kldce/llsvm/pusb/pusb_kernel/lbe） | 继承默认 ∅ | 继承默认 {2} | None | False |

`tabular_only` 不单独声明，派生自 `native_architectures == frozenset()`。

### 3.3 Registry 侧（`registry/metadata.py`）

`AlgorithmMetadata` 增加同名 4 字段，默认值与基类一致，并增加派生 property：

```python
@property
def is_tabular_only(self) -> bool:
    return self.native_architectures == frozenset()
```

`_SYNC_FIELDS`（`registry/registry.py:124-133`）增加这 4 个字段名。现有同步机制**只同步类显式声明的字段**（`_sync_class_metadata_to_registry` 跳过基类继承默认），因此：

- 传统算法：类不声明 → 注册表条目保持 tabular 默认，两者天然一致；
- 深度算法：类显式声明 → 注册表自动镜像。

`frozenset` 引用共享安全（不可变）。同步函数中现有 tuple→list 转换逻辑不影响 frozenset。

## 4. 消费点

### 4.1 Pipeline gate 并行校验（`workflows/pipeline.py:219-236` 区域）

将并行校验提取为 `workflows/_models.py` 中的纯函数（便于单测）：

```python
def check_architecture_capability(cls, architecture: str) -> None:
    """architecture="cnn" 时校验签名检查与能力声明结论一致，不一致抛错。"""
```

`architecture="cnn"` 时的逻辑：

1. 签名结论：`declares_encoder_parameter(cls)`（现有逻辑不变）；
2. 能力结论：`"cnn" in cls.native_architectures and 4 in cls.input_ndims and cls.encoder_parameter is not None`；
3. **两者不一致 → `PipelineError`**（声明与签名漂移，fail-loud）；
4. 能力不支持 CNN 时，错误信息引用能力声明（如 "declares tabular_only: input_ndims={2}")。

`architecture="mlp"` 路径行为完全不变（默认值）。输入侧维度校验复用现有 `validate_pu_X_y(allow_nd=...)` 机制，阶段 0 仅在错误路径上引用能力声明增强提示信息，不新增输入校验逻辑。

### 4.2 `list-methods` 加列（`cli/info.py`）

表格从 5 列扩为 7 列：

```
Name / Family / Prior / Status / Input / Arch / Auto-inst
```

- Input：`2` 或 `2,4`；
- Arch：`mlp` / `mlp,cnn` / `tabular`（`meta.is_tabular_only` 时）。

数据直接从 `meta` 读取（同步后已有），无需逐行 resolve 类。

## 5. 统一 encoder 输出校验

新建 pu_toolbox/estimators/deep/_validation.py（`core/validation.py` 保持 torch-free）：

```python
def validate_encoder_features(features, *, encoder_param_name: str) -> int:
    """校验 encoder 输出：ndim == 2、torch.isfinite 全真、feature_dim >= 1；返回 feature_dim。"""
```

接入点（现有 probe 逻辑委托）：

- `infomax_pu.py:158-165`（InfoMaxPUClassifier.fit 与 InfoMaxPURepresentation.fit 两处 probe）；
- `weighted_contrastive_pu.py:169-173`。

有限性检查与项目输入契约（拒绝 NaN/Inf）对齐，属预期内的行为收紧。

## 6. 契约测试（tests/contract/test_capability_declarations.py，测试先行）

四组不变量，对全部 17 个注册分类器断言：

1. **声明合法性**：
   - `input_ndims` 非空且 ⊆ {2, 4}；
   - `native_architectures` ⊆ {"mlp", "cnn"}；
   - `encoder_parameter` 为 None 或真实存在于 `__init__` 签名；
   - `trains_encoder=True ⇒ encoder_parameter is not None`；
2. **跨机制一致性**：`"cnn" in native_architectures ⇐ encoder_parameter is not None`，且与 `declares_encoder_parameter(cls)` 结论一致；
3. **注册表同步**：`get_metadata(name)` 的 4 字段 == 类属性值（同时保证默认值对齐）；
4. **派生语义**：`is_tabular_only ⇔ native_architectures == frozenset()`。

另加 `check_architecture_capability` 的单元测试：签名/能力漂移场景抛错、一致场景通过。

## 7. 文档变更

- `docs/dev/architecture.md` 注册表 schema 节补充 4 字段 + `is_tabular_only`（`metadata.py` docstring 引用该节，保持引用准确）；
- 新建 docs/dev/new_algorithm_template.md：声明清单 + 测试清单（§9 最低要求 + 契约测试自动门禁说明）+ 最小示例片段；
- 完成时更新 `docs/dev/process_checklist.md`（dev-workflow 步骤 4）。

## 8. 兼容性影响清单

| 现有测试 | 影响 |
|---|---|
| `tests/unit/cli/test_info.py` | 表头/列断言更新（+Input/+Arch 两列） |
| 注册表元数据相关单测 | 若存在字段枚举/数量断言需更新（实现时 grep 确认） |
| `tests/integration/test_pipeline_deep.py` | 不变——签名检查仍生效，现有算法能力一致，无新错误路径 |
| 传统算法全部测试 | 零变化（继承默认） |

## 9. 验收标准

1. 快速测试全绿（预期内更新仅 test_info.py 等）；新增契约测试 4 组不变量全绿；
2. 现有 WConPU/InfoMax MLP/CNN 集成测试**不改一行且通过**（零回归的最强证明）；
3. 7 项质量门禁全过；
4. `list-methods` 7 列输出对齐可读。

## 10. 实施顺序

1. 契约测试先行（红）；
2. 基类字段 + `AlgorithmMetadata` 字段 + `_SYNC_FIELDS`（绿）；
3. 各深度算法类显式声明（绿）；
4. `check_architecture_capability` 纯函数 + pipeline 接入；
5. `list-methods` 加列 + `test_info.py` 更新；
6. `validate_encoder_features` + 两估计器接入；
7. 文档（architecture.md、new_algorithm_template.md、process_checklist.md）；
8. 全量验证：快速测试 + 7 门禁。
