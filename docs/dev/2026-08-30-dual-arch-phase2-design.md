# 双架构能力契约（阶段 2）设计规范：nnpu encoder 试点

- 日期：2026-08-30
- 状态：已与用户确认（brainstorming 三决策 + 两批设计呈现均获批）
- 上游：dual_architecture_plan.md §5 阶段 2（"以 nnpu 为首个试点"）
- 前置：阶段 0（能力契约）与阶段 1（双架构整理）已完成；本机无 CUDA，GPU 执行级测试在 `D:\CodexProject\FunctionTest` 验收环境执行（用户已配置 CUDA）

## 1. 背景与范围

上游计划阶段 2「以 `nnpu` 为首个试点」：增加可选 `encoder=None`；`None`
继续使用原有默认网络；传入 encoder 时只替换表征网络，不改变 PU loss
数学目标；并定义 encoder→head 的组合结构及与 `model` 的共存/优先级。

nnpu 现状（`pu_toolbox/estimators/risk/nnpu.py`）：

- `model: torch.nn.Module | None` 为**完整网络**（输出 raw score g(x)），
  默认 `nn.Linear(d, 1)`；fit 内 `copy.deepcopy(self.model)` 后训练；
- 能力声明：`native_architectures = {"mlp"}`、`input_ndims = {2}`、
  `encoder_parameter = None`、`trains_encoder = False`；
- 训练循环、early stopping、optimizer 重建、decision_function 全部围绕
  单一 `self.model_`；
- 传统 PU benchmark 的 nnpu 调优基线（r1，beta 0.3）基于 Linear 默认。

**范围内**：

1. nnpu 新增 `encoder` 参数 + encoder→head 组合结构（fit 内组合）；
2. 能力字段升级为 MLP/CNN 双架构声明（随之自动获得 pipeline/UI/报告
   联动，见 §4）；
3. 回归测试：二维/四维输入、CV fold 隔离、seed、设备、保存加载；
4. CPU/GPU 执行级测试：新增 `gpu` pytest marker 入 tests/，在
   FunctionTest（CUDA）环境执行。

**范围外（明确不做）**：

- `dist_pu`：计划原文"试点稳定后再评估，避免同时改造多个训练循环"；
- 阶段 3（self_pu / dgpu，长期搁置）、阶段 4（adapter_architectures，
  YAGNI）；
- 公共 compose helper 提取：试点阶段组合逻辑留在 nnpu 内，dist_pu
  接入时再提取（届时本阶段测试即安全网，YAGNI）。

## 2. 已确认决策

| 决策点 | 结论 |
|---|---|
| encoder 与 model 共存语义 | **model 复用为 head**：encoder 提供时 model 被解释为 head（接在 encoder 后输出 score），`nn.Sequential(encoder, head)` 组合；encoder=None 时 model 语义不变（完整网络）。同传合法，非互斥 |
| 默认网络 | encoder=None 时保持现状 `nn.Linear(d, 1)`。计划原文"原有 MLP"为笔误（泛指原有默认网络），本文档勘误；表格行为、benchmark 基线、调优结果全部不动 |
| 组合实现方式 | 方案 A：fit 内组合为单一 `model_ = nn.Sequential(encoder_, head)`，训练循环/early stopping/optimizer/decision_function 零改动 |
| 默认 head | model 为 None 时 head = `nn.Linear(representation_dim, 1)`（representation_dim 由 probe 得出） |
| head 维度不匹配 | 不预校验，由 torch 前向自然报 shape error（与现状 model 语义一致，YAGNI） |
| GPU 测试组织 | 新增 `gpu` marker 注册进 pyproject，测试写入仓库 tests/，无 CUDA 自动 skip；FunctionTest 环境装包后 `pytest -m gpu` 真执行 |
| CV 隔离测试 | 参数化扩展现有 `tests/integration/test_cv_fold_isolation.py`（数据协议泛化为 classifier 工厂），nnpu 复用同一断言套件 |
| 既有测试策略 | 新测试放新文件；既有 `test_nnpu.py`、红线文件（test_pipeline_deep.py / test_classifier_baseline.py）零改动 |

## 3. nnpu 类改造设计

### 3.1 构造函数

`__init__` 新增 keyword-only 参数（紧跟 `model` 之后，与 infomax/wconpu
先例同为 keyword-only）：

```python
def __init__(
    self,
    model: torch.nn.Module | None = None,
    *,
    encoder: torch.nn.Module | None = None,
    class_prior: float | None = None,
    ...  # 其余参数逐字不变
)
```

docstring 补充 encoder 语义：替换表征网络；提供时 model 退化为 head；
encoder 与 model 同传合法，组合使用。Module 对象被 sklearn
clone/get_params 深拷贝——与 infomax 既有模式一致（已验证可用）。

### 3.2 fit 变更

仅 `self.encoder is not None` 时走新分支，encoder=None 分支逐字不动：

1. `validate_pu_X_y(..., allow_nd=self.encoder is not None)`——
   四维输入放行（与 infomax 先例一致）；**validation_data 分支的
   `validate_pu_X_y` 同样传入 `allow_nd`**（nnpu.py:301-305 现状无该
   参数，否则 encoder 模式下带 early stopping 的四维 X_val 会被拒绝）；
2. `encoder_ = copy.deepcopy(self.encoder).to(device)`——fit 内深拷贝，
   与 CV fold 隔离语义（"共享构造 + 逐 fit 深拷贝"）一致；
3. probe：`encoder_.eval()` + `X[:1]` 前向 →
   `validate_encoder_features(probe.flatten(start_dim=1),
   encoder_param_name="encoder")` 得 `representation_dim`
   （BatchNorm 需 eval 模式，同 infomax 先例）；
4. head 确定：`model` 非 None → `copy.deepcopy(self.model)` 复用为
   head；否则 `nn.Linear(representation_dim, 1)`；head 与 encoder 同
   device；
5. `self.model_ = nn.Sequential(encoder_, head)`——此后训练循环、
   early stopping（validation_data 路径）、optimizer 重建
   （`type(self.optimizer)(self.model_.parameters(), **defaults)` 自动
   含 encoder 参数）、`_decision_function`、`evaluate_pu_risk` 全部
   复用现状代码，零改动。

PU loss 数学目标不变：`_nnpu_train_step` 的输入仍是
`self.model_` 输出的 scores，nnPU 风险估计器（Kiryo et al. 2017）语义
不受影响——只替换了表征网络。

### 3.3 能力字段升级

```python
native_architectures = frozenset({"mlp", "cnn"})
input_ndims = frozenset({2, 4})
encoder_parameter = "encoder"
trains_encoder = True
```

## 4. 平台联动（声明驱动，零额外代码）

- **Pipeline**：`check_architecture_capability` + `fresh_estimator`
  自动注入共享 encoder → CV fold 隔离语义自动继承（fit 内 deepcopy
  保证）；
- **UI**：`cnn_candidates()` 自动包含 nnpu（阶段 1 纯函数从 registry
  元数据推导）；骨架 selectbox 已有；
- **`list-methods`**：能力列自动更新；
- **报告 provenance**：CNN 运行时自动报 `native_cnn` + backbone +
  encoder 摘要（`in_channels` 为图像通道数）；MLP 运行时
  `native_mlp` + encoder None；
- **注册表**：能力字段类属性权威 + `_SYNC_FIELDS` 自动镜像，
  `builtin_methods.py` 无需改动（已确认条目无显式能力字段）。

**声明变化的涟漪（既有测试必然更新）**：

| 文件 | 变更 |
|---|---|
| `tests/unit/ui/test_cnn_candidates.py` | 锁定集合 `{"infomax_pu", "weighted_contrastive_pu"}` → 增加 `"nnpu"` |
| `tests/contract/test_capability_declarations.py` | `_EXPECTED_DECLARATIONS` 的 nnpu 条目 → `({"mlp","cnn"}, {2,4}, "encoder", True)` |
| `tests/unit/cli/test_info.py:87` | list-methods Input/Arch 列精确断言同步为 `"2,4"` / `"mlp,cnn"`（Task 1 实施发现，与 infomax 同行风格一致） |

## 5. 测试设计

| 文件 | 内容 |
|---|---|
| `tests/unit/estimators/test_nnpu_encoder.py`（新建） | ① encoder + 二维输入：训练生效、`model_` 为 Sequential、encoder 权重在变；② encoder + 四维输入：训练 + predict round-trip；③ encoder + 自定义 head 同传：组合正确、head 参数参与训练；④ encoder 输出非法（非 2D）→ ValueError；⑤ 无 encoder + 四维输入 → 拒绝（fail-fast）；⑥ encoder=None 默认路径 = `nn.Linear(d,1)` 回归；⑦ 同 seed 两次 fit 权重一致（seed 确定性） |
| `tests/integration/test_cv_fold_isolation.py`（参数化扩展） | 现有数据协议（`[1×4, 0×8, 1×4, 0×8]` 交错布局）泛化为 classifier 工厂参数，nnpu 复用断言套件（防假阳性 / 起点无污染 / 折间隔离 / 模板不被训练） |
| `tests/integration/test_nnpu_pipeline_cnn.py`（新建） | `PUPipeline(architecture="cnn", classifier="nnpu")` 端到端：fit_evaluate 成功、provenance = `native_cnn` + backbone + encoder 摘要；`architecture="mlp"` 对照报 `native_mlp` + encoder None；save/load round-trip（pickle dump/load 后 predict 一致） |
| `tests/unit/estimators/test_nnpu_gpu.py`（新建，gpu marker） | CUDA 可用时：nnpu + cnn13 真实训练 + predict + 参数设备一致性；`torch.cuda.is_available()` 不满足自动 skip。本机 skip，FunctionTest 环境 `pytest -m gpu` 真执行 |
| `pyproject.toml` | markers 列表注册 `"gpu: tests that require a CUDA GPU (auto-skip without one)"` |

## 6. 兼容性影响清单

| 项 | 影响 |
|---|---|
| encoder=None 默认路径 | 逐字不动：现有 `tests/unit/estimators/test_nnpu.py`、`check_baseline_configs` 门禁锁定的 nnpu 基线配置（beta=0.3 等）、七轮调优结果、benchmark 基线全部有效 |
| 红线文件 | `tests/integration/test_pipeline_deep.py`、`tests/contract/test_classifier_baseline.py` 零改动 |
| 既有 test_nnpu.py | 零改动（新 encoder 分支全在新文件） |
| sklearn clone / get_params | encoder 为 Module 被 deepcopy——与 infomax 既有模式一致 |
| UI / CLI / run_config | 声明驱动自动生效；encoder 由 pipeline 管理（`_MANAGED_CLASSIFIER_PARAMS` 已含），用户侧无感知 |

## 7. 验收标准

1. 新增测试全绿（单元 + CV 隔离 + pipeline 端到端）；
2. 快速测试零回归，两个红线文件全程未改；
3. 7 项质量门禁全过（pyproject gpu marker 注册后 format 等门禁同步）；
4. 报告目检：nnpu + `architecture="cnn"` → provenance `native_cnn` +
   encoder 摘要；`architecture="mlp"` → `native_mlp` + encoder None；
5. `list-methods` 中 nnpu 行显示 cnn 能力；
6. GPU 测试在 FunctionTest（CUDA）环境实际执行通过（用户环境验收）；
7. 设计文档蒸馏进 `dual_architecture_plan.md` §5 阶段 2「实施结果」，
   `process_checklist.md` 登记未发布条目。

实施结果摘要见 dual_architecture_plan.md §5 阶段 2。
