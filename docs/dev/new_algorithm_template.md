# 新算法接入模板

## 1. 必做声明

新算法必须：

1. 实现 API 契约（fit/predict/decision_function/get_params/set_params）；显式声明
   `label_semantics`（`"pu"`、`"pnu"` 或监督目标的 `"pn"`），说明主 `fit(X, y)` 的
   标签**含义**，不能仅靠 `{0, 1}` 取值校验区分 PU 与 PN；
2. 在类属性块声明 4 个架构能力字段（`BasePUClassifier` 有 tabular 默认值，
   深度算法必须显式声明）：

   | 字段 | 合法值 | 说明 |
   |---|---|---|
   | native_architectures | ⊆ {"mlp","cnn"} | 原生架构路径；∅ = tabular_only（派生） |
   | input_ndims | ⊆ {2,4}，非空 | 支持输入维度 |
   | encoder_parameter | None 或构造函数参数名 | 接收注入 encoder 的构造函数参数名（声明性元数据；Pipeline 依构造函数签名经该参数注入 encoder，不以本字段驱动注入） |
   | trains_encoder | bool | 是否端到端训练注入的 encoder |

3. 在 registry/builtin_methods.py 注册 AlgorithmMetadata（含
   implementation_status=NATIVE 仅当有真实训练逻辑；未实现必须 API_ONLY）；
4. 声明 sample_weight_support / backend / requires_class_prior 等既有字段。

## 2. 自动门禁（无需手写）

- 契约测试 tests/contract/test_capability_declarations.py：声明合法性、
  注册表同步、tabular_only 派生、签名一致性，且注册分类器必须在类自身显式声明
  `label_semantics`——新算法漏声明直接失败；
- tests/contract/test_classifier_baseline.py：API 契约 + 基线行为；
- check_doc_links Rule 1：文档引用路径必须真实存在。

## 3. 若声明支持 CNN（native_architectures 含 "cnn"）

必须提供：

- CNN smoke training 测试；
- 输入/输出形状测试（encoder 输出经 validate_encoder_features 校验）；
- CV fold 隔离测试（fold 间权重不泄漏）；
- 固定 seed 测试；
- GPU 执行级测试：挂 `gpu` pytest marker（pyproject 已注册，无 CUDA 自动 skip，
  验收环境 `pytest -m gpu` 真执行；先例 `tests/unit/estimators/test_nnpu_gpu.py`）；
- save/load/predict round-trip 测试；
- 不支持架构的 fail-fast 测试（PUPipeline architecture 校验）。

## 4. 最小示例

深度算法类属性块（假设 MLP+CNN 双架构）：

    native_architectures = frozenset({"mlp", "cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True
    label_semantics = "pu"

fit 内 probe（若 encoder 非 None）：

    representation_dim = validate_encoder_features(
        probe.flatten(start_dim=1), encoder_param_name="encoder"
    )

传统表格算法：可继承架构能力的 tabular 默认，但注册方法仍须显式声明
`label_semantics = "pu"`；三值 PNU 方法声明 `"pnu"`。监督 oracle 声明 `"pn"`。
