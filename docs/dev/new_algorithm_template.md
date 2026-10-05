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

3. 在 registry/builtin_methods.py 注册 AlgorithmMetadata（`implementation_status` 取
   NATIVE 仅当有真实训练逻辑；未实现必须 API_ONLY；写在哪一栏见第 5 条）；
4. 声明只在注册表字面量里写的那些字段：`aliases` / `deprecated_aliases` / `paper` /
   `supports_sparse` / `supports_gpu` / `upstream_url` / `license` / `training_cost`；
   若方法依赖额外干净真值 support set，置 `requires_clean_support=True`，并使缺失支持集时的
   `fit` 直接报错。
5. 那 8 个条目字段（`family` / `assumption` / `scenario` / `requires_class_prior` /
   `implementation_status` / `source_status` / `backend` / `maturity`）写在哪一栏，**逐字段**由
   「你的类是否声明该字段」决定，与条目是否绑定无关：类属性块声明了的，注册时同步进 registry，
   `AlgorithmMetadata(...)` 里就**不得**再写（写了也会被覆盖）；**类没声明的**必须写在字面量里，
   否则只能落到 dataclass 默认值。所以「已绑定」不等于「一律不写」：`class_prior_estimation`
   是已绑定条目，但它的类一个字段都不声明，8 个全写在字面量；`pusb` / `lbe` 的
   `requires_class_prior` 同理（基类默认值不算「类声明」）。未实现（`api_only`）的方法没有可绑定的
   类，这 8 个字段全写在字面量里。两种情况都由 `tests/test_builtin_methods.py` 的
   `test_static_entries_do_not_redeclare_class_fields` 守住。**已知边界**：那 5 个仅类字段
   （4 个架构能力字段与 `label_semantics`）不允许出现在任何条目字面量里，`api_only` 方法因此只能
   取 dataclass 默认值——见 `docs/dev/single_source_map.md` 的登记。

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
