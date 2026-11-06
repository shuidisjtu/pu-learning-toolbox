# 用 Python API 定制四角色 PU 实验

前置条件：完成[快速开始](../quickstart.md)，准备带真实二元标签的数据，用于构造可控的 PU 实验。
本教程使用 CPU 合成数据，介绍 `ExperimentRunner` 的直接调用和策略扩展。
官方 Survey 脚本的运行流程见[运行指南](../../research/pu_survey/runbook.md)；
接口签名见 [API 参考](../reference/api.md#四角色运行与策略注入)。

## 1. 先运行完整示例

在仓库根目录执行：

```powershell
uv sync --python 3.11
uv run python examples/minimal/14_diy_experiment.py
```

[示例源码](../../../examples/minimal/14_diy_experiment.py)运行两组实验，每组训练三个 uPU 候选：

| 实验 | 标签生成 | 训练 | 选模 |
|---|---|---|---|
| `basic` | 内置 SCAR | 内置 `FitTrainer` | 独立 PA 与 OA |
| `custom` | 带自定义策略标识的 SCAR | 仅在 train 拟合标准化器 | 固定阈值的 PU 选模与独立 OA |

终端显示各协议选中的候选编号、阈值和测试 Accuracy。添加 `--seed 11` 可更换随机种子。
保存运行 manifest 时，指定一个尚不存在的输出目录：

```powershell
uv run python examples/minimal/14_diy_experiment.py --output-dir tmp/diy-experiment
```

输出 `basic.json` 和 `custom.json`，包含生成记录、候选参数、选择结果、测试指标和成本。
示例直接使用通用实验接口，配置中省略 `survey_protocol` 和 `recipe_registry`。

## 2. 准备四路数据

四份 `DatasetPart` 的输入标签都使用真实二元标签：`1` 为正类，`0` 为负类，
并声明 `view="clean"`。runner 为 train 和 pu_val 分别生成 PU 视图，
其中 `1` 为已标记正例，`0` 为未标记样本。

| 角色 | 运行时用途 |
|---|---|
| `train` | 生成 PU 标签并训练候选；预处理统计在这里拟合 |
| `pu_val` | 生成 PU 标签，供 PA 或自定义 PU 策略选模 |
| `clean_val` | 真实标签验证集，供 OA 选模 |
| `test` | 候选和阈值选定后，计算最终指标 |

`indices` 使用同一原始数据源中的全局样本身份，四路内部和相互之间均须唯一。
切分后的子集继续保留原始索引。`test` 设置 `for_selection=False`。

下面的片段可独立运行，按合成总体先验 0.5 构造四路输入并完成一次实验：

```python
import numpy as np

from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.experiment import DatasetPart, ExperimentRunner, FitTrainer

rng = np.random.RandomState(7)
y_true = np.tile([0, 1], 300)
X = rng.normal(size=(600, 5)) + y_true[:, None]
order = rng.permutation(len(y_true))
parts = []
for start, stop in ((0, 300), (300, 400), (400, 500), (500, 600)):
    ids = order[start:stop]
    parts.append(DatasetPart(
        X=X[ids], labels=y_true[ids], view="clean", indices=ids,
        for_selection=start != 500,
    ))

runner = ExperimentRunner(
    seed=7,
    class_prior=0.5,
    config={"c": 0.4, "trainer": FitTrainer()},
)
result = runner.fit(
    UPUClassifier(class_prior=0.5, fit_intercept=False),
    *parts,  # train, pu_val, clean_val, test
)
print(result.test_metrics)
```

这里的输入是带完整标签的 benchmark 数据。已有真实 P/U 数据可直接使用估计器或
[`PUPipeline`](pipeline.md)；四角色 runner 的输入契约要求上述完整标签视图。

## 3. 修改参数、候选和运行输出

候选池中的每个字典通过估计器的 `set_params` 应用。runner 克隆模型后逐候选训练，
随后让 PA 和 OA 各自选择候选与阈值。

```python
runner = ExperimentRunner(
    seed=7,
    class_prior=0.5,
    threshold_candidates=np.linspace(0, 1, 21),
    manifest_path="tmp/my-experiment.json",
    config={
        "c": 0.4,
        "trainer": FitTrainer(),
        "candidates": [{"reg_lambda": value} for value in (0.001, 0.01, 0.1)],
        "split_ref": {"dataset": "my_dataset", "split_seed": 7},
    },
)
```

常用配置与使用场景如下：

| 参数位置 | 用途 |
|---|---|
| `seed` | 标签生成与未显式设置 `random_state` 的候选模型种子 |
| `generator` | 标签生成策略实例；默认 `SCARGenerator()` |
| `protocols` | 选模策略实例列表；默认 `[ProtocolPA(), ProtocolOA()]` |
| `class_prior` | 总体正类先验；PA 必需，来源为生成设置或独立先验估计 |
| `threshold_candidates` | 内置 PA/OA 的验证侧归一化阈值网格，默认 0–1 上 11 点 |
| `manifest_path` | 保存运行 manifest；省略时从 `result.manifest` 读取 |
| `config["c"]` | 正类标记率，默认 0.1 |
| `config["candidates"]` | 候选参数字典列表，默认 `[{}]` |
| `config["trainer"]` | 训练器实例；经典单点用 `FitTrainer()`，默认是 `DeepFitTrainer()` |
| `config["split_ref"]` | 数据来源与划分说明；应记录实际数据版本和索引身份 |
| `config["os_or_ts"]` | 请求估计器支持的训练风险视图；省略时沿用估计器默认行为 |
| `config["architecture"]` | 显式声明 `"mlp"` 或 `"cnn"`，用于能力校验 |

新模型需支持 sklearn 的克隆与参数接口，声明 `label_semantics="pu"` 和实际输入能力，
并由 `decision_function(X)` 返回逐样本有限分数。完整数据、模型和轨迹契约见
[API 参考](../reference/api.md#实验层experiment)。
自定义 selector 返回 `threshold=None` 时，模型须提供返回 0/1 的 `predict(X)`。

## 4. 扩展标签生成器

`Generator` 负责生成标签视图，特征数组由调用方准备。
继承该接口，实现 `generate(X, y_true, c, seed)`，返回标签数组与元信息字典。
示例复用 SCAR 抽样，增加应用自定义的标记策略版本：

```python
from pu_toolbox.experiment import SCARGenerator
from pu_toolbox.experiment.protocols import Generator

class AuditedSCARGenerator(Generator):
    output_view = "pu"

    def generate(self, X, y_true, c, seed=None):
        labels, metadata = SCARGenerator().generate(X, y_true, c, seed)
        return labels, {**metadata, "policy_id": "diy-scar-v1"}
```

通过 `generator=AuditedSCARGenerator()` 注入。生成器分别接收 train 和 pu_val，
元信息写入 `result.manifest["generation"]`。替换采样逻辑时，应记录实际标记率、
机制、种子和标签身份，并保证 PU 视图中的已标记行来自真实正类。
PA 的 pu_val 需要同时包含已标记正例与 U。改用 SAR 时，根据标记机制选择相应评价准则；
官方 SAR 实验使用 OA。

## 5. 扩展训练器

继承 `Trainer`，实现 `fit` 并返回 `RunTrajectory`。该方法接收训练 PU 标签和可选的
`val_pu=(X_val, y_pu_val)`。示例在每个候选的 train 上拟合标准化器，然后把标准化器
与已拟合分类器一起放入预测 Pipeline：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pu_toolbox.experiment import FitTrainer
from pu_toolbox.experiment.protocols import Trainer

class ScaledFitTrainer(Trainer):
    trains_on_real_labels = False

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        scaler = StandardScaler()
        trajectory = FitTrainer().fit(
            estimator, scaler.fit_transform(X), y,
            class_prior=class_prior, os_or_ts=os_or_ts,
        )
        trajectory.model = Pipeline([("scale", scaler), ("classifier", trajectory.model)])
        return trajectory
```

通过 `config={"trainer": ScaledFitTrainer(), ...}` 注入**实例**。
验证与测试的 `decision_function` 自动复用这个已拟合的标准化器。
自写训练循环时，经典单点可返回：

```python
from pu_toolbox.experiment.tracking import EpochRecord, RunTrajectory

trajectory = RunTrajectory(
    epochs=[EpochRecord(epoch=1, metrics={})],
    model=fitted_model,
)
```

深度模型可使用内置 `DeepFitTrainer` 及逐 epoch checkpoint；配置入口包括
`capture_epoch_checkpoints`、`checkpoint_dir`、`checkpoint_bytes_per_component` 和
`reclaim_unselected_checkpoints`。自定义训练器负责产生自己的逐 epoch 权重与轨迹；
`SelectionArtifact.epoch` 使用轨迹中从 1 开始的位置，`checkpoint_index` 指向所选权重。
权重恢复与磁盘预算契约见 [API 参考](../reference/api.md#实验层experiment)。

## 6. 扩展选模策略

继承 `SelectionProtocol`，实现 `select`，返回 `SelectionArtifact`。
示例在验证侧归一化阈值 0.5 上，使用内置 PU proxy accuracy 选择候选：

```python
from dataclasses import replace

from pu_toolbox.experiment import ProtocolOA, ProtocolPA
from pu_toolbox.experiment.protocols import SelectionProtocol

class FixedThresholdPA(SelectionProtocol):
    name = "PA-fixed"

    def select(self, trajectories, val_part, threshold_candidates=None, *, class_prior=None):
        artifact = ProtocolPA().select(
            trajectories, val_part, np.array([0.5]), class_prior=class_prior,
        )
        return replace(artifact, protocol=self.name)
```

通过 `protocols=[FixedThresholdPA(), ProtocolOA()]` 注入。完整示例同时注入这三个自定义策略。
`name` 与返回的 `artifact.protocol` 保持一致，每个策略使用唯一名称。

普通 `SelectionProtocol` 子类接收生成后的 pu_val。需要干净验证标签的自定义 OA 策略应继承
`ProtocolOA`，runner 据此路由 clean_val；修改协议名称时同步更新返回的 artifact。
自行定义阈值与分数变换时，把验证侧的 `val_score_min` 和 `val_score_scale` 保存在
`artifact.metrics`，runner 在最终评测时复用它们。

## 7. 读取结果并继续预测

```python
print(result.selections)      # Selected run, epoch, threshold and validation metrics
print(result.test_metrics)    # Independent test metrics
print(result.failures)        # Candidate failures and retries
print(result.resources)       # Training and tuning costs
print(result.manifest["selection"]["OA"]["candidate_index"])
```

`SelectionArtifact.run_index` 对应成功训练的轨迹列表；原始候选编号从 manifest 的
`selection[name]["candidate_index"]` 读取。

继续使用经典模型预测时，同时取出 `result.selected_models[name]` 和
`result.selections[name]`，复用验证侧分数变换与所选阈值。
完整示例的 `predict_selected(result, name, X_new)` 封装了这一过程。
这使后续预测与报告中的分类规则一致。

## 8. 下一步

- 使用自己的数据替换四份 `DatasetPart`，保留全局索引与 train-only 预处理。
- 从候选参数字典开始调整，再按需要替换生成器、训练器或选择策略。
- 查阅 [API 参考](../reference/api.md#四角色运行与策略注入)了解参数契约，
  查阅[实验层设计](../../dev/experiment_layer.md)了解接口的设计依据。
- 按官方 Survey 配方执行及生成工作簿时，使用[运行指南](../../research/pu_survey/runbook.md)。
