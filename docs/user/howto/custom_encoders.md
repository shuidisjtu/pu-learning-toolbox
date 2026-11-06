# 注入 encoder 与使用冻结特征

前置条件：安装 `uv sync --extra torch`，了解[表征与训练路径](../concepts/representations.md)
和[先验目标](../concepts/class_prior_semantics.md)。本教程用 CPU 合成图像，分别运行
端到端 nnPU 与冻结 encoder 后的传统 uPU。

## 1. 运行完整示例

```powershell
uv run python examples/minimal/18_custom_encoders.py
uv run python examples/minimal/18_custom_encoders.py --output-dir tmp/encoder-demo
```

输出目录使用新路径，包含 `settings.json`、`native.pkl` 和 `frozen.pkl`。
[示例 18](../../../examples/minimal/18_custom_encoders.py)的四角色规模为 32/12/12/12，
输入为 `(N,1,8,8)`、取值在 `[0,1]` 的图像；总体先验为 0.5。
train 按固定数 SCAR 标记一半正例，已标记比例为 0.25，原生 U 先验换算为 1/3。
两条路径均按固定预算训练并使用模型默认阈值，测试标签用于最终评测。

| 路径 | encoder 使用方式 | 训练参数与记录 |
|---|---|---|
| `native_cnn` | nnPU 每次 fit 深复制模板，训练副本与分类头 | 原生 U 先验、初始化/拟合权重哈希、训练轮数与模型保存加载检查 |
| `cnn_feature_adapter` | 同一初始随机模板以 eval/no-grad 抽取四路特征 | 原生 U 先验、外部固定来源、split/representation/feature 哈希 |

示例的随机模板记录 seed 与 `pretraining="none"`；两条路径按各自训练设置解释指标。

## 2. 构造模板并训练 nnPU

下面片段可独立运行。先固定初始化 seed，再创建模板；`random_state` 控制拟合过程。
encoder 输出有限二维特征，nnPU 推断特征维度后接单输出线性头。

```python
import numpy as np
import torch
from torch import nn

from pu_toolbox.estimators.risk import NonNegativePUClassifier
from pu_toolbox.preprocessing import make_scar_labels
from pu_toolbox.prior import population_to_native_unlabeled

torch.manual_seed(7)
template = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(16, 4),
)
rng = np.random.RandomState(7)
y_true = np.tile([0, 1], 16)
X = rng.uniform(size=(32, 1, 8, 8)).astype(np.float32)
y_pu = make_scar_labels(y_true, c=0.5, random_state=7)
pi_u = population_to_native_unlabeled(0.5, y_pu.mean(), input_sampling="os")
model = NonNegativePUClassifier(
    encoder=template, class_prior=pi_u, batch_size=8,
    max_epochs=2, random_state=7, device="cpu",
).fit(X, y_pu)
assert model.encoder_ is not template
assert model.decision_function(X).shape == (32,)
```

直接调用 nnPU 的默认 `fit` 时，`class_prior` 指向原生 U；显式换算使用本训练分区的
已标记比例。`PUPipeline` 则接收总体先验，按分类器声明和每折计数换算，详见
[Pipeline 指南](pipeline.md)。

模板与 `model.encoder_` 分别保留初始化和拟合状态。多候选或多折可以复用初始化模板，
每次 fit 获得独立副本。自定义 `model=nn.Linear(4,1)` 可替换分数头。

## 3. 替换共享骨干

通过同一个 `encoder` 参数换入库内骨干：

```python
from pu_toolbox.estimators.deep import build_encoder

torch.manual_seed(7)
shared = build_encoder("cnn", backbone="cnn13_no_bn", in_channels=1)
shared.eval()
with torch.no_grad():
    assert shared(torch.zeros(2, 1, 32, 32)).shape == (2, 512)
```

`build_encoder` 内嵌通道归一化，默认均值/标准差为 0.5。微型模板直接使用输入数值；
自行添加归一化时，将参数写入表征设置。
完整示例可选运行共享骨干，两种命令使用 `(N,1,32,32)`：

```powershell
uv run python examples/minimal/18_custom_encoders.py --backbone cnn13_no_bn --max-epochs 1
uv run python examples/minimal/18_custom_encoders.py --backbone resnet18 --max-epochs 1
```

ResNet 依赖 torchvision，使用随机初始化 `weights=None`。精确骨干参数见
[build_encoder 契约](../reference/api.md#build_encoder)。

## 4. 冻结 encoder 后训练传统方法

四路 clean 数据先组成 `DatasetBundle`，使用同一 encoder 提取特征。
下面片段承接前文的 `rng` 与 `template`，构造四路数据后生成 train 的 PU 标签：

```python
from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.experiment import DatasetBundle, DatasetPart, adapt_image_bundle_to_features

parts = []
offset = 0
for role, size in (("train", 32), ("pu_val", 12), ("clean_val", 12), ("test", 12)):
    parts.append(DatasetPart(
        X=rng.uniform(size=(size, 1, 8, 8)).astype(np.float32),
        labels=np.arange(size) % 2, view="clean",
        indices=np.arange(offset, offset + size), for_selection=role != "test",
    ))
    offset += size
bundle = DatasetBundle(*parts)
y_pu = make_scar_labels(bundle.train.labels, c=0.5, random_state=7)
pi_u = population_to_native_unlabeled(0.5, y_pu.mean(), input_sampling="os")

features, representation = adapt_image_bundle_to_features(
    bundle, template, feature_version="my-fixed-encoder-v1",
    backbone_manifest={"name": "tiny", "seed": 7, "pretraining": "none"},
    encoder_fit_scope="fixed_external", batch_size=8, device="cpu",
)
frozen_model = UPUClassifier(
    class_prior=pi_u, fit_intercept=False, random_state=7,
).fit(features.train.X, y_pu)
print(representation["representation_sha256"])
```

adapter 在 eval/no-grad 下抽取，检查权重哈希，随后恢复各子模块模式，保留四角色的索引和标签。
`fixed_external` 声明固定外部模板来源。冻结由 train 拟合的 encoder 时，使用
`encoder_fit_scope="train_partition_only"` 并传入 `encoder_fit_indices=bundle.train.indices`；
同时在 `backbone_manifest` 记录训练数据、监督类型和权重版本。

## 5. 保存加载与继续预测

端到端模型包含拟合后的 encoder 和头；冻结路径保存 encoder 与分类器：

```python
import pickle

restored = pickle.loads(pickle.dumps(model))
np.testing.assert_array_equal(restored.decision_function(X), model.decision_function(X))
```

从自己保存的 `native.pkl` 读取后，直接调用 `decision_function(X_new)`。
`frozen.pkl` 读取结果为 `(encoder, classifier)`，先用保存的 encoder 对同样预处理的
新图像以 eval/no-grad 抽取特征，再调用分类器：

```python
encoder, classifier = pickle.loads(pickle.dumps((template, frozen_model)))
X_new = bundle.test.X
encoder.eval()
with torch.no_grad():
    new_features = encoder(torch.as_tensor(X_new, dtype=torch.float32)).cpu().numpy()
predictions = classifier.predict(new_features)
```

保存后的 CPU 示例在回归中核对两条路径的预测一致性。数据切分、训练预算和初始化来源
随设置文件保存；比较方式见[表征概念](../concepts/representations.md)，
独立选模与自定义训练器见[四角色实验教程](custom_experiments.md)。
