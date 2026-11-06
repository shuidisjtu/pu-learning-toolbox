# 表征、encoder 与训练路径

## 1. 方法与输入表征共同决定实验含义

同一个 PU 方法在二维特征、可训练图像 encoder 和冻结特征上运行，学习任务和计算预算不同。
比较结果时需要同时记录输入形状、encoder / head、初始化或预训练来源及训练范围。

| 当前训练路径 | 输入与训练过程 | 应记录的信息 |
|---|---|---|
| `native_2d` | 方法直接训练二维特征 | 特征来源、训练侧预处理、输入维数 |
| `native_cnn` | 方法端到端训练图像网络 | backbone、head、初始化、归一化、更新范围 |
| `cnn_feature_adapter` | 冻结 encoder 提取二维特征，再训练 PU 方法 | encoder 状态、预训练/拟合范围、特征与 split 身份 |

路径词表和比较门禁在
[feature_adapter.py](../../../pu_toolbox/experiment/feature_adapter.py)。
冻结特征与端到端 CNN 分别分组；组内核对 backbone、训练范围和计算预算。

## 2. 当前能力声明

估计器与 registry 已使用四个字段描述能力：

| 字段 | 含义 |
|---|---|
| `native_architectures` | 方法原生支持的 `mlp` / `cnn` 架构 |
| `input_ndims` | 实际接受的数组维数，例如二维特征或四维 NCHW |
| `encoder_parameter` | 可注入 encoder 的构造参数名；没有则为 `None` |
| `trains_encoder` | 方法是否训练该 encoder |

`input_ndims` 记录接受的数组维数，`native_architectures` 记录原生架构；
实际消费点决定图像经 encoder 处理还是展平。
字段权威定义见 [core/base.py](../../../pu_toolbox/core/base.py)，
架构进度见[双架构方案](../../dev/dual_architecture_plan.md)。

## 3. 共享 encoder 与方法生命周期

当前 `pu_toolbox.estimators.deep.build_encoder` 支持 `architecture="mlp"`（返回 `None`，
使用方法内置路径）及 `architecture="cnn"`。CNN 可用 backbone 为
`cnn13`、`cnn13_no_bn`、`resnet18`、`resnet50`；输入为 NCHW，
默认归一化均值/标准差为每通道 0.5。ResNet 当前使用随机初始化 `weights=None`。

nnPU 已提供 `encoder` 注入入口：每次 `fit` 深复制到 `encoder_` 后训练，
从而隔离候选/折间状态。其他方法的生成器、多网络组件和 head 生命周期见各估计器实现。
实现见
[nnpu.py](../../../pu_toolbox/estimators/risk/nnpu.py) 和
[vision.py](../../../pu_toolbox/estimators/deep/vision.py)。

架构能力可用下面的小片段核对，无需训练或下载权重：

```python
from pu_toolbox.registry import get_algorithm

model_class = get_algorithm("nnpu")
assert "cnn" in model_class.native_architectures
assert 4 in model_class.input_ndims
assert model_class.encoder_parameter == "encoder"
assert model_class.trains_encoder
```

精确签名见 [API 参考](../reference/api.md)，注入、共享骨干替换和保存加载见
[encoder 教程](../howto/custom_encoders.md)与[示例 18](../../../examples/minimal/18_custom_encoders.py)。

## 4. 冻结特征与数据隔离

`adapt_image_bundle_to_features` 对四角色数据使用同一 encoder，以 eval / 无梯度方式抽特征，
前后检查 encoder 状态哈希，保留原始索引与标签视图，并返回表示身份元数据。
它不训练 encoder。

`encoder_fit_scope="fixed_external"` 表示调用者声明 encoder 来自固定外部来源；
`"train_partition_only"` 要求提供恰好等于 train 的拟合索引。
`train_partition_only` 记录调用者声明的拟合范围，索引校验确认该范围与 train 一致。
预训练数据、权重版本与监督使用范围写入 `backbone_manifest`。

归一化、降维和可训练 encoder 都只能在允许的训练角色拟合。
比较时保留 split、encoder 状态和表示哈希，
并将 PN oracle 纳入同样的表征和训练预算条件。
研究组合入口见[自定义实验](../howto/custom_experiments.md)。

## 5. 来源

Chen et al. (2026), Appendix D.1（数据集内表征与骨干约定）：
[PU-Bench](https://openreview.net/forum?id=tb8DabMbMq)。
参考工程实现见
[固定版本 model_factory.py](https://github.com/XiXiphus/PU-Bench/blob/2d95a19eefd72e66ff30128ec2f65e1d1d4cc077/train/utils/model_factory.py)。

论文的统一比较条件与工具箱具体 encoder 生命周期是不同证据；后者以上述真实代码为准。
官方实验选用哪些 backbone，见 [Survey 协议](../../research/pu_survey/pu_survey_protocol.md)。
