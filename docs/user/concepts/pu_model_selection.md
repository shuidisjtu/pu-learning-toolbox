# PU 选模：PA、OA 与监督角色

## 1. 先规定允许读取的监督

研究实验需要分别选择候选超参数、checkpoint 和阈值，再在 test 上评测。
当前 [ExperimentRunner](../howto/custom_experiments.md)用四个角色表达这些边界：

| 角色 | 可用监督 | 用途 |
|---|---|---|
| train | PU 标签 | 训练候选及拟合预处理 |
| pu_val | PU 标签 | PA 或自定义 PU 选模 |
| clean_val | 完整真实标签 | OA 选模 |
| test | 完整真实标签 | 所选模型的最终评测 |

角色应在原始来源级隔离。PA 读取 `view="pu"` 中的 PU 标签；
四角色 runner 当前接收 clean 输入，再由 Generator 生成 train/pu_val 视图。
已有真实 PU 数据的日常内部 CV 使用 `PUPipeline`；四角色实验使用 `ExperimentRunner`。

## 2. PA 的条件和公式

PA（proxy accuracy，代理准确率）使用 PU 验证标签与总体类先验计算选模分数。
以下采用 Wang et al. 的定义，设已标记正例集合为 $`L`$，原生未标记集合为 $`U`$，
$`\hat Y=\mathbf1\{f(X)\ge\tau\}`$ 为模型在阈值 $`\tau`$ 下的预测类别。

PA 与总体 Accuracy 的期望排序一致需要以下条件：已标记正例代表目标总体的正类分布
（如 SCAR 标记），使用正确的总体先验 $`\pi_{population}`$，并按验证数据的采样设计
选取总体分布的估计池。两种采样设计对应的计算方式如下。

**OS：同一总体样本中标记一部分正例。** 完整验证分区 $`L\cup U`$ 来自目标总体，
第二项在该分区上计算预测负类的比例：

```math
PA_{OS}=2\pi_{population}\widehat P(\hat Y=1\mid S=1)
        +\widehat P_{L\cup U}(\hat Y=0),
```

**TS：独立取得额外正例 P 与总体混合 U。** 原生 $`U`$ 来自目标总体，
第二项在 $`U`$ 上计算预测负类的比例：

```math
PA_{TS}=2\pi_{population}\widehat P_L(\hat Y=1)
        +\widehat P_U(\hat Y=0).
```

两式都用第一项估计正类召回率的加权贡献，用第二项估计目标总体中的预测负类比例。
总体估计池的选择由采样设计决定，相关定义见[PU 输入采样](pu_sampling.md)。
设真正类召回率为 $`a`$、负类上预测正类的比例为 $`b`$，则总体上：

```math
E[PA]=2\pi a+1-\pi a-(1-\pi)b
     =Accuracy+\pi.
```

对于独立于验证数据的固定预测规则和阈值，期望 PA 与总体 Accuracy 相差常数 $`\pi`$，
因此具有相同的期望排序。有限验证集的经验 PA 存在抽样波动；选模后的表现通过独立 test
评测。PA 是 Accuracy+π 的代理分数，可超过 1。先验值改变正例项权重，也可能改变最优阈值。

工具箱的 `proxy_accuracy` 与 `ProtocolPA` 按 **OS 采样设计下的 PA 计算式**选模，入口在
[experiment/strategies.py](../../../pu_toolbox/experiment/strategies.py)。

```python
import numpy as np
from pu_toolbox.experiment.strategies import proxy_accuracy

scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
y_pu = np.array([1, 1, 0, 0, 0])
pa = proxy_accuracy(scores, y_pu, threshold=0.5, class_prior=0.4)
assert abs(pa - 1.2) < 1e-12  # 2 * 0.4 * 1 + 2 / 5
```

工具箱的官方 Survey 在 SCAR 条件下比较 PA 与 OA，在 SAR 条件下使用 OA；
自定义 SAR 准则记录其额外识别假设与监督预算。

## 3. OA 的条件和公式

OA（oracle accuracy）使用 clean_val 的完整真实标签计算验证准确率。
其中 oracle 表示选模阶段可以读取真实验证标签。设
$`D_{\mathrm{val}}=\{(x_i,y_i)\}_{i=1}^{n_{val}}`$，则：

```math
OA(f,\tau)=\frac{1}{n_{val}}\sum_{i=1}^{n_{val}}
\mathbf1\!\left[\mathbf1\{f(x_i)\ge\tau\}=y_i\right].
```

使用条件是 clean_val 具有可靠的正负类真值、代表目标总体，并与训练和最终测试数据
按原始来源隔离。对于独立于验证数据的固定预测规则和阈值，经验 OA 的期望等于
目标总体 Accuracy。

OS 和 TS 研究均可从目标总体独立取得 clean_val，按同一 OA 公式选模。
公式直接比较预测类别与真实类别，输入为模型分数、阈值和真实标签。
SCAR/SAR 描述 PU 数据的标记机制；OA 所需的验证监督为完整真实标签。
TS 中额外 P 与 U 的拼接表具有自己的类先验，clean_val 按目标总体分布抽取。

工具箱的 `ProtocolOA` 按此准则在 clean_val 上选择模型，所选模型的泛化表现由 test 评测。

## 4. PA 与 OA 的独立选择

当前 `ProtocolPA` 与 `ProtocolOA` 分别从同一训练候选轨迹选择超参数、checkpoint 和阈值，
并分别计算 test 指标。PA 和 OA 分别保留候选、checkpoint、阈值及 test 结果。

内置协议对各自验证角色上的非恒定 score 做 min-max 归一化，再扫描阈值网格；
恒定 score 保留原值，选择记录中的 affine 参数为 `None`。
阈值并列时保留网格中首先出现的值，默认升序网格下即较小阈值。
精确契约见 [API 参考](../reference/api.md)。
独立选择的可执行入口见[示例 14](../../../examples/minimal/14_diy_experiment.py)。

如果训练算法还需要独立 clean support，应记录其来源、规模及使用阶段；
`PA+clean-support` 与纯 PU 的 PA 分开解释。OA 的 clean_val 标签、训练期 support
和最终 test 标签分别记录监督预算与使用阶段。

## 5. 形成可比较的实验

- 同一 seed 的方法复用底层 split、表征和同一标记配置；扫描标记率时明确标签复用规则。
- 预处理统计仅在 train 拟合；记录预训练模型的来源和拟合范围。
- 记录候选数、epoch 预算、checkpoint 及阈值网格，共享训练成本只计一次。
- PN oracle 与 PU 对照共享 split、表征和预算，以完整训练标签训练并通过 OA 选择。
- 报告先验来源、采样设计、监督预算和多 seed 统计；先验、候选与阈值确定后，在 test 上评测。

日常调参见[模型调优](../howto/model_tuning.md)，研究协议的具体设置见
[Survey 协议](../../research/pu_survey/pu_survey_protocol.md)。

## 6. 来源

Wang et al. (2026), §3、式(3)/(5)/(6)、Proposition 1，§3.3 的 PAUC 扩展：
[Accessible, Realistic, and Fair Evaluation](https://openreview.net/forum?id=5R11h5o44C)。

PA 在 OS 采样设计下的固定参考实现对照见
[实现基线](../../dev/pu_research_semantics.md#15-pa-在-os-采样设计下的参考实现差异)。
