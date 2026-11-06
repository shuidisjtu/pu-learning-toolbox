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

角色应在原始来源级隔离。PA 读取的 `view="pu"` 不暴露真实标签；
四角色 runner 当前接收 clean 输入，再由 Generator 生成 train/pu_val 视图。
已有真实 PU 数据的日常内部 CV 入口是 `PUPipeline`，其契约不等同于四角色实验。

## 2. PA 的条件和公式

Wang et al. 的 proxy accuracy 在相应 SCAR / 代表性正例和正确总体先验条件下定义为：

```math
PA_{OS}=2\pi_{population}\widehat P(\hat Y=1\mid S=1)
        +\widehat P_{L\cup U}(\hat Y=0),
```

```math
PA_{TS}=2\pi_{population}\widehat P_L(\hat Y=1)
        +\widehat P_U(\hat Y=0).
```

OS 的第二项在完整验证分区上计算；TS 的第二项只在原生 U 上计算。
设真正类召回率为 $`a`$、负类上预测正类的比例为 $`b`$，则总体上：

```math
E[PA]=2\pi a+1-\pi a-(1-\pi)b
     =Accuracy+\pi.
```

该期望针对独立于验证数据的固定预测规则和阈值；扫描后取最大经验 PA 还存在选择偏差。
有限验证集的经验 PA 存在抽样波动。PA 是 Accuracy+π 的代理分数，可超过 1。
先验值改变正例项权重，也可能改变最优阈值。

工具箱目前的 `proxy_accuracy` 实现 **OS 分支**，入口在
[experiment/strategies.py](../../../pu_toolbox/experiment/strategies.py)。
TS 分支与 PAUC 数值 API 尚未作为公共能力提供。

```python
import numpy as np
from pu_toolbox.experiment.strategies import proxy_accuracy

scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
y_pu = np.array([1, 1, 0, 0, 0])
pa = proxy_accuracy(scores, y_pu, threshold=0.5, class_prior=0.4)
assert abs(pa - 1.2) < 1e-12  # 2 * 0.4 * 1 + 2 / 5
```

上述排序结论适用于具有代表性的已标记正例。官方 Survey 的 SAR 比较使用 OA；
自定义 SAR 准则记录其额外识别假设与监督预算。

## 3. OA 与独立选择

OA 在代表目标总体的 clean validation 上计算真实 Accuracy，不需要先验项。
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

## 4. 形成可比较的实验

- 同一 seed 的方法复用底层 split、表征和同一标记配置；扫描标记率时明确标签复用规则。
- 预处理统计仅在 train 拟合；记录预训练模型的来源和拟合范围。
- 记录候选数、epoch 预算、checkpoint 及阈值网格，共享训练成本只计一次。
- PN oracle 与 PU 对照共享 split、表征和预算，以完整训练标签训练并通过 OA 选择。
- 报告先验来源、采样设计、监督预算和多 seed 统计；test 不参与先验、候选或阈值选择。

日常调参见[模型调优](../howto/model_tuning.md)，研究协议的具体设置见
[Survey 协议](../../research/pu_survey/pu_survey_protocol.md)。

## 5. 来源与参考实现差异

Wang et al. (2026), §3、式(3)/(5)/(6)、Proposition 1，§3.3 的 PAUC 扩展：
[Accessible, Realistic, and Fair Evaluation](https://openreview.net/forum?id=5R11h5o44C)。

论文 OS 公式与当前工具箱一致。
[PUBench 的固定版本实现](https://github.com/wu-dd/PUBench/blob/a9a62b05b0f222c72aff4df8992307376c78d682/lib/misc.py)
以及 [PU-Bench 的固定版本实现](https://github.com/XiXiphus/PU-Bench/blob/2d95a19eefd72e66ff30128ec2f65e1d1d4cc077/train/utils/metrics.py)
在 OS 第二项使用观测 PU 标签准确率。若 L 占比为 $`q`$、L 内预测正类比例为 $`a`$，
该替换较论文公式多出 $`q(2a-1)`$，会影响排序；引用时须写明采用的定义。
