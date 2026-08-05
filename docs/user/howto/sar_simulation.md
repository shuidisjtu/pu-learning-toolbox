# SCAR / SAR 数据模拟指南

## 1. 适用范围

该模块用于构造可控的 PU benchmark。设真实类别为 `Y`，是否观察到正标签为 `S`：

```math
S=1 \Longrightarrow Y=1,
\qquad
P(S=1\mid Y=0,X)=0.
```

SCAR 使用常数 `P(S=1|Y=1,X)=c`；SAR 允许该概率随特征变化。真实标签和 propensity
在现实 PU 训练中通常不可见，模拟器保留它们只是为了评估算法。

## 2. 公共接口

### 2.1 只计算 propensity

```python
from pu_toolbox.preprocessing import make_sar_propensity

propensity = make_sar_propensity(
    X,
    y_true,
    mechanism="linear",
    label_frequency=0.4,
    strength=1.5,
)
```

返回值表示 `P(S=1|Y,X)`，所以真实负类位置固定为零。函数会校准截距，使正类中的
propensity 均值等于 `label_frequency`。

### 2.2 从完整标签生成 PU 标签

```python
from pu_toolbox.preprocessing import make_sar_labels

y_pu, propensity = make_sar_labels(
    X,
    y_true,
    mechanism="nonlinear",
    label_frequency=0.4,
    random_state=42,
    return_propensity=True,
)
```

`y_pu` 使用项目规范：`1` 为观察到的正类，`0` 为 U。默认
`ensure_labeled=True`，当小样本 Bernoulli 抽样没有选中任何正类时，会选择 propensity
最高的真实正类，以保证下游估计器可以训练。研究纯采样分布时可显式关闭该保护。

### 2.3 直接生成合成数据

```python
from pu_toolbox.preprocessing import make_sar_dataset

X, y_pu, y_true, propensity = make_sar_dataset(
    n_samples=1000,
    n_features=8,
    class_prior=0.3,
    separation=2.0,
    mechanism="linear",
    label_frequency=0.4,
    strength=1.5,
    random_state=42,
)
```

训练模型时只能传入 `X, y_pu`。`y_true` 用于分类评估，`propensity` 用于标记机制恢复
评估；将二者用于模型训练或超参数选择会造成真值泄漏。

## 3. 机制定义

| 机制 | 定义 | 用途 |
|---|---|---|
| `scar` | 正类 propensity 为常数 | SCAR 控制组 |
| `linear` | 标准化特征投影经过 sigmoid | 单调 instance-dependent bias |
| `nonlinear` | 在线性投影上加入中心化二次项 | 非线性 selection bias |

`feature_weights` 控制投影方向，默认所有特征等权并进行 L2 归一化。`strength=0` 时，
线性和非线性机制退化为常数 propensity，可作为实现正确性的边界检查。

## 4. 标记率语义

`label_frequency` 是正类 propensity 的目标均值，不是每次随机抽样后必须精确达到的比例：

```math
\frac{1}{n_+}\sum_{i:Y_i=1}e(X_i)
=\text{label_frequency}.
```

实际标记比例由 Bernoulli 抽样产生，会围绕目标波动。正式报告应同时保存：

- 目标平均 propensity；
- 实际 `sum(y_pu)/sum(y_true)`；
- 正类 propensity 标准差；
- 数据 seed 和机制强度。

仅比较实际标记数明显不同的实验，可能把样本量收益误判为 SAR 方法收益。

## 5. SCAR/SAR 对比 benchmark

```bash
python -m benchmarks.assigned_methods.run \
  --config benchmarks/assigned_methods/configs/scar_sar_comparison.json \
  --output benchmarks/assigned_methods/results/scar_sar_comparison
```

配置对每个 seed 复用相同的数据生成过程和目标标记率，依次运行 SCAR、线性 SAR 和
非线性 SAR。除分类指标外，还报告：

- `posterior_spearman`；
- `posterior_kendall`；
- `pairwise_ranking_accuracy`；
- `propensity_mse_positive`；
- `propensity_spearman_positive`。

SCAR 的真实 propensity 是常数，因此 propensity rank correlation 在数学上未定义，
结果中记录为空值，而不是伪造为零。

## 6. 常见错误

- **负类出现 `y_pu=1`**：违反 PU 单边标记机制，应视为数据生成错误。
- **用全数据真实标签调参**：只允许在合成诊断中使用，不能冒充 PU-only 模型选择。
- **混淆 propensity 均值和实际标记率**：前者已校准，后者有抽样波动。
- **忽略特征尺度**：模拟器在真实正类内标准化后构造 propensity，外部自定义机制也应
  明确尺度处理。
- **只报告 accuracy**：selection-bias 方法还应报告排序和 propensity 恢复质量。
