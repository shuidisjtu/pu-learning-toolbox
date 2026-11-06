# 生成 SCAR/SAR 数据

> 前置条件：先完成 [快速开始](../quickstart.md)。
> 概念：SCAR/SAR 的数学定义与三种生成机制见 [concepts/scar_sar.md](../concepts/scar_sar.md)。

本模块用于构造可控的 PU benchmark。真实标签用于模拟标记机制和分类评估，
propensity 用于标记机制恢复评估，训练接口接收生成后的 PU 视图。

## 1. 公共接口

### 1.1 只计算 propensity

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

### 1.2 从完整标签生成 PU 标签

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

### 1.3 直接生成合成数据

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

训练模型传入 `X, y_pu`。`y_true` 用于分类评估，`propensity` 用于标记机制恢复评估；
额外 clean support、OA 选模等监督使用按[数据角色](custom_experiments.md#2-准备四路数据)分别记录。

> **默认机制是 SAR**：`make_sar_dataset` 不传 `mechanism` 时解析为 `"linear"`（标记依赖特征）
> 并发出 `UserWarning`。类先验估计器与多数 PU 分类器假设 SCAR；SCAR 场景请显式传
> `mechanism="scar"`（区别见 [concepts/scar_sar.md](../concepts/scar_sar.md)）。

## 2. 标记率语义

`label_frequency` 是正类 propensity 的目标均值：

```math
\frac{1}{n_{+}}\sum_{i\,:\,Y_{i}=1} e(x_{i}) = \bar c.
```

其中 $`\bar c`$ 即传入的 `label_frequency`，$`e(x)`$ 为正类标记倾向。
OS 总体的已标记比例满足 $`q=\pi_{population}\bar c`$，符号见
[SCAR/SAR 概念](../concepts/scar_sar.md#1-两种标记机制)。

实际标记比例由 Bernoulli 抽样产生，会围绕目标波动。正式报告应同时保存：

- 目标平均 propensity；
- 实际 `sum(y_pu)/sum(y_true)`；
- 正类 propensity 标准差；
- 数据 seed 和机制强度。

比较实验时同时核对实际标记数与机制，分析标记数量及选择偏好的作用。

### 2.1 对照 Bernoulli 与具名定额策略

```powershell
uv run python examples/minimal/17_sar_policies.py --seed 7 --label-frequency 0.4
```

[示例 17](../../../examples/minimal/17_sar_policies.py)对同一个 OS clean 池运行四种政策，
每种输出标签哈希、目标/实际标记率、计数、seed 和评分来源。

| 策略 | 机制参数与评分来源 | 抽样政策 | seed=7 的实际标记数 |
|---|---|---|---:|
| linear | 正类内标准化投影，`strength=1.5`，等权特征方向 | 独立 Bernoulli，`ensure_labeled=False` | 63 |
| nonlinear | 同一投影加入中心化二次项，`strength=1.5` | 独立 Bernoulli，`ensure_labeled=False` | 61 |
| LBE-A | 本 clean 池真实标签拟合 LogisticRegression，`k=10`，90% 权重加 10% 均匀平滑 | 定额加权无放回 | 64 |
| LBE-B | 同一拟合范围，`k=10, shrink_coef=1` | 定额加权无放回 | 64 |

池中有 160 个真正例，请求标记率为 0.4。Bernoulli 的期望计数为 64；
LBE 配额为 `min(n_pos, max(1, round(c*n_pos)))`，使用半数取偶并保证至少一例。
LBE 权重按正例内的相对偏好解释，公式和来源见
[具名机制对照](../concepts/scar_sar.md#6-论文定义与具名实现)。

在四角色实验中，Generator 分别在 train 和 pu_val 生成视图，辅助评分在各自允许的
clean 分区拟合。可将 `SARLBEAGenerator()` 或 `SARLBEBGenerator()` 传给
`ExperimentRunner(generator=...)`，并通过 `protocols=[ProtocolOA()]` 使用 clean 验证选模。
组件替换步骤见[实验教程](custom_experiments.md#4-扩展标签生成器)。

## 3. SCAR/SAR 对比 benchmark

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
结果中记录为空值。

## 4. 常见错误

- **负类出现 `y_pu=1`**：违反 PU 单边标记机制，应视为数据生成错误。
- **监督角色混用**：按 train、pu_val、clean_val、test 分别记录训练、PU 选模、OA 和最终评测所用的标签。
- **混淆 propensity 均值和实际标记率**：前者已校准，后者有抽样波动。
- **忽略特征尺度**：模拟器在真实正类内标准化后构造 propensity，外部自定义机制也应
  明确尺度处理。
- **只报告 accuracy**：selection-bias 方法还应报告排序和 propensity 恢复质量。

## 下一步

- 检查生成数据的质量与假设证据：[data_profiling.md](data_profiling.md)
- 精确参数契约：[API 参考](../reference/api.md)
