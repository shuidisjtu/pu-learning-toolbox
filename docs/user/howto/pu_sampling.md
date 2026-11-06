# 构造同池 OS 与独立 TS 数据

前置条件：安装工具箱，了解[采样设计](../concepts/pu_sampling.md)和
[先验目标](../concepts/class_prior_semantics.md)。本教程使用小规模 CPU 高斯数据。

## 1. 从同一 clean 池定额标记

`make_fixed_count_labels` 在真实正类中均匀无放回选择指定数量，保留输入行序。
例如一个 40 个正类、60 个负类的完整池，标记 20 个正例后，U 中剩余 20 个正类和 60 个负类。

```python
import numpy as np
from pu_toolbox.preprocessing import make_fixed_count_labels

y_true = np.array([1] * 40 + [0] * 60)
y_pu = make_fixed_count_labels(y_true, n_labeled=20, random_state=7)
assert y_pu.sum() == 20
assert y_true[y_pu == 0].mean() == 0.25
```

如果按比例指定配额，使用 `make_scar_labels(y_true, c=0.5, random_state=7)`；
若需要 Bernoulli 或特征相关标记，见 [SCAR/SAR 模拟](sar_simulation.md)。
`make_pu_labels(..., mechanism="fixed_count", n_labeled=20)` 提供统一分派入口。

## 2. 独立取得正例 P 与总体混合 U

`make_gaussian_case_control_data` 从正类条件高斯分布取得 P，再独立从目标总体取得 U。
目标总体由 `class_prior` 及正负类均值确定，每个 U 样本的类别独立抽取。

```python
import numpy as np
from pu_toolbox.preprocessing import make_gaussian_case_control_data

X, y_pu, y_true, metadata = make_gaussian_case_control_data(
    n_p=80, n_u=800, class_prior=0.4, n_features=2, random_state=7,
)
assert X.shape == (880, 2)
assert np.all(y_true[y_pu == 1] == 1)
assert metadata["population_class_prior"] == 0.4
assert metadata["unlabeled_class_prior"] == y_true[y_pu == 0].mean()
assert len(set(metadata["sample_ids"])) == 880
```

输出按 P 后 U 排列，`y_pu` 为训练角色标签，`y_true` 用于合成审计与评测。
`metadata` 保存目标总体先验、U 和拼接表经验先验、样本身份、分布参数和抽样政策。
`sample_ids` 的 `positive:` 与 `population:` 前缀标记两个独立来源中的抽样事件。
这些身份在单次输出中唯一；合并多个生成结果时，由调用者增加批次命名空间。
整数 seed 可复现全部输出；传入 `RandomState` 时沿用其当前随机流，元数据中的 seed 为 `None`。

## 3. 运行成对比较示例

```powershell
uv run python examples/minimal/15_pu_sampling.py --seed 7
```

[示例 15](../../../examples/minimal/15_pu_sampling.py)先生成独立 TS，
再把其中总体抽样得到的 U 作为 clean 来源池，构造 OS 条件。
两组共享目标总体 0.4 和同一总体抽样，便于观察移除已标记正例后的 U 变化。
输出分别显示目标参数、有限样本经验比例、标记数量及抽样政策。

研究数据按原始身份先切分 train / pu_val / clean_val / test，再在允许的分区生成标签。
完整四角色组合见[自定义实验](custom_experiments.md)。

## 4. 迁移已有调用

| 已有用法 | 对应入口 |
|---|---|
| `make_case_control_labels(y_true, n_labeled, random_state)` | `make_fixed_count_labels(y_true, n_labeled, random_state)` |
| `make_pu_labels(..., mechanism="case_control")` | 改为 `mechanism="fixed_count"`，其余参数保留 |
| 以独立 TS 为目标的 Gaussian 教学代码 | `make_gaussian_case_control_data(..., class_prior=...)`，接收四个返回值 |

旧 `case_control` 名字保留同池行为与 seed 轨迹，发出 `FutureWarning`，计划在 2.0 移除。
新定额入口支持全部为正类的输入；旧别名延续原有双类别输入校验。
`make_gaussian_pu_data` 继续提供“全部正例可见、U 来自负类”的教学场景，
返回第三项为该拼接表的真实正类比例。

参数有效域和完整返回字段见 [API 参考](../reference/api.md#数据生成)。
