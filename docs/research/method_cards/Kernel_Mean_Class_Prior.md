# Method Card: Kernel Mean Class-Prior Estimation（KM1 / KM2）

## 1. 方法定位与状态

| 项目 | 内容 |
|---|---|
| 论文 | *Mixture Proportion Estimation via Kernel Embedding of Distributions* |
| 作者 | Harish G. Ramaswamy, Clayton Scott, Ambuj Tewari |
| 会议 | ICML 2016（PMLR 48） |
| 任务 | 从可靠正样本与未标记混合样本估计正类先验 |
| 公共接口 | `KernelMeanPriorEstimator` |
| 可选算法 | `variant="km1"`、`variant="km2"` |
| 当前实现 | NumPy/sklearn 原生实现，Frank-Wolfe simplex QP |
| 首个集成点 | InfoMax-PU 表示空间中的 class-prior estimator |

该方法只估计 `pi=P(Y=1)`，不直接训练分类器。InfoMax-PU 论文在表示学习后使用
kernel-mean estimator；但没有说明使用 KM1 还是 KM2，因此 paper-protocol 配置暂时显式锁定
KM1，并把该选择保留为复现差距。

## 2. 问题设定与可识别性

设未标记分布为 `F`，可靠正类分布为 `H`，未知负类分布为 `G`：

```math
F=(1-\kappa)G+\kappa H,\qquad 0\le\kappa<1.
```

输入样本满足：

```math
X_U\overset{i.i.d.}{\sim}F,\qquad X_P\overset{i.i.d.}{\sim}H.
```

在标准 SCAR/case-control PU 设定中，`H=p(x|Y=1)`，故 `kappa` 即正类先验。若可靠正例
受选择偏差影响，`H` 不再等于总体正类条件分布，此估计量的目标也随之改变。

仅凭两个分布不能无条件识别任意 mixture proportion。论文的可辨识结论依赖
`G` 相对于 `H` 的不可约性；若负类分布本身含有可分解出的 `H` 成分，算法估计的是最大可行
mixture proportion，而不一定是生成过程写下的名义比例。

## 3. RKHS 距离曲线

令 `phi(x)` 为核 `k` 对应的 RKHS feature map，分布均值嵌入为
`mu_Q=E_Q[phi(X)]`。重参数化：

```math
\lambda=\frac{1}{1-\kappa},\qquad
\kappa=\frac{\lambda-1}{\lambda}.
```

则候选负类均值为：

```math
\lambda\mu_F+(1-\lambda)\mu_H.
```

算法计算该点到所有概率分布均值嵌入集合 `C` 的距离：

```math
d(\lambda)=
\inf_{w\in C}
\left\|\lambda\mu_F+(1-\lambda)\mu_H-w\right\|_{\mathcal H}.
```

当 `lambda` 仍处于可行 mixture 区域时，距离接近零；越过真实拐点后，距离近似线性增长。
KM1/KM2 通过距离曲线斜率超过阈值的位置估计拐点。

## 4. 经验 QP 与数值求解

拼接 `n_U+n_P=N` 个样本，记 Gram matrix 为 `K`。对候选 `lambda` 构造：

```math
u_\lambda=
\left[
\frac{\lambda}{n_U}\mathbf 1_{n_U},
\frac{1-\lambda}{n_P}\mathbf 1_{n_P}
\right].
```

经验距离通过 probability simplex 上的二次规划获得：

```math
\widehat d(\lambda)^2
=\min_{v\ge0,\ \mathbf 1^Tv=1}
(u_\lambda-v)^T K(u_\lambda-v).
```

作者 Python 2.7 代码使用 CVXOPT。项目为避免新增重型 solver 依赖，使用带精确线搜索的
Frank-Wolfe 算法求同一个 simplex QP，并记录每次求解的迭代数与最终 dual gap。该变化是
数值后端差异，不是目标函数变化；正式复现仍应检查 `max_qp_gap` 和容差敏感性。

## 5. KM1 与 KM2

算法用有限差分估计斜率：

```math
\widehat d'(\lambda)
\approx
\frac{\widehat d(\lambda+\epsilon/2)-\widehat d(\lambda)}{\epsilon/2}.
```

KM1 使用理论阈值：

```math
\tau_{KM1}=\frac{1}{\sqrt{\min(n_U,n_P)}}.
```

KM2 使用启发式阈值。与作者代码一致，默认由初始斜率和经验 RKHS 分布距离加权：

```math
\tau_{KM2}=0.8s_{initial}+0.2\|\widehat\mu_F-\widehat\mu_H\|_{\mathcal H}.
```

在 `[1, lambda_upper_bound]` 上二分搜索首个超过阈值的位置，最后转换为
`kappa=(lambda-1)/lambda`。实现会同时计算 KM1/KM2，并由 `variant` 决定
`class_prior_` 返回哪一个。

## 6. 核选择与预处理

实现采用 RBF kernel：

```math
k(x,x')=\exp\left(-\frac{\|x-x'\|^2}{2\sigma^2}\right).
```

未显式指定 `kernel_width` 时，默认（`width_selection="relative"`，2026-08-10 起）取
`kernel_width_scale × sqrt(median pairwise squared distance)`（`kernel_width_scale=0.1`）。
该固定相对比例是尺度不变、数据自适应的；作者代码的 max-MMD 5 档宽度搜索
（`width_selection="mmd_grid"`）保留为可选路径 —— 探针显示 max-MMD 准则在常规 SCAR
数据上系统性偏选宽带宽、低估类先验（默认 km1 0.30→0.59、km2 0.31→0.46，真值 0.5），
论文协议复现（如 InfoMax-PU）需显式 pin `width_selection="mmd_grid"`。

`standardize=False` 与作者 kernel routine 的直接输入行为一致。若上游没有稳定尺度，应在
训练数据内标准化并记录统计量；InfoMax-PU 在已学习表示空间中调用该估计器。

## 7. API 与参数

```python
from pu_toolbox.prior import KernelMeanPriorEstimator

estimator = KernelMeanPriorEstimator(
    variant="km1",
    epsilon=0.04,
    lambda_upper_bound=8.0,
    km2_final_slope_weight=0.2,
    max_qp_iter=2000,
    qp_tolerance=1e-7,
    max_samples_per_group=None,
    standardize=False,
    random_state=0,
)
estimator.fit(X, y_pu)
pi_hat = estimator.estimate()
```

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `variant` | `"km1"` | 选择公开估计值 |
| `kernel_width` | `None` | 固定 RBF 宽度；空值时执行候选选择 |
| `width_selection` | `"relative"` | `"relative"`（0.1×中位距离，默认）/ `"mmd_grid"`（作者 5 档搜索） |
| `kernel_width_scale` | `0.1` | `relative` 模式下宽度与中位距离的比值 |
| `epsilon` | `0.04` | 有限差分步长与二分停止尺度 |
| `lambda_upper_bound` | `8.0` | 作者代码的搜索上界 |
| `km2_final_slope_weight` | `0.2` | KM2 阈值中的最终斜率权重 |
| `max_qp_iter` | `2000` | 每次 Frank-Wolfe 最大迭代数 |
| `qp_tolerance` | `1e-7` | Frank-Wolfe dual-gap 容差 |
| `max_samples_per_group` | `None` | 可复现的组内降采样上限 |
| `standardize` | `False` | 是否在估计器内部标准化 |

拟合后保存 `km1_estimate_`、`km2_estimate_`、`class_prior_`、`kernel_width_`、
`distribution_distance_`、`thresholds_` 和 `diagnostics_`。

## 8. 复杂度、测试与边界

- Gram matrix 的时间和内存均为 `O(N^2)`；全量大数据运行前应估算内存。
- 每个候选宽度需要一次 kernel matrix 评估；每个二分点包含两次 simplex QP。
- `max_samples_per_group` 可控制成本，但会改变统计估计量，必须进入实验配置和结果记录。
- 测试覆盖 identity-kernel QP 金标准、固定 seed 降采样、KM1/KM2 分派、有限区间输出、未拟合调用、非法参数和不可区分分布。
- `class_prior_` 被裁剪到 `[0,1]`；下游分类器仍要求严格位于 `(0,1)`，退化估计会被拒绝。
- InfoMax 的 20-seed paper-protocol 尚未实际执行；现阶段不能声称复现论文数值。

## 9. 参考资料

1. Ramaswamy, Scott, Tewari. *Mixture Proportion Estimation via Kernel Embedding of Distributions*. ICML 2016.
2. 论文页面：<https://proceedings.mlr.press/v48/ramaswamy16.html>
3. 作者软件页面：<https://web.eecs.umich.edu/~cscott/code.html#kmpe>
4. InfoMax-PU：<https://arxiv.org/abs/1710.05359>
