# Method Card: LBE

## 1. 待办与注意

### 1.1 待办

- [x] 明确 LBE 的潜变量：真实类别 `Y` 与是否被标记 `S` 分离建模。
- [x] 实现 `P(y=1|x)` 分类器和 `P(s=1|y=1,x)` propensity 模型的交替估计。
- [x] 对未标记样本计算 EM latent positive posterior。
- [x] 接入 `BasePUClassifier`、`predict_label_proba` 和 registry。
- [x] 增加 synthetic SAR 数据上的 API、有限值和 propensity 范围测试。
- [ ] 将当前线性 soft-label EM 与论文的完整 likelihood/Adam 实现逐项对齐。
- [ ] 增加神经网络版本、正则项、初始化策略和 official benchmark。
- [ ] 对多随机初始化报告 identifiability 和局部最优敏感性。

### 1.2 注意

- LBE 处理的是 instance-dependent labeling bias：正样本被标记的概率依赖 `x`。
- 未标记样本不是负类；对 U 的真实 `Y` 必须通过潜变量后验估计。
- `P(s=1|x)` 可以由 `P(y=1|x)P(s=1|y=1,x)` 分解，但两个因子在有限样本下可能存在可辨识性和局部最优问题。
- 当前实现用线性 logistic 模型和交替 soft-label 更新近似论文的 EM + Adam 框架，不能直接宣称复现论文神经网络实验。
- 当前 `class_prior=None` 时使用工程初始化值；这不是论文给出的普适先验估计器，也不能替代 ReCPE/penL1。
- `predict_label_proba` 输出的是被标记概率 `P(S=1|x)`，不是类别概率 `P(Y=1|x)`。

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Instance-Dependent Positive and Unlabeled Learning with Labeling Bias Estimation |
| Authors | Chen Gong, Qizhou Wang, Tongliang Liu, Bo Han, Jia You, Jian Yang, Dachen Tao |
| Venue | IEEE Transactions on Pattern Analysis and Machine Intelligence |
| Year | 2022 |
| Volume / Pages | 44(8), 4163–4176 |
| Family | `bias_aware` |
| Scenario | `single_training_set`、`selection_biased` |
| Assumption | SAR / instance-dependent labeling |
| Requires class prior | 论文图模型本身不要求直接观测负例；项目接口允许可选初始化先验 |
| Requires propensity | 方法显式估计 `P(S=1|Y=1,X)` |
| Requires negative samples | 否 |
| Source | [论文信息页](https://research.polyu.edu.hk/en/publications/instance-dependent-positive-and-unlabeled-learning-with-labeling-/) |
| Code record | [LBE_TPAMI21.rar](https://gcatnjust.github.io/ChenGong/code/LBE_TPAMI21.rar) |

论文摘要说明：标记概率不仅由类别决定，还依赖观测特征；作者建立 `P(S,Y|X)` 图模型，通过 EM 和 Adam 同时学习 labeling probability 与 classifier。该摘要信息是项目将 LBE 放入 `bias_aware/SAR` family 的依据。

## 3. 问题设定

令：

- `X`：输入特征；
- `Y in {0,1}`：真实类别；
- `S in {0,1}`：是否有观测正标签。

PU 观测关系为：

```math
S=1 \Longrightarrow Y=1,
```

但：

```math
S=0 \centernot\Longrightarrow Y=0.
```

在 SAR 机制下定义：

```math
p_\theta(y=1\mid x)=r_\theta(x),
```

```math
p_\phi(s=1\mid y=1,x)=c_\phi(x).
```

如果真实负例不会被标为正例，则联合观测概率为：

```math
p(s=1\mid x)=r_\theta(x)c_\phi(x),
```

```math
p(s=0\mid x)=1-r_\theta(x)c_\phi(x).
```

因此 P/U 数据只直接观测到 `S`，而 `Y` 在 U 样本上是潜变量。

### 3.1 与 SCAR 的区别

| 模型 | propensity | 典型接口 |
|---|---|---|
| SCAR | `c_phi(x)=c` 常数 | Elkan-Noto、uPU、nnPU |
| SAR | `c_phi(x)` 随 `x` 变化 | LBE、PUSB |

LBE 的目标不是仅用一个常数校准模型，而是把“样本本身是否容易被标记”建模出来。

## 4. 符号与记号

| 论文符号 | 含义 | 开发侧对应 |
|---|---|---|
| `x` | 特征 | `X` |
| `y` | 潜在真实标签 | U 中不可观测 |
| `s` | 是否被标记 | `(y_pu == 1).astype(float)` |
| `r_theta(x)` | 类别后验 `P(Y=1|X=x)` | `classifier_.predict_proba(X)[:,1]` |
| `c_phi(x)` | labeling propensity `P(S=1|Y=1,X=x)` | `propensity_model_.predict_proba(X)[:,1]` |
| `r_theta(x)c_phi(x)` | 观测正标签概率 `P(S=1|X=x)` | `predict_label_proba(X)` |
| `q_i` | U 样本 latent `P(Y=1|S=0,x_i)` | `_latent_positive_probability_` |
| `theta` | 类别模型参数 | classifier pipeline 参数 |
| `phi` | propensity 模型参数 | propensity pipeline 参数 |
| `k` | labeled positive 数量 | `n_positive_` |
| `n` | 总训练样本数量 | `_X_shape_[0]` |

## 5. 似然与 EM 更新

### 5.1 观测似然

在 `S=1` 的已标记正样本上，`Y=1` 是确定的，因此：

```math
\log p(s=1\mid x)=\log r_\theta(x)+\log c_\phi(x).
```

在 `S=0` 的未标记样本上，真实类别未知，需要对 `Y=0` 和 `Y=1` 边缘化：

```math
\log p(s=0\mid x)=
\log\left(1-r_\theta(x)c_\phi(x)\right).
```

总观测对数似然可写为：

```math
\mathcal L(\theta,\phi)=
\sum_{i:S_i=1}
\left[\log r_\theta(x_i)+\log c_\phi(x_i)\right]
+\sum_{i:S_i=0}
\log\left[1-r_\theta(x_i)c_\phi(x_i)\right].
```

实际论文使用其图模型对应的 likelihood parameterization；实现时应以论文完整公式和官方代码变量为最终核对依据。

### 5.2 E-step：未标记样本的潜变量后验

对 `S=0` 的样本：

```math
q_i
=P(Y_i=1\mid S_i=0,x_i)
=\frac{r_\theta(x_i)(1-c_\phi(x_i))}
{1-r_\theta(x_i)c_\phi(x_i)}.
```

对已标记正样本：

```math
q_i=1,qquad S_i=1.
```

此后可以用 `q_i` 作为类别模型的 soft target。

### 5.3 M-step：类别模型

固定 `q_i` 后，类别模型近似最小化 soft-label logistic loss：

```math
\min_\theta
-\sum_i\left[q_i\log r_\theta(x_i)
+(1-q_i)\log(1-r_\theta(x_i))\right].
```

项目实现通过两份样本副本和权重 `q_i`、`1-q_i` 调用 sklearn logistic regression，得到可运行的 soft-label 更新。

### 5.4 M-step：propensity 模型

propensity 只对真实正类有意义。固定 `q_i` 后，正类相关样本使用：

```math
\min_\phi
-\sum_i q_i\left[s_i\log c_\phi(x_i)
+(1-s_i)\log(1-c_\phi(x_i))\right].
```

已标记正样本的 `q_i=1,s_i=1`；未标记样本以 `q_i` 作为“潜在正类”的样本权重，并以 `s_i=0` 作为 propensity 的负观测。

### 5.5 交替停止

项目当前停止条件为 latent posterior 最大变化量小于 `1e-5`，或达到 `n_em_iter`。论文的完整实现还涉及 Adam 优化、参数初始化和 likelihood 迭代细节；不能用当前停止条件替代论文实验 protocol。

## 6. 当前实现算法流程

```text
输入：X、s，其中 s=1 表示 labeled positive，s=0 表示 U

1. 校验 P/U 均非空。
2. 初始化 q_i：P 设为 1，U 设为 class_prior 或工程初值。
3. 重复 n_em_iter 次：
   a. 用 q 作为 soft target 更新 classifier: r_theta(x)=P(Y=1|x)；
   b. 根据 classifier 更新 propensity model: c_phi(x)=P(S=1|Y=1,x)；
   c. 用 q_i = r_i(1-c_i)/(1-r_i c_i) 更新 U 的 latent posterior；
   d. 若 q 变化低于容差则提前停止。
4. 保存 classifier、propensity model 和 q。
5. 对新样本输出类别概率 r(x)，输出标记概率 r(x)c(x)。
```

## 7. 超参数、初始化与数值约束

| 参数 | 当前默认值 | 含义 | 复现注意 |
|---|---:|---|---|
| `n_em_iter` | 20 | 最大交替次数 | 论文需与 optimizer stop protocol 对齐 |
| `max_iter` | 1000 | logistic solver 最大迭代次数 | 项目适配 |
| `C` | 1.0 | 两个 logistic 模型的正则倒数 | 项目适配 |
| `class_prior` | `None` | U latent posterior 初始化强度 | 不是当前实现的独立 CPE |
| `random_state` | 0 | logistic 初始化随机种子 | 当前固定为 0 |

### 7.1 数值稳定性

- `r` 和 `c` 被限制到 `[1e-5, 1-1e-5]`。
- 计算 `1-r*c` 时使用下界 `1e-5`，避免除零。
- `q_i` 最终应截断或验证在 `[0,1]`。
- 当 P/U 极度不平衡时，propensity 的加权 logistic 可能退化；应报告 warning 或增加正则化。
- 多个 EM 固定点都是可能的，正式实验必须使用多随机初始化或固定初始化 protocol。

## 8. API 接口与项目落点

### 8.1 构造函数

```python
class LBEClassifier(BasePUClassifier):
    def __init__(self, *, max_iter=1000, n_em_iter=20, C=1.0):
        ...
```

### 8.2 API 语义

| API / 属性 | 约定 |
|---|---|
| `fit(X, y_pu, *, class_prior=None, sample_weight=None)` | P/U 标签协议；`class_prior` 只用于初始化；当前 `sample_weight` 接口保留但未用于 EM |
| `decision_function(X)` | 类别模型 logit `logit(P(Y=1|x))` |
| `predict_proba(X)` | 两列类别概率 `[P(Y=0|x), P(Y=1|x)]` |
| `predict_label_proba(X)` | 观测正标签概率 `P(S=1|x)=r(x)c(x)` |
| `classifier_` | 类别后验模型 |
| `propensity_model_` | labeling propensity 模型 |
| `_latent_positive_probability_` | 训练数据最终 latent `q_i` |
| `_class_prior` | 当前训练集类别概率的工程诊断值，不等于显式 CPE 输出 |

### 8.3 模块落点

| 模块 | 责任 | 状态 |
|---|---|---|
| `pu_toolbox/estimators/bias_aware/lbe.py` | 线性 logistic soft-label EM | ✅ |
| `pu_toolbox/estimators/bias_aware/__init__.py` | 导出 `LBEClassifier` | ✅ |
| `pu_toolbox/registry/builtin_methods.py` | LBE 元数据、SAR 标记和 lazy binding | ✅ |
| `tests/unit/estimators/test_bias_aware.py` | API 和 propensity 范围测试 | ✅ |
| `benchmarks/sar/lbe/` | SAR 合成数据和论文数据集 benchmark | ⏳ |

## 9. 测试与验收标准

### 9.1 API 和数学性质

- P/U 缺失时统一校验失败。
- `predict_proba` 每行和为 1，值在 `[0,1]`。
- `predict_label_proba` 值在 `[0,1]`。
- 对任意新样本验证 `P(S=1|x) <= P(Y=1|x)`，因为 `c(x) <= 1`。
- `q_i=1` 对所有已标记正样本成立。
- 固定输入和初始化时，EM 结果可重复。

### 9.2 SAR 合成测试

生成：

```math
Y\sim Bernoulli(\pi),
\qquad
S\sim Bernoulli(Y\cdot c(X)),
```

其中 `c(X)` 为已知 sigmoid propensity。测试应比较：

- LBE 估计的 `c_hat(X)` 与真实 `c(X)` 的 Brier/AUC；
- `r_hat(X)` 与真实 `P(Y=1|X)` 的 log-loss/AUC；
- LBE 与 SCAR baseline 在 `c(X)=constant` 和 `c(X)` 变化两种设置下的差异；
- 随正样本数、U 样本数、propensity 强度变化的稳定性。

### 9.3 论文复现

- 采用论文相同的线性 logistic 和非线性网络两条实验路径；
- 记录 EM 初始化、Adam 学习率、训练轮数、正则项和 early stopping；
- 报告类别模型和 labeling model 两套指标；
- 不能只报告分类 accuracy 而忽略 propensity estimation 误差。

## 10. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact`/`official_related` 记录需以压缩包许可证和源码对应论文版本复核 |
| Implementation status | `NATIVE`，当前为线性 clean-room EM |
| 当前实现可声称 | 可运行 SAR 接口、类别后验、labeling propensity 和 latent posterior |
| 当前实现不可声称 | 已复现论文的完整 Adam/深度网络实验、理论收敛条件或表格结果 |
| 主要风险 | 潜变量模型可能存在局部最优；正类过少时两个 logistic 模型会互相补偿；propensity 与 class posterior 的分解需要数据机制支持 |
| 下一步 | 逐式核对官方源码、补充 likelihood regression tests 和 paper-like SAR benchmark |

