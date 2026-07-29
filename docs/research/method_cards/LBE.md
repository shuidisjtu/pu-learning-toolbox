# Method Card: LBE

## 1. 待办与注意

### 1.1 待办

- [x] 明确 LBE 的潜变量：真实类别 `Y` 与是否被标记 `S` 分离建模。
- [x] 实现 `P(y=1|x)` 分类器和 `P(s=1|y=1,x)` propensity 模型的交替估计。
- [x] 对未标记样本计算 EM latent positive posterior。
- [x] 接入 `BasePUClassifier`、`predict_label_proba` 和 registry。
- [x] 增加 synthetic SAR 数据上的 API、有限值和 propensity 范围测试。
- [x] 补充线性/神经变体、SAR 数据、初始化敏感性和 propensity 评估协议。
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
| `benchmarks/assigned_methods/` | SAR 合成 runner、官方归档锁和结果 | ✅ linear EM；⏳ official MLP |

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

## 10. 复现实验协议

### 10.1 实现变体

实验结果必须明确区分：

| 变体 | 分类模型/propensity 模型 | 优化 | 定位 |
|---|---|---|---|
| `linear_em` | 两个线性 logistic | 当前 soft-label 交替更新 | 项目 baseline |
| `neural_lbe` | 论文网络 | EM + Adam/官方优化流程 | 论文级复现 |
| `oracle_propensity` | 分类模型 + 真实 `e(x)` | 诊断路径 | 分解误差上界 |

`linear_em` 可以验证接口和机制，但不能替代论文中的 `neural_lbe`。

### 10.2 SAR 合成协议

使用与 PUSB 可共享的生成器产生 `(X,Y,S,e(X))`，训练时只传入 `(X,S)`。至少覆盖 SCAR、
线性 SAR、非线性 SAR，并扫描类别先验、平均标记率、偏置强度和类别 overlap。每个正式
设置至少运行 20 个数据 seed；每个数据 seed 再运行 5 个模型初始化，以分离采样方差与
局部最优敏感性。

拆分必须按真实 `Y` 和观测 `S` 联合分层；验证/测试中保留 `Y` 仅用于评估。归一化统计量、
初始化先验和 early stopping 均只使用训练/验证部分。

### 10.3 训练记录与模型选择

论文级配置需从官方代码锁定网络层、激活、Adam 学习率、batch size、E/M 更新频率、
正则项、最大 epoch、初始化和停止条件。每次运行保存：

- observed log-likelihood 与各组成项；
- `q` 的均值、范围和最大迭代变化；
- `r_hat(X)`、`c_hat(X)`、`r_hat(X)c_hat(X)`；
- 最佳 epoch、停止原因和数值 warning。

主模型选择指标应是验证集 observed likelihood 或论文指定的无真实标签准则。使用隐藏
`Y/e(X)` 选超参数只能进入 `oracle` 消融。

### 10.4 对照、指标与消融

对照至少包括 P/U logistic、SCAR baseline、PUSB、`linear_em`、`neural_lbe` 和
`oracle_propensity`。在相同 split 上报告：

- 类别后验 `r_hat`：ROC-AUC、PR-AUC、log-loss、Brier score；
- propensity `c_hat`：在真实正类子集上的 MAE、Brier score 和 rank correlation；
- 观测标签概率 `r_hat*c_hat`：对 `S` 的 log-loss/Brier score；
- 分类：独立验证阈值下的 balanced accuracy 和 F1；
- 可辨识性：不同初始化间预测方差、失败率和 likelihood 差异。

消融至少包含固定常数 propensity、移除正则、不同 `q` 初始化和不同 E/M 更新频率。

### 10.5 真实数据、产物与验收

真实数据的类别映射、selection mechanism、split 和预处理以论文/官方代码 manifest 为准。
若真实 propensity 不可观测，只报告类别与观测标签指标，并明确 propensity 无 ground truth，
不得用训练拟合优度冒充 propensity 恢复精度。

建议落点为 `benchmarks/sar/lbe/`。每个结果目录保存配置、代码 commit、数据 manifest、
逐 seed/初始化日志、预测和聚合表。验收要求：

- 恒等式 `predict_label_proba = r_hat*c_hat` 在保存预测上成立；
- 已标记正例的 latent `q=1`，所有概率有限且在 `[0,1]`；
- 同时有 SCAR/SAR、线性/神经网络和初始化敏感性结果；
- 论文级结果使用官方优化流程，且多初始化汇总不只选取最好一次；
- 与论文数字不一致时报告差异来源，不根据测试标签反向修改 protocol。

当前 linear-EM clean-room 运行使用 seed `0..4`，得到 ROC-AUC
`0.9887 ± 0.0108`，正类 propensity rank Spearman 为 `0.9030 ± 0.0516`。额外的
10-seed 配对 benchmark 在 SCAR、线性 SAR、非线性 SAR 下得到 posterior pairwise
ranking accuracy `0.6283`、`0.8694`、`0.9061`。SCAR 的真实 propensity 为常数，
propensity rank correlation 按未定义值处理。官方
`LBE_TPAMI21.rar` 已锁定 SHA-256
`79cc2c3635a6bcefef1d12832cc9e29be4c0c42a6c31ce6e7b44c6aeac504c6a`，
但其 CUDA MLP + Adam 实验尚未执行。

## 11. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact`/`official_related` 记录需以压缩包许可证和源码对应论文版本复核 |
| Implementation status | `NATIVE`，当前为线性 clean-room EM |
| 当前实现可声称 | 可运行 SAR 接口、类别后验、labeling propensity 和 latent posterior |
| 当前实现不可声称 | 已复现论文的完整 Adam/深度网络实验、理论收敛条件或表格结果 |
| 主要风险 | 潜变量模型可能存在局部最优；正类过少时两个 logistic 模型会互相补偿；propensity 与 class posterior 的分解需要数据机制支持 |
| 下一步 | 逐式核对官方源码、补充 likelihood regression tests 和 paper-like SAR benchmark |
