# Method Card: LBE

## 1. 论文信息

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

### Assumptions

LBE 处理的是 instance-dependent labeling bias：正样本被标记的概率依赖 `x`。令 `Y in {0,1}` 为真实类别，`S in {0,1}` 为是否有观测正标签，PU 观测关系为：

```math
S=1 \Longrightarrow Y=1,
```

但：

```math
S=0 \centernot\Longrightarrow Y=0.
```

在 SAR 机制下：

```math
p_\theta(y=1\mid x)=r_\theta(x),
\qquad
p_\phi(s=1\mid y=1,x)=c_\phi(x).
```

如果真实负例不会被标为正例，则联合观测概率为：

```math
p(s=1\mid x)=r_\theta(x)c_\phi(x),
\qquad
p(s=0\mid x)=1-r_\theta(x)c_\phi(x).
```

因此 P/U 数据只直接观测到 `S`，而 `Y` 在 U 样本上是潜变量。

---

## 2. 问题设定与符号

论文摘要说明：标记概率不仅由类别决定，还依赖观测特征；作者建立 `P(S,Y|X)` 图模型，通过 EM 和 Adam 同时学习 labeling probability 与 classifier。

### 2.1 与 SCAR 的区别

| 模型 | propensity | 典型接口 |
|---|---|---|
| SCAR | `c_phi(x)=c` 常数 | Elkan-Noto、uPU、nnPU |
| SAR | `c_phi(x)` 随 `x` 变化 | LBE、PUSB |

LBE 的目标不是仅用一个常数校准模型，而是把“样本本身是否容易被标记”建模出来。

| 论文符号 | 含义 |
|---|---|
| `x` | 特征 |
| `y` | 潜在真实标签（U 中不可观测） |
| `s` | 是否被标记 |
| `r_theta(x)` | 类别后验 `P(Y=1|X=x)` |
| `c_phi(x)` | labeling propensity `P(S=1|Y=1,X=x)` |
| `r_theta(x)c_phi(x)` | 观测正标签概率 `P(S=1|X=x)` |
| `q_i` | U 样本 latent `P(Y=1|S=0,x_i)` |
| `theta` | 类别模型参数 |
| `phi` | propensity 模型参数 |
| `k` | labeled positive 数量 |
| `n` | 总训练样本数量 |

---

## 3. 核心公式

### 3.1 观测似然

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

### 3.2 E-step：未标记样本的潜变量后验

对 `S=0` 的样本：

```math
q_i
=P(Y_i=1\mid S_i=0,x_i)
=\frac{r_\theta(x_i)(1-c_\phi(x_i))}
{1-r_\theta(x_i)c_\phi(x_i)}.
```

对已标记正样本：

```math
q_i=1,\qquad S_i=1.
```

此后可以用 `q_i` 作为类别模型的 soft target。

### 3.3 M-step：类别模型

固定 `q_i` 后，类别模型近似最小化 soft-label logistic loss：

```math
\min_\theta
-\sum_i\left[q_i\log r_\theta(x_i)
+(1-q_i)\log(1-r_\theta(x_i))\right].
```

### 3.4 M-step：propensity 模型

propensity 只对真实正类有意义。固定 `q_i` 后，正类相关样本使用：

```math
\min_\phi
-\sum_i q_i\left[s_i\log c_\phi(x_i)
+(1-s_i)\log(1-c_\phi(x_i))\right].
```

已标记正样本的 `q_i=1,s_i=1`；未标记样本以 `q_i` 作为“潜在正类”的样本权重，并以 `s_i=0` 作为 propensity 的负观测。

---

## 4. 算法概要

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

---

## 5. 论文边界

- LBE 处理的是 instance-dependent labeling bias：正样本被标记的概率依赖 `x`。
- 未标记样本不是负类；对 U 的真实 `Y` 必须通过潜变量后验估计。
- `P(s=1|x)` 可以由 `P(y=1|x)P(s=1|y=1,x)` 分解，但两个因子在有限样本下可能存在可辨识性和局部最优问题。
- `predict_label_proba` 输出的是被标记概率 `P(S=1|x)`，不是类别概率 `P(Y=1|x)`。

---

## 6. 实现注记

- 当前实现用线性 logistic 模型和交替 soft-label 更新近似论文的 EM + Adam 框架，不能直接宣称复现论文神经网络实验。
- 当前 `class_prior=None` 时使用工程初始化值；这不是论文给出的普适先验估计器，也不能替代 ReCPE/penL1。
- 实际论文使用其图模型对应的 likelihood parameterization；实现时应以论文完整公式和官方代码变量为最终核对依据。
- 项目实现通过两份样本副本和权重 `q_i`、`1-q_i` 调用 sklearn logistic regression，得到可运行的 soft-label 更新。
- 停止条件：latent posterior 最大变化量小于 `1e-5`，或达到 `n_em_iter`。论文的完整实现还涉及 Adam 优化、参数初始化和 likelihood 迭代细节；不能用当前停止条件替代论文实验 protocol。
- 数值稳定性：
  - `r` 和 `c` 被限制到 `[1e-5, 1-1e-5]`；
  - 计算 `1-r*c` 时使用下界 `1e-5`，避免除零；
  - `q_i` 最终应截断或验证在 `[0,1]`；
  - 当 P/U 极度不平衡时，propensity 的加权 logistic 可能退化；应报告 warning 或增加正则化；
  - 多个 EM 固定点都是可能的，正式实验必须使用多随机初始化或固定初始化 protocol。

结果记录（benchmark 数字与官方源码 SHA-256 锁以 benchmarks 产物为准）：

> 结果见 `benchmarks/assigned_methods/results/clean_room_multiseed/`（当前 linear-EM clean-room 结果与官方 `LBE_TPAMI21.rar` SHA-256 锁）

---

## 7. 论文实验参考

### 7.1 SAR 合成生成式

```math
Y\sim Bernoulli(\pi),
\qquad
S\sim Bernoulli(Y\cdot c(X)),
```

其中 `c(X)` 为已知 sigmoid propensity。

### 7.2 SAR 合成协议

使用与 PUSB 可共享的生成器产生 `(X,Y,S,e(X))`，训练时只传入 `(X,S)`。至少覆盖 SCAR、线性 SAR、非线性 SAR，并扫描类别先验、平均标记率、偏置强度和类别 overlap。拆分必须按真实 `Y` 和观测 `S` 联合分层；验证/测试中保留 `Y` 仅用于评估。

### 7.3 对照基线

对照至少包括 P/U logistic、SCAR baseline、PUSB、`linear_em`、`neural_lbe` 和 `oracle_propensity`（真实 `e(x)` 的诊断路径，用于分解误差上界）。在相同 split 上报告：

- 类别后验 `r_hat`：ROC-AUC、PR-AUC、log-loss、Brier score；
- propensity `c_hat`：在真实正类子集上的 MAE、Brier score 和 rank correlation；
- 观测标签概率 `r_hat*c_hat`：对 `S` 的 log-loss/Brier score；
- 分类：独立验证阈值下的 balanced accuracy 和 F1；
- 可辨识性：不同初始化间预测方差、失败率和 likelihood 差异。

### 7.4 超参数（论文含义）

| 参数 | 论文含义 |
|---|---|
| `n_em_iter` | EM 最大交替次数；论文需与 optimizer stop protocol 对齐 |
| `class_prior` | U latent posterior 初始化先验；不是独立的 CPE |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `official_exact`/`official_related` 记录需以压缩包许可证和源码对应论文版本复核 |
| Implementation status | `NATIVE`，当前为线性 clean-room EM |
| 当前实现可声称 | 可运行 SAR 接口、类别后验、labeling propensity 和 latent posterior |
| 当前实现不可声称 | 已复现论文的完整 Adam/深度网络实验、理论收敛条件或表格结果 |
| 主要风险 | 潜变量模型可能存在局部最优；正类过少时两个 logistic 模型会互相补偿；propensity 与 class posterior 的分解需要数据机制支持 |
