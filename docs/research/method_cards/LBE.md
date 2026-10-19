# Method Card: LBE

> 参数契约（签名、参数表、返回结构）以 [API 参考](../../user/reference/api.md) 为权威；本文档只记论文研究内容、实现边界与复现状态。

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

本节线性说明对应 `lbe` / `LBEClassifier`；独立的 `lbe_neural` 变体见 §10，不替换本节实现或历史结果。

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

> 结果见 `benchmarks/assigned_methods/results/clean_room_multiseed/`（当前 linear-EM clean-room 结果）；官方 `LBE_TPAMI21.rar` SHA-256 锁见 `benchmarks/assigned_methods/configs/official_sources.lock.json`

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
| Source status | 作者发表页当前直接链接包于 2026-10-19 复核，SHA-256 与旧官方锁一致；用户另提供包为不同变体、来源版本未证实。发表页只写明 research-only / 商用需许可，不构成明确软件许可证；详见[机器来源记录](../pu_survey/data/lbe_source_review_20261012.json) |
| Implementation status | `lbe` 为线性 clean-room EM；`lbe_neural` 为独立实验性双网络 generalized EM（§10），均不等于论文数值复现 |
| 当前实现可声称 | 可运行 SAR 接口、类别后验、labeling propensity 和 latent posterior |
| 当前实现不可声称 | 已复现论文的完整 Adam/深度网络实验、理论收敛条件或表格结果 |
| 主要风险 | 潜变量模型可能存在局部最优；正类过少时两个 logistic 模型会互相补偿；propensity 与 class posterior 的分解需要数据机制支持 |

## 9. 2026-10-12 用户源码包复核与修复

收到的 `Instance-Dependent PU Learning via Graphical Model(1).rar` 同时包含
`LBE_MLP` 与 `LBE_LF`；源码/数据逐文件摘要、公式结论、配方差异和问题位置见
[机器可读复核记录](../pu_survey/data/lbe_source_review_20261012.json)。未执行作者代码或将其复制进仓库，
没有发现明确软件许可证。2026-10-19 对作者发表页当前 code 链接只读下载并比对摘要，确认其与旧官方锁 `79cc2c36…` 字节相同；用户包 `86034d5b…` 为不同变体，来源版本仍不能确认，不能覆盖官方来源锁。

| 项 | 旧官方锁 `79cc2c36…` | 新用户包 `86034d5b…` |
|---|---|---|
| 每个模型预训练 | 100 steps | 20 steps |
| weight decay | 0.01 | 0.001 |
| classifier small-loss 保留比例 | 80% | 全部 |
| propensity 预训练 | observed-label balanced BCE | `s+(1-s)*r` 加权 BCE |
| EM 主循环 | 20 E × 40 M | 相同；文件头未使用的 `M_STEPS=20` 不能作为实际预算 |

两个包的 `em.py` 和两份数据字节相同；`reader.py` 不同，新包没有旧包的 NaN 权重回退。
E-step 的联合概率归一化与本项目
`q=r(1-c)/(1-r*c)` 相符；M-step 的 classifier soft target 与 propensity 的 `q` 权重也相符。
**不需要把 U 当负类或给 propensity 去掉权重**。线性 penalized logistic 解法与作者 Adam/MLP
的优化轨迹、初始化、预算并不等价；本轮仍未新增神经网络路径或复现论文表格。

2026-10-19 对同一包内 `LBE_LF` 脚本做了额外静态控制流核查（未执行 MATLAB）：breast 与 toydata 主入口均实际调用 `MStep_Adam`，并将 `MStep`/GD 调用注释掉；两份被调用的 Adam 函数都会覆盖传入的 `Theta1/Theta2`，但初始化不同——breast 两者均重置为随机向量，toydata 的 `Theta1` 重新由 `glmfit(TrainingFeature,PIndicator)` 拟合、`Theta2` 随机化。故该源码分支每个 M-step 不以外层传入参数作为 Adam 起点。GD 替代实现也有自己的初始化代码，但不是主入口启用的分支。逐文件摘要及位置记于[机器来源复核记录](../pu_survey/data/lbe_source_review_20261012.json) 的 `source_behavior_findings_20261019`。这只描述该用户包的实际控制流，不推断论文公式要求 warm-start，也不改变 `lbe` / `lbe_neural` 的实现、官方来源锁或历史实验。

已修复：

- 所有概率预测入口要求成功拟合；重训失败后不能继续假装旧模型有效；
- `max_iter` / `n_em_iter` 须为正整数，`C` 须有限且为正，初始化先验须为 `(0,1)` 内实数；
- 增加 `n_iter_`、`converged_`、`observed_log_likelihood_history_`。
  最后一项是观测似然之和，不含正则项；不能据此保证单调或全局收敛；
- 官方 preflight 数据目录由错误的 `dataset/` 改为源码实际使用的 `datasets/`；
- 台账显式记录校准不适用：原生 OS 的类别/标记联合似然不接 TS 的 U∪P 风险校准，
  不是“TS 适配待实现”；`calibration_applied=false` 与原生 OS 拒绝路径保持不变。

源程序不能不加检查地放入正式协议：MATLAB demo 用训练真值优化 ROC 阈值，不属于纯 PU-only PA；
MLP demo 在切分前计算全量统计，`train.py` 的 `fea_norm` 未被训练使用；SCAR `type=3` 还会返回
未初始化的 `model`。其反复 test accuracy 只是当前代码的诊断，不能据此认定已按 test 选模，
但迁移时必须保持 test 与训练/选模隔离。这些差异均已登记，不修改旧 645-run 实验事实。

本轮新增 32 项源码公式/接口/边界/确定性测试；连同台账契约与 pilot 拒收路由共 62 项定向回归通过。
在基线 `25fc71d` 的线性实现与修复版上，用同一合成输入、默认及显式 0.3 初始化各运行 5 轮，
类别概率和标记概率的最大差均为 0；这仅证明该回归样例未改变有效训练行为，不是正式数值复现。

## 10. 2026-10-14 独立神经网络变体

新增 `NeuralLBEClassifier` / `lbe_neural`（别名 `neural_lbe`），原 `lbe`、旧协议及历史数值不变。
模块独立实现本卡 E/M 公式：在 log space 归一化联合概率；每轮 E 后验 detach，classifier 使用 soft BCE，
propensity 使用后验加权的 observed-label BCE。每个 M 更新重抽训练行，不访问任何验证/test/隐藏类别真值。

默认配方明确选用 **用户包 `86034d5b…` 的 MLP 变体**，并非将其当作旧官方包：
MLP100/tanh，两个模型分别 20 次预训练，20 E × 40 M，重抽 10% 行，wd=1e-3；
源配方差异沿用 §9。完整预训练共 40 次、EM 共 800 次 Adam 更新，classifier 快照仅按 20 个 EM 轮保存。
源代码未复制、执行或再分发；公开来源归属/许可问题依然待审。

稳定 BCE/logits（不同于 source probability+epsilon）、CPU 分批与梯度累积、微小样本至少抽一行、
为两个网络分别复制并训练 CNN encoder，均为明确的 benchmark adaptations。
对含 BN/dropout 的注入网络，分批/推理模式与源全批 MLP 的行为不能宣称相同。
standalone 序列化保存两网络；runner 保存的是 classifier 推理快照，不承诺完整训练状态续跑。

原生 **OS 联合观测似然** 不做 U∪P 风险校准；显式 TS fail-loud，台账与 fit 接口同时登记，
不是将“有 os_or_ts 参数”当成已校准。`class_prior` 仅校验而不影响该源配方。
潜变量局部最优/可辨识性和适配后的数值对齐均未由合成测试解决。

实现/API/预算的[工程交付](../pu_survey/delivery/lbe_neural_integration_20261014.md)及
[机器记录](../pu_survey/data/lbe_neural_engineering_20261014.json)保留 HENG958 独立复核、
正式协议/配方/资源批准、GPU/真实数据与原始制品前置；不提前进入主榜。
