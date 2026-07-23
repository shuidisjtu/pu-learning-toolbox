# Method Card: LLSVM（Large-Margin Label-Calibrated SVM）

## 1. 待办与注意

### 1.1 待办

- [x] 实现 `LLSVMClassifier`：只接受正样本与未标记样本（P/U），以式（9）的平滑非凸目标训练线性判别函数。
- [x] 将类先验 $`\pi=P(y=+1\mid x\in U)`$ 显式作为 `class_prior` 输入，或接入现有 `penL1` 估计器；禁止静默设为 `0.5`。$`t=2\pi-1`$。
- [x] 实现 minibatch SGD、训练历史及数值稳定的损失计算；优化器、学习率、epoch 和 batch size 必须可配置。
- [x] 完成 P/U 标签协议、输入校验、目标/梯度单元测试，以及合成双高斯数据端到端测试。
- [ ] 建立 paper-like benchmark；论文的 OpenML/CIFAR/GermanCredit 比较不能直接视为本项目复现结果。

### 1.2 注意

- 方法只适用于 P/U：$`X_P`$ 的标签为 $`+1`$，$`X_U`$ 为未标记。它不需要已知负样本，也不需要 propensity。
- 核心归纳偏置是：P 与隐藏负类在特征空间中形成可分簇，未标记样本应远离决策边界。若类别高度重叠、特征不具备聚类/间隔结构，hat 项可能造成过度自信预测。
- 仅使用正样本 hinge 项和未标记 hat 项会把所有训练点推向正类；标签校准项是防止该退化的必要部分，不能省略。
- 训练目标非凸；SGD 只保证得到局部解。需要固定随机种子、保存最优验证 checkpoint，并报告多随机种子方差。
- 论文用 $`\operatorname{penL1}`$ 估计类先验；先验误差会直接影响 $`t`$ 和边界位置，论文的泛化界不覆盖该估计误差。
- **[项目适配]** LLSVM 已完成 native 实现（2026-07-23）。`penL1` 已实现并作为默认类先验估计器。官方代码已审阅，实现以代码为准（见 §4.3）。

---

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Large-Margin Label-Calibrated Support Vector Machines for Positive and Unlabeled Learning |
| Authors | Chen Gong, Tongliang Liu, Jian Yang, Dacheng Tao |
| Venue | IEEE Transactions on Neural Networks and Learning Systems, 30(11), 3471-3482 |
| Year | 2019 |
| DOI | `10.1109/TNNLS.2019.2892403` |
| Family | `risk_estimation` / label-calibrated PU classifier |
| Setting | 二分类 P/U；线性实值判别函数 $`f_\omega(x)=\omega^\top \bar{x}`$ |
| Requires class prior | `True`：未标记集中的正类先验 $`\pi`$，用于 $`t=2\pi-1`$ |
| Requires propensity | `False` |
| Requires negative samples | `False` |
| GPU required | `False`（SGD 张量实现可支持 GPU） |
| Source status | `official_exact`：论文 + 官方 MATLAB 代码包已审阅；**实现以代码为准** |

### Assumptions

```math
X_P\sim p(x\mid y=+1),\qquad
X_U\sim p(x)=\pi p(x\mid y=+1)+(1-\pi)p(x\mid y=-1).
```

其中 $`0<\pi<1`$。论文的目标还隐含 P 与潜在负类存在可利用的低密度间隔/聚类结构；这不是标准无偏 PU 风险估计的分布无关假设。

---

## 3. 符号与记号

| 符号 | 含义 | 开发侧对应 |
|---|---|---|
| $`P,U`$ | 正样本集、未标记集 | `X_pos`, `X_unlabeled` |
| $`p,u,n=p+u`$ | P、U、总样本数 | `n_positive`, `n_unlabeled` |
| $`\bar{x}`$ | 末尾增广常数 1 的特征 | 显式截距或增广设计矩阵 |
| $`f_\omega(x)=\omega^\top\bar{x}`$ | 实值 score；符号决定类别 | `decision_function(X)` |
| $`\alpha`$ | 正样本平方 hinge 权重 | `positive_weight` |
| $`\beta`$ | 未标记 Gaussian-like hat 权重 | `unlabeled_margin_weight` |
| $`\gamma`$ | 标签校准权重 | `calibration_weight` |
| $`\pi`$ | U 中正类先验 | `class_prior` |
| $`t=2\pi-1`$ | U 的平均标签上界 | `calibration_target_` |
| $`A`$ | 压缩函数缩放参数（论文取 2，代码取 10） | `squash_scale` |
| $`\Phi_A(z)=\frac{A}{\pi}\arctan z`$ | 将 score 压缩到 $`[-A/2, A/2]`$；论文写为 $`\frac{2}{\pi}\arctan z`$ | `_squash(score)` |

---

## 4. 核心公式

### 4.1 原始建模动机

论文先以正样本的 hinge、U 上的 hat loss 和 U 平均软标签约束建模。hat loss 为

```math
h(z)=\max(1-|z|,0),
```

它惩罚 $`z\in[-1,1]`$，推动未标记点远离边界。校准约束是

```math
\frac1u\sum_{x\in U}\Phi(f_\omega(x))\le t+\eta,\qquad \eta\ge0.
```

这解释了 $`t`$：若 U 中正类比例为 $`\pi`$，真实标签均值为 $`\pi-(1-\pi)=2\pi-1`$。原始形式含非光滑、跨 U 样本耦合的项，**不应直接作为 minibatch 实现目标**。

### 4.2 实际训练目标（论文式 9）

用平方 hinge、Gaussian-like 近似和 Jensen 上界后，最小化：

```math
J(\omega)=\frac12\lVert\omega\rVert_2^2
+\frac{\alpha}{p}\sum_{x\in P}[\max(1-f_\omega(x),0)]^2
+\frac{\beta}{u}\sum_{x\in U}\exp[-3f_\omega(x)^2]
+\frac{\gamma}{u}\sum_{x\in U}[\max(\Phi(f_\omega(x))-t,0)]^2.
```

- 第一项：$`\ell_2`$ 正则；若使用独立 `intercept`，建议不正则化截距。**[项目适配]**
- 第二项：让标记正样本 score 至少为 1。
- 第三项：在 score 为 0 时取最大值 1，促使 U 离开边界；它使目标非凸。
- 第四项：逐个约束 U 的软标签均值上界的可分上界，校准“全判正”的偏置。

对 U，第三和第四项的梯度（未含正则）分别为：

```math
\nabla_\omega\frac{\beta}{u}e^{-3f^2}
=-\frac{6\beta}{u}f e^{-3f^2}\bar{x},
```

```math
\nabla_\omega\frac{\gamma}{u}[\max(\Phi(f)-t,0)]^2
=\frac{4\gamma}{\pi u(1+f^2)}\max(\Phi(f)-t,0)\bar{x}.
```

实现时优先让自动微分计算梯度，并用上述式子做小批量数值梯度校验。

### 4.3 论文 vs 官方代码偏差（实现以代码为准）

官方 MATLAB 代码包（`LLSVM_TNNLS19.rar`）中的目标函数和梯度与论文式 (9) 存在以下差异。代码是作者实际运行实验的版本，**本项目以代码为准实现**。

| # | 项目 | 论文 (§4.2) | 官方代码 | 说明 |
|---|---|---|---|---|
| 1 | 指数系数 | $`\exp[-3f^2]`$ | $`\exp[-5f^2]`$ | 梯度中 $`-6f`$ → $`-10f`$；更窄的 hat，对边界附近惩罚更集中 |
| 2 | 压缩函数 | $`\Phi(z)=\frac{2}{\pi}\arctan z`$（固定） | $`\Phi_A(z)=\frac{A}{\pi}\arctan z`$，$`A=10`$ | 引入可配缩放参数 $`A`$，校准项和梯度均随之变化 |
| 3 | P 项归一化 | $`\frac{\alpha}{p}\sum_P`$ | $`\alpha\sum_P`$（不除 $`p`$） | 等价于将 $`\alpha`$ 吸收了 $`p`$ 倍；跨数据集时需注意尺度 |
| 4 | U 指数项归一化 | $`\frac{\beta}{u}\sum_U`$ | $`\beta\sum_U`$（不除 $`u`$） | 同理 |
| 5 | 增广常数 | 1 | 10 | `fit_intercept` 对应的偏置列值 |

代码的实际训练目标：

```math
J_{\text{code}}(\omega)
=\alpha\sum_{x\in P}[\max(1-f_\omega(x),0)]^2
+\beta\sum_{x\in U}\exp[-5f_\omega(x)^2]
+\frac{\gamma}{u}\sum_{x\in U}\!\left[\max\!\left(\tfrac{A}{\pi}\arctan f_\omega(x)-t,\,0\right)\right]^2
```

对应的 U 项梯度（实现目标）：

```math
\nabla_\omega\,\beta\,e^{-5f^2}=-10\beta\,f\,e^{-5f^2}\,\bar{x}
```

```math
\nabla_\omega\,\frac{\gamma}{u}\!\left[\max\!\left(\Phi_A(f)-t,0\right)\right]^2
=\frac{2A\gamma}{\pi\,u\,(1+f^2)}\max\!\left(\Phi_A(f)-t,0\right)\bar{x}
```

> 正则化项：代码在 SGD 梯度中加 $`\omega\times\text{BatchSize}`$，`ComputeCost` 不含正则项。建议实现时采用标准 $`\frac{\lambda}{2}\lVert\omega\rVert^2`$ 正则并在 cost 中统一计算，以便验证收敛。

### 4.4 类先验与阈值

论文以 $`\operatorname{penL1}`$（du Plessis, Niu, Sugiyama, 2015）在网格 $`\{0.05,0.10,\ldots,0.95\}`$ 上估计 $`\pi`$，随后设 $`t=2\pi-1`$。

**[项目适配]** 若当前 `penL1` 支持连续优化或不同候选网格，应复用其 API；LLSVM 只消费最终 $`class_prior_`$，不复制一套先验估计实现。允许用户传入可信先验以跳过估计。

---

## 5. 算法概要

1. 校验 P/U 均非空，标签仅为 $`\{+1,0\}`$，且 $`0<\pi<1`$。
2. 若未提供 `class_prior`，调用先验估计器在 P/U 上估计 $`\pi`$；令 $`t=2\pi-1`$。
3. 初始化线性参数（含或不含截距）；固定随机种子后打乱训练索引。
4. 对每个 epoch 和 minibatch，按式（9）计算 P 与 U 项的批量估计、反向传播并更新参数。
5. 在验证集按任务指标（首选 AUC；有可靠阈值标注时可用 F1/accuracy）选择 $`\alpha,\beta,\gamma`$、学习率和早停 checkpoint。
6. 保存 $`class_prior_`$、$`calibration_target_`$、最终目标分量和训练历史，提供可诊断输出。

论文固定步长 $`\tau=0.01`$、$`N=40`$ 个 minibatch；官方代码实际使用步长 $`5\times10^{-6}`$、$`N=20`$ 个 minibatch、$`3000`$ epochs。**实现以代码参数为默认值**，论文值仅作参考。论文仅在训练开始时 shuffle 一次；工程实现应默认每 epoch shuffle，并使其可配置。**[项目适配]**

---

## 6. API 接口与项目落点（拟议）

### 6.1 公共 API 与数据协议

| API / 决策点 | 约定 |
|---|---|
| `fit(X, y, *, class_prior=None, sample_weight=None)` | sklearn 风格入口；$`y\in\{+1,0\}`$。 |
| `class_prior` | 优先使用显式值；为 `None` 时接入 `penL1`。不可估计或估计失败时抛出明确异常。 |
| `decision_function(X)` | 返回 $`f_\omega(x)`$，不返回压缩后的 $`\Phi(f)`$。 |
| `predict(X)` | 默认以 score 0 为阈值输出 $`\{-1,+1\}`$；阈值校准属于独立决策，不能与训练期 $`t`$ 混淆。 |
| `predict_proba(X)` | 不应直接由 score 或 $`\Phi`$ 宣称概率；仅在另加后校准器时提供。 |
| 稀疏支持 | **[项目适配]** 线性 score 可支持 CSR；实现前确认批量切片、优化器和公共 validation 的稀疏契约。 |

### 6.2 构造参数（拟议）

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `class_prior` | `None` | U 中正类先验；`None` 时估计。 |
| `alpha` | `2.0` | 正样本平方 hinge 权重，$`>0`$。（论文用 1，代码用 2） |
| `beta` | `1.0` | U 的间隔/hat 近似权重，$`>0`$。 |
| `gamma` | `10.0` | 标签校准权重，$`>0`$；必须经验证集选择。（论文用 100，代码用 10） |
| `squash_scale` | `10.0` | 压缩函数缩放 $`A`$。（论文用 2，代码用 10） |
| `learning_rate` | `5e-6` | SGD 初始步长。（论文称 0.01，代码用 5e-6） |
| `max_epochs` | `3000` | 最大训练 epoch。（论文未明确，代码用 3000） |
| `batch_size` | `None` | 批量大小；`None` 时分为 20 批（代码默认）。 |
| `fit_intercept` | `True` | 是否拟合截距。 |
| `shuffle` | `True` | 每 epoch 是否打乱。 |
| `random_state` | `None` | 初始化与 shuffle 随机种子。 |
| `prior_estimator` | `"penl1"` | 仅在未传 `class_prior` 时使用。**[项目适配]** |

### 6.3 拟合属性与模块

| 项目 | 建议 |
|---|---|
| 拟合属性 | `coef_`, `intercept_`, `class_prior_`, `calibration_target_`, `n_positive_`, `n_unlabeled_`, `loss_history_`, `objective_components_`。 |
| estimator | `pu_toolbox/estimators/classic/llsvm.py`：`LLSVMClassifier`。**[项目适配：建议路径]** |
| loss | `pu_toolbox/losses/llsvm.py`：可测试的 objective components 与 stable `atan`/`exp` 计算。**[项目适配：建议路径]** |
| registry | `llsvm`、family=`RISK_ESTIMATION`、scenario=`CASE_CONTROL`、`requires_class_prior=True` |
| 状态 | `implementation_status=NATIVE` / `source_status=OFFICIAL_EXACT` |

---

## 7. 验证与复现要点

- 目标分解：对固定 score 比较实现的四个分量与式（9）；确认 U 的校准项是**样本均值的平方上界**（逐样本平方的均值），不是原始均值的平方。
- 梯度：小数据、双精度下对 P/U 批量做有限差分；分别覆盖 hinge 激活/非激活、$`f=0`$、校准激活/非激活。
- 数据协议：拒绝缺 P、缺 U、非法标签、非有限特征、$`\pi\notin(0,1)`$；验证显式先验覆盖估计器输出。
- 行为：双高斯 P/U 合成集上，含校准项的边界不应退化为“全部为正”；去掉校准项仅作为受控消融测试。
- 可复现性：相同 `random_state` 应得到一致参数/历史；不同种子至少报告均值与标准差。
- 基准：复用相同 PU 划分与先验估计协议，比较 WSVM、DH、nnPU、LPM 等；不要把论文表格数值复制为项目结果。

---

## 8. 论文结论的可用边界

论文给出在特征范数有界时的 margin 泛化界：界随 P、U 样本量增大而降低，且依赖 $`\alpha+\beta+\gamma t^2`$、margin $`\rho`$ 和样本数。它支持“更多 P/U 数据有助于泛化”的定性判断，但不提供超参数默认值、全局最优保证或先验误差保证。

论文实验在 synthetic DoubleGaussian、4 个 OpenML 数据集、CIFAR cat-vs-dog 特征和 GermanCredit 上，大多数设置优于其比较基线；作者也明确指出 $`\alpha,\beta,\gamma`$ 敏感，仍需调参。开发侧应将这些结论作为 benchmark 假设，而非性能承诺。
