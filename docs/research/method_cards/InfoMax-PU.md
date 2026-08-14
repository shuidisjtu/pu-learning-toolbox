# Method Card: InfoMax PU / PURL

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Information-Theoretic Representation Learning for Positive-Unlabeled Classification |
| Authors | Tomoya Sakai, Gang Niu, Masashi Sugiyama |
| Venue | Neural Computation, 33(1):244-268 |
| Year | 2021（arXiv 初稿为 2017） |
| DOI | `10.1162/neco_a_01337` |
| arXiv | `1710.05359` |
| 方法类型 | PU 表示学习 + 下游类先验估计/PU 分类 |
| 核心原则 | 最大化输入表示与真实类别之间的 squared-loss mutual information |
| 是否需要类先验 | 表示学习阶段不需要；下游 PU 分类器通常需要 |
| 官方源码 | 未发现作者公开的该论文完整实现 |
| Source status | `not_found` |

### Assumptions

输入为独立采样的正样本与未标记样本：

```math
\mathcal{X}_{\mathrm P}
=\{x_i^{\mathrm P}\}_{i=1}^{n_{\mathrm P}}
\overset{\mathrm{i.i.d.}}{\sim}p(x\mid y=+1),
```

```math
\mathcal{X}_{\mathrm U}
=\{x_k^{\mathrm U}\}_{k=1}^{n_{\mathrm U}}
\overset{\mathrm{i.i.d.}}{\sim}p(x),
```

其中

```math
p(x)=\theta_{\mathrm P}p(x\mid y=+1)
     +\theta_{\mathrm N}p(x\mid y=-1),
\qquad
\theta_{\mathrm N}=1-\theta_{\mathrm P}.
```

论文采用 case-control PU 场景。表示学习阶段只需要区分
`\mathcal X_P` 和 `\mathcal X_U` 的来源，不需要观察负样本，也不需要已知
`\theta_P`。

---

## 2. 问题设定与符号

论文提出的核心对象是表示学习器 PURL（Positive-Unlabeled Representation
Learning），目标是学到一个低维表示，使表示与真实类别之间的平方损失互信息
（SMI）最大化，再在表示空间上做类先验估计与下游 PU 分类。

| 论文符号 | 含义 |
|---|---|
| $\`\mathcal X_P\`$ / $\`\mathcal X_U\`$ | 正样本集 / 边缘分布未标记集 |
| $\`\theta_P\`$ / $\`\theta_N\`$ | 正类 / 负类先验，$\`\theta_N=1-\theta_P\`$ |
| $\`\mathrm{SMI}(X,Y)\`$ | 平方损失互信息（Pearson divergence 形式） |
| $\`r^*(x)\`$ | 真实密度比 $\`p(x\mid y=+1)/p(x)\`$ |
| $\`w(x)\`$ | 密度比模型，最小化 $\`J_{\mathrm{PU}}\`$ 以最大化 PU-SMI 下界 |
| $\`\phi(x)\`$ / $\`\beta\`$ | 线性参数模型的特征映射 / 系数 |
| $\`\lambda_{\mathrm{PU}}\`$ | 密度比目标的 L2 正则系数 |
| $\`v(x)\`$ / $\`g(v)\`$ | 表示映射 / 密度比头，$\`w(x)=g(v(x))\`$ |

---

## 3. 核心公式

### 3.1 Squared-Loss Mutual Information

平方损失互信息定义为：

```math
\mathrm{SMI}(X,Y)
=\sum_{y\in\{-1,+1\}}\frac{p(y)}{2}
\int
\left(
\frac{p(x,y)}{p(x)p(y)}-1
\right)^2p(x)\,\mathrm dx.
```

SMI 是联合分布 `p(x,y)` 与独立分布 `p(x)p(y)` 之间的 Pearson divergence：

- `SMI >= 0`；
- 当且仅当 `X` 与 `Y` 独立时 `SMI = 0`；
- 与直接估计 KL 型互信息相比，其经验估计可以转化为最小二乘密度比问题。

### 3.2 PU-SMI 恒等式

论文 Theorem 1 将 SMI 改写为只涉及正类条件分布和边缘分布的形式：

```math
\mathrm{PU\text{-}SMI}
=\frac{\theta_{\mathrm P}}{2\theta_{\mathrm N}}
\int
\left(
\frac{p(x\mid y=+1)}{p(x)}-1
\right)^2p(x)\,\mathrm dx.
```

并证明：

```math
\mathrm{PU\text{-}SMI}=\mathrm{SMI}.
```

令真实密度比为：

```math
r^*(x)=\frac{p(x\mid y=+1)}{p(x)}.
```

直接分别估计分子和分母再取比值会放大密度估计误差，因此论文采用直接密度比估计。

### 3.3 PU-SMI 下界与经验目标

对任意函数 `w(x)`，论文 Theorem 2 给出：

```math
\mathrm{PU\text{-}SMI}
\ge
\frac{\theta_{\mathrm P}}{\theta_{\mathrm N}}
\left(-J_{\mathrm{PU}}(w)-\frac12\right),
```

其中：

```math
J_{\mathrm{PU}}(w)
=\frac12\mathbb E_{x\sim p(x)}[w(x)^2]
-\mathbb E_{x\sim p(x\mid y=+1)}[w(x)].
```

等号当且仅当：

```math
w(x)=r^*(x).
```

经验目标为：

```math
\widehat J_{\mathrm{PU}}(w)
=
\frac{1}{2n_{\mathrm U}}
\sum_{k=1}^{n_{\mathrm U}}w(x_k^{\mathrm U})^2
-
\frac{1}{n_{\mathrm P}}
\sum_{i=1}^{n_{\mathrm P}}w(x_i^{\mathrm P}).
```

训练通过最小化 `\widehat J_PU` 来最大化 PU-SMI 下界。未知比例
`\theta_P/\theta_N` 只是正的乘法常数，因此不影响最优 `w` 或表示映射的学习。

### 3.4 线性参数模型

若：

```math
w(x)=\beta^\top\phi(x),
```

加入 L2 正则后的目标为：

```math
\min_\beta
\frac12\beta^\top\widehat H_{\mathrm U}\beta
-\beta^\top\widehat h_{\mathrm P}
+\frac{\lambda_{\mathrm{PU}}}{2}\|\beta\|_2^2,
```

其中：

```math
\widehat H_{\mathrm U}
=\frac1{n_{\mathrm U}}\sum_k\phi(x_k^{\mathrm U})\phi(x_k^{\mathrm U})^\top,
\qquad
\widehat h_{\mathrm P}
=\frac1{n_{\mathrm P}}\sum_i\phi(x_i^{\mathrm P}).
```

解析解为：

```math
\widehat\beta
=
(\widehat H_{\mathrm U}+\lambda_{\mathrm{PU}}I)^{-1}
\widehat h_{\mathrm P}.
```

论文证明，在有界且线性独立的 basis 等条件下：

```math
\|\widehat\beta-\beta^*\|_2
=
O_p(n_{\mathrm P}^{-1/2}+n_{\mathrm U}^{-1/2}),
```

PU-SMI 估计误差也具有相同的最优参数收敛阶。

### 3.5 PURL 表示学习

将密度比模型分解为：

```math
w(x)=g(v(x)),
```

其中：

- `v: R^d -> R^m` 为表示映射，`m < d`；
- `g: R^m -> R` 为密度比头。

若表示满足：

```math
p(y\mid x)=p(y\mid v(x)),
```

则 `v(x)` 是关于类别的充分表示。根据 SMI 数据处理性质，最大化表示与类别之间的
SMI 可以寻找保留类别信息的低维表示。

---

## 4. 算法概要

### 4.1 PURL 交替优化

论文 Algorithm 1 采用交替优化：

1. 固定 `v`，更新 `g` 以最小化 `\widehat J_PU(g o v)`；
2. 固定 `g`，更新 `v` 以最小化同一目标，从而增大 PU-SMI 下界；
3. 重复直到停止条件满足。

### 4.2 论文实验流水线（从表示学习到 PU 分类）

```text
P/U 输入
  -> PURL 学习低维表示
  -> 在表示空间估计 class prior
  -> 使用估计的 class prior 训练 nnPU 分类器
  -> 测试分类
```

用户若已知 class prior，可跳过估计步骤并直接传入 `class_prior`。

---

## 5. 论文边界

- 论文提出的核心对象是表示学习器，而不是单独的最终分类器。工具箱中的
  `InfoMaxPUClassifier` 将 PURL、类先验估计和下游 nnPU 分类器组合成统一接口，
  但必须在文档和元数据中保留这一区别。
- PURL 最大化的是类别依赖信息，不保证表示唯一。
- 神经密度比模型存在尺度和局部最优问题。
- 自动 class-prior 估计会把第二阶段误差传递给最终分类器。
- 表示学习阶段只接受 `P`/`U` 数据，不接受真实负标签作为额外监督（case-control 设定）。
- `density_ratio` 输出是密度比估计，不是概率；需要概率输出必须由下游模型单独校准，
  不能把 density ratio 直接冒充概率。
- 只运行 PURL 不能直接得到最终类别预测，必须接下游分类器。
- 论文对 MNIST/Fashion-MNIST 使用全连接网络和展平输入，不需要额外假设 CNN。

---

## 6. 实现注记

> 见 ADR-0014 #3（Fashion-MNIST 60/60 trials：ROC-AUC 0.8547±0.0819 / 0.6313±0.3077 /
> 0.3340±0.2873，AUC<0.5 的 seed 数 0/20、7/20、15/20；KM1 类先验估计不随受控先验变化，
> 约 0.875——诊断性负结果）

- **诊断性负结果（一句话结论）**：在 Fashion-MNIST 受控协议下，KM1 类先验估计在
  `pi=0.3/0.5/0.7` 三组均约为 `0.875`、未跟随受控 U 先验变化——该表示上的类先验估计
  对先验变化不敏感（具体数字以 ADR-0014 #3 与 benchmarks REPORT 为准）。
- **分步优化**：论文报告同时更新 `g` 和 `v` 的稳定性较差，工具箱接口保留
  `ratio_steps` 与 `encoder_steps` 分步，而不是把两者固定成一次联合反向传播。
- **paper-protocol 的类先验估计器**：paper-protocol 通过结构化配置注入
  `KernelMeanPriorEstimator(variant="km1")`；默认 `prior_estimator=None` 仍保留 penL1，
  以维持现有 API 行为。论文未说明使用 KM1 还是 KM2，当前 KM1 是显式临时选择。

结果记录（benchmark 数字与 runner 状态以 benchmarks 产物为准）：

> 结果见 `benchmarks/deep_pu/results/infomax_fashion_protocol_matrix/REPORT.md`
> （Fashion-MNIST 60/60 trials 协议矩阵）；preflight 记录见
> `benchmarks/deep_pu/results/infomax_fashion_protocol_preflight/` 与
> `benchmarks/deep_pu/official_preflight/preflight.json`

---

## 7. 论文实验参考

论文 benchmark 包括：

- LIBSVM：`ijcnn1`、`phishing`、`mushrooms`、`a9a`；
- MNIST；
- Fashion-MNIST；
- 20 Newsgroups。

主要设置：

| 项目 | 论文设置 |
|---|---|
| 网络结构 | 普通数据 `d-60-20-1`，文本 `d-30-10-1` |
| 隐藏层 | ReLU 和 batch normalization |
| 优化器 | SGD，学习率 `0.001` |
| 正则化 | weight decay `0.0005`；gradient noise `0.01` |
| 更新节奏 | density-ratio head 更新 4 个 mini-batch，表示映射更新 1 个 mini-batch |
| 样本数 | `n_P=1000`，`n_U=2000`；验证集 `n_P=50`，`n_U=200` |
| 下游分类器 | `m-300-300-300-1`，所有隐藏层 ReLU 和 batch normalization |
| 下游训练 | 图像/普通数据 Adam 200 epoch，文本 AdaGrad 300 epoch |
| nnPU 参数 | `beta=0`、`gamma=1` |
| 试验次数 | 结果报告 20 次试验的均值与标准误 |

---

## 8. 源码状态与复现风险

| 字段 | 内容 |
|---|---|
| Source status | `not_found`：未发现作者公开的完整实现，工程实现属 clean-room |
| 实现状态 | `InfoMaxPURepresentation` + paper-style nnPU MLP，状态 `NATIVE`；论文完整 benchmark 待运行 |
| 复现差距 | 论文未公开 MNIST/Fashion-MNIST 类别分组编号与 mini-batch size；类别分组 `[0,1,2,3,4]`、batch size `256`、KM 变体（KM1）与 test-prior 规则均为显式临时工程选择；固定 epoch 协议不使用验证集早停 |
| 结果声明 | `run_manifest.json` 固定 `paper_claim=false`，当前结果不能标记为论文结果 |
| 主要风险 | 自动先验估计的误差会传播给下游分类器；clean-room 表示与论文网络结果的比较需谨慎 |

参考资料：

1. Sakai, Niu, Sugiyama. *Information-Theoretic Representation Learning for Positive-Unlabeled Classification*. Neural Computation 33(1), 2021.
2. DOI: <https://doi.org/10.1162/neco_a_01337>
3. arXiv: <https://arxiv.org/abs/1710.05359>
4. 作者发表列表：<https://t-sakai-kure.github.io/publications.html>
