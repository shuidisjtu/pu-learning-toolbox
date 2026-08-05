# Method Card: InfoMax PU / PURL

## 1. 方法定位与当前状态

| 项目 | 内容 |
|---|---|
| 论文方法名 | Positive-Unlabeled Representation Learning（PURL） |
| 工具箱注册名 | `infomax_pu` |
| 方法类型 | PU 表示学习 + 下游类先验估计/PU 分类 |
| 核心原则 | 最大化输入表示与真实类别之间的 squared-loss mutual information |
| 是否需要类先验 | 表示学习阶段不需要；下游 PU 分类器通常需要 |
| 当前实现 | `InfoMaxPURepresentation` + paper-style nnPU MLP，状态 `NATIVE` |
| 复现级别 | clean-room core + paper-protocol 网络；论文完整 benchmark 待运行 |

注意：论文提出的核心对象是表示学习器，而不是单独的最终分类器。工具箱中的
`InfoMaxPUClassifier` 将 PURL、类先验估计和下游 nnPU 分类器组合成统一接口，
但必须在文档和元数据中保留这一区别。

## 2. 论文信息

| 字段 | 内容 |
|---|---|
| Paper | Information-Theoretic Representation Learning for Positive-Unlabeled Classification |
| Authors | Tomoya Sakai, Gang Niu, Masashi Sugiyama |
| Venue | Neural Computation, 33(1):244-268 |
| Year | 2021（arXiv 初稿为 2017） |
| DOI | `10.1162/neco_a_01337` |
| arXiv | `1710.05359` |
| 官方源码 | 未发现作者公开的该论文完整实现 |
| Source status | `not_found` |

## 3. 问题设定

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

## 4. Squared-Loss Mutual Information

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

## 5. PU-SMI 恒等式

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

## 6. PU-SMI 下界与经验目标

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

## 7. 线性参数模型

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

## 8. PURL 表示学习

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

论文 Algorithm 1 采用交替优化：

1. 固定 `v`，更新 `g` 以最小化 `\widehat J_PU(g o v)`；
2. 固定 `g`，更新 `v` 以最小化同一目标，从而增大 PU-SMI 下界；
3. 重复直到停止条件满足。

论文报告同时更新 `g` 和 `v` 的稳定性较差，因此工具箱接口保留
`ratio_steps` 与 `encoder_steps`，而不是把两者固定成一次联合反向传播。

## 9. 从表示学习到 PU 分类

论文实验流水线为：

```text
P/U 输入
  -> PURL 学习低维表示
  -> 在表示空间估计 class prior
  -> 使用估计的 class prior 训练 nnPU 分类器
  -> 测试分类
```

因此工具箱提供两个层次：

- `InfoMaxPURepresentation`：只实现论文核心 PURL，提供 `fit/transform`；
- `InfoMaxPUClassifier`：组合 PURL、类先验估计器和 nnPU 分类器。

用户若已知 class prior，可跳过估计步骤并直接传入 `class_prior`。

## 10. API 规格

### 10.1 表示学习器

```python
InfoMaxPURepresentation(
    representation_dim=20,
    hidden_dim=60,
    ratio_steps=4,
    encoder_steps=1,
    max_epochs=200,
    learning_rate=1e-3,
    weight_decay=5e-4,
    batch_norm=False,
    representation_activation=False,
    batch_size=None,
    gradient_noise=0.0,
    random_state=None,
    device="cpu",
)
```

核心方法：

- `fit(X, y_pu)`；
- `transform(X)`；
- `fit_transform(X, y_pu)`；
- `density_ratio(X)`；
- `get_training_history()`。

### 10.2 分类器

```python
InfoMaxPUClassifier(
    class_prior=None,
    representation_dim=20,
    representation_epochs=200,
    classifier_epochs=200,
    representation_ratio_steps=4,
    representation_encoder_steps=1,
    representation_weight_decay=5e-4,
    representation_batch_norm=False,
    representation_activation=False,
    representation_batch_size=None,
    representation_gradient_noise=0.0,
    classifier_hidden_dims=(),
    classifier_batch_norm=False,
    classifier_optimizer="adam",
    classifier_learning_rate=1e-3,
    classifier_weight_decay=0.0,
    classifier_batch_size=256,
    prior_estimator=None,
    random_state=None,
    device="cpu",
)
```

当 `class_prior=None` 时，分类器在表示空间调用项目 class-prior estimator。
paper-protocol 通过结构化配置注入 `KernelMeanPriorEstimator(variant="km1")`；默认
`prior_estimator=None` 仍保留 penL1，以维持现有 API 行为。`fit` 可接收独立
`validation_data=(X_validation, y_validation_pu)`，并使用同一已拟合 PURL 转换验证集后传给 nnPU。
默认值保持原有 clean-room 线性 head；论文网络需要显式设置
`classifier_hidden_dims=(300, 300, 300)`、`classifier_batch_norm=True`。普通数据使用
`classifier_optimizer="adam"`，文本数据使用 `"adagrad"`。

## 11. 参数与输入验证

- `y_pu` 统一归一化为 `{1, 0}`；
- P 与 U 均不能为空；
- `representation_dim`、`hidden_dim`、epoch 和 batch size 必须为正；
- 输入必须为有限二维数值数组，不支持 sparse；
- 表示阶段不接受真实负标签作为额外监督；
- `density_ratio` 输出应为有限一维数组，但神经估计不保证天然非负；
- 若需要概率输出，必须由下游模型单独校准，不能把 density ratio 直接冒充概率。

## 12. 论文实验协议

论文 benchmark 包括：

- LIBSVM：`ijcnn1`、`phishing`、`mushrooms`、`a9a`；
- MNIST；
- Fashion-MNIST；
- 20 Newsgroups。

主要设置：

- 普通数据采用 `d-60-20-1` 网络，文本采用 `d-30-10-1`；
- hidden layer 使用 ReLU 和 batch normalization；
- SGD，学习率 `0.001`；
- weight decay `0.0005`；
- gradient noise `0.01`；
- density-ratio head 更新 4 个 mini-batch，表示映射更新 1 个 mini-batch；
- `n_P=1000`，`n_U=2000`；
- 验证集 `n_P=50`，`n_U=200`；
- 下游分类器使用 `m-300-300-300-1`，所有隐藏层使用 ReLU 和 batch normalization；
- 下游图像/普通数据使用 Adam 训练 200 epoch，文本使用 AdaGrad 训练 300 epoch；
- nnPU 参数 `beta=0`、`gamma=1`；
- 结果报告 20 次试验的均值与标准误。

项目已在 `benchmarks/deep_pu/` 提供统一 runner、锁定论文配置和 3-seed 合成
clean-room 结果。该结果使用短周期 MLP，不满足本节网络、epoch、数据和 20 次试验协议，
因此 `run_manifest.json` 固定为 `paper_claim=false`。

`configs/official_data_infomax_fashion_protocol.json` 已锁定论文网络深度、BN、优化器、
gradient noise、样本数、epoch 和 20 seeds，并通过 CPU preflight。论文只写明将
MNIST/Fashion-MNIST 十类分成两组，没有给出类别编号，也没有报告 mini-batch size；当前
配置把 `[0,1,2,3,4]` 和 batch size `256` 明确标记为临时工程选择。runner 已从训练集
之外确定性划分 `50 P + 200 U` validation，并接入原生 KM1/KM2 class-prior estimator；
论文未说明使用 KM1 还是 KM2，当前 KM1 是显式临时选择。固定 epoch 协议不使用验证集早停。
这些未公开细节和尚未执行的 20-seed 全量实验使配置仍为 `paper_protocol`，不能标记为论文结果。

## 13. 测试与验收

### 13.1 数学测试

- 手工张量核对 `\widehat J_PU`；
- 增大正样本 density-ratio 输出时，目标中的正样本项下降；
- 增大未标记样本输出绝对值时，平方项上升；
- 线性模型解析解满足一阶最优条件。

### 13.2 API 测试

- `fit_transform` 输出形状正确；
- 固定 seed 时结果可复现；
- `transform` 前调用抛出 NotFittedError；
- classifier 能自动估计并记录 `class_prior_`；
- structured runner config 能构造 KM estimator，并记录真实/估计先验与绝对误差；
- train/validation 索引互斥，验证集经过同一 PURL 映射；
- registry alias `information_theoretic_pu` 可解析。

### 13.3 行为测试

- 可分合成 PU 数据上的表示优于随机表示；
- 训练历史为有限值；
- 分类器输出 `{0,1}`；
- `decision_function` 方向为分数越高越偏正类。

## 14. 局限与复现风险

- PURL 最大化的是类别依赖信息，不保证表示唯一；
- 神经密度比模型存在尺度和局部最优问题；
- 自动 class-prior 估计会把第二阶段误差传递给最终分类器；
- 论文没有提供可确认的完整官方源码，工程实现属于 clean-room；
- 论文对 MNIST/Fashion-MNIST 使用全连接网络和展平输入，不需要额外假设 CNN；
- 论文未公开图像类别分组编号、mini-batch size 或完整源码；
- 只运行 PURL 不能直接得到最终类别预测。

## 15. 参考资料

1. Sakai, Niu, Sugiyama. *Information-Theoretic Representation Learning for Positive-Unlabeled Classification*. Neural Computation 33(1), 2021.
2. DOI: <https://doi.org/10.1162/neco_a_01337>
3. arXiv: <https://arxiv.org/abs/1710.05359>
4. 作者发表列表：<https://t-sakai-kure.github.io/publications.html>
