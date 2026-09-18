# Method Card: GradPU

> 参数签名以 [API 参考](../../user/reference/api.md) 为准。本卡记录论文目标、工程实现和复现边界；当前接入不表示 Survey P3.2 验收完成。

## 论文与适用范围

| 字段 | 内容 |
|---|---|
| Paper | GradPU: Positive-Unlabeled Learning via Gradient Penalty and Positive Upweighting |
| Authors | Songmin Dai, Xiaoqiang Li, Yue Zhou, Xichen Ye, Tong Liu |
| Venue | AAAI 2023, 37(6), 7296–7303 |
| DOI | [10.1609/aaai.v37i6.25889](https://doi.org/10.1609/aaai.v37i6.25889) |
| 原文 | [AAAI 论文及 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/25889) |
| 官方代码 | 尚未核实到可用的官方实现；`source_status=not_found` |
| 类先验 | 目标函数不使用；`fit(class_prior=...)` 仅为统一接口兼容 |
| 数据 | 可靠正样本 P 与未标记样本 U；不需要负类真值 |

论文的泛化界在 SCAR 条件下推导；另在正样本选择偏差情形做了实验。注册表中的 SAR / `selection_biased` 表示该实验情形，**不代表已有一般 SAR 理论保证**。

## 核心目标

令 $r_\theta(x)$ 为网络原始输出，$g_\theta(x)=\tanh r_\theta(x)\in[-1,1]$。论文式 (5)–(7) 为：

```math
\tilde x=\lambda x_P+(1-\lambda)x_U,\quad \lambda\sim U(0,1),
\qquad
L_{GP}=\mathbb E_{\tilde x}\|\nabla_{\tilde x}r_\theta(\tilde x)\|_2^2,
```

```math
w_P(x;\beta)=1-\beta\log\frac{1+g_\theta(x)}{2},
```

```math
\widehat R_{GradPU}=
\frac{1}{|P|}\sum_{x\in P} w_P(x;\beta)|g_\theta(x)-1|
+\frac{1}{|U|}\sum_{x\in U}|g_\theta(x)+1|
+\alpha L_{GP}.
```

输入梯度按论文实现说明取 **tanh 前原始输出**；正样本权重不按权重和重新归一化。计算中用 `logsigmoid(2r)` 等价表示 $\log((1+\tanh r)/2)$，避免接近 $-1$ 时的数值下溢。每步从 P/U 批次重采样至共同大小构造插值；$\beta$ 从 0 线性升至 `beta_max`。

## 工程实现与边界

- 注册名 `gradpu`（别名 `grad_pu`），公开类 `GradPUClassifier`。PyTorch 可选依赖；支持 CPU 和指定 CUDA 设备，输入仅为稠密二维数值特征。
- 默认网络为单隐层 `MLP128`；可传入输出单个原始分数的自定义 `torch.nn.Module`。含 BatchNorm 的网络被拒绝，因为论文发现其与该梯度正则不兼容。
- `decision_function` 返回 $[-1,1]$ 分数，`predict` 在 0 阈值处产生 0/1 标签；不提供校准概率。非空 `sample_weight` 会报错，避免误以为权重参与优化。
- 保留各 epoch 的目标分量、$\beta$、更新步数和 checkpoint 回调；支持工具箱训练轨迹与权重恢复。
- 论文实验使用 MNIST/FashionMNIST 四层 MLP300 和 CIFAR10 CNN13，并调节学习率与模型选择。本实现尚**未**提供 CNN 路径、论文协议图像预处理、复现实验或论文数值宣称。默认超参数是可用起点，不是论文最优参数。

## 验证状态与后续门禁

已有公式 golden test、参数/边界、随机性、接口/注册、训练轨迹及权重往返测试；CUDA 执行测试在具备 GPU 的门禁上运行。正式 P3.2 仍需：在 P2.0b 数据/协议冻结后接入 Survey ledger 与矩阵，补齐图像骨干和论文基准（如果决定宣称复现），并完成多 seed GPU smoke、资源记录及合作者审阅。本卡不将这些待办写成已完成。
