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

- 注册名 `gradpu`（别名 `grad_pu`），公开类 `GradPUClassifier`。支持 CPU/CUDA、二维特征及显式无 BatchNorm encoder 的四维图像。
- 默认网络为单隐层 `MLP128`；可传入输出单个原始分数的自定义 `torch.nn.Module`。含 BatchNorm 的网络被拒绝，因为论文发现其与该梯度正则不兼容。
- `decision_function` 返回 $[-1,1]$ 分数，`predict` 在 0 阈值处产生 0/1 标签；不提供校准概率。非空 `sample_weight` 会报错，避免误以为权重参与优化。
- 保留各 epoch 的目标分量、$\beta$、更新步数和 checkpoint 回调；支持工具箱训练轨迹与权重恢复。
- 论文实验使用 MNIST/FashionMNIST 四层 MLP300 和去 BN 的 CIFAR CNN13。本实现已有无 BN CNN 工程路径，但论文预处理、模型细节、调度与数值复现仍未完成。

### 原生图像工程路径

`encoder` 经 deepcopy 解冻训练，输出有限二维特征，后接默认 MLP 或用户原始分数头。
四维图像缺 encoder 时拒绝；含 BatchNorm 时仍明确拒绝。
显式骨干 `cnn13_no_bn` 是工具箱现有13卷积布局的独立无 BN 变体，**不改默认 cnn13**，
不是作者完整网络复现。公开 pipeline 可选此名称，报告记录实际骨干。
插值与输入梯度发生于原始 NCHW 图像，不在冻结特征上代算；create_graph 保留二阶反传。
P/U 数据留 CPU、优化批次上卡；预测分批 eval 并恢复模式，完整空间 shape 与空输入校验。
公式解析卷积例子、真实梯度图、参数更新、种子/clone/pickle/epoch 回放及 GPU 测试见
`test_grad_pu_cnn.py`；公共流水线、不同折 encoder 隔离有对应集成测试。
正式资源/选模/来源审阅仍未完成，默认共享 BN 骨干不会被静默替换。

## 验证状态与后续门禁

2026-10-05 补充[来源/实验流程核查](../pu_survey/gradpu_source_review_20261005.md)：
印刷页 7300 明确 CNN13 去 BatchNorm，原训练集划出 P/500 validation 后剩余作为 U。
理论 U~p(x) 与发布实验分开登记；TS 校准不是原实验协议直接重放。
作者代码仍未确认，补充材料和正式无 BatchNorm 图像规格继续待办；工程路径见上节。

已有公式 golden test、参数/边界、随机性、接口/注册、训练轨迹及权重往返测试。方法台账已登记；`fit(os_or_ts="ts")` 的 U 风险及输入梯度插值中的边缘角色逐批取 `U∪P`，正例权重、超参数与训练先验口径不变；逐 run 实际视图以 manifest 为准。2026-09-28 全局 PyTorch 2.6.0+cu124 环境下的 A6000 原有 GPU 测试和 TS 特征路径单次 CUDA smoke 通过；不替代 frozen-lock、多 seed 或正式资源验收。正式 P3.2 仍需：冻结矩阵之外的候选规格、原生图像骨干与论文基准（若决定宣称复现）、多 seed 资源记录及合作者审阅。
