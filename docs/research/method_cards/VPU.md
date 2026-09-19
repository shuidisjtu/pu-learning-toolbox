# Method Card: Variational PU (VPU)

> 参数签名与行为契约以 [API 参考](../../user/reference/api.md) 为准。本卡区分论文目标、工具箱适配和 Survey 验收状态。

## 论文与来源

| 字段 | 内容 |
|---|---|
| Paper | A Variational Approach for Learning from Positive and Unlabeled Data |
| Authors | Hui Chen, Fangqing Liu, Yin Wang, Liyue Zhao, Hao Wu |
| Venue | NeurIPS 2020 |
| 原文 | [NeurIPS 论文](https://papers.nips.cc/paper/2020/file/aa0d2a804a3510442f2fd40f2100b054-Paper.pdf) |
| 作者代码 | [HC-Feynman/vpu](https://github.com/HC-Feynman/vpu)，审阅锁定 `603bcc1e628795f57a5ac87e5b0b977273b7cf91`，MIT |
| 来源状态 | `official_related`：公式与关键训练逻辑对照作者代码，网络和实验协议为工具箱适配 |

## 目标、假设与输出

用非负函数 $\Phi_\theta(x)$ 近似正类后验。论文式 (6) 的变分目标为

```math
L_{var}=\log\mathbb E_{x\sim f}[\Phi_\theta(x)]-
\mathbb E_{x\sim f_P}[\log\Phi_\theta(x)].
```

其中边缘分布 $f$ 的训练批次来自完整 **P∪U 池**，而非只用 U；这与作者 `vpu.py` 的 `x_loader` 及 `dataset/dataset_cifar.py` 的池定义一致。论文式 (8)–(9) 的 MixUp 正则为

```math
\tilde x=\gamma x_P+(1-\gamma)x_f,\quad
\tilde\Phi=\gamma+(1-\gamma)\Phi_\theta(x_f),\quad
L_{reg}=\mathbb E[(\log\tilde\Phi-\log\Phi_\theta(\tilde x))^2],
\quad \gamma\sim\mathrm{Beta}(\alpha,\alpha).
```

训练目标是 $L_{var}+\lambda L_{reg}$，不需要数值类先验；统一 `fit(class_prior=...)` 参数仅接受并校验，不参与损失或结果元数据。MixUp 目标保留梯度，对齐锁定的作者实现。训练后按 P∪U 训练池的最大 $\log\Phi$ 归一化；`decision_function` 返回 $\log(\Phi_{norm}/0.5)$，`predict` 以 0 分数为阈值，`predict_proba` 返回归一化正类分数和补数。**这些分数未经独立概率校准**，不应在假设不成立时当作可靠后验。

方法要求可靠的正样本与未标记样本、SCAR 标记机制；论文还使用正类存在纯区域的可识别性假设。若是 SAR/选择偏差或不存在纯正类区域，不应仅因“无需先验”就推断其估计有效。PA 选模时可用 PU 验证集计算变分风险，不读取真实验证标签。

## 当前实现边界

- `vpu`（别名 `variational_pu`）注册为实验性原生方法，支持稠密二维特征、CPU 和显式 CUDA；PyTorch 为可选依赖。默认两层 MLP64 适配工具箱，不是论文 UCI 的七层 MLP300 或 CIFAR CNN，因此未宣称论文精度复现。
- mini-batch 从正样本池和完整 P∪U 池分别有放回抽样；优化器 Adam `betas=(0.5,0.99)`。每轮保存分量损失、总风险、可选 PU 验证风险和更新次数；归一化层包含在权重快照中，支持逐轮 checkpoint 恢复。
- 训练、预测拒绝非有限值或错误维度；非空 `sample_weight` 明确报错。`predict_proba` 是 VPU 自身归一化分数，不是独立校准器。外推样本的归一化分数可能超过 1；概率接口会截到 `[0,1]`，而原始 `decision_function` 保留未截断值以便审计。
- 当前为 **P3.1 技术预集成**：未进入 Survey 方法台账与冻结执行矩阵，也未完成共享 backbone、图像路径、公开数值对照、多 seed GPU/资源记录及双人复核。不能列入正式 pilot 或 P3.1 验收完成项。
