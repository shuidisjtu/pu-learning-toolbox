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

## 训练视图（OS/TS）

**原生假设为 `ts`**：变分目标的边缘项要求 $f$ 是训练总体的边缘分布，因此边缘池必须是 $D_U\cup D_P$。作者实现的 X loader 覆盖「正例与未标记」、且 P 池是 X 池的子集而非互斥划分，故本实现把完整 `X` 作边缘池与作者实现**逐集合等价**。四层证据与逐行核对见 [VPU 采样假设审计](../pu_survey/vpu_sampling_audit.md)。

- **OS 对照**：存在且可解释，但它是**协议定义的消融**（边缘池被限制到 $p(x\mid s=0)$），**不是** VPU 在其原生假设下的运行；引用 OS 结果时不得读作原生结果。
- **不是 `both`**：`both` 可辩（genuine OS 数据集定义下 `X` 本身也是 $p(x)$ 样本），但 registry 只登记 `CASE_CONTROL`，改 registry 属方法学定位变更，超出当前切片。
- **验证路径：source diagnostic，不作选模准则**（决策 D22）：上游与本实现都把验证变分风险定义在**合并边缘池** $U_{val}\cup P_{val}$ 上（本实现 `vpu.py:152-158`、`:222-231`），故此处**与上游一致**，不存在口径分歧。协议 §2.3 的「验证集和测试集保持原始 OS 协议」约束的是**参与选参或裁决**的视图；本量定位为 source diagnostic——**可以**按上游定义计算、记录并作 os/ts 对照，**不得**充当 PA/OA 选模准则。上文「PA 选模时可用 PU 验证集计算变分风险」说的是论文的方法能力；在本项目 Survey 协议下以它选模，须先改用 OS 验证视图并单独成文（D22 ③(c)）。此处**不得**引用 D17②/D19⑥——两者均不覆盖 VPU。
- **接线状态**：截至 2026-09-28，`VPUClassifier.fit` **尚未**接受 `os_or_ts`，视图接线与台账条目属 P3.1 后续切片。接线后显式 `os` 请求的语义将由「静默跑 TS」变为「真跑 OS」，属可观测变化。

## 当前实现边界

- `vpu`（别名 `variational_pu`）注册为实验性原生方法，支持稠密二维特征、CPU 和显式 CUDA；PyTorch 为可选依赖。默认两层 MLP64 适配工具箱，不是论文 UCI 的七层 MLP300 或 CIFAR CNN，因此未宣称论文精度复现。
- mini-batch 从正样本池和完整 P∪U 池分别有放回抽样；优化器 Adam `betas=(0.5,0.99)`。每轮保存分量损失、总风险、可选 PU 验证风险和更新次数；归一化层包含在权重快照中，支持逐轮 checkpoint 恢复。
- 训练、预测拒绝非有限值或错误维度；非空 `sample_weight` 明确报错。`predict_proba` 是 VPU 自身归一化分数，不是独立校准器。外推样本的归一化分数可能超过 1；概率接口会截到 `[0,1]`，而原始 `decision_function` 保留未截断值以便审计。
- 2026-09-19 在 RTX A6000（限定 0 号卡）完成 CUDA smoke；该单次技术 smoke 不替代正式多 seed GPU/资源验收。
- **与上游的既有适配缺口（与训练视图无关，2026-09-28 亲验作者代码后登记）**：上游每 20 epoch 将学习率减半并重建优化器（`vpu.py:39-41`），本实现**无此衰减**；学习率默认值本实现为 `3e-4`，上游为 `3e-5`（`run.py:11`），相差一个数量级；`max_epochs` 100 vs 50、`batch_size` 128 vs 500，且每 epoch 迭代数由数据量推导而上游固定 `val_iterations=30`。加上网络规模差异，本实现**不是作者训练脚本的复现**。
- 当前为 **P3.1 技术预集成**：采样假设已审计并裁决为原生 `ts`（2026-09-28，见审计记录），但**方法台账条目、训练视图接线、共享 backbone、图像路径、公开数值对照、多 seed GPU/资源记录及双人复核均未完成**，且不进入冻结执行矩阵。不能列入正式 pilot 或 P3.1 验收完成项。
