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

训练目标是 $L_{var}+\lambda L_{reg}$，不需要数值类先验；统一 `fit(class_prior=...)` 参数仅接受并校验，不参与损失或结果元数据。MixUp 目标保留梯度，对齐锁定的作者实现。训练后按**边缘池**的最大 $\log\Phi$ 归一化（该池随训练视图变化，见下文「训练视图」节）；`decision_function` 返回 $\log(\Phi_{norm}/0.5)$，`predict` 以 0 分数为阈值，`predict_proba` 返回归一化正类分数和补数。**这些分数未经独立概率校准**，不应在假设不成立时当作可靠后验。

方法要求可靠的正样本与未标记样本、SCAR 标记机制；论文还使用正类存在纯区域的可识别性假设。若是 SAR/选择偏差或不存在纯正类区域，不应仅因“无需先验”就推断其估计有效。PA 选模时可用 PU 验证集计算变分风险，不读取真实验证标签。

## 训练视图（OS/TS）

**原生假设为 `ts`**：变分目标的边缘项要求 $f$ 是训练总体的边缘分布，因此边缘池必须是 $D_U\cup D_P$。作者实现的 X loader 覆盖「正例与未标记」、且 P 池是 X 池的子集而非互斥划分，故本实现把完整 `X` 作边缘池与作者实现**逐集合等价**。四层证据与逐行核对见 [VPU 采样假设审计](../pu_survey/admission/vpu_sampling_audit.md)。

- **OS 对照**：存在且可解释，但它是**协议定义的消融**（边缘池被限制到 $p(x\mid s=0)$），**不是** VPU 在其原生假设下的运行；引用 OS 结果时不得读作原生结果。
- **不是 `both`**：`both` 可辩（genuine OS 数据集定义下 `X` 本身也是 $p(x)$ 样本），但 registry 只登记 `CASE_CONTROL`，改 registry 属方法学定位变更，超出当前切片。
- **验证路径：source diagnostic，不作选模准则**（决策 D22）：验证风险的**函数形式**与上游一致，都是 `logsumexp(验证边缘池)/n - mean(log Φ(验证正例))`；工具箱把整份验证 PU 分区作为边缘池，P 项取其中的已标记正例（`vpu.py:256-262`、`:351-366`；stream 路径 `:462-482`）。上游 `val_x_loader` 同样是整份验证分区，但其 `val_p_loader` 由 `dataset_cifar.py:13-22` 的 `get_val_labeled` 从**全量 clean 类标签**中抽样，并非 `val_x_loader` 的正例子集；故这是**已登记的采样差异**，不能写成逐样本一致。D22 的定位不变：本量**可以**按上游目标计算、记录并作 os/ts 对照，**不得**充当 PA/OA 选模准则。协议 §2.3 的「验证集和测试集保持原始 OS 协议」约束的是**参与选参或裁决**的视图。上文「PA 选模时可用 PU 验证集计算变分风险」说的是论文的方法能力；在本项目 Survey 协议下以它选模，须先改用 OS 验证视图并单独成文（D22 ③(c)）。此处**不得**引用 D17②/D19⑥——两者均不覆盖 VPU。详见[2026-10-18 只读复核](../pu_survey/admission/vpu_d21_d22_review_20261018.md)。
- **接线状态（2026-09-28 已接线）**：`fit` 现接受 `os_or_ts: str = "ts"`——默认值取 `ts`，故裸 `fit` 保持论文目标不变。两个池都由核心层 `pu_toolbox.core.training_views` 给出的角色位置产生；边缘池在 `ts` 下还原为完整分区**自身的源顺序**，故固定 seed 下抽样轨迹与接线前一致。新增 `n_loss_unlabeled_` 记录实际进入边缘池的行数。显式 `os` 请求的语义由「静默跑 TS」变为「真跑 OS」，属可观测变化（同一变更修好了路由转发，决策 D23）。**方法台账条目已加入**（`pu_toolbox/experiment/method_ledger.json` 的 `vpu` 条目；同文件的 `fillers` 注记亦列有 `vpu`，属人工分工记录、无代码消费），视图接线之外的 P3.1 项仍未完成。

## 当前实现边界

- `vpu`（别名 `variational_pu`）注册为实验性原生方法，支持稠密二维特征及注入可训练 encoder 的四维图像、CPU 和显式 CUDA；PyTorch 为可选依赖。默认两层 MLP64 适配工具箱，注入 encoder 后仍接该头，不是论文 UCI 的七层 MLP300 或 CIFAR CNN，因此未宣称论文精度复现。
- mini-batch 从正样本池与**边缘池**分别有放回抽样；边缘池随训练视图变化（`ts` 为完整分区、`os` 为未标记行），正例池不随视图变化。优化器 Adam `betas=(0.5,0.99)`；每 epoch 的迭代数与批大小上界由**边缘池**的行数推出（`ts` 下与接线前的完整 `X` 相同）。每轮保存分量损失、总风险、可选 PU 验证风险和更新次数；归一化层包含在权重快照中，支持逐轮 checkpoint 恢复。
- 训练、预测拒绝非有限值或错误维度；非空 `sample_weight` 明确报错。`predict_proba` 是 VPU 自身归一化分数，不是独立校准器。外推样本的归一化分数可能超过 1；概率接口会截到 `[0,1]`，而原始 `decision_function` 保留未截断值以便审计。
- 2026-09-19 在 RTX A6000（限定 0 号卡）完成 CUDA smoke；2026-09-28 又通过全局 PyTorch 2.6.0+cu124 环境下的原生与 TS 特征路径单次 CUDA smoke。这些技术 smoke 不替代 frozen-lock 或正式多 seed GPU/资源验收。
- **与上游的适配差异（2026-10-15 补充）**：显式 `optimizer_schedule="source_reset_halving"` 已支持作者零基 epoch19/39…训练前减半并重建 Adam 的时序，`iterations_per_epoch=30` 可显式固定更新数；默认仍为 `constant` / 数据量推导，不静默改变已有运行。学习率默认 `3e-4` vs 作者 `3e-5`、epochs100 vs50、batch128 vs500，以及小型 MLP/有放回配对采样等差异仍保留；采用两个开关也**不是作者训练脚本或论文数值复现**。见[源调度复核](../pu_survey/admission/vpu_schedule_review_20261015.md)。
- 2026-10-15 新增 **P3.2 工程接线**：encoder deepcopy 后联合训练（对 encoder 全部参数执行 `requires_grad_(True)`，会覆盖模板中已有的冻结设置；BatchNorm 按训练模式更新运行统计），训练/验证池留 CPU、设备端逐批传输；归一化、验证、推理及权重恢复有分批契约。新路径不扁平化图像；验证使用全局 float64 归约，仍仅为 D22 source diagnostic。旧二维无 encoder 的 auto 路径保留 resident 与旧数值运算。见[交付](../pu_survey/delivery/vpu_encoder_stream_20261015.md)。
- 当前仍为 **P3.1 技术预集成**：采样假设已审计并裁决为原生 `ts`（2026-09-28，见审计记录），训练视图与方法台账已接入，但**正式共享 backbone 规格、作者图像增强/网络、公开数值对照、多 seed GPU/资源记录及双人复核均未完成**，且不进入冻结执行矩阵。不能列入正式 pilot 或 P3.1 验收完成项。
