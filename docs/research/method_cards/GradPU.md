# Method Card: GradPU

> 参数签名以 [API 参考](../../user/reference/api.md) 为准。本卡记录论文目标、工程实现和复现边界；当前接入不表示 Survey P3.2 验收完成。

2026-11-04 D56 候选修正：`build_wconpu_backbone` 对 `cnn13_no_bn` 新增显式
`initialization="relu_gain"`（默认仍 `pytorch_default`）；将同 seed 的原 Conv 权重乘
sqrt(6)，从 PyTorch 默认初始化方差改为 ReLU fan-in 方差，不增加 RNG 抽样，不增加 BN。
原因是原 13 层 no-BN 堆叠的弱信号与 Adam 耦合 L2 项共同造成退化；这不是已核实的
原论文初始化。候选工厂 `gradpu_initialization` 记录不同初始化身份/权重摘要，冻结
runner 不传该参数，旧默认、协议、注册表与训练配方不变。初始化修正本身不保证稳定：
ADNI 短测提示原 lr=1e-3 仍可能使 tanh 饱和，lr=1e-4 仅为待验证候选。

编码器退化问题仍未解决，完整预算稳定性与新配方仍待验证。
GradPU 的后续实验暂时搁置，历史诊断/恢复脚本已退出工具箱。

执行策略边界：PR #162 的 split/chunk 默认模型为确定性的逐样本网络，
但原校验只排除了 BatchNorm。注入 Dropout 的 CPU 两轮反例中，joint 与 chunk=3
的参数最大差为 0.0038243，不能套用确定性模型的等价测试结论。因此启用 GP 分块且
alpha>0 时，现明确拒绝非零 Dropout、随机 RReLU 和带 dropout 的 RNN/attention；
不关闭用户模型的随机层、不自动改配方，可显式取消分块。alpha=0、未分块或零 Dropout
保持可用。验证覆盖拒绝路径与确定性 MLP 参数/轨迹回归；自定义随机操作、状态更新或
跨样本耦合不能靠模块枚举完全检测，调用者仍须验证逐样本确定性。

## 论文与适用范围

| 字段 | 内容 |
|---|---|
| Paper | GradPU: Positive-Unlabeled Learning via Gradient Penalty and Positive Upweighting |
| Authors | Songmin Dai, Xiaoqiang Li, Yue Zhou, Xichen Ye, Tong Liu |
| Venue | AAAI 2023, 37(6), 7296–7303 |
| DOI | [10.1609/aaai.v37i6.25889](https://doi.org/10.1609/aaai.v37i6.25889) |
| 原文 | [AAAI 论文及 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/25889) |
| 官方代码 | 截至 2026-10-19，AAAI 正式论文页只列 PDF，按标题/作者检索未确认作者仓库；`source_status=not_found` |
| 类先验 | 目标函数不使用；`fit(class_prior=...)` 仅为统一接口兼容 |
| 数据 | 可靠正样本 P 与未标记样本 U；不需要负类真值 |

论文的泛化界在 SCAR 条件下推导；另在正样本选择偏差情形做了实验。注册表中的 SAR / `selection_biased` 对应该实验情形。

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

2026-10-18 修复[逐子模块推理模式恢复](PULDA.md)：
预测及异常退出保留混合子模式、RNG和参数标志，不改变无BN拒绝或训练目标。

2026-10-18 根据 #120 补[无 BN/private 图像接线](PULDA.md)：
独立技术构造入口固定 `cnn13_no_bn`、train-only 归一化、seed/hash 和可训练状态，
仅单列 method-private/no-BN CNN，保留 BN fail-closed 与 paper-only 身份。
2026-10-21 #120 按 v2 §2.5/Q1 登记为架构例外：论文明确移除 BN（与梯度正则不兼容），
单列并保留 SHOULD 级共享架构对照。200 快照资源 profile 转 P3.3，正式 recipe/runner
分层仍待接线。

`encoder` 经 deepcopy 解冻训练，输出有限二维特征，后接默认 MLP 或用户原始分数头。
四维图像缺 encoder 时拒绝；含 BatchNorm 时仍明确拒绝。
显式骨干 `cnn13_no_bn` 是工具箱现有13卷积布局的独立无 BN 变体，**不改默认 cnn13**，
公开 pipeline 可选此名称，报告记录实际骨干。
插值与输入梯度发生于原始 NCHW 图像，不在冻结特征上代算；create_graph 保留二阶反传。
P/U 数据留 CPU、优化批次上卡；预测分批 eval 并恢复模式，完整空间 shape 与空输入校验。
公式解析卷积例子、真实梯度图、参数更新、种子/clone/pickle/epoch 回放及 GPU 测试见
`test_grad_pu_cnn.py`；公共流水线、不同折 encoder 隔离有对应集成测试。
正式资源/选模/来源审阅仍未完成，默认共享 BN 骨干不会被静默替换。

### 显存执行策略（2026-11-03）

`GradPUClassifier` 新增两个执行层参数：`backward_mode={"joint","split"}` 与
`gp_chunk_size`（默认 `joint` / `None`）。`split` 先把 P/U 分支反传并释放其图，再反传
GP 二阶分支，最后只做一次 optimizer step；`gp_chunk_size` 再把 GP 插值按行分块。
两者只改变反传调度，不改损失、batch、精度、epoch 预算或 RNG 顺序；实际取值写入
`training_provenance_.activation_memory`，CLI/campaign 作为执行策略登记。默认 `joint`
时行为与旧实现一致。

ADNI（seed0、c=0.1、batch 256、200 epoch）实测：

| 配置 | 峰值 allocated / reserved | 用时 |
|---|---:|---:|
| joint（原始实现） | 48.379 / 60.309 GiB | 87.95 min |
| split + chunk 162 | 21.477 / 29.881 GiB | 92.44 min |

- 200 epoch 损失轨迹最大差：ADNI 6.9e-7、CIFAR-10 4.8e-7；test 指标完全相同；
- 纯 split 在 20 epoch 对照中与 joint 耗时基本持平；GP 分块有额外开销：
  ADNI +5.1%、CIFAR +11.8%；
- 48GB 卡优先纯 split；需要 32GB 余量再启用分块。

**图像编码器塌缩（2026-11-03 首次完整运行发现）**：ADNI/CIFAR 的 joint 与 split
都输出常数 score 0.9842（std 5.96e-8），编码器权重有漂移但末层特征 std=0；
2D MLP 正常。判断为 `cnn13_no_bn` 初始化/无 BN ReLU 堆叠导致的编码器塌缩，与
split/chunk 无关；修复前 ADNI/CIFAR 的 GradPU 数值只能作执行诊断，不能作方法效果结论。
正式 recovery 路径仍默认 `cpu_offload`，尚未接入 split/chunk。

## 验证状态与后续门禁

2026-10-18 根据 #121 补实际调度/batch/运行声明：
保留固定LR默认，显式支持epoch线性/余弦至初值0.1倍；曲线不是论文给定。
记录实际Adam LR、分支batch、TS风险/GP行数、更新数及SAR部分匹配边界，
经 `training_provenance_` 写入runner每候选manifest。补齐CNN/回放/fold证据；
正式学习率、batch语义、候选池和阶段资格仍未决定，完整资源profile未执行。

阶段记录：
pu 阶段快照保留局部位置和累计 optimizer 更新数；梯度插值/二阶梯度与 TS 风险逻辑不改。

P1-2 公开对照准备（`p3_public_comparison_20261005.md`，已归档于 tag `survey-docs-archive-20261025`）核 Table1 GradPU
CIFAR n_P1000/3000 为90.1±0.2/91.9±0.1，不是相邻 Sup. 列90.5/92.9。
三 runs 的 deviation 暂不强定 SD/SEM，validation标签/选模尚待补充材料；全部 pending。

2026-10-05 补充来源/实验流程核查：
印刷页 7300 明确 CNN13 去 BatchNorm，原训练集划出 P/500 validation 后剩余作为 U。
理论 U~p(x) 与发布实验分开登记；TS 校准不是原实验协议直接重放。
作者代码仍未确认，补充材料和正式无 BatchNorm 图像规格继续待办；工程路径见上节。

五方法准入证据包（`p3_admission_evidence_20261005.md`，已归档于 tag `survey-docs-archive-20261025`）已登记当前完整构造默认值、
max_epochs 推断快照和 optimizer_steps_ 成本，补 raw-score 梯度惩罚的标量/二阶反传对照。
论文配方/验证标签与网格、阶段选模准入仍待审。

已有公式 golden test、参数/边界、随机性、接口/注册、训练轨迹及权重往返测试。方法台账已登记；`fit(os_or_ts="ts")` 的 U 风险及输入梯度插值中的边缘角色逐批取 `U∪P`，正例权重、超参数与训练先验口径不变；逐 run 实际视图以 manifest 为准。2026-09-28 全局 PyTorch 2.6.0+cu124 环境下的 A6000 原有 GPU 测试和 TS 特征路径单次 CUDA smoke 通过。正式 P3.2 仍需：冻结矩阵之外的候选规格、原生图像骨干与论文基准（若决定宣称复现）、多 seed 资源记录及合作者审阅。

2026-10-18 复核：论文全文已核对（AAAI 2023，8 页，SHA-256
`c9ec9499389007546c5e3020edd98a8e17f221511b380663fc65efedeeca9244`）；
未找到作者代码，按 paper-only 复核。Eq. (5)–(7)、未归一化正例权重、tanh 前
raw-score 梯度惩罚、β 线性调度和阈值 0 与实现一致；本轮未发现新的公式/实现缺陷。
图像路径按方案 B 使用现有 `cnn13_no_bn`（method-private），不进入标准
ResNet-18 主榜；BN fail-closed 保留，详见
[#120](https://github.com/shuidisjtu/pu-learning-toolbox/issues/120)。
2026-10-18 用户接受 paper-only 身份。学习率调度、batch 语义、网络/epoch、TS/SAR 声明与证据补全
见 [#121](https://github.com/shuidisjtu/pu-learning-toolbox/issues/121)。
有条件接受工程证据，但固定学习率与论文调度差异、阶段/预算/资源/公开数值
仍待解决。

2026-10-29：独立 `run_candidate_image_experiment.py` 已补 `adni` 替代数据槽位，
与 CIFAR/MNIST/Fashion-MNIST 共用 train-only 归一化和私有 no-BN CNN 路径。
依据为 D1 的替代数据决定，不是原论文新增 ADNI 实验；不改损失、TS 校准或冻结预算。
计划、manifest 的 split_ref 与结果摘要显式记录「不是临床 ADNI」、未知许可及
未核实 subject 隔离；仓库来源收据不能替任意输入 NPZ 证明真实来源。
CPU 合成灰度短测核对了全部已付 epoch、PA/OA 选模及序列化回放。
