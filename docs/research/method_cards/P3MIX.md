# Method Card: P3Mix（完整工程实现，待方法复核与正式准入）

Li, C., Li, X., Feng, L., & Ouyang, J. “Who Is Your Right Mixup Partner in Positive and Unlabeled Learning?” ICLR 2022. 正文与附录来源为本地用户提供的 `30_who_is_your_right_mixup_partne.pdf`（14页）；正确 OpenReview 入口为 [NH29920YEmj](https://openreview.net/forum?id=NH29920YEmj)。本轮首次逐页核对原文 §2、式 (1)–(8)、Algorithm 1、实验设置与附录。PDF 本身没有作者代码链接；当前未确认作者实现仓库，不将独立实现称为作者源码复现。

## 论文方法核对

- 基础 P3Mix 将 U 暂按伪负例处理，使用式 (1)–(3) 的分组目标 `mean_P + beta * mean_U`，对两组分别求均值。
- 式 (4) 使用 `lambda' = max(lambda, 1-lambda)`，`lambda ~ Beta(alpha, alpha)`；软标签随特征一起混合。
- 式 (5) 对边缘伪负例从正例候选池均匀选 mixup 伙伴，其余 P/U 行从当前 P∪U 小批次均匀抽取伙伴。
- 式 (6) 的边缘条件是 `1-gamma <= f(x;Theta) <= gamma`，区间端点含入；但原文紧接着明确规定 γ=0.5 时集合为空（即使预测值恰为 0.5），γ=1 时包含全部 U。实现对 γ=0.5 采用该显式退化规则，其余范围按闭区间处理。
- 式 (7) 每个 epoch 结束时，以训练 P 上预测 Bernoulli 熵最高的 top-k 行刷新候选池；Algorithm 1 初始候选池随机初始化。
- P3Mix-E 在式 (8) 中加入 teacher 正则 `mean log(1 - <f(x;Theta), y_tilde>)`；teacher 对 mixup 后样本给目标，并在 Algorithm 1 的每个 epoch 末按 EMA 更新。实现把 agreement 截至 `1-1e-7`，避免完全一致时 `log(0)`。Algorithm 1 写明 Θ 与 eΘ 均随机初始化，但未说明二者是否独立；本实现复制 student 初始化 teacher，是待复核的工程选择。
- P3Mix-C 在 mixup 前，将 `f(x;Theta) > gamma` 的 U 伪负标签改为正类。E 与 C 是论文定义的两个独立变体，不合并为一个方法。

正文实验设置（§3.1，第 6 页）说明 Adam、alpha=1、k=100、beta ∈ {0.8, 0.9, 1.0}、gamma ∈ {0.85, 0.9, 0.95}；P3Mix-E 的 eta 在 {1,2,3,4,5} 中选。Fashion-MNIST 用 LeNet-5，CIFAR-10 与 STL-10 用 7-layer CNN；每个数据集 5 次独立运行，另报告 500 个验证样本，但未说明验证样本的抽样框、类别组成或选模用途。论文未给出完整学习率/epoch/batch 配置或 mean-teacher EMA 衰减；Algorithm 1 说 Θ/eΘ 均随机初始化，但未说明是否独立，本工具箱使用 student 副本并将其标为工程选择。论文也没有作者代码地址。结果表是公开文献锚点，不代表本项目复现结果；CIFAR-10 结构化读数、PDF 摘要和重复表格位置见[机器可读来源锚点](../pu_survey/data/p3mix_literature_anchors_20261019.json)，状态仍为待负责人复核，不并入冻结对照矩阵。

## Toolbox 实现

`pu_toolbox/estimators/deep/p3mix.py` 现提供已注册 `P3MixClassifier`，`variant` 为 `base`、`e`、`c`。完整路径包含 MLP 与显式注入 CNN encoder、训练 P 独占候选身份、随机初始化池、逐 epoch 熵 top-k 刷新、边缘 U mixup、两组独立损失、EMA teacher 和高置信标签修正。类先验不参与训练，clean validation/test 标签不读取；仅支持原文 OS 单训练集合视图，TS 校准明确拒绝。

论文没有规定默认 backbone、batch 划分、学习率、epoch、EMA 衰减。因此 toolbox 对应默认值是工程接口，不能当成原文 recipe；如需论文数值复现，应先锁定所用数据切分、backbone、训练预算、选模准则和 E 的 EMA/eta 搜索。来源代码状态在 registry/ledger 登为未找到；NATIVE 表示本地独立实现。

数值超参数要求有限实数，拒绝 bool 和字符串；`gamma∈[0.5,1]`、`teacher_decay∈[0,1)`、`early_learning_weight≥0`（E 变体须 `>0`），而 `learning_rate`、`beta`、`mixup_alpha` 须大于零。整数预算参数拒绝 bool。

`tests/unit/estimators/test_p3mix.py` 覆盖 batch 方程、边界、epoch 末训练 P top-k 熵池刷新、身份、三变体短拟合、EMA 参数/缓冲更新、CNN encoder 实际更新、推断、失败重训和注册信息。当前实现和测试属于工程集成证据，尚无公开 benchmark 行为对照、GPU 多 seed 复现或合作者签署。

## 当前状态与准入边界

P3Mix 已由“仅数学组件、未注册”推进为可训练的实验级分类器并写入工具箱 registry 与方法台账。Survey 冻结矩阵、P3 候选准入草稿和正式实验配方未因此改变；没有签署准入、正式预算或资源 profile。P3.1 仍需方法负责人复核论文映射和工程适配，并补齐按正式协议的 benchmark/资源验证后，才可考虑准入。
