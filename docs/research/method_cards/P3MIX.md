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

论文 Table 1 指定了各图像任务的 backbone；Algorithm 1 第 3 行要求合并 P∪U shuffle 后切批。工具箱默认 MLP 和分别 shuffle P/U、循环较短角色的 batching 都是工程适配，后者会增加较短角色的行使用次数。论文未给出完整学习率、epoch、batch size 或 EMA 衰减；如需论文数值复现，应先锁定数据切分、backbone、训练预算、选模准则和 E 的 EMA/eta 搜索。`training_provenance_` 保存原始角色大小与累计行使用次数，`epoch_provenance_` 逐轮披露角色使用/重复次数，便于核对实际训练量。来源代码状态在 registry/ledger 登为未找到；NATIVE 表示本地独立实现。

数值超参数要求有限实数，拒绝 bool 和字符串；`gamma∈[0.5,1]`、`teacher_decay∈[0,1)`、`early_learning_weight≥0`（E 变体须 `>0`），而 `learning_rate`、`beta`、`mixup_alpha` 须大于零。整数预算参数拒绝 bool。

公开 mixup helper 先将输入与候选提升到共同浮点 dtype，再替换伙伴并混合；整数输入搭配带小数候选时保留候选数值。二维特征与四维图像均有凸组合和输入不变性回归。

角色行次只统计优化 batch 的原始 P/U 主样本，不另计 mixup 伙伴、池刷新或 teacher forward；它披露的是 batching 复用量，不能当作完整计算量或资源上界。计数只在 epoch 成功结束时提交；拟合中途异常时，该 epoch 已实际前向的行不进入收据，因此应读作“已完成轮次”的实际使用量。

`tests/unit/estimators/test_p3mix.py` 覆盖 batch 方程、边界、epoch 末训练 P top-k 熵池刷新、身份、三变体短拟合、EMA 参数/缓冲更新、CNN encoder 实际更新、推断、失败重训和注册信息。当前实现和测试属于工程集成证据，尚无公开 benchmark 行为对照、GPU 多 seed 复现或合作者签署。

## 当前状态与准入边界

2026-10-31 补齐显式 `batching_policy="merged_shuffle"`：按 Algorithm 1 第 3 行合并 P∪U，
每轮打乱并按总 `batch_size` 切批，尾批保留、不重复任何主样本；原默认 `role_balanced`
保持按角色切批与循环补齐，`batch_size` 仍为每角色上限，避免改变旧调用。
合并切批可能没有某一角色；论文没有定义空组均值，本实现令该组损失贡献为零，
不丢弃或重采样该批。P3Mix-C 的损失角色仍按原始 P/U 标签区分，而非修正后的伪标签。
这一空组约定是显式工程选择，不声称论文完整复现；两种路径均披露实际更新数、角色行次与复用量。
单样本尾批的 BatchNorm 使用运行统计并恢复原训练模式，避免丢行/重复行；这也是工程约定，非论文事实。
三变体的 singleton/尾批、CNN CV/refit、确定性与序列化均有 CPU 回归；OS/先验/校准口径不变，
未修改冻结预算或比较矩阵，也不据此解除正式准入阻断。

2026-10-20 再核 PDF 后补齐输入隔离：各次 student/teacher 前向使用私有当前 batch，encoder
probe 使用独立行副本，避免自定义 in-place 网络污染式 (4) 输入或验证/测试评分数组。
三变体均补充实际卷积的公开 `PUPipeline` 两折 CV+全量 refit、pickle、四角色 PA/OA
checkpoint 与回收回归，详情见[原文交叉表追加记录](../pu_survey/admission/p3mix_paper_review_20261019.md)。
这是技术接入验收，新增临时 batch 内存仍需在正式资源 profile 中实测。

下面为显式短预算 CPU 接口示例，不是论文配方；`X_images` 为 NCHW，`s_observed` 为观测
PU 标签。工程 backbone 不是论文的 7-layer CNN，正式选模与超参仍须独立批准。
示例中的总体先验供 Pipeline 的 PU 风险评价使用，不进入 P3MIX 的训练目标。

```python
from pu_toolbox import PUPipeline

pipeline = PUPipeline(
    classifier="p3mix", architecture="cnn", backbone="cnn13", device="cpu",
    classifier_params={
        "variant": "c", "max_epochs": 2, "batch_size": 16, "pool_size": 100,
    },
    cv=2, random_state=0,
)
# report = pipeline.fit_evaluate(X_images, s_observed, class_prior=pi_population)
```

P3Mix 已由“仅数学组件、未注册”推进为可训练的实验级分类器并写入工具箱 registry 与方法台账。Survey 冻结矩阵、P3 候选准入草稿和正式实验配方未因此改变；没有签署准入、正式预算或资源 profile。P3.1 仍需方法负责人复核论文映射和工程适配，并补齐按正式协议的 benchmark/资源验证后，才可考虑准入。
