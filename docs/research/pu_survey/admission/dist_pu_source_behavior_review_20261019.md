# Dist-PU 锁定源码训练流程复核（2026-10-19）

## 范围与身份

复核对象为 `benchmarks/assigned_methods/configs/official_sources.lock.json` 锁定的作者仓库
`Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective`
@ `cb74be1a87176fd38270873c06374e53905b7354`。只做只读来源核对，没有复制、执行或修改作者源码。
本记录补足方法行为/工具箱实现差异；不属于合作者方法签署、公开 benchmark 复现、P3 准入或协议批准。

## 源码行为核对

| 主题 | 锁定源码行为 | 工具箱实现 | 影响 |
|---|---|---|---|
| 训练阶段 | `train.py` 先执行 `warm_up()`，再执行 `dist_PU()`；默认各 60 epochs。 | 单一全量张量训练循环，没有阶段切换。 | 参数更新轨迹与论文官方程序不同。 |
| 数据批次 | `train_loader` 用 `batch_size=256`、shuffle；每批一次优化器更新。 | `batch_size` 当前不参与训练，每 epoch 对全量矩阵一次更新。 | 参数名不能被解释成已实现 mini-batch。 |
| 损失 | warm-up 使用基础损失；PU 阶段在原批次上计算 `base_loss`，加动态熵项，并对 mixup 输出计算熵和 soft-target Mixup BCE。 | 默认 `expectation` 为正例 BCE、均值平方对齐、熵和预测均值 Mixup；`author_histogram` 仅替换分布损失组件为 P/U histogram L1，其他训练阶段与熵/Mixup路径仍是 toolbox 当前实现。 | histogram 项已可选复用；整体仍不是作者完整算法流程。 |
| 伪标签时序 | PU 阶段在初始化时以模型更新伪标签；每个 batch 使用当前伪标签，强制已标记 P 目标为 1，优化后再写回该批未标记行预测。 | 每个 epoch 从当前模型输出生成 mixup soft targets；没有持久的逐样本伪标签状态。 | 对训练历史的依赖和梯度输入不同。 |
| 优化器与日程 | warm-up Adam 使用 `warm_up_lr`、`warm_up_weight_decay`；进入 PU 阶段后重建 Adam，使用另一组学习率/权重衰减，并采用余弦 scheduler。动态 `co_entropy` 按 epoch 递增。 | 一个 Adam，无 weight decay 参数/阶段切换/scheduler；熵权重固定。 | 当前 estimator 不是作者训练 schedule 的重放。 |
| scheduler 计步 | `CosineAnnealingLR(..., T_max=pu_epochs)`；warm-up 的 `utils.train()` 与 PU 的 `train_mixup()` 都在每个 mini-batch optimizer step 后调用 `scheduler.step()`。 | 没有 scheduler。 | 作者程序的 scheduler 计数单位是 batch，不能把默认 `pu_epochs=60` 简写成 60 次 epoch 级衰减。 |
| 结构与数据 | 作者代码按 CIFAR-10、Fashion-MNIST、Alzheimer 构造不同网络/数据处理；仓库 README 给出其环境。 | 当前类为二维输入的小型 MLP，不含作者图像 backbone、数据增强或数据 loader。 | 不得将当前 clean-room 输出与论文图像结果宣称为复现。 |
| 选模 | 官方 `train.py` 每 epoch 在 test loader 上计算 accuracy，并以最高 accuracy 记录 best。 | estimator 本身不读取 test/clean 标签；由实验层选模。 | 不能照抄官方 test-based 选择；项目实验仍须遵循本仓库预注册的 PA/OA 规则。 |

上述差异可在锁定源码 `train.py` 的参数/loader、阶段设置及 `train_mixup()` 中核对；
分布损失 `losses/distributionLoss.py` 使用 histogram 分布比较，并在 batch 内分别计算 P 与 U 的
label-distribution loss，`frac_prior=1/(2*prior)`。因此，方法卡 §3 的均值平方对齐目标是论文直觉/clean-room
抽象，不等同于作者发布程序的 histogram loss。作者源码同时设置 `co_mu=0.002`、`co_entropy=0.004`、
`alpha=6`、`co_mix_entropy=0.04`、`co_mixup=5`，并非工具箱类当前固定默认值。

## 结论与整改边界

当前 `DistPUClassifier` 应继续称为“clean-room 小型 MLP 变体/核心目标探索”，不应标为官方源码忠实实现，
也不应以尚未使用的 `batch_size` 暗示已支持批训练。作者训练流程的忠实适配需要独立 opt-in 路径和明确
的模型/数据/优化器接口；其中 test-loader 选模必须替换为项目批准的验证选择协议，不能复制作者 test 泄漏。

另给既有 estimator 增加显式 `distribution_loss="author_histogram"` 组件，复现锁定源码
`LabelDistributionLoss(num_bins=1, proxy='polar', dist='L1')` 的 P/U histogram 项；默认
`expectation` 路径与既有损失保持不变。新组件仍复用单阶段全量训练、当前熵/Mixup 与项目 OS/TS 角色接线，
不能称为完整作者程序重放。拟合后的 `training_provenance_` 记录实际视图/校准、分布损失模式、未生效的
`batch_size`、优化步数/epoch 及 `upstream_training_recipe_reproduced=false`；对应数学和短拟合回归通过。
正式准入、历史 pilot、冻结矩阵和预算均不变。
后续若实现完整两阶段适配，必须作为新显式 recipe/路径，单独测算阶段和 checkpoint 成本，并保持旧 clean-room
路径可重现；正式数据、backbone、PA/OA 选择和预算仍须走协议审批。

## 复核依据

- 作者锁定训练入口：[`train.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/train.py)
- 作者分布损失：[`losses/distributionLoss.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/losses/distributionLoss.py)
- 作者熵定义：[`losses/entropyMinimization.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/losses/entropyMinimization.py)
- 作者 scheduler 更新位置：[`utils.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/utils.py) 与 `train.py::train_mixup()`。
- 本地实现与登记：[Dist-PU 方法卡](../../method_cards/Dist-PU.md)、`pu_toolbox/estimators/risk/dist_pu.py`、`pu_toolbox/experiment/method_ledger.json`。
