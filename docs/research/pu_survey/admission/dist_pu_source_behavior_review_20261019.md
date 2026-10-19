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
| 损失 | warm-up 使用基础损失；PU 阶段在原批次上计算 `base_loss`，加动态熵项，并对 mixup 输出计算熵和 soft-target Mixup BCE。 | 实现包含正例 BCE、均值对齐平方项、逐样本二元熵与预测均值目标 Mixup BCE。 | 两者是不同的算法/目标实现，不是仅仅 backbone 不同。 |
| 伪标签时序 | PU 阶段在初始化时以模型更新伪标签；每个 batch 使用当前伪标签，强制已标记 P 目标为 1，优化后再写回该批未标记行预测。 | 每个 epoch 从当前模型输出生成 mixup soft targets；没有持久的逐样本伪标签状态。 | 对训练历史的依赖和梯度输入不同。 |
| 优化器与日程 | warm-up Adam 使用 `warm_up_lr`、`warm_up_weight_decay`；进入 PU 阶段后重建 Adam，使用另一组学习率/权重衰减，并采用余弦 scheduler。动态 `co_entropy` 按 epoch 递增。 | 一个 Adam，无 weight decay 参数/阶段切换/scheduler；熵权重固定。 | 当前 estimator 不是作者训练 schedule 的重放。 |
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

另给既有 estimator 增加 run-level `training_provenance_`：明确记录实际视图/校准、单阶段全量实现、
`batch_size` 未生效、每 epoch optimizer step 数、实际训练 epoch 数及非作者 recipe 身份。该元数据不改变损失、
默认参数、优化轨迹、校准角色、历史 pilot、冻结矩阵或资源预算；本轮对应执行回归通过。
后续若实现完整两阶段适配，必须作为新显式 recipe/路径，单独测算阶段和 checkpoint 成本，并保持旧 clean-room
路径可重现；正式数据、backbone、PA/OA 选择和预算仍须走协议审批。

## 复核依据

- 作者锁定训练入口：[`train.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/train.py)
- 作者分布损失：[`losses/distributionLoss.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/losses/distributionLoss.py)
- 作者熵定义：[`losses/entropyMinimization.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/losses/entropyMinimization.py)
- 本地实现与登记：[Dist-PU 方法卡](../../method_cards/Dist-PU.md)、`pu_toolbox/estimators/risk/dist_pu.py`、`pu_toolbox/experiment/method_ledger.json`。
