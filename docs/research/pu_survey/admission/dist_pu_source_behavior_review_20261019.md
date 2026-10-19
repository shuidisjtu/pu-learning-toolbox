# Dist-PU 锁定源码训练流程复核（2026-10-19）

## 范围与身份

复核对象为 `benchmarks/assigned_methods/configs/official_sources.lock.json` 锁定的作者仓库
`Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective`
@ `cb74be1a87176fd38270873c06374e53905b7354`。只做只读来源核对，没有复制、执行或修改作者源码。
本记录补足方法行为/工具箱实现差异；不属于合作者方法签署、公开 benchmark 复现、P3 准入或协议批准。

## 源码行为核对

| 主题 | 锁定源码行为 | 工具箱实现 | 影响 |
|---|---|---|---|
| 训练阶段 | `train.py` 先执行 `warm_up()`，再执行 `dist_PU()`；默认各 60 epochs。 | 默认 `clean_room` 仍单阶段；显式 `author_staged` 才分 warm-up/PU 两阶段。 | staged 路径适配来源时序；整体实现仍不等同作者程序。 |
| 数据批次 | `train_loader` 用 `batch_size=256`、shuffle；每批一次优化器更新。 | `author_staged` 使用有效 batch size（默认 256）；`clean_room` 保持全量更新且忽略该兼容参数。 | provenance 区分两条训练路径，不再把默认的无效参数误作批训练。 |
| 损失 | warm-up 使用基础损失；PU 阶段在原批次上计算 `base_loss`，加动态熵项，并对 mixup 输出计算熵和 soft-target Mixup BCE。 | 默认 `expectation` 保持正例 BCE/矩对齐 clean-room 目标；`author_staged` 强制 `author_histogram`，按来源拆分两个阶段的 histogram、U 熵、mixed 熵及 mixup BCE。 | 仅 staged 路径复用作者主要损失与时序；clean-room 默认目标不变。 |
| 伪标签时序 | PU 阶段在初始化时以模型更新伪标签；每个 batch 使用当前伪标签，强制已标记 P 目标为 1，优化后再写回当前批行预测。 | `author_staged` 在 warm-up 后按训练行身份初始化，并在每个 PU batch 读取/写回当前行的持久伪标签；`clean_room` 仍无持久伪标签。 | staged 复现来源状态依赖；不宣称精确复放原始 DataLoader/RNG 序列。 |
| 优化器与日程 | warm-up Adam 使用 `warm_up_lr`、`warm_up_weight_decay`；进入 PU 阶段后重建 Adam，使用另一组学习率/权重衰减，并采用余弦 scheduler。动态 `co_entropy` 按 epoch 递增。 | `author_staged` 使用两套 Adam/weight decay 与动态熵；默认 `clean_room` 仍是旧单 Adam/fixed weights。 | 只有 staged 路径适配分阶段优化器日程。 |
| scheduler 计步 | 两阶段均用 `CosineAnnealingLR(..., T_max=pu_epochs)`。warm-up 的 `utils.train()` 在完整 DataLoader 循环后调用一次 `scheduler.step()`；PU 阶段 `train_mixup()` 则在每个 mini-batch optimizer step 后调用。 | `author_staged` 分别按 epoch 和 batch step，且记录实际 scheduler step counts；默认 `clean_room` 无 scheduler。 | cadence 不同：warm-up 每 epoch 一步，PU 每 batch 一步。不能把两阶段都概括成 batch cadence，也不能将 `T_max=pu_epochs` 当成两阶段统一的 epoch 级 scheduler 语义。 |
| 结构与数据 | 作者代码按 CIFAR-10、Fashion-MNIST、Alzheimer 构造不同网络/数据处理；仓库 README 给出其环境。 | 当前类为二维输入的小型 MLP，不含作者图像 backbone、数据增强或数据 loader。 | 不得将当前 clean-room 输出与论文图像结果宣称为复现。 |
| 选模 | 官方 `train.py` 每 epoch 在 test loader 上计算 accuracy，并以最高 accuracy 记录 best。 | estimator 本身不读取 test/clean 标签；由实验层选模。 | 不能照抄官方 test-based 选择；项目实验仍须遵循本仓库预注册的 PA/OA 规则。 |

## 2026-10-19 author_staged 适配补记

新增 `training_recipe="author_staged"` 与 `distribution_loss="author_histogram"` 的显式组合，保留默认
`clean_room` 路径。来源逐项根据锁定 `train.py`、`utils.py`、`losses/factory.py`、
`losses/distributionLoss.py`、`losses/entropyMinimization.py`、`customized/mixup.py` 和
`dataTools/mixupDataset.py` 核对：

- warm-up 以作者基础 histogram loss 加 `co_mu` 的 U 熵优化；warm-up 结束后用模型对全训练集生成
  pseudo scores，按 dataset row identity 保存。
- PU 阶段每个 batch 按样本 identity 读取持久 pseudo scores，Observed P target 强制为 1；mixup 使用
  `Beta(alpha,alpha)`、batch 内 permutation，源代码默认 `is_bias=False`，故不做 `max(λ,1−λ)`；
  另计算 raw batch histogram、动态 U 熵、mixed-score 熵及双目标 BCE。
- optimizer、weight decay 和 scheduler 在阶段切换时重建；warm-up scheduler 每 epoch 一步，PU scheduler
  每个 mini-batch 一步，二者 `T_max=pu_epochs`；动态熵系数按来源公式随 PU epoch 增长。
- 作者 `MixupDataset.update_psudos()` 的初始预测使用不打乱的 loader（默认 test_batch_size=128）；
  toolbox 使用同样的 128-row evaluation chunks，保留 CPU pseudo-label 向量并按训练行身份索引，避免 batch
  shuffle 后错位。输入 feature/label 矩阵也留在 CPU，只将当前训练 mini-batch 搬到解析后的设备；这减少
  整体数据显存占用，但不声称复放作者的 DataLoader/worker/RNG 完整序列。

这条路径仅称为**来源审计的作者流程适配**，不标记为 upstream recipe replay：当前仍是 toolbox MLP，
没有作者数据集专属模型、原始预处理/增强、split 或完整 search；class prior 由 estimator 参数提供；
作者每轮使用 test accuracy 选 best 的行为明确不移植。`training_provenance_` 记录阶段、配置、
实际 optimizer/scheduler steps、伪标签时序、校准视图和 `upstream_training_recipe_reproduced=false`。

TS 路径对每个 minibatch 中的 marginal histogram/entropy role 将该批 P∪U 行纳入，同时保留已知 P
监督 histogram term；这是项目 §2.3 在 minibatch 目标上的工程映射，不是原作者流程的一部分。默认 OS
clean-room 结果、历史 Pilot、冻结矩阵和正式准入均不变。短 CPU 测试只能证明路径执行、阶段记账和数学
组件，不证明公开数值复现、训练效果或资源可接受性。

上述差异可在锁定源码 `train.py` 的参数/loader、阶段设置及 `train_mixup()` 中核对；
分布损失 `losses/distributionLoss.py` 使用 histogram 分布比较，并在 batch 内分别计算 P 与 U 的
label-distribution loss，`frac_prior=1/(2*prior)`。因此，方法卡 §3 的均值平方对齐目标是论文直觉/clean-room
抽象，不等同于作者发布程序的 histogram loss。作者源码同时设置 `co_mu=0.002`、`co_entropy=0.004`、
`alpha=6`、`co_mix_entropy=0.04`、`co_mixup=5`，并非工具箱类当前固定默认值。

scheduler 的代码位置：[warm-up `utils.py::train()` 第 14–30 行](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/utils.py#L14-L30)
在 batch 循环外 step；[PU `train.py::train_mixup()` 第 909–958 行](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/train.py#L909-L958)
在 batch 循环内 step。scheduler 两次分别于 `train.py` 第 837–840 行和 896–899 行创建，均传入 `pu_epochs`。

## 结论与整改边界

默认 `DistPUClassifier` 仍是“clean-room 小型 MLP 变体/核心目标探索”。显式选择
`author_staged` + `author_histogram` 时启用经来源审计的两阶段 mini-batch 流程适配；此时 `batch_size` 生效。
两者都不标为官方程序忠实重放：staged adapter 未移植官方数据、backbone、预处理、augmentation 或 split，
也不复制作者 test-loader 选模。作者 histogram 单组件在 clean-room 模式下仍仅替换分布损失项。
两条路径均记录 `training_provenance_`；现有 clean-room 参数更新和历史 Pilot 保持不变。正式配方、图像结构、
数据/PA-OA 选择和预算仍须走协议审批；本适配不代表新准入或公开数值复现。

## 复核依据

- 作者锁定训练入口：[`train.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/train.py)
- 作者分布损失：[`losses/distributionLoss.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/losses/distributionLoss.py)
- 作者熵定义：[`losses/entropyMinimization.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/losses/entropyMinimization.py)
- 作者损失组装：[ `losses/factory.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/losses/factory.py)；pseudo-label 容器：[ `dataTools/mixupDataset.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/dataTools/mixupDataset.py)
- 作者 scheduler 更新位置：[`utils.py` @ `cb74be1`](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective/blob/cb74be1a87176fd38270873c06374e53905b7354/utils.py) 与 `train.py::train_mixup()`。
- 本地实现与登记：[Dist-PU 方法卡](../../method_cards/Dist-PU.md)、`pu_toolbox/estimators/risk/dist_pu.py`、`pu_toolbox/experiment/method_ledger.json`。
