# Self-PU 锁定作者源码行为复核（2026-10-19）

## 范围与结论

本记录是对官方仓库固定快照 `VITA-Group/Self-PU`
[`a0e332ae4f8110e2490d597876e36bf837e1060f`](https://github.com/VITA-Group/Self-PU/tree/a0e332ae4f8110e2490d597876e36bf837e1060f)
的静态源码核查。未执行上游代码，也不推断其报告数值如何生成。核对范围仅限
`train_2s2t_mix.py` 的 meta 训练/验证数据流，以及对应的 Dataset/PULoss 行为。

**结论：不能把该仓库的 `train_with_meta` 直接视为“在独立 clean validation 上做监督式 meta 校准”的忠实实现。**
锁定源码把官方测试 split 作为 meta 分支输入；meta 梯度使用该 split 的无标签预测熵，而后每个 epoch
又使用同一测试 split 的真值准确率评估并选取 checkpoint。这是作者仓库的可观察代码行为，不代表论文数学定义，
也不能据此证明论文中的报告实验采用了相同的数据流程。

## 可复核证据

| 待核事实 | 锁定源码位置 | 观察结果 |
|---|---|---|
| meta loader 的数据来源 | `train_2s2t_mix.py:116–118`（MNIST）及 `163–165`（CIFAR）；对应 `datasets.py:120–128`、`cifar_datasets.py:97–115` | `dataset_test` 以 `split='test'` 构建；`make_dataset` / `make_cifar_dataset` 的非 train 分支从传入数据的 `_testX/_testY` 构造 PN 集，返回真值作为 `Y/T`。 |
| meta 分支实际接入哪个 loader | `train_2s2t_mix.py:198–199, 246–250` | 同一个 `dataloader_test` 被传给 `train_with_meta(...)`，随后又传给 `validate(...)`。 |
| meta 梯度对测试批次使用什么目标 | `train_2s2t_mix.py:608, 647, 824–850`；第二个 student 对应 `948–963` | `train_with_meta` 从该 loader 取 `val_data` 和 `val_labels`。更新权重的 `l_g_meta` 是 `-mean(sum(p*log(p))) * 2`，即负预测熵目标；`val_labels` 不进入此目标或其梯度，只被移到设备并在 epoch 10 的诊断文件中一并保存。故真实测试标签不是这个 meta 梯度的监督标签，但测试特征参与了训练更新。 |
| 测试真值是否参与性能选择 | `train_2s2t_mix.py:250, 254–272`；`validate` 的 `1062–1108` | 每 epoch 在相同 loader 上根据 `T` 计算四个 student/EMA 的准确率；`max(all_accuracy)` 被用于覆盖保存 `model_best.pth.tar`。这使测试标签参与 checkpoint 选择，不能把这些值当作独立一次性 test 估计。 |
| 来源所用 PU 风险组件 | `utils/util.py:215–225, 248–264` | `PULoss(loss_fn='sigmoid_eps')` 将批内 `eps` 传到 sigmoid 风险估计；此事实只用于识别来源分支，不表示工具箱估计器与上游训练逐步等价。 |

源码链接均固定在上述 commit，而不是浮动分支：[训练入口](https://github.com/VITA-Group/Self-PU/blob/a0e332ae4f8110e2490d597876e36bf837e1060f/train_2s2t_mix.py)、[MNIST 数据集](https://github.com/VITA-Group/Self-PU/blob/a0e332ae4f8110e2490d597876e36bf837e1060f/datasets.py)、[CIFAR 数据集](https://github.com/VITA-Group/Self-PU/blob/a0e332ae4f8110e2490d597876e36bf837e1060f/cifar_datasets.py)、[PU loss](https://github.com/VITA-Group/Self-PU/blob/a0e332ae4f8110e2490d597876e36bf837e1060f/utils/util.py)。

## 对工具箱实现和结果解释的影响

- 该发现是官方代码行为审计，不等同于论文算法定义的推翻，也不单凭源代码推断作者论文结果存在何种偏差；论文报告协议与代码运行对应关系仍需原始运行记录才能确定。
- 工具箱 `SelfPUClassifier` 的 clean-meta 路径通过显式 `validation_data` 接收有标签 support；`SelfPUCleanSupportTrainer` 对 support 的身份、角色和额外标签预算作审计，内部选择与最终外部 OA/test 分离。不得为追求“源码一致”而把测试样本重新接入训练、meta 梯度或 checkpoint 选择。
- 冻结 Pilot 没有给 PU 方法 clean validation，仍是无 meta 消融；本审计不改变历史结果、校准状态、PA 资格、冻结矩阵或准入状态。
- 当前实现与官方源码在 meta 外层目标（监督标签损失 vs. 无标签熵目标）、数据流（独立 support vs. 作者入口中的 test split）和选择隔离上并非逐行复现。论文/方法卡若称 clean-validation meta，应明确这是方法定义/工具箱协议，不应把它误写成上述锁定源码行为。

## 尚需外部确认

1. HENG958/RV5 确认冻结 Pilot 中 Self-PU 是无 clean-label meta 的消融，并确认其 PA 资格结论。
2. 若要声称论文数值复现，须取得论文实验数据划分、验证集角色及 checkpoint 选择的原始协议/日志；当前源码审计不能替代这些材料。
3. 若启动 clean-support 正式实验，须另行批准 support 数据来源、标签预算、角色隔离、外部 OA/test 及资源配方。

本审计是锁定源码的静态读取，无代码执行、无训练、无数据访问；不改作者代码、工具箱训练目标、冻结实验制品或正式 recipe。
