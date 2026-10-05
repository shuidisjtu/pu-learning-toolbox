# Method Card: Split-PU

> 当前为二维表格特征的技术预集成，不是论文 CIFAR 结果复现或 Survey P3.2 验收。

| 字段 | 内容 |
|---|---|
| 论文 | Xu 等，*Split-PU: Hardness-aware Training Strategy for Positive-Unlabeled Learning*，ACM MM 2022；[arXiv:2211.16756](https://arxiv.org/abs/2211.16756) |
| 官方代码 | [`loadder/SplitPU_MM2022`](https://github.com/loadder/SplitPU_MM2022)，核对 commit `82fb9730597a65a12156736419d4588675bc24d5` |
| 源码位置 | `main.py`、`splitpu_utils.py:split/train_splitpu`、`utils.py:JSDLoss/hard_loss/sim_loss`、`nnpu_utils.py:train_nnpu` |
| 先验 | `class_prior` 为 U 分布正类比例，只用于 nnPU teacher 预训练 |

## 训练阶段

1. 用 P/U 的 nnPU 风险训练 teacher；输入仅有观测 PU 标签。
2. 随机初始化 temporary 模型，拟合 teacher 的硬预测；在 U 上的预测一致率达到阈值可提前停止。teacher/temporary 对 U 的预测不一致即 hard，其他为 easy。
3. 随机初始化 student；对 P 用 BCE，对 easy U 用 teacher 软预测的加权 Jensen–Shannon 散度；对 hard U 用弱/强扰动预测一致性、teacher 低层特征 MSE 与 student 表征余弦一致性。下一轮以上一轮 student 为 teacher，默认两轮。

第一轮默认 hard/feature/sim 权重为 0.3/0.3/0.1；第二轮按官方 `main.py` 收窄为 0.01/0/0。

公开类 `SplitPUClassifier`，注册名 `split_pu`（别名 `split-pu`）；默认二维 MLP，可选 CPU/CUDA。`decision_function` 是 raw logit，0 为预测阈值，不宣称校准概率。非空 `sample_weight` 被拒绝。训练历史、每 epoch checkpoint 回调和 `state_dict` 权重恢复可用；若小样本集出现 U 全部一致/不一致，以 teacher 最低/最高 margin 的一个 U 维持两个分支非空。

## 差异与门禁

- 官方使用 CIFAR CNN、图像弱/强增强和带 predictor 的高层 SimSiam 一致性；本版用二维特征的 Gaussian 扰动与隐层余弦一致性，是**结构保留的表格适配**，不提供图像 backbone，不可直接拿论文准确率作数值裁决。
- 官方代码训练中反复读取测试标签报告准确率；本实现的 `fit` 无真实标签入口，不使用 test set 决定阶段或权重。PA/OA 由外部 runner 分开执行。
- 已配置公式、阶段、接口、确定性、checkpoint 和 CUDA smoke 测试。方法台账已登记；`fit(os_or_ts="ts")` 会在 **nnPU teacher 的未标记风险项**逐 mini-batch 使用 `U ∪ P`，正例项与先验保持不变。temporary 的一致率、easy/hard 划分及 student 的 U 分支仍只取原始 U，避免已知 P 被当伪负例。台账默认 `ts-compatible`，实际视图以 run manifest 为准。2026-09-28 全局 Python/PyTorch 环境下的 A6000 原有 GPU 测试和 TS 特征路径单次 CUDA smoke 均通过；不替代 frozen-lock、多 seed 或正式资源验收。公开论文数值对照、Survey 矩阵、合作者复核仍待完成；本技术预集成不进入冻结 pilot/主榜。
