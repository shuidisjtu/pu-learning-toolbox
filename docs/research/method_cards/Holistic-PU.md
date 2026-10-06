# Method Card: Holistic-PU

2026-10-07 补[阶段技术留痕](../pu_survey/p3_stage_budget_review_20261007.md)：
warmup/pseudo_pn 快照保留阶段内 epoch 和累计更新数；不改变 fixed warmup、
OS/TS 适用性或预算，也不由技术快照记录决定正式阶段资格。

同日补[原始停止/阶段来源核查及 CNN 工程路径](../pu_survey/holistic_source_cnn_review_20261007.md)：
原生注入 CNN 已能两阶段端到端训练；LZO 与后段重新初始化仍未实现，不宣称完整论文复现。

Wang et al., NeurIPS 2023；[原论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/d5c0f9585592bad5251133813893a6c0-Paper-Conference.pdf)
式 (1)、(4)/(5)、(8)、§2.4 的最终 CE。[作者仓库](https://github.com/wxr99/HolisticPU/tree/4d4ce7d6ba29722995d308293374c0f475d988d3)
已锁定 `4d4ce7d6ba29722995d308293374c0f475d988d3`。独立实现，不执行/复制作者代码，未确认许可证。

## 组件路径与来源分歧

平衡 P/U 批次预热并记录 U 的正类概率轨迹；按全部时间对计算 signed robust trend，
以二分自然断点给 U 伪标签，继续在 P+伪标签 U 上监督训练。
工具箱正类为 1，原文/代码正类为 0；不能直接抄其标签数值。

| 项 | 论文目标 / 本分类器 | 固定 commit 发布代码 |
|---|---|---|
| 趋势 | 全部 i<j 的 signed log 影响函数，scale=2 | `train_phase1` 尾部：邻差 `log(1+d+d²/2)`；负差不等于论文 signed 形式 |
| 划分 | 式 (8)：SSE左/n左 + SSE右/n右 | `jenkspy.jenks_breaks` 的 SSE左+SSE右；不能将两者视为同目标 |
| 正断点处理 | 当前无额外常数过滤 | `utils/misc.py:101-105` 的 `three_sigma` 实际取 x<0.2/9，不是三倍标准差 |
| 停止 | 论文 §2.2/§2.3 用 LZO；本分类器仍是固定预算 `fixed_warmup_budget_not_LZO` | `main.py:284` / `train.py:213` 按固定 warming_steps；`train.py:168` 混合验证调用被注释，不据此声称论文 LZO |
| 后段训练 | §2.4 的伪标签 CE；补充 Algorithm 2 页22 要求重新初始化，本分类器继续同模型/Adam 是适配 | `main.py:329-335` 删除预热模型后新建；`model/loss.py:loss_ft` 渐进目标、强弱增强未复现 |

`holistic_trend_scores` 明确区分 `paper_pairwise` 与 `author_adjacent`（后者 scale 必须 1）；
`holistic_natural_break` 区分 `paper_variance` 与 `author_sse`。
classifier 只走论文目标，不在正式结果中静默换 variant。常数轨迹不能识别两类，直接拒绝，
不凭空捏造类别或先验。两个函数均有独立 scalar / brute-force 数学对照。

## 采样、预算与接口

标准 CIFAR 分支 `dataset/cifar.py:74-93` 在 `p_u_split` 中用 `setdiff1d` 排除 P，登记 OS。
`lt_p_u_split` 为另一长尾独立采样分支，不推广本声明。当前无同形 TS risk，显式 TS 拒绝。
外部先验不使用；`estimated_unlabeled_prior_` 仅为 U 内伪正例率，不冒充已校准总体先验。
普通 PU-only 不需要 clean support，不在 fit 中读取任何真值验证/test 标签。

`HolisticPUClassifier` / `holistic_pu`，二维 MLP 或注入端到端 CNN、CPU/CUDA。`warmup_epochs>=2`，
`max_epochs` 为后段轮数；`optimizer_steps_` 包括全部阶段。
`prediction_trajectory_` 为 U×warmup 的概率数组，`pseudo_label_indices_` 保留训练行身份，
`pseudo_labels_` / `trend_scores_` / `breakpoint_` 支持对账。
sample_weight 非空拒绝。`epoch_callback` 在两个阶段每轮结束时调用，零起始 epoch 连续编号；
`history_["phase"]` 明确区分 `warmup` / `pseudo_pn`，`checkpoint_epoch_count` 为两阶段轮数之和。
`EpochCheckpointTrainer` 可保存各轮分类器权重及 CPU 回放；完整轨迹需另存可信 pickle。
权重快照不是优化器/RNG/伪标签状态恢复，也不是训练断点续跑。候选阶段能否进入 PA/OA
选择尚未预注册，工程 hook 不代表正式选模协议已通过；不改历史冻结结果。
内存预算需计 n_U×warmup×8 bytes，pairwise 时间代价 O(n_U×warmup²)；自然断点 O(n_U log n_U)，
不构造全样本二次矩阵。不是 LZO 或完整官方图像实验复现。

`encoder` 必须输出二维 batch×features；4-D NCHW 不允许默默 flatten。每次 fit deepcopy
模板并训练编码器，两个阶段继续同一副本；模板权重/BatchNorm 不污染其它折。
原始数据/目标留 CPU，优化批次、轨迹扫描、预测按 `batch_size` 分批上设备。
eval 扫描不更新 BN，预测即使失败也恢复全部层的原模式；单行尾批用 BN 运行统计，
不丢行且保留 affine 参数梯度。这些是独立工程约定，非作者 CNN7/增强/EMA 配方。

公式、暴力划分、40,000 行内存形态 smoke、平衡 minibatch、常数轨迹拒绝、种子/clone/pickle
及预算边界见 `tests/unit/estimators/test_holistic_pu.py`。正式候选、图像/停止规格、数字对照、
GPU/多 seed 正式资源记录及负责人复核待完成；不进入历史冻结 pilot。
`tests/unit/estimators/test_holistic_pu_cnn.py` 补两阶段编码器更新、singleton、种子/refit、
输入/输出契约、模式/BN 隔离与阶段回放；公共 pipeline 的 CNN 及折间隔离也有集成测试。
