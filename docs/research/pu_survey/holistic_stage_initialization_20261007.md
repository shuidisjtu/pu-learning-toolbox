# Holistic-PU：后段模型/优化器重新初始化

状态：**工程实现 / pending_method_owner_review**。承接
[来源复核](holistic_source_cnn_review_20261007.md) 和[后续优先级](post_pilot_priority_plan.md)。
不改旧冻结协议/结果，不进行真实数据跑批，不代填负责人签署或正式候选。

## 1. 来源与明确边界

会议[摘要页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d5c0f9585592bad5251133813893a6c0-Abstract-Conference.html)
所附 Supplemental ZIP 的 `attachment.pdf` 页22 Algorithm 2 第14步要求后段重新初始化；
该附件是匿名投稿稿，原始摘要及版本边界见上轮来源复核。
[作者锁定 main.py](https://github.com/wxr99/HolisticPU/blob/4d4ce7d6ba29722995d308293374c0f475d988d3/main.py)
329–335 删除预热模型并新建，344 行新建优化器。复核时 commit 和已登记 SHA-256 均一致。
本轮只独立实现这一初始化步骤，不复制/执行作者代码；作者后段多模型、渐进目标、
增强、EMA、SGD/调度及 LZO 均不能由本步骤推导为完成。

## 2. 接口与运行语义

`HolisticPUClassifier(pseudo_pn_initialization=...)`：

| 值 | 网络 | 优化器 | 兼容/来源含义 |
|---|---|---|---|
| `continue`（默认） | 继续预热权重和 BN | 继续 Adam moments/step | 保留旧适配调用，不声称完整来源初始化 |
| `reinitialize` | 新建 MLP；CNN deepcopy 外部模板后对参数层调用 reset_parameters，再新建 head；BN 运行统计重置 | 新建 Adam，state 最初为空 | 对齐后段重初始化步骤，不将仅复制预训练权重叫 fresh |

选择发生在预热轨迹记录、趋势划分和伪标签产生之后；因此同 seed/模板下，两变体的
预热轨迹和伪标签逐 bit 一致，最终网络可以不同。外部 encoder 模板始终不修改。
数据和目标继续留 CPU、优化按批上设备，不消费任何新增真值标签/selection/test。
MLP/CNN13/ResNet-18 的公共 pipeline 已有技术测试；可用 classifier_params 显式传此参数。

重新初始化要求编码器每个直接参数所有者有可调用的 `reset_parameters`。
缺失时训练前 fail-closed；legacy weight_norm 的 reset_parameters 只改临时计算 weight、
不能可靠重置 weight_g/weight_v，因此明确拒绝。固定归一化常数等无参数 buffer 保留。
自定义 reset 方法须满足自己的参数/状态重置契约；不泛称任意用户模型都可复现作者初始化分布。

## 3. 预算、恢复及登记

`pseudo_pn_initialization_` / `pseudo_pn_optimizer_reset_` 标注实际路径。
`stage_optimizer_steps_={warmup:..., pseudo_pn:...}` 分阶段计实际更新，
`optimizer_steps_` 和 `history_["optimizer_steps"]` 始终累计，包括被丢弃的预热训练成本。
不得因新 Adam 的 step 从1开始而把全局已付成本归零。

checkpoint 仍是 warmup/pseudo_pn 两阶段、连续 epoch，存储上界仍为
warmup_epochs+max_epochs；两阶段架构相同，权重快照可用最终模板独立恢复预热权重。
重建后 state_dict 的键/形状若与预热不同，拒绝继续训练，避免产生无法跨阶段回放的快照。
快照不带优化器续训或 LZO 状态，不改变正式 PA/OA 候选阶段资格。
旧二维默认调用的分数/轨迹/伪标签须保持兼容；外部参数显式记录选择变体。

台账和独立五方法扩展草稿同步实现、初始化差异及文档引用；新增
`pseudo_pn_initialization_variant_recipe_decision` 准入 blocker。
候选池/预算/协议仍 null，`admitted=false`、负责人决定为空，校准字段仍 false。

## 4. 测试与继续项

`test_holistic_pu_initialization.py` 直接插桩真实 Adam：新旧实例/参数不共享，
预热和后段局部 step 与全局 cost 对账；检查新 CNN 权重和清空 BN，不只检查一个 reset 布尔值。
另覆盖默认兼容、种子/clone/refit/pickle、所有阶段快照、非法参数与不可重置编码器拒绝。
公共 pipeline 测试覆盖 CNN13/ResNet-18 的显式选项和报告参数留痕。

CPU 隔离环境：Python 3.11.12 / torch 2.13.0+cpu；不是正式 frozen-lock/GPU 验收。
定向模型/阶段/台账契约 **120 passed / 2 skipped**；包含公共 pipeline 的另一组 **48 passed**。
11 项静态门禁通过，P1-1/P1-2/五方法预算-选模绑定保持通过；冻结文件未改。
父提交 `2406a80` 与默认 continue 分支的 MLP/CNN 各 seed 0/1/7 合成探针：分数、轨迹、
伪标签逐 bit 一致，累计更新各13。手工改变后段架构的拒绝探针：未标记 fitted，
保留9次预热成本，不进入 PN 训练；这些不是公开数据数值复现。

最终全量快速回归（`not slow and not e2e`）：**3128 passed / 56 skipped / 36 deselected**，
161 warnings，199.44s，exit0。没有运行远端 CI、真实数据正式批次或其它 Python/CUDA 矩阵。

下一独立项为 LZO 的预热终点选择：正例 mixup 构造、验证集合/随机流、候选与平局、
趋势截断/状态恢复、额外成本及与外部 PA/OA 隔离。来源没有给清楚的选择需明确标记
工程口径，不以未调用的作者 validation helper 冒充完整原实验；正式准入仍交负责人审核。
