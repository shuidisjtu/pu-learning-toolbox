# Method Card: CVIR

> 技术预集成，不是 BBE/TEDⁿ 的实现，也不是 Survey v1 正式跑批方法。

## 来源与核心口径

| 字段 | 内容 |
|---|---|
| 论文 | Garg 等，*Mixture Proportion Estimation and PU Learning: A Modern Approach*，NeurIPS 2021，[原文 Algorithm 2，印刷页6](https://proceedings.neurips.cc/paper_files/paper/2021/file/47b4f1bfdf6d298682e610ad74b37dca-Paper.pdf) |
| 官方仓库 | [`acmi-lab/PU_learning`](https://github.com/acmi-lab/PU_learning)，锁定 `5e5e350dc0588de95a36eb952e3cf5382e786aec`，Apache-2.0 |
| 本项目入口 | `CVIRClassifier`，注册名 `cvir`，二维 MLP 或注入可训练 CNN 的四维图像，PyTorch |
| 必需先验 | `unlabeled_positive_prior = α_U = P(Y=1 \mid X\in U)`，即原始未标记池的混合比例 |

算法每轮先按当前正类分数给**原始 U** 排序，保留分数最低的约 `1−α_U` 部分为临时负例；随后按 `α_U·L_P + (1−α_U)·L_N` 更新模型。保留数有限样本下取 `max(1, floor((1−α_U)·n_U))`，相同分数稳定排序。默认先做一轮 P-v-U warm-start，再运行固定 20 轮 CVIR；这两个预算是本工具箱适配选择，原文 Algorithm 2 使用收敛停机。`class_prior` 参数会显式拒绝，避免把公共接口传来的总体 `π` 无声当作 `α_U`。

## 台账与校准

原文 U 为总体边缘分布，原生 TS 下分布参数 `α_U=π_population`；这不意味着有限样本中实际正类比例恰好等于它。OS 移走标记 P 后，U 的先验通常不同，不能无条件拿总体 π 替代。`requires_class_prior=False` 只表示公共总体先验门禁不适用，不代表无需先验。

2026-10-17 按 #117 明确三条路径：直接组件 `input_sampling="ts"` 接受调用方提供的独立 P/边缘 U，必须显式给 `unlabeled_positive_prior` 与 `prior_source`；它不执行 OS 校准，也不从 U 真值估算比例。默认 `input_sampling="os"` 保留既有训练行为，但只称 OS 工程适配。实验层新增 `CVIRKnownPriorTrainer`，要求显式 α_U/来源，仅允许 SCAR OS；总体 π 留给 PA，不转发到训练。两者数值和来源分开写 manifest，调用方来源声明不冒充统计验证。

OS→TS 校准仍未支持：排序、保留数量及损失均作用于原始 U，不新增 P 候选。`calibration_hooked=false`，显式校准 `os_or_ts="ts"` 拒绝。没有预先断言简单合并必然足够或必然错误。`input_sampling="ts"` 与 `os_or_ts="ts"` 是不同契约；后者在 Survey 表示训练风险校准，不是直接组件接收原生 TS 数据。冻结协议/矩阵不变，正式准入仍需审批。

## 适配边界与下一步

2026-10-15 来源核查确认：发布代码实际使用 balanced CE、`n_U-int(α*n_U)` 保留数、zip 原始 loader 后 mask、P/U 合并 forward、默认 SGD；warm-start 只在 TEDn 分支启用。本实现继续采用论文加权目标与既有默认，不静默改写成发布代码；`official_exact` 只指源码可得性。逐项位置、文件摘要与判读见[来源/图像复核](../pu_survey/admission/cvir_source_cnn_review_20261015.md)。

新工程路径 deepcopy encoder 联合训练；原始图像留 CPU，全局 U 排序、训练与预测均分批上设备。旧二维 auto 保留 resident 的全量排序/预测。CPU-stream 仍需主机完整输入和 O(n_U) 排序分数，不是磁盘流式；不同 forward 批形可能带来末位误差。`optimizer_steps_` 与 `epoch_provenance_` 记录两阶段全部成本；快照包含 encoder/头及阶段上下文，不支持优化器续训，也不批准任何阶段选模资格。

本 CVIR 实现没有接入 BBE 或 TEDⁿ 交替过程。2026-10-17 另补[独立 BBE](BBE.md)：
论文 Algorithm 1 统计量与 train/holdout 隔离的二维 PvU 评分入口，显式估计 α_U，
不自动提供总体 π，不改变这里的固定先验训练行为。另补独立 [TEDⁿ](TEDn.md)
端到端工程路径（未加权目标、内部留出动态估计、MLP/注入 CNN、训练误差停止）；
它没有接入本 CVIR 或冻结 Survey。
仍无论文图像/文本配方、原始训练预算、公开数值复现或多 seed GPU 证据。2026-09-28 的 RTX A6000 单次 CUDA 技术 smoke 仍是旧二维路径，不覆盖新 CNN，也不是 frozen-lock 或多 seed 验收。需先由合作者复核 `α_U` 来源与 TS-OS 方案，再决定是否扩展 Survey 的先验字段、选模/视图协议与执行矩阵。新的工程测试只证明合成 CPU 契约，不能读作论文数值复现。
