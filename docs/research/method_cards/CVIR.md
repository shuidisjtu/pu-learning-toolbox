# Method Card: CVIR

2026-10-22 P3.2 补齐：新增显式 `fit(os_or_ts="ts")` 的训练风险池校准，
以及 `CVIRPopulationPriorTrainer` 的总体 π → 实际风险池先验换算。
默认 OS/独立 native-TS 与旧 `CVIRKnownPriorTrainer` 不变；
见[交付、差异理由与验证范围](../pu_survey/delivery/cvir_calibration_20261022.md)。
这是独立工程入口，尚未接入正式 protocol v3 或 Survey 跑批。

2026-10-20 工程补核：修复 resident/streamed 评分覆盖逐层模式及 in-place encoder
改写输入的问题，提前拒绝非法学习率/回调与复数输入；见[输入/推理验收](../pu_survey/admission/cvir_source_cnn_review_20261015.md)。
默认训练目标、原始 U 候选、α_U、TS 校准缺口与正式准入边界均不变。

> 技术预集成，不是 BBE/TEDⁿ 的实现，也不是 Survey v1 正式跑批方法。

## 来源与核心口径

| 字段 | 内容 |
|---|---|
| 论文 | Garg 等，*Mixture Proportion Estimation and PU Learning: A Modern Approach*，NeurIPS 2021，[原文 Algorithm 2，印刷页6](https://proceedings.neurips.cc/paper_files/paper/2021/file/47b4f1bfdf6d298682e610ad74b37dca-Paper.pdf) |
| 官方仓库 | [`acmi-lab/PU_learning`](https://github.com/acmi-lab/PU_learning)，锁定 `5e5e350dc0588de95a36eb952e3cf5382e786aec`，Apache-2.0 |
| 本项目入口 | `CVIRClassifier`，注册名 `cvir`，二维 MLP 或注入可训练 CNN 的四维图像，PyTorch |
| 必需先验 | `unlabeled_positive_prior` 为实际风险池的混合比例；默认是原始 U 的 α_U，显式校准后为总体 π |

算法每轮先按当前正类分数给**原始 U** 排序，保留分数最低的约 `1−α_U` 部分为临时负例；随后按 `α_U·L_P + (1−α_U)·L_N` 更新模型。保留数有限样本下取 `max(1, floor((1−α_U)·n_U))`，相同分数稳定排序。默认先做一轮 P-v-U warm-start，再运行固定 20 轮 CVIR；这两个预算是本工具箱适配选择，原文 Algorithm 2 使用收敛停机。`class_prior` 参数会显式拒绝，避免把公共接口传来的总体 `π` 无声当作 `α_U`。

## 台账与校准

原文 U 为总体边缘分布，原生 TS 下分布参数 `α_U=π_population`；这不意味着有限样本中实际正类比例恰好等于它。OS 移走标记 P 后，U 的先验通常不同，不能无条件拿总体 π 替代。`requires_class_prior=False` 只表示公共总体先验门禁不适用，不代表无需先验。

2026-10-17 按 #117 明确三条路径：直接组件 `input_sampling="ts"` 接受调用方提供的独立 P/边缘 U，必须显式给 `unlabeled_positive_prior` 与 `prior_source`；它不执行 OS 校准，也不从 U 真值估算比例。默认 `input_sampling="os"` 保留既有训练行为，但只称 OS 工程适配。实验层新增 `CVIRKnownPriorTrainer`，要求显式 α_U/来源，仅允许 SCAR OS；总体 π 留给 PA，不转发到训练。两者数值和来源分开写 manifest，调用方来源声明不冒充统计验证。

**本 Survey 的基础数据按协议 §2.3 始终按 OS 抽样**；不为原生 TS 方法重新独立抽样 TS 数据。
2026-10-22 已接入显式 **TS-compatible 校准**：先将训练风险池替换为原始 `U∪P`
（稳定次序 U 后 P），再按同一混合先验做全局排序、截断和临时负例训练；P 的正例项保留。
warm-start 的风险池同步替换。不能在筛完负例之后追加 P，那会改变 CVIR 的非线性截断目标。
这是协议 §2.3 允许的全局池实现，不等同独立 TS 抽样；不改变 PA/OA 验证或 test 视图。
`calibration_hooked=true` 表示接口已实现，台账默认 `calibration_applied=false` 保留，逐 run
按请求记录实际值。`input_sampling="ts"` 仍只声明调用方提供原生 P/边缘 U，不能同时请求
`os_or_ts="ts"` 再校准一次。

新 `CVIRPopulationPriorTrainer` 只允许 SCAR、显式 known-prior 来源和未预置 α_U/source
的 estimator，防止覆盖独立声明。按协议 §3，OS 用 `(π*n−n_L)/(n−n_L)`，校准 TS 用 π；
只读观测 PU 计数，不读隐藏训练真值，不用请求的 c 代替实际 n_L，不裁剪不可能/端点先验。
选择侧的 PA 仍使用总体 π；训练侧由 adapter 绑定风险池先验，
不传给 `estimator.fit(class_prior=...)`。
拟合期间临时绑定 α/source，成功或异常均恢复构造参数；实际值保留在 fitted 属性与 manifest。
原生 TS 直接调用和旧 OS adapter 不变。正式协议/预算、真实数据与 GPU 验收仍待，
不改冻结矩阵/历史结果，不把工程一致性称为论文效果复现。

## 适配边界与下一步

2026-10-31 补齐 `scripts/run_candidate_image_experiment.py --method cvir --view os/ts`
独立共享 ResNet-18 图像执行入口（仅 SCAR，不是论文 backbone/正式配方）。显式给训练预算、
`input_sampling=os`、`data_device_policy=cpu_stream` 与总体先验来源；实际 P 计数决定 OS 的 α_U，
校准 TS 用总体 π。不向 fit 转发 `class_prior`，临时 α/source 在成功或异常后恢复。
逐阶段验证 warm-start/截断训练的更新数、风险池、保留负例身份及排序前向数；
两阶段快照均进入独立 PA/OA 选模与回收，属明确工程选择，不批准正式阶段资格。
共享 ResNet 的训练 BN 若会收到 singleton 小批，计划/拟合前明确拒绝该预算，不丢行、
不重采样、也不静默切换 BN 模式。CPU 原始 fit/图像适配、OS/TS 与快照回放回归
仅验证工程一致性；核心算法、历史来源摘要与正式准入阻断保持不变。

2026-10-15 来源核查确认：发布代码实际使用 balanced CE、`n_U-int(α*n_U)` 保留数、zip 原始 loader 后 mask、P/U 合并 forward、默认 SGD；warm-start 只在 TEDn 分支启用。本实现继续采用论文加权目标与既有默认，不静默改写成发布代码；`official_exact` 只指源码可得性。逐项位置、文件摘要与判读见[来源/图像复核](../pu_survey/admission/cvir_source_cnn_review_20261015.md)。

新工程路径 deepcopy encoder 联合训练；原始图像留 CPU，全局 U 排序、训练与预测均分批上设备。旧二维 auto 保留 resident 的全量排序/预测。CPU-stream 仍需主机完整输入和 O(n_U) 排序分数，不是磁盘流式；不同 forward 批形可能带来末位误差。`optimizer_steps_` 与 `epoch_provenance_` 记录两阶段全部成本；快照包含 encoder/头及阶段上下文，不支持优化器续训，也不批准任何阶段选模资格。

本 CVIR 实现没有接入 BBE 或 TEDⁿ 交替过程。2026-10-17 另补[独立 BBE](BBE.md)：
论文 Algorithm 1 统计量与 train/holdout 隔离的二维 PvU 评分入口，显式估计 α_U，
不自动提供总体 π，不改变这里的固定先验训练行为。另补独立 [TEDⁿ](TEDn.md)
端到端工程路径（未加权目标、内部留出动态估计、MLP/注入 CNN、训练误差停止）；
它没有接入本 CVIR 或冻结 Survey。
仍无论文图像/文本配方、原始训练预算、公开数值复现或多 seed GPU 证据。2026-09-28 的
RTX A6000 单次 CUDA 技术 smoke 仍是旧二维路径，不覆盖新 CNN，也不是 frozen-lock
或多 seed 验收。先验换算/训练视图的独立工程接入已补齐；正式 protocol v3 profiles/units、
runner 绑定、配方预算与真实来源的运行验收仍待。新的测试只证明合成 CPU 契约，
不能读作论文数值复现。
