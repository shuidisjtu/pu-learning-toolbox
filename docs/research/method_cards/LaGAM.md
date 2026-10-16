# Method Card: LaGAM

> 当前为二维特征/注入 CNN 的技术预集成，**不符合 PU-only PA 训练口径**，也未获 Survey P3.2 正式验收。

| 字段 | 内容 |
|---|---|
| 论文 | Long 等，*Positive-Unlabeled Learning by Latent Group-Aware Meta Disambiguation*，CVPR 2024；[论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Long_Positive-Unlabeled_Learning_by_Latent_Group-Aware_Meta_Disambiguation_CVPR_2024_paper.pdf) 与 [补充材料 Algorithm 1](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Long_Positive-Unlabeled_Learning_by_CVPR_2024_supplemental.pdf) |
| 官方代码 | [`llong-cs/LaGAM`](https://github.com/llong-cs/LaGAM)，核对 commit `362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8` |
| 源码位置 | `train.py:Trainer.train_loop`（元标签、mixup、warmup）；`utils/utils_algo.py:run_kmeans`；`utils/utils_loss.py:BCELoss/ContLoss` |
| 先验 | 不需要类先验；`fit(class_prior=...)` 只作接口兼容，不进入优化 |

## 核心机制与本实现

LaGAM 先用观测 PU 标签 warmup；之后每轮聚类当前表征形成潜在组。每个训练批次对分类器层做一次可微虚拟更新，再用**独立干净 support set** 的分类损失对训练标签扰动求元梯度，从梯度方向产生 U 的候选标签；观测 P 恒固定为正，U 的候选标签经 EMA 更新。实际训练包含平衡 BCE、Beta(4,4) mixup、双视图 instance InfoNCE 和同潜在组对比损失。

公开类 `LaGAMClassifier`，注册名 `lagam`（别名 `la_gam`），使用二维 MLP 或独立复制后联合训练的注入 CNN、Gaussian 双视图与 sklearn KMeans。支持 CPU/CUDA、随机种子、训练历史、epoch 回调与权重推理恢复；`decision_function` 为 raw logit。非空 `sample_weight` 会报错。作者 ResNet/projection 配方、Faiss GPU 聚类、图像增强、classifier cutoff/kNN augment、400 epoch 调度未重放，**不得声称数值复现**。

2026-10-15 来源补记：发布源码的虚拟 classifier 更新筛选 ndim>1，因此不更新 bias。新增 `meta_update_bias=False` 仅对齐该参数范围，默认 True 保留旧行为。原硬观测元目标/稳定 BCE、EMA 端点时序、optimizer 等与作者仍不同；标量与两列扰动的等价只在同损失/目标/范围下成立。来源文件摘要与逐项边界见[来源/CNN 复核](../pu_survey/admission/lagam_source_cnn_review_20261015.md)。

新图像路径强制 CPU 数据分批，训练/聚类/预测 batch_size、support_batch_size 各自限流；CPU 仍需全量嵌入。CNN 虚拟步冻结 eval 特征并恢复模式，避免 support 更新真实 BN/dropout；实际分类/对比端到端训练。旧二维 auto 保留 resident。`optimizer_steps_` / `virtual_steps_` / `clustering_passes_` 分列成本；max_epochs 已含 warmup，两阶段快照上下文不是选模批准，也不支持训练续跑。

## 协议门禁：干净 support set

`fit(X, y_pu, *, support_data=(X_support, y_clean))` 要求调用方显式提供**独立于 PU 训练池、选模验证集和测试集**的带真值 0/1 标签 support set，且两类均非空。没有该参数会直接报错；训练器不会从隐藏真值或 `clean_val` 自动提取。官方 `train.py` 明确从 `valid_loader` 取真值标签做元更新，因此标准 PA 口径不允许它参赛；标准 OA 若把同一 clean validation set 同时用于训练元更新和选模也会泄漏，需要另设不重叠 support/selection 分割与标签预算。这超出当前冻结的 Survey v1 矩阵，必须先获得协议扩展和合作者审阅。现有 `EpochCheckpointTrainer` 也不传 `support_data`，因此不能把这项技术预集成包装成已接通正式 runner。

注册元数据 `requires_clean_support=true`；普通 PU-only 推荐器会排除该方法，以免自动工作流挑中一个无法安全训练的候选。

发布 `utils/utils_data.py:182-193` 从 U 排除已标记 P，属于 OS split。2026-10-17 按 Issue #115 将 registry 改为 SINGLE_TRAINING_SET、台账改为 os，来源收据保留原冲突并补整改记录。此定位描述作者发布实验，不断言理论仅适用 OS；不新增 TS 接口或校准。source_status=official_exact 只表示作者源码可得，不证明数值重放。support 数组共享内存将拒绝，但不同副本的样本/角色独立仍须协议索引证据。

## 验证与待办

2026-10-17 新增 `LaGAMCleanSupportTrainer`，仅供独立 SCAR OS/终点 OA；
训练前核四角色、先验估计、support 的全局 IDs 与适用分组，支持集不得来自
选模/test。实训回归确认虚拟元更新发生，manifest 显著标注额外监督/预算，
修改 test 不影响训练或选模。旧 `EpochCheckpointTrainer`/frozen Survey 不变；
这不是正式候选池准入，未替代实际数据支持集、来源配方与标签预算审批。
详见[算法收尾与调用示例](../pu_survey/reviews/algorithm_completion_20261017.md)。

单元测试覆盖平衡 BCE、独立二阶元梯度、组对比损失、支持集 fail-closed、CNN 模式/缓冲、分批/成本及推理回放。2026-09-28 A6000 单次 CUDA smoke 为旧二维路径，不覆盖新 CNN，也不替代 frozen-lock 或多 seed 资源验收。台账仍为 `run_view=os-compatible`、`calibration_applied=false`：无安全的同形 U 风险替换，校准也不能消除独立 support 监督。标准 PA 不适用；仍需 support/选模互斥协议、采样定位、来源许可/全文、公开数值、候选矩阵及合作者审阅。
