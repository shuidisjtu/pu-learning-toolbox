# Method Card: LaGAM

> 当前为二维表格特征的技术预集成，**不符合 PU-only PA 训练口径**，也未获 Survey P3.2 正式验收。

| 字段 | 内容 |
|---|---|
| 论文 | Long 等，*Positive-Unlabeled Learning by Latent Group-Aware Meta Disambiguation*，CVPR 2024；[论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Long_Positive-Unlabeled_Learning_by_Latent_Group-Aware_Meta_Disambiguation_CVPR_2024_paper.pdf) 与 [补充材料 Algorithm 1](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Long_Positive-Unlabeled_Learning_by_CVPR_2024_supplemental.pdf) |
| 官方代码 | [`llong-cs/LaGAM`](https://github.com/llong-cs/LaGAM)，核对 commit `362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8` |
| 源码位置 | `train.py:Trainer.train_loop`（元标签、mixup、warmup）；`utils/utils_algo.py:run_kmeans`；`utils/utils_loss.py:BCELoss/ContLoss` |
| 先验 | 不需要类先验；`fit(class_prior=...)` 只作接口兼容，不进入优化 |

## 核心机制与本实现

LaGAM 先用观测 PU 标签 warmup；之后每轮聚类当前表征形成潜在组。每个训练批次对分类器层做一次可微虚拟更新，再用**独立干净 support set** 的分类损失对训练标签扰动求元梯度，从梯度方向产生 U 的候选标签；观测 P 恒固定为正，U 的候选标签经 EMA 更新。实际训练包含平衡 BCE、Beta(4,4) mixup、双视图 instance InfoNCE 和同潜在组对比损失。

公开类 `LaGAMClassifier`，注册名 `lagam`（别名 `la_gam`），当前使用二维 MLP、Gaussian 双视图与 sklearn KMeans。支持 CPU/CUDA、随机种子、训练历史、epoch 回调与 `state_dict` 权重恢复；`decision_function` 为 raw logit。非空 `sample_weight` 会报错。官方的 ResNet 图像骨干、Faiss GPU 聚类、图像增强、classifier cutoff/kNN augment、400 epoch 调度未实现，**不得声称数值复现**。

## 协议门禁：干净 support set

`fit(X, y_pu, *, support_data=(X_support, y_clean))` 要求调用方显式提供**独立于 PU 训练池、选模验证集和测试集**的带真值 0/1 标签 support set，且两类均非空。没有该参数会直接报错；训练器不会从隐藏真值或 `clean_val` 自动提取。官方 `train.py` 明确从 `valid_loader` 取真值标签做元更新，因此标准 PA 口径不允许它参赛；标准 OA 若把同一 clean validation set 同时用于训练元更新和选模也会泄漏，需要另设不重叠 support/selection 分割与标签预算。这超出当前冻结的 Survey v1 矩阵，必须先获得协议扩展和合作者审阅。现有 `EpochCheckpointTrainer` 也不传 `support_data`，因此不能把这项技术预集成包装成已接通正式 runner。

注册元数据 `requires_clean_support=true`；普通 PU-only 推荐器会排除该方法，以免自动工作流挑中一个无法安全训练的候选。

## 验证与待办

单元测试覆盖平衡 BCE、二阶元标签方向、组对比损失、支持集 fail-closed、确定性、回调权重往返和 CUDA smoke 配置。正式接入仍需独立 support set 协议、官方行为对照、GPU 实跑、台账/矩阵登记及审阅。
