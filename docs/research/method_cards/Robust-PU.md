# Method Card: Robust-PU

> 本卡记录技术预集成，不表示 Survey P3.2 正式验收或论文数值复现。

## 来源与适用范围

| 字段 | 内容 |
|---|---|
| 论文 | Zhu 等，*Robust Positive-Unlabeled Learning via Noise Negative Sample Self-correction*，KDD 2023；[arXiv:2308.00279](https://arxiv.org/abs/2308.00279) |
| 官方代码 | [`woriazzc/Robust-PU`](https://github.com/woriazzc/Robust-PU)，核对 commit `34d950f2c6e56510855a922acb5f84b6459773ef` |
| 来源位置 | `main.py` 的 `pre_train`、`weighted_dataloader`、`train_episode`、`train`；`spl_utills.py` 的 `TrainingScheduler`、`calculate_spl_weights`；`lossFunc.py` 的 `nnpu_loss`、`bce_loss` |
| 先验语义 | 依官方 `utils.py:get_pu_data`，`class_prior` 是**未标记池 U 内的正类比例**，不是把 P+U 拼接后计算的比例；只用于 nnPU 预训练 |
| 输入 | 已切好的可靠正样本 P 与未标记样本 U，稠密二维数值特征；不接收隐藏真值标签 |

## 算法与实现

官方默认先用 nnPU 风险预训练，再按当前模型对每个样本的“困难度”计算自步权重。对 P 的困难度是 `softplus(-logit / temper_p)`；对 U 当作候选负例，是 `softplus(logit / temper_n)`。阈值从 `alpha_p/n` 线性增长至 `max_thresh_p/n`；默认 Welsch 权重为 `exp(-困难度 / 阈值²)`，还提供官方代码中的 hard 与 linear 两种规则。`phi` 对**同一原始样本索引**的权重作跨轮移动平均。随后把 P 标为 1、U 标为 0，用不按权重和重新归一化的加权 BCE 更新网络。

公开类 `RobustPUClassifier`，注册名 `robust_pu`（别名 `robust-pu`）。默认 MLP100；也可传入每行输出一个 raw logit 的 PyTorch 模型。支持显式 CPU/CUDA、固定随机种子、训练历史、每轮 checkpoint 回调与 `state_dict` 保存/恢复。`decision_function` 返回 raw logit，0 为预测阈值；不宣称概率校准。`sample_weight` 不支持，传入即报错。预训练与每个自步 episode 均产生一个 checkpoint；权重与输入顺序绑定，避免官方脚本全局 moving-weight 缓存随 shuffle 错位。

## 与论文/官方脚本不同的部分

- 本实现只有二维表格 MLP 路径，没有官方 CIFAR CNN、图像增强和论文训练预算；默认预训练 10 epoch + 20 episode 是轻量起点，非最优复现配置。
- 官方脚本用真实验证标签选预训练及最终模型，并读取真实训练标签做诊断日志；本实现的 `fit` **没有这些标签入口**，也不在内部做真实标签选模。PA/OA 只能由工具箱实验 runner 在各自协议路径处理。
- 官方有更多 scheduler、hardness、SPL 与重启分支；这里只实现默认的 logistic hardness、linear scheduler、Welsch 权重，并额外保留 hard/linear 权重规则。不得将这个子集声称为官方完整功能等价。
- 来源许可证未在核对的仓库中确认；本代码为独立实现，不复制官方源码。

## 验证与剩余门禁

本地单元测试覆盖权重公式、参数错误、接口、确定性、checkpoint、权重往返和 CPU 训练；CUDA smoke 测试已配置，需在有可用 GPU 的门禁执行。正式 P3.2 尚需 GPU 实跑证据、公开行为/论文协议对照、方法台账与 Survey 矩阵接入，并由合作者复核；在 P2.0b/c 未签署前不得将此方法混入正式 pilot 或主榜。
