# Method Card: Robust-PU

原生 CNN 工程路径（2026-10-05）：`encoder=nn.Module` deepcopy/解冻训练，有限二维特征
接默认 MLP 或用户 raw-logit 头；四维图像缺 encoder 时拒绝，不静默 flatten。
probe、自步权重扫描及推断均 eval，不更新 BatchNorm；预测恢复模式并校验完整空间 shape。
图像/观测标签保留 CPU，自步权重按原始行在 CPU 更新，优化批次上设备；不把权重跟随
shuffle 重新编号。TS 风险池替换仍只在 nnPU 预训练，自步伪负例仍为原始 U。
阶段计数/快照、BN 真实训练更新与模板未污染、种子/clone/pickle/权重回放、GPU 驻留见
`test_robust_pu_cnn.py`；公共 pipeline、不同 CV 折有集成测试。
图像微型 smoke 不替代作者增强/scheduler、正式预算、多 seed 资源或数值验收。

资源接口补齐（2026-10-05）：`checkpoint_epoch_count=pretrain_epochs+episodes`，
预训练每轮和每个完整 episode 各一份快照；`inner_epochs` 不额外产生快照，全部更新计入
`optimizer_steps_`（每次 fit 重置）。runner 按此数量估算峰值，显式协议预算仍优先；
回收不是峰值优化，正式 bytes/component 和候选预算仍须实测/预注册。
独立优化器 step 插桩、两阶段快照/回放和失败不重试见 `test_multistage_candidate_accounting.py`。

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

- 本实现支持二维 MLP 和注入 CNN 的四维图像工程路径，非官方 CIFAR 网络/图像增强/完整训练预算；默认预训练 10 epoch + 20 episode 是轻量起点。
- 官方脚本用真实验证标签选预训练及最终模型，并读取真实训练标签做诊断日志；本实现的 `fit` **没有这些标签入口**，也不在内部做真实标签选模。PA/OA 只能由工具箱实验 runner 在各自协议路径处理。
- 官方有更多 scheduler、hardness、SPL 与重启分支；这里只实现默认的 logistic hardness、linear scheduler、Welsch 权重，并额外保留 hard/linear 权重规则。不得将这个子集声称为官方完整功能等价。
- 来源许可证未在核对的仓库中确认；本代码为独立实现，不复制官方源码。

## 准入证据补全（2026-10-05）

P1-2 [公开对照准备](../pu_survey/p3_public_comparison_20261005.md)补原表 CIFAR/Spambase
三档 π_U 的 error读数与10 trials；锁定 main.py:515 明确 mean×100、std分数，
误差棒归一化不是 numeric验收。论文PN把U当负类，不是全监督oracle；论文的prior-free
不能照搬到本工具箱需要先验的nnPU warmup组合，标签/采样和选模预算继续待审。

见[五方法证据包](../pu_survey/p3_admission_evidence_20261005.md)。上游 `main.py:71-84`
预训练恢复 best clean-validation 模型，`:396-406` 最终按 clean-validation/patience
早停并恢复；本实现固定预算、从末轮预训练继续，不以内部 clean-val 做阶段决定。
作者 parser 默认 pre_epochs=400/epochs=100（`:554-594`），不是我方默认 10/20。
新增 SPL 三规则边界/停止梯度对照，沿用真实 step 和快照上界插桩测试。
PA/OA 外置不能消除阶段初始化差异；正式选模/候选预算和未决许可仍须负责人复核。

## 验证与剩余门禁

本地单元测试覆盖权重公式、参数错误、接口、确定性、checkpoint、权重往返和 CPU 训练。方法台账已登记；`fit(os_or_ts="ts")` 会在 **nnPU 预训练的未标记风险项**逐 mini-batch 使用 `U ∪ P`，正例项与先验保持不变。后续自步伪负例仍只取原始 U；若跳过预训练（`pretrain_epochs=0`），请求 `ts` 会显式报错，避免 manifest 误记校准。台账默认 `ts-compatible`，每次实际视图以 run manifest 为准。2026-09-28 全局 Python/PyTorch 环境下的 A6000 原有 GPU 测试和 TS 特征路径单次 CUDA smoke 均通过；这不替代 frozen-lock、多 seed 或正式资源验收。正式 P3.2 仍需公开论文数值对照、Survey 矩阵接入及合作者复核；本技术预集成不得混入冻结 pilot 或主榜。
