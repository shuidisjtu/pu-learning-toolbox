# nnPU 来源与行为核查（2026-10-19）

范围：交叉核对 Kiryo et al. NIPS 2017 论文 Algorithm 1、作者 Chainer `nnPUlearning` 的可见损失/训练代码、toolbox PyTorch loss 与训练循环。属于单方技术审计，不是方法负责人签署、正式准入或实验数值复现。

## 来源与可复核版本

- 原论文：[NIPS 2017 PDF](https://papers.nips.cc/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf)。式 (6) 定义报告的非负风险；Algorithm 1（印刷第 4 页）对每对 mini-batch `X_p^i, X_u^i` 计算负风险分支 `r_i`，按 `r_i ≥ -β` 使用完整 uPU 梯度，否则沿 `-∇r_i` 更新并用 `γη` 折扣步长。
- 作者代码：[kiryor/nnPUlearning](https://github.com/kiryor/nnPUlearning)，论文 §5 明确给出仓库地址。可直接阅读的 [`pu_loss.py`](https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py) 将报告值 `self.loss` 与反传用 `self.x_out` 分开：校正分支报告 `R_p^+ - β`，反传 `-γ r`；官方 [`train.py`](https://github.com/kiryor/nnPUlearning/blob/master/train.py) 用 Chainer `SerialIterator` 迭代合并的 `XYtrain`。本仓库 `official_sources.lock.json` 尚无该仓库 immutable commit，故以上 URL 指向可变 `master`，是源码行为线索而非版本锁证明。

## 对照结论

| 项目 | 论文 / 作者代码 | toolbox 实现 | 结论 |
|---|---|---|---|
| 风险分解 | `r=R_u^-−πR_p^-`，报告值为 `πR_p^+ + max(0,r)`（β=0 时）；β>0 时 Algorithm 1 容忍到 `−β` | `NonNegativePULoss.evaluate` 单独计算 `R_p^+`,`r`,`uPU`,`nnPU` 报告风险 | 与式 (6) 及 β 扩展一致。 |
| 校正分支反传 | 官方 `pu_loss.py` 在 `r < −β` 时输出报告 `πR_p^+−β`，而 backward source 是 `−γr`，不是 `max` 的直接导数 | `_nnpu_train_step` 用 detached 正风险构建报告量，并让 `opt_loss=-γr` 决定梯度 | 核心梯度行为一致；本地单元测试对两分支直接核验。 |
| mini-batch 角色/构成 | 论文 Algorithm 1 将 `(X_p^i,X_u^i)` 作为成对角色 mini-batch；作者 `train.py` 则将合并 `XYtrain` 交给随机 Chainer iterator，loss 按批次内标签数分别归一化（每类计数用 `max(1,count)` 防零） | toolbox 拆出 P/U 两个 loader、独立 shuffle，批大小各自受组大小限制，较短 loader 循环到最长 loader；TS 时同一个 P batch 仍进正例风险并另拼入 U 风险输入 | toolbox 在角色层面遵循论文伪代码，但 mini-batch 组成/每 epoch 步数不等于作者合并迭代器的实现；不可声称逐代码复现。其数值影响需在批准 recipe 下比较，不能靠单元测试断定方向。 |
| 优化器/模型 | 论文允许外部 SGD-like 优化器；官方训练脚本按配置选择 Chainer 模型与优化器 | clean-room PyTorch，默认 MLP/Adam，支持注入 encoder 和 GPU | 工程扩展，不等同论文全部网络、数据处理与 200-epoch 实验协议。 |
| TS 视图 | 原论文为 case-control `P~p(x|y=1), U~p(x)`，无 TS-OS 校准规则 | 项目协议允许逐训练 mini-batch 将 P batch 同时放入 U 风险角色；验证继续按 OS 角色 | 项目校准叠加，RV6 方法学确认仍待；非原文结论。 |

## 现有测试覆盖

- `tests/unit/estimators/test_nnpu.py` 检验公式风险、β/γ 分支、报告风险与优化量区别、梯度方向及训练状态。
- `tests/unit/losses/test_nnpu_loss.py` 检验 sigmoid risk golden、非负风险性质和参数边界。
- `tests/unit/estimators/test_nnpu_ts_view.py` 检验 TS mini-batch 角色替换及视图 gate。
- 上述测试支撑 toolbox 内部数学与协议不变量；由于官方仓库未锁定 commit，不能证明这些分支来自某一固定 Chainer 代码版本，更不能证明完整 NIPS 表格数值已重现。

结论：nnPU 核心校正梯度与论文/可见官方实现一致；PyTorch 实现、P/U loader 组成与 TS 校准均是需披露的适配层。应补官方源码 immutable commit/hash 后再做版本化文件级复核。该记录不改算法默认值、冻结结果、准入或 owner review 状态。
