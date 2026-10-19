# nnPU 来源与行为核查（2026-10-19）

范围：交叉核对 Kiryo et al. NIPS 2017 论文 Algorithm 1、作者 Chainer `nnPUlearning` 的可见损失/训练代码、toolbox PyTorch loss 与训练循环。属于单方技术审计，不是方法负责人签署、正式准入或实验数值复现。

## 来源与可复核版本

- 原论文：[NIPS 2017 PDF](https://papers.nips.cc/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf)。式 (6) 定义报告的非负风险；Algorithm 1（印刷第 4 页）对每对 mini-batch `X_p^i, X_u^i` 计算负风险分支 `r_i`，按 `r_i ≥ -β` 使用完整 uPU 梯度，否则沿 `-∇r_i` 更新并用 `γη` 折扣步长。
- 作者代码：[kiryor/nnPUlearning](https://github.com/kiryor/nnPUlearning)，论文 §5 明确给出仓库地址。GitHub 提交历史显示观察到的最新 `master` revision 为 [`a92c5365c07490c871e338c07f6e31d3eae2ca37`](https://github.com/kiryor/nnPUlearning/commit/a92c5365c07490c871e338c07f6e31d3eae2ca37)，提交日期 2022-12-13，提交只改动 `dataset.py`；现已写入 `benchmarks/assigned_methods/configs/official_sources.lock.json`。锁定文件 [`pu_loss.py`](https://github.com/kiryor/nnPUlearning/blob/a92c5365c07490c871e338c07f6e31d3eae2ca37/pu_loss.py) 将报告值 `self.loss` 与反传用 `self.x_out` 分开：校正分支报告 `R_p^+ - β`，反传 `-γ r`；[`train.py`](https://github.com/kiryor/nnPUlearning/blob/a92c5365c07490c871e338c07f6e31d3eae2ca37/train.py) 用 Chainer `SerialIterator` 迭代合并的 `XYtrain`。来源 pin 证明审计目标可复核，不意味着 toolbox 逐行重放或数值复现。
- 2026-10-20 对该 immutable commit 做独立干净检出，逐一比较 `pu_loss.py`、`train.py`、`dataset.py`、`model.py`、`README.md`、`LICENSE` 的工作区原始字节与对应 Git blob；六个文件的 SHA-256 写入来源锁 `sources.nnpu.file_sha256`，摘要口径为 `sha256_git_blob_bytes`。该口径绑定原始 Git 对象字节，Windows 工作区若做 CRLF 转换应通过 `git show <commit>:<path>` 重算。此次仅阅读文件，没有导入/执行 Chainer 或下载实验数据。

## 对照结论

| 项目 | 论文 / 作者代码 | toolbox 实现 | 结论 |
|---|---|---|---|
| 风险分解 | `r=R_u^-−πR_p^-`，报告值为 `πR_p^+ + max(0,r)`（β=0 时）；β>0 时 Algorithm 1 容忍到 `−β` | `NonNegativePULoss.evaluate` 单独计算 `R_p^+`,`r`,`uPU`,`nnPU` 报告风险 | 与式 (6) 及 β 扩展一致。 |
| 校正分支反传 | 官方 `pu_loss.py` 在 `r < −β` 时输出报告 `πR_p^+−β`，而 backward source 是 `−γr`，不是 `max` 的直接导数 | `_nnpu_train_step` 用 detached 正风险构建报告量，并让 `opt_loss=-γr` 决定梯度 | 核心梯度行为一致；本地单元测试对两分支直接核验。 |
| mini-batch 角色/构成 | 论文 Algorithm 1 将 `(X_p^i,X_u^i)` 作为成对角色 mini-batch；作者 `train.py` 则将合并 `XYtrain` 交给随机 Chainer iterator，loss 按批次内标签数分别归一化（每类计数用 `max(1,count)` 防零） | toolbox 拆出 P/U 两个 loader、独立 shuffle，批大小各自受组大小限制，较短 loader 循环到最长 loader；TS 时同一个 P batch 仍进正例风险并另拼入 U 风险输入 | toolbox 在角色层面遵循论文伪代码，但 mini-batch 组成/每 epoch 步数不等于作者合并迭代器的实现；不可声称逐代码复现。其数值影响需在批准 recipe 下比较，不能靠单元测试断定方向。 |
| 优化器/模型 | `train.py:100-104` 固定 Adam 并附加梯度 WeightDecay(0.005)；CIFAR preset 在 `:70-76` 设 batch=500、lr=1e-5、CNN，通用 CLI 默认 100 epochs（`:41`）；`model.py:117-178` 是九个卷积、BN 与三个全连接层 | clean-room PyTorch，默认单层线性头、Adam(lr=1e-3)、weight_decay=0；支持显式模型/encoder 和 GPU | 结构、优化器正则、学习率和 epoch 口径均需独立冻结；源码默认不是论文 200-epoch 设置的充分重建。 |
| CIFAR 数据/先验与 P/U 行身份 | `dataset.py:102-107` 正类为 `{0,1,8,9}`；`:122-130` 区分 `n_P+n_U=n_train` 与 `n_U=n_train`，前者从 U 排除 P，后者把已标记 P 追加回正例 U 段；返回先验为实际 U 中的正类比例。`exp-cifar` 的 P=1000、U=50000 走后者 | 本项目原始 split 的 P/U 身份分离，TS 校准在逐训练批次向 U 风险角色另拼 P；人口 π 由协议/调用方提供 | 作者整体 U 对同一训练数据的复用，不等于独立 case-control 抽样，也不等于 toolbox 逐 mini-batch 的校准复制；P/U 数量、角色频率和先验口径须分别登记。 |
| 图像预处理 | `dataset.py:94-98` 转 float32 后除以 255，无通道均值/标准差归一化 | Pilot 使用 train-only 通道统计，encoder 为协议指定结构 | 预处理不同，即使 CIFAR 正类映射相同也不足以直接裁决数值。 |
| 验证/测试使用 | `train.py:223-227,243-256` 在训练集合上构造 PU evaluator，并逐 epoch 监控独立测试集合的 error；可见主程序没有选中 checkpoint 保存或独立 clean/PU 验证集划分 | 项目 PA/OA 从对应独立验证角色选 checkpoint，test 留作最终评价 | 不能把作者 test error 曲线、训练 PU proxy 或最终状态自动认作本项目 PA/OA 选中模型。 |
| TS 视图 | 原论文为 case-control `P~p(x|y=1), U~p(x)`，无 TS-OS 校准规则 | 项目协议允许逐训练 mini-batch 将 P batch 同时放入 U 风险角色；验证继续按 OS 角色 | 项目校准叠加，RV6 方法学确认仍待；非原文结论。 |

## 现有测试覆盖

- `tests/unit/estimators/test_nnpu.py` 检验公式风险、β/γ 分支、报告风险与优化量区别、梯度方向及训练状态。
- `tests/unit/losses/test_nnpu_loss.py` 检验 sigmoid risk golden、非负风险性质和参数边界。
- `tests/unit/estimators/test_nnpu_ts_view.py` 检验 TS mini-batch 角色替换及视图 gate。
- 上述测试支撑 toolbox 内部数学与协议不变量；官方 commit 和关键文件字节现已绑定，但没有执行两套框架的逐步数值对照，因此不证明完整训练轨迹或 NIPS 表格数值已重现。
- 2026-10-20 来源/registry 契约、assigned preflight、nnPU estimator、TS 视图与 loss 专项共 69 项通过，0 跳过、失败或错误；两条极大 β 的预期 warning 来自现有边界用例。JUnit 在本机 `/tmp/nnpu-source-snapshot-20261020.xml`；六文件的 Git blob/工作区摘要一致性和文档链接门禁均通过。

## 本地生命周期补核（2026-10-20）

独立复现了两项 toolbox 状态问题：成功拟合后以非法先验重训会抛出异常，但旧 `_is_fitted=True` 仍允许预测；评分则递归执行 `eval()`，覆盖调用方的混合 BN/Dropout 模式并永久保留。前者也影响参数绑定失败、epoch 回调异常或取消，后者在前向或部分 `eval()` 失败时同样会遗留状态。

`NonNegativePUClassifier.fit` 现使用既有 `invalidate_failed_fit`：参数绑定前撤销公开拟合状态，异常/取消原样传播，成功后仍由 estimator 自行设置完成状态。该保护不回滚模型权重、history 或回调写出的外部制品。评分逐子模块保存/直接恢复 training 标志，覆盖正常前向、前向异常和部分 eval 转换异常；空输入直接返回空分数，不访问模型。fit 成功路径的目标、梯度、步数、epoch 编号、默认值和 TS 路由保持原行为；`predict_proba` 仍不提供。

回归位于 `tests/unit/estimators/test_nnpu_lifecycle.py`：校验非法先验/未知参数后旧预测失效、四种回调异常只传播一次、后续成功重训、与未包装训练的分数/history 一致、签名/clone/pickle，以及混合模式和 BN buffers 在评分后不变。候选专用的 16 方法源码身份草稿不因此扩展为全方法准入或训练批准。

生命周期、既有 nnPU/loss/TS、encoder、checkpoint 与管线专项共 107 项通过、4 项 CUDA 测试跳过，0 失败或错误（本机 JUnit `/tmp/nnpu-lifecycle-20261020.xml`）。当前工作区全仓快速回归 `python -m pytest tests/ -m 'not slow and not e2e'` 共 4973 项通过、59 项跳过、0 失败/错误，用时 481.880 秒，JUnit `/tmp/pu-independent-followup-fast-20261020.xml`。15 项本地静态/证据门禁通过；可选条件未实际执行的正式 registry 门禁、GPU 与 slow/e2e 不计作验收。现有 NVML、代价/消融、收敛及非正式 checkpoint 容量警告仍存在，不声明零 warning。

结论：nnPU 核心校正梯度与论文/锁定官方实现一致；PyTorch 实现、P/U loader 组成与 TS 校准均是需披露的适配层。2026-10-20 已补齐六个关键上游文件的 hash，原文档与台账中“未锁定 commit”“未保存上游 hash”的旧限制同步撤除；完整 Chainer 运行复现、逐步数值对照及相同实验协议证据仍未交付。该记录不改算法默认值、冻结结果、准入或 owner review 状态。
