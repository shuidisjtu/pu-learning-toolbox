# P3Mix 原文核对与实现交叉表（2026-10-19）

范围：用户提供的 `30_who_is_your_right_mixup_partne.pdf`，PDF 14 页，ICLR 2022 论文（SHA-256：`8599cffab378b0e3dfb7586a55cef482587b7b29f187619833c00b0545d5f528`）。核对重点是论文登记、公式与训练时序映射、协议差异是否透明；不代表方法负责人签署或 Survey 正式准入。正确 OpenReview ID 为 `NH29920YEmj`，来源页列于[方法卡](../../method_cards/P3MIX.md)。本轮没有找到论文作者代码链接；因此源代码状态登记为 `not_found`，不推断作者没有私有/未索引实现。

| 原文断言 | 出处 | 当前实现/登记 | 复核边界 |
|---|---|---|---|
| 目标为 P 均值加 β 倍 U 均值，U 暂按伪负类处理 | §2，式 (1)–(3)，印刷页 3 | 训练损失保留原始 P/U 行角色，各自求 BCE 均值；不按 batch 行数合池 | P3Mix-C 修改 U 的软标签后，仍按原 U 角色归入 U 项 |
| 修改 mixup 系数为 λ′=max(λ,1−λ)，并同步混合软标签 | §2.1，式 (4)，印刷页 4 | `p3mix_heuristic_batch` | α=1 是论文实验设置；工具箱仍允许显式配置 |
| 边缘伪负例从候选正例池选伙伴，其他样本从当前小批次 P∪U 选伙伴 | §2.1，式 (5)，印刷页 4 | P3Mix batch helper 和训练循环；只用当前训练批/训练 P 候选池 | 两组独立 shuffle 后按角色切批并循环较短组，是保证两个均值非空的工程 batching 方案；论文未精确定义每个 batch 的角色采样实现 |
| 边缘 U 条件为 1−γ≤f(x;Θ)≤γ | §2.1，式 (6)，印刷页 5 | 闭区间；gamma 范围 [0.5,1]；γ=0.5 时显式返回空集 | 公式在 γ=0.5 时退化为 `{f=0.5}`，与作者紧接着的“γ=0.5 implies empty”文字说明有张力；代码遵从该显式端点说明，新增精确 0.5 回归测试 |
| 每个 epoch 以正例预测熵 top-k 更新候选池 | §2.1，式 (7)，Algorithm 1 第 12 行，印刷页 5 | 初始池从训练中观察到的 P 均匀无放回抽样，训练 P 上每 epoch 结束刷新 top-k 熵 | 只保留训练行身份；不读 clean labels/validation/test。k 大于训练 P 数时取全部 P |
| P3Mix-E 用 mean teacher 产生 mixup 行的软目标，最小化 log(1−〈f,目标〉) 正则并 EMA 更新 teacher | §2.2，式 (8)，Algorithm 1 第 1、9–13 行，印刷页 4–5 | variant=`e`：teacher 对混合输入输出 sigmoid 两类概率；按式 (8) 计算正则，agreement 以 `1−1e−7` 为上限避免 `log(0)`；每个 epoch 末按算法 1 更新 teacher state 的 EMA；当前以 student 副本初始化 teacher | Algorithm 1 明确将 Θ 与 eΘ 都随机初始化，但未说明两者是否独立；复制 student 是本实现的显式工程决定。原文未列 EMA decay；工具箱默认 0.99。η 默认 1.0 取自论文搜索集合，需正式 recipe 冻结 |
| P3Mix-C 将 f(x;Θ)>γ 的 U 行伪标签在 mixup 前改为正类 | §2.2，印刷页 5；Algorithm 1 第 7 行 | variant=`c`；改变目标标签但保留该行原始 U 损失组归属 | P3Mix-E 与 P3Mix-C 是分别报告的独立变体，API 不把两者合并 |
| Adam；α=1、k=100；β 搜索 `{0.8,0.9,1.0}`；γ 搜索 `{0.85,0.9,0.95}`；E 的 η 搜索 `{1,2,3,4,5}` | §3.1，印刷页 6 | 对应为可配置 estimator 参数；工程默认 beta=1、gamma=.9、alpha=1、pool_size=100、eta=1 | 论文未在正文给出学习率、batch size、epoch 数、mean teacher 衰减及足以重建的完整调参/选模流程；不能宣称数值复现 |
| 作者的图像实验用 LeNet-5（Fashion-MNIST）或 7-layer CNN（CIFAR-10/STL-10） | §3.1 表 1，印刷页 6 | 默认 MLP；图像要求用户显式注入可训练 encoder | 注入 encoder 是 toolbox 工程路径，不是论文 backbone；共享特征/初始化与正式资源仍须审批 |
| 报告五次运行并以 accuracy 展示主结果；还含消融、敏感度和信用卡 fraud 附录 | §3.2、表 2–4，印刷页 6–8、14 | 仅登记文献出处，不把文献数值复制到本项目矩阵 | 本轮没有用本项目数据进行 benchmark，也未调整冻结矩阵/历史 pilot |

## 公开数值锚点（来源记录，不并入冻结矩阵）

同一组 CIFAR-10 读数已另存为可机器读取的[锚点清单](../data/p3mix_literature_anchors_20261019.json)，其中绑定本地 PDF SHA-256、表格位置、指标单位、五次运行与协议事实。清单保持 `pending_method_owner_review`，且明确不改冻结矩阵；E/C 在 Table 2 与 Table 3 的重复列出被记为同篇论文内的重复报告，不视作独立样本。

原文 §3.1–3.2 称每个数据集/方法独立运行 5 次，表 2 与表 3 的指标为 classification accuracy（mean ± standard deviation）。CIFAR-10 的两种合成 PU 任务及原文值如下（百分数）：

| 原文任务 | 正类映射 | 来源表 | P3Mix | P3Mix-E | P3Mix-C | Supervised |
|---|---|---|---:|---:|---:|---:|
| CIFAR-10-1 | `{0,1,8,9}` | Table 3, p. 7 | 87.0 ± 1.1 | 88.2 ± 0.4 | 88.7 ± 0.4 | — |
| CIFAR-10-2 | `{2,3,4,5,6,7}` | Table 3, p. 7 | 84.3 ± 0.6 | 84.7 ± 0.5 | 87.9 ± 0.5 | — |
| CIFAR-10-1 | `{0,1,8,9}` | Table 2, p. 6 | — | 88.2 ± 0.4 | 88.7 ± 0.4 | 91.3 ± 0.3 |
| CIFAR-10-2 | `{2,3,4,5,6,7}` | Table 2, p. 6 | — | 84.7 ± 0.5 | 87.9 ± 0.5 | 91.3 ± 0.3 |

数据与协议出处：§3.1 印刷页 5 说明沿用 CIFAR-10 默认 0–9 标签，按 (0,1,8,9) 与 (2,3,4,5,6,7) 构造两种互反任务；每个数据集随机抽取训练集中的 1,000 个正例，并报告使用 500 个 validation instances。**论文未说明这 500 个样本的抽样框、类别组成或如何用于模型/超参选择**，因此不能推断为固定比例 clean validation，也不能直接映射为本项目 PA/OA 选模口径。表 1（p. 6）指定 CIFAR-10 backbone 为 7-layer CNN。表 2 的 supervised 行是全监督参照，不是 PU 可比方法。论文在该处没有列出每次运行的数值或置信区间，所以不能从 mean±std 还原 seed 级观测。

与本项目的对应关系仅是**候选外部锚点**：CIFAR-10-1 正类映射与本项目 `{0,1,8,9}` 一致，但论文 7-layer CNN 与本项目 ResNet-18 小图 stem 不同；论文的 1,000 个已标记正例、validation 构造、训练/预处理细节和本项目 split/label budget/选模口径须逐项核对后才能判断可比性。Table 2/3 的相同 E/C 数字是同一论文中的重复报告，不是两份独立实验。上述值只补入本来源审计文档，**不新增或改写** `survey_comparison_v1.json`、冻结协议、既有 anchor、Pilot 结果或准入状态。

## 工程接入状态

实现位于 `pu_toolbox/estimators/deep/p3mix.py`，分类器为 `P3MixClassifier`，registry/台账标为 NATIVE、实验级、未准入。当前只支持原文 OS 单训练集合路径；`os_or_ts="ts"` 明确拒绝。外部 `class_prior` 不参与目标；`sample_weight` 不支持。图像输入必须有 encoder，不做静默 flatten。每组 batch 上限为 `batch_size`，因此单次训练 forward 上限为两组之和；训练输入保留 CPU，按 batch 上卡。

实现是以全文公式/伪代码为基础的干净工程适配。缺失的 recipe 信息没有伪造成论文事实。局部测试验证三变体短拟合、两组损失、C 的高置信修正、E 的式 (8) 与稳定化、EMA 更新、真实 epoch 结束时训练 P top-k 熵池身份、失败重训失效以及推断模式恢复；它们不验证正式表现、论文数值、完整数据协议或资源上界。teacher 复制 student 初始化与原文“均随机初始化”之间的关系仍需方法负责人确认，不应解释为已证明等价。

2026-10-19 追加训练模式回归：候选池每轮刷新会暂时将 student 切至 `eval()`；新增两 epoch 前向轨迹断言，确认每次刷新后训练模式恢复，且下一 epoch 的优化批次仍处于 `train()`。这是生命周期/模式隔离回归，不改变训练算法或正式 recipe。

## 未解除的前置

- HENG958 按 RV8 复核本交叉表、batching/网络/优化器/EMA 工程决定和实现边界；审阅状态仍 pending。
- 正式数据/label map、PA/OA 选模规则、P3Mix 训练预算、超参搜索范围和资源规格须单独登记批准。
- 论文只报告 500 个 validation instances，未给抽样框、标签组成和选模用途；正式 recipe 不得把它直接解释为 clean-label validation 或 PA/OA 验证集。
- 需要实际执行批准后的公开数据复现与多 seed/GPU 资源验收；本轮不改 `survey_protocol_v1.json`、冻结矩阵或 645-run 结果。
