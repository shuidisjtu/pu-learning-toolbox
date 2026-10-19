# P3Mix 原文核对与实现交叉表（2026-10-19）

范围：用户提供的 `30_who_is_your_right_mixup_partne.pdf`，PDF 14 页，ICLR 2022 论文（SHA-256：`8599cffab378b0e3dfb7586a55cef482587b7b29f187619833c00b0545d5f528`）。核对重点是论文登记、公式与训练时序映射、协议差异是否透明；不代表方法负责人签署或 Survey 正式准入。正确 OpenReview ID 为 `NH29920YEmj`，来源页列于[方法卡](../../method_cards/P3MIX.md)。本轮没有找到论文作者代码链接；因此源代码状态登记为 `not_found`，不推断作者没有私有/未索引实现。

| 原文断言 | 出处 | 当前实现/登记 | 复核边界 |
|---|---|---|---|
| 目标为 P 均值加 β 倍 U 均值，U 暂按伪负类处理 | §2，式 (1)–(3)，印刷页 3 | 训练损失保留原始 P/U 行角色，各自求 BCE 均值；不按 batch 行数合池 | P3Mix-C 修改 U 的软标签后，仍按原 U 角色归入 U 项 |
| 修改 mixup 系数为 λ′=max(λ,1−λ)，并同步混合软标签 | §2.1，式 (4)，印刷页 4 | `p3mix_heuristic_batch` | α=1 是论文实验设置；工具箱仍允许显式配置 |
| 边缘伪负例从候选正例池选伙伴，其他样本从当前小批次 P∪U 选伙伴；训练集合合并 shuffle 后切成小批次 | §2.1，式 (5)；Algorithm 1 第 3、6 行，印刷页 4 | P3Mix batch helper 和训练循环；只用当前训练批/训练 P 候选池 | 论文第 3 行明确写合并 P∪U shuffle；工具箱分别 shuffle P/U、按角色切批并循环较短组，以保证两个均值非空。该方案会重复使用较短角色的部分训练行，是采样适配差异，不能称为原文的逐行 batching 重放 |
| 边缘 U 条件为 1−γ≤f(x;Θ)≤γ | §2.1，式 (6)，印刷页 5 | 闭区间；gamma 范围 [0.5,1]；γ=0.5 时显式返回空集 | 公式在 γ=0.5 时退化为 `{f=0.5}`，与作者紧接着的“γ=0.5 implies empty”文字说明有张力；代码遵从该显式端点说明，新增精确 0.5 回归测试 |
| 每个 epoch 以正例预测熵 top-k 更新候选池 | §2.1，式 (7)，印刷页 5；Algorithm 1 第 12 行，印刷页 4 | 初始池从训练中观察到的 P 均匀无放回抽样，训练 P 上每 epoch 结束刷新 top-k 熵 | 只保留训练行身份；不读 clean labels/validation/test。k 大于训练 P 数时取全部 P |
| P3Mix-E 用 mean teacher 产生 mixup 行的软目标，最小化 log(1−〈f,目标〉) 正则并 EMA 更新 teacher | §2.2，式 (8)，Algorithm 1 第 1、9–13 行，印刷页 4–5 | variant=`e`：teacher 对混合输入输出 sigmoid 两类概率；按式 (8) 计算正则，agreement 以 `1−1e−7` 为上限避免 `log(0)`；每个 epoch 末按算法 1 更新 teacher state 的 EMA；当前以 student 副本初始化 teacher | Algorithm 1 明确将 Θ 与 eΘ 都随机初始化，但未说明两者是否独立；复制 student 是本实现的显式工程决定。原文未列 EMA decay；工具箱默认 0.99。η 默认 1.0 取自论文搜索集合，需正式 recipe 冻结 |
| P3Mix-C 将 f(x;Θ)>γ 的 U 行伪标签在 mixup 前改为正类 | §2.2，印刷页 5；Algorithm 1 第 7 行 | variant=`c`；改变目标标签但保留该行原始 U 损失组归属 | P3Mix-E 与 P3Mix-C 是分别报告的独立变体，API 不把两者合并 |
| Adam；α=1、k=100；β 搜索 `{0.8,0.9,1.0}`；γ 搜索 `{0.85,0.9,0.95}`；E 的 η 搜索 `{1,2,3,4,5}` | §3.1，印刷页 6 | 对应为可配置 estimator 参数；工程默认 beta=1、gamma=.9、alpha=1、pool_size=100、eta=1 | 论文未在正文给出学习率、batch size、epoch 数、mean teacher 衰减及足以重建的完整调参/选模流程；不能宣称数值复现 |
| 作者的图像实验用 LeNet-5（Fashion-MNIST）或 7-layer CNN（CIFAR-10/STL-10） | §3.1 表 1，印刷页 6 | 默认 MLP；图像要求用户显式注入可训练 encoder | 注入 encoder 是 toolbox 工程路径，不是论文 backbone；共享特征/初始化与正式资源仍须审批 |
| 报告五次运行并以 accuracy 展示主结果；还含消融、敏感度和信用卡 fraud 附录 | §3.2、表 2–4，印刷页 6–8、14 | 仅登记文献出处，不把文献数值复制到本项目矩阵 | 本轮没有用本项目数据进行 benchmark，也未调整冻结矩阵/历史 pilot |

## 公开数值锚点（来源记录，不并入冻结矩阵）

Table 2 的 CIFAR-10 比较读数及 Table 3 的 P3Mix 消融读数已另存为可机器读取的[锚点清单](../data/p3mix_literature_anchors_20261019.json)，其中绑定本地 PDF SHA-256、表格位置、指标单位、五次运行与协议事实。清单保持 `pending_method_owner_review`，且明确不改冻结矩阵；E/C 在 Table 2 与 Table 3 的重复列出被记为同篇论文内的重复报告，不视作独立样本。

原文 §3.1–3.2 称每个数据集/方法独立运行 5 次，表 2 与表 3 的指标为 classification accuracy（mean ± standard deviation）。CIFAR-10 的两种合成 PU 任务及原文值如下（百分数）：

| 原文任务 | 正类映射 | 来源表 | P3Mix | P3Mix-E | P3Mix-C | Supervised |
|---|---|---|---:|---:|---:|---:|
| CIFAR-10-1 | `{0,1,8,9}` | Table 3, p. 7 | 87.0 ± 1.1 | 88.2 ± 0.4 | 88.7 ± 0.4 | — |
| CIFAR-10-2 | `{2,3,4,5,6,7}` | Table 3, p. 7 | 84.3 ± 0.6 | 84.7 ± 0.5 | 87.9 ± 0.5 | — |
| CIFAR-10-1 | `{0,1,8,9}` | Table 2, p. 6 | — | 88.2 ± 0.4 | 88.7 ± 0.4 | 91.3 ± 0.3 |
| CIFAR-10-2 | `{2,3,4,5,6,7}` | Table 2, p. 6 | — | 84.7 ± 0.5 | 87.9 ± 0.5 | 91.3 ± 0.3 |

Table 2 还报告八种 PU 比较方法（accuracy, mean ± standard deviation；按 CIFAR-10-1 / CIFAR-10-2 顺序）：uPU `76.5 ± 2.5 / 71.6 ± 1.4`；nnPU `84.7 ± 2.4 / 83.7 ± 0.6`；nnPU+mixup `87.2 ± 0.6 / 85.8 ± 1.2`；Self-PU `85.1 ± 0.8 / 83.9 ± 2.6`；PAN `87.0 ± 0.3 / 82.8 ± 1.0`；VPU `86.8 ± 1.2 / 82.5 ± 1.1`；MIXPUL `87.0 ± 1.9 / 87.0 ± 1.1`；PULNS `87.2 ± 0.6 / 83.7 ± 2.9`。Appendix A 称 uPU/nnPU、Self-PU、PAN、VPU、MIXPUL 使用公开代码；nnPU+mixup 是论文描述的单独变体；PULNS 用作者自编 Python/三层 MLP selector；Supervised 是全监督分类器。机器锚点逐方法记录了这些来源陈述和可见设置。GitHub 当前将论文引用的 Self-PU 与 MIXPUL 地址分别重定向到 `VITA-Group/Self-PU` 和 `weitongseu/MixPUL`；这只记录当前仓库身份，不代表 Table 2 跑批所用的不可变版本。附录没有为 Table 2 跑批固定各仓库不可变版本，也未充分说明各法的验证/选模及 split/seed 绑定；故这些依然是论文报告值，不是可据此逐行重放的共同协议或 toolbox 复现。

数据与协议出处：§3.1 印刷页 5 说明沿用 CIFAR-10 默认 0–9 标签，按 (0,1,8,9) 与 (2,3,4,5,6,7) 构造两种互反任务；每个数据集随机抽取训练集中的 1,000 个正例，并报告使用 500 个 validation instances。**论文未说明这 500 个样本的抽样框、类别组成或如何用于模型/超参选择**，因此不能推断为固定比例 clean validation，也不能直接映射为本项目 PA/OA 选模口径。表 1（p. 6）指定 CIFAR-10 backbone 为 7-layer CNN。表 2 的 supervised 行是全监督参照，不是 PU 可比方法。论文在该处没有列出每次运行的数值或置信区间，所以不能从 mean±std 还原 seed 级观测。

与本项目的对应关系仅是**候选外部锚点**。冻结协议字节摘要为 `e0263cc4…`，CIFAR split index 摘要为 `42a9b4a0…`；可机器核对的差异清单见锚点 JSON 的 `project_protocol_comparison`。主要对照如下：

| 维度 | P3Mix 论文 | 本项目冻结 Pilot | 可比性判定 |
|---|---|---|---|
| 正类定义 | CIFAR-10-1 `{0,1,8,9}`；CIFAR-10-2 `{2,3,4,5,6,7}` | 单一正类 `{0,1,8,9}` | 仅 CIFAR-10-1 映射相同；CIFAR-10-2 是互补的正类任务 |
| 标记正例预算 | 固定 1,000 | SCAR `c={0.1,0.3,0.5}`；本地五个索引 split 各有 18,000 个训练正例，对应 1,800 / 5,400 / 9,000 个标记正例 | 均不相同；c 轴不能被当成固定 `n_P=1000` |
| 训练/验证角色 | 论文称有 500 个 validation instances，但未说明抽样框、类别组成或选模用途 | 训练 45,000；独立 `pu_val` 与 `clean_val` 各 2,500；PA 使用 PU proxy accuracy，OA 使用 clean-label accuracy | 划分和选择协议不一致，论文验证用途无法映射 |
| Backbone / 预处理 | 7-layer CNN；其 Table 2/3 运行的完整预处理/增强绑定未在本审计中确立 | 端到端 ResNet-18 或随机冻结 ResNet-18 特征；train-only 通道统计；无增强 | 模型和预处理路径不同，不作逐数对照 |
| 重复设计 | 5 次运行，未给本 Pilot split 身份 | 五个 split seed（0–4） | 重复数相同不代表同一 split 或配对重复 |

因此**不可直接做数值裁决**；本次只是把协议差异结构化，不新增或改写 `survey_comparison_v1.json`、冻结协议、既有 anchor、Pilot 结果或准入状态。Table 2/3 的相同 E/C 数字是同一论文中的重复报告，不是两份独立实验。Table 2 的 `Supervised` 两个 CIFAR 行也记录为全监督参照（91.3 ± 0.3），但其训练样本量/标签预算未充分说明；它们不是 PU 方法、不是同预算对照，也不等同于本项目 PN oracle。

## 工程接入状态

实现位于 `pu_toolbox/estimators/deep/p3mix.py`，分类器为 `P3MixClassifier`，registry/台账标为 NATIVE、实验级、未准入。当前只支持原文 OS 单训练集合路径；`os_or_ts="ts"` 明确拒绝。外部 `class_prior` 不参与目标；`sample_weight` 不支持。图像输入必须有 encoder，不做静默 flatten。每组 batch 上限为 `batch_size`，因此单次训练 forward 上限为两组之和；训练输入保留 CPU，按 batch 上卡。

实现是以全文公式/伪代码为基础的干净工程适配。缺失的 recipe 信息没有伪造成论文事实。局部测试验证三变体短拟合、两组损失、C 的高置信修正、E 的式 (8) 与稳定化、EMA 更新、真实 epoch 结束时训练 P top-k 熵池身份、失败重训失效以及推断模式恢复；它们不验证正式表现、论文数值、完整数据协议或资源上界。teacher 复制 student 初始化与原文“均随机初始化”之间的关系仍需方法负责人确认，不应解释为已证明等价。

2026-10-19 追加训练模式回归：候选池每轮刷新会暂时将 student 切至 `eval()`；新增两 epoch 前向轨迹断言，确认每次刷新后训练模式恢复，且下一 epoch 的优化批次仍处于 `train()`。这是生命周期/模式隔离回归，不改变训练算法或正式 recipe。

2026-10-20 数值与采样成本补核：公开 `p3mix_heuristic_batch` 原先在输入的原始 dtype 中写入候选伙伴；整数输入搭配浮点候选时会截断候选小数。现先按输入、候选与 float32 的共同 dtype 提升，再进行伙伴替换和式 (4) 混合。二维/四维、整数/float32 输入搭配 float64 候选的回归直接对照凸组合并确认输入不被改写。classifier 原本将训练输入统一为 float32，此修复不调整其默认训练轨迹。

同轮明确 Algorithm 1 第 3 行的合并 shuffle 与工具箱角色循环的差异：`training_provenance_` 记录工程选择、原始 P/U 行数和累计参与优化的行数；`epoch_provenance_` 逐轮记录 P/U 使用次数与重复次数，供训练预算和来源行为复核。例如 P=5、U=10、每角色 batch_size=4 时，一轮实际使用 9 个 P 行次、10 个 U 行次，P 中有 4 个重复行次。回归使用模型实际看到的原始训练行身份核对收据，覆盖 P 较少、U 较少和相同大小三种情况；这只披露现有 batching 行为，没有将其批准为正式配方。

行次的统计单位是每个优化 batch 中的原始 P/U **主样本角色**：不另计 mixup 伙伴引用、候选池熵刷新、原始输入打分或 teacher forward。同一个训练正例作为别人的伙伴时不会增加这里的 P 主样本计数。因此这些字段用于核对角色循环，不代表原始样本参与所有运算的总次数、完整前向量、计算成本或资源上界。

本轮 P3MIX estimator/math、文献锚点、候选源码绑定及台账来源/registry 一致性专项共 106 项通过，0 跳过、失败或错误；JUnit 在本机 `/tmp/p3mix-source-followup-20261020.xml`。格式、文档链接、测试质量及候选源码绑定门禁通过。该测试范围不包含正式数据训练、GPU 验收或作者结果重放。

## 未解除的前置

- HENG958 按 RV8 复核本交叉表、batching/网络/优化器/EMA 工程决定和实现边界；审阅状态仍 pending。
- 正式数据/label map、PA/OA 选模规则、P3Mix 训练预算、超参搜索范围和资源规格须单独登记批准。
- 论文只报告 500 个 validation instances，未给抽样框、标签组成和选模用途；正式 recipe 不得把它直接解释为 clean-label validation 或 PA/OA 验证集。
- 需要实际执行批准后的公开数据复现与多 seed/GPU 资源验收；本轮不改 `survey_protocol_v1.json`、冻结矩阵或 645-run 结果。
