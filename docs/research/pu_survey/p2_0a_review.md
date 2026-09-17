# P2.0a 合作者复核包

状态日期：2026-09-17。**已由 shuidisjtu 于 2026-09-17 签署验收**（记录见 §4）。
主责 HENG958；复核人 shuidisjtu。依据 [执行计划](survey_execution_plan.md) §2.0，
代码和测试通过不等同于合作者同意，不得由自动化代签或自行清除 `collaborator_review`。

## 1. 复核对象

- 交付提交：`83236d8`（P2.0a 交付对象：矩阵、装配、runner 门禁与 CPU/GPU smoke）。
  验证时的 HEAD 为 `0c90f27`（含 A3a 修复与五项协议返工）；两者不同是预期的，
  交付对象不因后续修复而前移。若需回退到 `83236d8` 重新验证，说明验证范围已变。
- 已签署的矩阵 `survey_protocol_v1.json` 实际版本为 `survey-v1.2`；
  根据合作者的“要求修改”审核，修正 KLDCE 可运行性、Self-PU 预算与矩阵校验。
  `selection_spec_kind=descriptive_documentation` 表明其中八项文字不是 runner 消费的
  可执行真相源；实际选模行为以代码和对应测试为准。
- 当前待审 canonical SHA-256：
  `faa9084c631a3852ed75d104c9c6dbe1514043a3f21db15e4d9bab674a42ff16`。
  若 JSON 内容变更，须重新计算，不能沿用此摘要签署。
- `--protocol survey-v1.2` 是当前入口；`survey-v1`/`survey-v1.1` 保留为当前规格别名，
  实际版本与 JSON 内容摘要以 manifest 为准，不能据 CLI 别名将结果记为旧版。
- 交付与测试证据：[P2.0a 交付](p2_0a_delivery.md)、
  [逐 epoch checkpoint 交付](epoch_checkpoint_delivery.md)。
- 不纳入本次验收：P2.0b 标签语义契约、P2.0c 对照预注册、正式跑批及主榜。

## 2. 决策清单（逐条确认，不默认通过）

| ID | 当前工程决定 | 复核时需要确认 |
|---|---|---|
| R1 | nnPU/Self-PU/MLP oracle 使用共享 MLP128；IMDB 固定 384 维输入 | 接受 score blueprint；确认 IMDB L2 归一化及 Spambase train-only 标准化实际数据口径 |
| R2 | mini-batch、Self-PU 双学生抽样、full-batch 分组；经典方法按闭式/迭代/内置 CV 真正成本记录 | 接受各预算真实单位及候选池；Self-PU 每 epoch 仅两次抽样优化更新，无匹配 mini-batch oracle，不得混排 |
| R3 | CIFAR 小图 ResNet-18，随机初始化、train-only 通道统计、当前不增强 | 接受该工程默认；若要求增强，先补真实训练 hook 并更新版本，不仅改注释 |
| R4 | adapter 使用随机、未训练、冻结的 ResNet 特征；按 split/seed 共享缓存 | 是否接受为 random-feature 工程补充；若研究要求已训练表征，当前 adapter 结果不能代表它 |
| R5 | native CNN 与 adapter 分榜；同路径仍检查预算/特征/标签摘要 | 接受比较边界与非正式产物隔离 |
| R6 | mini-batch MLP/adapter oracle 已接入；native-CNN oracle、Dist-PU full-batch oracle 尚缺 | 接受阶段 A 书面对齐决定；缺失对照继续单列或不可运行，后续专项实现 |
| R7 | Self-PU 本次不接入 clean validation，是明确的 no-meta 消融；逐 epoch 同时保存两个 teacher | 接受消融报告范围；完整 OA meta-reweighting 与 PA-ineligible 路径另立任务，不能向 PA 泄漏真实验证标签 |
| R8 | 固定预算一次训练，逐 epoch 权重离线独立选模；同分取较早候选/epoch/teacher/阈值 | 接受 checkpoint 语义、保留策略、存储成本与保存加载证据；它不支持 optimizer/RNG 续训 |
| R9 | PA 现有实现仍是 PU 分离度代理，OA 用真实验证 Accuracy 与阈值网格 | **未决**：锁定正式 PA 准则与阈值规则；该代理不等于协议 PA Accuracy，SCAR PA 正式验收仍阻断 |
| R10 | A6000/CUDA 12.4 技术 smoke 可运行；环境记录来自现有服务器 Python | 认可技术验证，但完整 pilot 仍需核对 uv.lock 环境、数据产物与独占/共享 GPU 窗口。**已确认（2026-09-17）**：win32 分支的 frozen-lock GPU 证据由复核人本人亲跑（8 passed / 0 skipped）；Linux 分支按方案 3 记录环境偏差并保持正式阻断。详见 §4 的 R10 与"冻结依赖 GPU 验证：已决议" |

R9 是新增发现的协议差距，不能因为 checkpoint 接线已完成就删除其正式阻断。
复核若要求修改任一规格，应记录理由、更新版本、重跑受影响测试和数据单元。

### 审核发现 A3：CIFAR split manifest 来源信息

- A3a：split 准备阶段未施加增强，却把 estimator 默认 `simaugment` 记入 manifest；
  合作者提交 `47f62c3` 已把此字段显式锁定为 `none`。
- A3b（[issue #52](https://github.com/shuidisjtu/pu-learning-toolbox/issues/52)）：
  大规模、HWC 存储转置而成的非连续 NCHW 数组在 `float32` 通道归约时累积精度丢失，
  45,000 张 CIFAR train 图的通道和饱和在约 `2**24`，导致各通道均值同为
  `2**24 / (45000*32*32) ≈ 0.364089`。`np.std` 的两遍算法复用这个饱和中心，std 因而
  同样被污染（旧值 `0.259735` vs 真值 `0.247096`）；"错误中心抬高"与"第二遍 float32
  累加压低"两个相反误差部分抵消，偏差仅约 5%，长期未被发现。工程代码已改用 `float64`
  累积均值与方差，并加入 4,100 张非连续图像的回归测试；输出图像仍为原来的 `float32`
  缩放表示。
- **历史产物已重建并验收（2026-09-17）**：在持有原始 CIFAR 数据的环境重新生成 5 个
  seed，逐 seed 比对 manifest 的 mean/std 与 `train.npz` 的 `float64` 重算值，并以
  解压后字节比对确认划分/标签/图像未变；跨 seed 统计量互不相同。新 manifest digest
  与命令见 [issue #52](https://github.com/shuidisjtu/pu-learning-toolbox/issues/52)
  （已关闭）。制品已迁入标准路径 `data/splits/cifar10/` 并重验通过，旧制品移入归档。
  该缺陷只出现在**制品生成路径**（`load_cifar10` 的非连续转置视图直通统计拟合）；
  运行路径经 `np.load` 读入连续数组、且 `prepare_image_bundle` 当场重算统计量，
  因此从未影响训练数值。A3b 的数据验收至此关闭。

### 审核修改 B：执行矩阵与 IMDB 来源信息

- 3 条 KLDCE 行均显式标为不可运行，并各自注明缺少 `flip_probability` 安全绑定；
  resolver 在读取数据前返回该原因。`h = 1 - c` 仅可作为未来 SCAR 方案候选，SAR
  应另行决定拒绝、估值及偏离记录，不能擅自套用。
- `load_protocol` 对审核状态、seeds、候选池、正式阻断项、`selection_spec` 及执行行
  必填字段和 JSON 布尔型进行 fail-closed 校验；不可运行行必须有原因。
- Self-PU 的实际训练为每个 epoch 对两位 student 各取一批 P/U 并各更新一次，而非
  全量遍历；独立预算 `two_student_sampled`、独立 comparability group，并有真实
  optimizer step 次数测试。该组不宣称具有匹配的 mini-batch oracle。
- 真实 SBERT 后端的 manifest 同时保留调用参数 `normalize_embeddings=false` 与
  `effective_output_normalization=l2_unit_norm`、
  `normalization_source=model_pipeline_module`；写入前对输出向量范数检查。
  注入测试 encoder 则按实际向量范数和注入来源记录，不伪称真实模型流水线。
  模型公开的 [modules.json](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/modules.json)
  列出 `Normalize` 模块；运行时数值检查仍是当前 revision 输出的最终判据。
- 上述工程修改响应审核；复核人已完成受影响范围的定向复验并签署（2026-09-17）：
  `review_status` 现为 `accepted`，`collaborator_review` 已从 `formal_blockers` 移除，
  逐条结论见 §4。

### 冻结依赖 GPU 验证：已决议（2026-09-17）

之前的全局 Python/CUDA 12.4 smoke 不能替代 frozen-lock 验证。`uv.lock` 的 Linux
PyTorch 2.13.0 依赖 CUDA 13 组件；前次沙箱外记录的驱动 550.54.14 低于 CUDA 13.0 GA
所需的 580.65.06（[NVIDIA CUDA 13.0 发布说明](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html)）。
该沙箱内 `nvidia-smi` 无法连接驱动，故未运行 frozen-lock GPU 套件。

**决议：方案 3——记录环境偏差并保持正式阻断**，已写入 `formal_blockers` 的
`linux_frozen_lock_environment_deviation`。需要澄清的是：**"frozen-lock 从未被验证"
并不成立**——win32 分支已由复核人本人在本机亲跑通过（8 passed / 0 failed / 0 skipped，
见 §4 的 R10）；未验证的是 **Linux 分支**。仍要求交付方提交该分支的实际失败/偏差证据
（"无法执行 + 原因 + 证据"），不得以全局 torch 2.6/cu124 的旧 smoke 顶替。

## 3. 复核前检查

1. 阅读 JSON 预算、执行单元与 `selection_spec`，不要仅阅读文档摘要。
2. 运行 checkpoint 专项与 GPU 测试，检查 PA/OA 选中不同 epoch 的反例。
3. 抽查 manifest：路径分组、完整 checkpoint 列表、各 epoch 验证指标、选择引用、实际版本。
4. 核对真实 split；工程合成数据 smoke 不等于 P1.4 真实产物已在本服务器验收。
5. 明确区分“接受工程规格”“接受方法消融范围”“允许正式 pilot”，三者不能相互替代。

## 4. 签署记录

```text
reviewer: shuidisjtu
review_date: 2026-09-17
delivery_commit: 83236d893c1cb1ef557129bfbb3e4114462617bf
verification_head: 0c90f27f73db7fa8f894089fc4c5254c790ffb95
reviewed_protocol_version: survey-v1.2
reviewed_protocol_sha256: b5b6b5f4945bf5bc1823eee6cddf487595d409b93f48911f0f1797b89ed2ff20
  （签署前候选值 faa9084c631a3852ed75d104c9c6dbe1514043a3f21db15e4d9bab674a42ff16。
    摘要包含 review_status 与 formal_blockers，故本次签署必然改变它；摘要未变即为失败。）

R1  接受。接受 score blueprint。书面写明实际数据口径：Spambase 为 train-only 逐特征
    z-score（复核人在真实制品上实测 train 特征均值 ~1e-8、std 1.0000，非全量拟合）；
    IMDB 的有效口径是 L2 归一化（实测 max|‖x‖−1| = 1.19e-07），来源为锁定 revision 模型
    内置的 Normalize 模块——manifest 记录的 normalize_embeddings=false 描述的是传给库的
    实参，不是有效输出行为。两者已同时记入 manifest（effective_output_normalization /
    normalization_source），该记录要求已由返工满足。
R2  接受。写明 Self-PU 的独立预算边界：two_student_sampled，200 epochs × 每 epoch 2 次
    optimizer step × 至多 512 样本，comparability group 与 minibatch 组不相交，且不宣称
    存在匹配的 mini-batch oracle。runner 侧守卫接受声明值、拒绝被篡改的步数。
R3  接受当前版本。ResNet-18 真实、随机初始化、train-only 统计被运行路径消费、非 none
    增强在运行路径直接拒绝。制品层 A3b 见 issue #52（provenance 缺陷，不影响数值——
    运行路径自行重算统计量）；5 个 seed 已于 2026-09-17 重建核验并迁入标准路径，
    issue 已关闭。A3a 已修复并在真实制品上复核为 augmentation=none。
R4  接受，并写明边界：adapter 行是随机、未训练、冻结的 ResNet 特征，属工程基线，
    不代表已训练表征；四路同状态、缓存绑定正确。
R5  条件接受。分组键与门禁函数存在且有测试，但当前没有生产调用方——强制分榜目前是
    库函数与落盘字段，尚无正式跑批路径经过它。条件：P2.1 聚合入口必须实现并强制调用
    分榜门禁。（作为 P2.1 前置条件记录，不阻塞本次工程交付接受。）
R6  接受书面边界、保持阻断。唯一 oracle 训练器为 PilotOracleMLP；native-CNN oracle 与
    full-batch oracle 仍缺，且缺失在协议中显式命名而非静默省略。相关对照继续单列或
    不可运行。
R7  接受消融、保持阻断。本次是明确的 no-meta 消融（拟合期告警确认），两 teacher 逐
    epoch 落盘，PA 路径不读内部选择。补记保留意见：technical_smoke 模式下无快照时 PA
    回退到内部 PU-risk 所选 teacher 的组合不加标注、且无测试。完整 OA meta-reweighting
    与 PA-ineligible 路径另立任务，不得向 PA 泄漏真实验证标签。
R8  条件接受。接受 checkpoint 语义、恢复机制与存储不变量（真实 2-epoch 探针、损坏
    fail-loud、weights_only、不存 optimizer/RNG/标签）。条件：(a) 跑批前置磁盘容量检查
    （磁盘占用未入资源统计、无代码护栏）；(b) checkpoint 覆盖不变量改为 runner 强制并
    补测试（当前由已保存的 component 集合反推，只存一个 teacher 时校验仍会通过）。
    （同样作为 P2.1 前置条件记录。）本机制不支持 optimizer/RNG 续训。
R9  保持阻断。PA 现有实现仍是 PU 分离度代理（threshold=None、不读真实标签），不等于
    协议 PA Accuracy。正式 SCAR PA 准则、阈值规则、tie-break 与文献 PA 定义一致性待
    专项任务锁定。不得因其他项通过而删除本阻断。
R10 接受 win32 frozen-lock smoke；Linux 分支记录环境偏差、保持正式阻断。
    本机（T600/win32，torch 2.14.0+cu126）由复核人本人亲手复现：8 passed / 0 failed /
    0 skipped，退出码 0，JUnit 时间戳 2026-09-16 19:52:28，uv.lock sha256 与产物记录
    一致，smoke 峰值显存 277.6 MiB allocated / 342.0 MiB reserved（smoke 口径）。
    Linux 分支：锁解析到 CUDA 13.0（要求驱动 ≥ 580.65.06），服务器驱动 550.54.14 不满足，
    采用方案 3——记录环境偏差并保持正式阻断，已记入 formal_blockers。

overall_decision: 接受 P2.0a 工程交付；不放行正式 P2.1。
    继续保持阻断：P2.0b 标签语义契约、P2.0c 对照预注册、PA 正式准则（R9）、缺失的
    CNN/full-batch oracle、完整 Self-PU OA meta-reweighting、Linux frozen-lock 环境偏差
    （方案 3）、P1.4 制品统一重建（IMDB/Spambase 待建；issue #52 的 CIFAR-10 部分已
    关闭）。P2.1 的启动另需兑现 R5 与 R8 的前置条件。
    IMDB 制品层（data/splits/imdb/ 的 5 个 manifest）不含本次返工新增的有效口径字段，
    并入 P1.4 三数据集统一重建，不在本次单独刷新——验收按代码与测试层进行。

evidence: 签署 PR 链接（见 §5）；本机 GPU 证据 JUnit 与三模态抽查记录见 PR 附件与
    复核材料；本协议的 canonical 摘要由 pu_toolbox.experiment.survey_protocol.digest 计算。
```

计算当前协议摘要：

```bash
uv run python -c 'from pu_toolbox.experiment.survey_protocol import load_protocol, digest; p=load_protocol(); print(p["protocol_version"], digest(p))'
```
