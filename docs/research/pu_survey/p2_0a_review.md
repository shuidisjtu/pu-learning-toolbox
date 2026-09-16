# P2.0a 合作者复核包

状态日期：2026-09-16。**材料已准备，尚未由 shuidisjtu 签署验收**。
主责 HENG958；复核人 shuidisjtu。依据 [执行计划](survey_execution_plan.md) §2.0，
代码和测试通过不等同于合作者同意，不得由自动化代签或自行清除 `collaborator_review`。

## 1. 复核对象

- 上次工程提交：`99c5bb7`（P2.0a 矩阵、装配、runner 门禁与 CPU/GPU smoke）。
- 当前待复核矩阵 `survey_protocol_v1.json` 的实际版本为 `survey-v1.2`；
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
| R10 | A6000/CUDA 12.4 技术 smoke 可运行；环境记录来自现有服务器 Python | 认可技术验证，但完整 pilot 仍需核对 uv.lock 环境、数据产物与独占/共享 GPU 窗口 |

R9 是新增发现的协议差距，不能因为 checkpoint 接线已完成就删除其正式阻断。
复核若要求修改任一规格，应记录理由、更新版本、重跑受影响测试和数据单元。

### 审核发现 A3：CIFAR split manifest 来源信息

- A3a：split 准备阶段未施加增强，却把 estimator 默认 `simaugment` 记入 manifest；
  合作者提交 `47f62c3` 已把此字段显式锁定为 `none`。
- A3b（[issue #52](https://github.com/shuidisjtu/pu-learning-toolbox/issues/52)）：
  大规模、HWC 存储转置而成的非连续 NCHW 数组在 `float32` 通道归约时累积精度丢失，
  45,000 张 CIFAR train 图的通道和饱和在约 `2**24`，导致各通道均值同为
  `2**24 / (45000*32*32) ≈ 0.364089`。工程代码已改用 `float64` 累积均值与方差，
  并加入 4,100 张非连续图像的回归测试；输出图像仍为原来的 `float32` 缩放表示。
- **历史产物未自动修复**：本服务器没有 `data/splits` 原始产物，不能据代码修复宣称
  既有 5-seed split manifest 已重建。P1.4 复核与正式跑批前，须在持有原始 CIFAR 数据的
  环境重生成 5 个 seed，并逐个比对 manifest 的 mean/std 与 `train.npz` 的 `float64`
  统计值；还须记录新 manifest digest。A3b 的数据验收在此之前保持开放。

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
- 上述工程修改响应审核，不等于复核人重新验收。`review_status` 仍为
  `pending_collaborator_review`，签署记录仍待本人填写。

### 冻结依赖 GPU 验证待共同决策

之前的全局 Python/CUDA 12.4 smoke 不能替代 frozen-lock 验证。当前 `uv.lock` 的
Linux PyTorch 2.13.0 依赖 CUDA 13 组件；前次沙箱外记录的驱动 550.54.14 低于
CUDA 13.0 GA 所需的 580.65.06。当前沙箱内 `nvidia-smi` 无法连接驱动，故未声称
重新验证了驱动状态，也未运行 frozen-lock GPU 套件。升级驱动、调整依赖锁，或暂时
保留阻断应由两位实施主体共同决定；决定前不得拿全局 torch 2.6/cu124 结果代签。
[NVIDIA CUDA 13.0 发布说明](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html)
给出最低驱动版本。

## 3. 复核前检查

1. 阅读 JSON 预算、执行单元与 `selection_spec`，不要仅阅读文档摘要。
2. 运行 checkpoint 专项与 GPU 测试，检查 PA/OA 选中不同 epoch 的反例。
3. 抽查 manifest：路径分组、完整 checkpoint 列表、各 epoch 验证指标、选择引用、实际版本。
4. 核对真实 split；工程合成数据 smoke 不等于 P1.4 真实产物已在本服务器验收。
5. 明确区分“接受工程规格”“接受方法消融范围”“允许正式 pilot”，三者不能相互替代。

## 4. 签署记录模板

请复核人本人在 PR review 或本文件提交明确记录；以下字段当前均为**待填写**：

```text
reviewer: shuidisjtu
review_date: 待填写
reviewed_code_commit: 待填写（本次代码提交后填完整 SHA）
reviewed_protocol_version: survey-v1.2
reviewed_protocol_sha256: 待填写（JSON canonical SHA-256，不是原始文件字节摘要）
R1 ... R10: 每条填写 接受 / 需修改 / 保持阻断，并附理由
overall_decision: 待填写（接受工程交付 / 要求修改；正式 pilot 门槛另核对）
evidence: 待填写（PR review 链接或本人提交 SHA）
```

计算待审 JSON 摘要：

```bash
python -c 'from pu_toolbox.experiment.survey_protocol import load_protocol, digest; p=load_protocol(); print(p["protocol_version"], digest(p))'
```

只有取得真实复核记录后才能按决定更新 `review_status`；本次不预填 `accepted`。
