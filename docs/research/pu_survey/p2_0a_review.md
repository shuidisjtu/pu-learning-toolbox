# P2.0a 合作者复核包

状态日期：2026-09-16。**材料已准备，尚未由 shuidisjtu 签署验收**。
主责 HENG958；复核人 shuidisjtu。依据 [执行计划](survey_execution_plan.md) §2.0，
代码和测试通过不等同于合作者同意，不得由自动化代签或自行清除 `collaborator_review`。

## 1. 复核对象

- 上次工程提交：`99c5bb7`（P2.0a 矩阵、装配、runner 门禁与 CPU/GPU smoke）。
- 本次补充：`survey_protocol_v1.json` 中 `protocol_version=survey-v1.1`，
  新增 `selection_spec`；预算与 backbone 未擅自修改。
- 准备材料时待审 canonical SHA-256：
  `ffeef333a88d4b67a042c9ec8e03d042598a204ba42ed7f15d3cee080137cce9`。
  若 JSON 内容变更，须重新计算，不能沿用此摘要签署。
- `--protocol survey-v1.1` 是当前入口；`survey-v1` 保留为当前规格别名，
  实际版本与 JSON 内容摘要以 manifest 为准，不能据 CLI 别名将结果记为旧版。
- 交付与测试证据：[P2.0a 交付](p2_0a_delivery.md)、
  [逐 epoch checkpoint 交付](epoch_checkpoint_delivery.md)。
- 不纳入本次验收：P2.0b 标签语义契约、P2.0c 对照预注册、正式跑批及主榜。

## 2. 决策清单（逐条确认，不默认通过）

| ID | 当前工程决定 | 复核时需要确认 |
|---|---|---|
| R1 | nnPU/Self-PU/MLP oracle 使用共享 MLP128；IMDB 固定 384 维输入 | 接受 score blueprint；确认 IMDB L2 归一化及 Spambase train-only 标准化实际数据口径 |
| R2 | mini-batch 与 full-batch 分组；经典方法按闭式/迭代/内置 CV 真正成本记录 | 接受各预算真实单位及候选池；不得将经典组或 Dist-PU 冒充同规格 mini-batch oracle 对照 |
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
reviewed_protocol_version: survey-v1.1
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
