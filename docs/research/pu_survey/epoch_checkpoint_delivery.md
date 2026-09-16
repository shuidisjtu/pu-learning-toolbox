# 独立逐 epoch checkpoint 选模交付

状态日期：2026-09-16。对应 P2.0a 后续训练链路补充；
复核入口见 [P2.0a 复核包](p2_0a_review.md)。

## 1. 执行关系

每个候选只训练一次，nnPU、Dist-PU、Self-PU、torch MLP oracle 的 `fit`
增加可选 `epoch_callback(epoch, self)`。回调在完整 epoch 后、内部最佳模型恢复前执行；
记录器不接收 `clean_val` 或 `test`。它仅保存网络权重、epoch 与 component。
Self-PU 两个 EMA teacher 同时保留，不能先用 PU 风险挑一个 teacher 后冒充 OA 自由选模。

Runner 对支持回调的内置 PU/监督 trainer 默认启用 `EpochCheckpointTrainer`：

```text
一次固定预算 fit → 逐 epoch 权重文件
                     ├─ PA：只读生成后的 pu_val → 独立候选/epoch/teacher 选择
                     └─ OA：只读 clean_val → 独立候选/epoch/teacher/阈值选择
各自恢复选中权重 → 独立 test 评测 → 各自 selection artifact 与 checkpoint 引用
```

经典无 epoch 方法仍是单点模型，不虚构 200 个 checkpoint。
第三方 trainer 不自动重写；若仅返回最终模型，manifest 保留 checkpoint 正式阻断。
直接使用旧 `DeepFitTrainer` 的 probe/内部选模行为不变；需要快照时可显式注入新 trainer。

## 2. 制品与保存加载

- `checkpoints.py`：权重记录、文件 SHA-256 校验、CPU architecture template、逐个恢复。
- `RunTrajectory.checkpoints`：与 `epochs` 的 1-based position 绑定，而不是假定源 epoch 从 1 开始。
- `SelectionArtifact.checkpoint_index`：每套协议独立指向该 trajectory 的快照。
- `RunResult.selected_models`：已恢复的 PA/OA 推理模型；OA 模型包含选中阈值的 VAL-side 规则。
- `candidate_runs[].epoch_checkpoints`：完整快照清单、teacher、摘要与逐 epoch PA/OA 指标。
- `selection.PA/OA.checkpoint`：选中制品的独立引用；选同一快照时可以共享只读权重文件，
  但返回的两个推理模型对象与选择记录仍独立。

指定 `manifest_path` 时默认在旁边的 `checkpoints/attempt-*` 独立目录持久化，
重试/重复执行不会覆盖前次权重。`config['checkpoint_dir']` 可指定保存根目录。
没有保存根目录时使用生命周期受控的临时目录；返回的推理模型仍可用，
manifest 的 checkpoint `path=null`、`persistent=false`，不能据此宣称可持久复现。

保存文件只含 tensor state dictionary，通过 `torch.load(..., weights_only=True)` 加载；
不序列化任意模型对象、原始数据、真实验证标签、optimizer 或 RNG 状态。
architecture template 只在 CPU 保留一次，每次仅恢复一个待评分 checkpoint，
避免将全部 CNN epoch 权重同时保存在 GPU/内存。约 44 MB 的 ResNet-18 权重保存 200 epoch
约需 8–9 GB/候选/seed，正式跑批前须确认磁盘容量；本次不自动删除历史制品。

跨进程加载需要调用方按已锁定协议构建相同网络，然后显式传入 template：

```python
from pu_toolbox.experiment.checkpoints import load_selected_checkpoint

# selection = manifest['selection']['OA']；template 为该路径实际网络结构。
selected = load_selected_checkpoint(selection, template, device='cpu')
scores = selected.decision_function(X_test)
predictions = selected.predict(X_test)
```

绝对路径需保持可用；移动制品时由调用方显式重定位路径，权重摘要仍须一致。
这是推理 checkpoint，不支持从该 epoch 原地继续 optimizer 训练。
缺文件、摘要损坏、非法 epoch/component 覆盖或 NaN 权重均报错，不回退为最终模型。

## 3. 协议与成本边界

实际训练满声明 epoch 且完整快照可供 PA/OA 独立选择，才解除本单元的
`per_epoch_independent_PA_OA_checkpoint_selection` 阻断；无 epoch 方法标为 `single_point_no_epoch`。
缩短训练、关闭 capture、旧自定义 trainer 或缺少持久化，均不能获得正式可复现资格。
协议版本更新为 `survey-v1.1`；旧 `survey-v1` 历史 manifest 仍是旧语义，不能改版本标签升级。

训练成本包括保存快照；离线逐 epoch 验证/恢复成本记入单配置成本与总调参成本，
并单列 `offline_checkpoint_validation_elapsed_seconds`、选择/分派开销。
最终 test 评测不冒充调参成本。逐 epoch 指标只含每套协议允许的验证结果，不保存标签。

**仍未完成**：PA 的正式 Accuracy/阈值准则（现有分离度代理显式阻断）、
P2.0b/c 验收、合作者签署、完整 Self-PU OA meta-reweighting、CNN/full-batch oracle。
本次 checkpoint 功能补齐不意味着所有单元 `formal_eligible=true`。

## 4. 验证

专项命令（合成数据工程测试，不是正式 benchmark）：

```bash
python -m pytest tests/unit/experiment/test_epoch_checkpoints.py \
  tests/unit/experiment/test_checkpoint_selection.py -q -m 'not gpu'
```

关键反例：PA 最优在 epoch 1、OA 最优在 epoch 2，分别恢复后 test 结果不同；
改变 clean_val 只影响 OA，改变 test 不改变任何选择；共同选择候选与 epoch、同分取最早；
损坏快照不回退；保存/加载复现阈值；记录过程不改变原训练的随机种子和优化器更新。
GPU 测试还覆盖 CUDA↔CPU 加载及 Dist-PU、Self-PU、MLP oracle 的逐 epoch 快照。

2026-09-16 验证记录：两个新增专项文件 **24 passed、4 deselected，1.60 秒**；
包括 float32 VAL-side 阈值边界反例：加载时保留相同归一化运算，
不通过反算 raw-score cutoff 偷换浮点判定。

沙箱外物理 0 号 RTX A6000（驱动 550.54.14、torch 2.6.0+cu124、CUDA 12.4）
执行四文件 `-m gpu`：**8 passed、30 deselected，4.72 秒**，无跳过。
通过 `CUDA_VISIBLE_DEVICES=0` 限制单卡，CPU 线程为 1，
PyTorch allocator 限额 `set_per_process_memory_fraction(0.04, 0)`（约 1947 MiB）；
测量结束 allocator 峰值 allocated 277.36 MiB、reserved 304.00 MiB，
不包含 CUDA 上下文等开销。保留一个 Self-PU no-meta 消融的预期警告，不视为完整论文实现。
GPU JUnit 报告：`/tmp/pu-toolbox-epoch-checkpoint-gpu.xml`，仅本地临时验证，不进入 Git。

最终完整快层命令 `python -m pytest tests/ -q -m 'not slow and not e2e and not gpu'`：
**1532 passed、2 skipped、44 deselected，112.89 秒**。
快层 JUnit 报告：`/tmp/pu-toolbox-epoch-checkpoint-fast.xml`。
格式、严格测试质量、结构、文档链接、API 覆盖、项目元数据、数学渲染、技能同步及基线检查通过；
文档检查保留一个既有 README 未列 `check_api_docs.py` 的非阻断提示。
离线 sdist/wheel 构建通过，包含 `checkpoints.py` 与当前矩阵 JSON；版本仍为 1.11.0，未发布。
所有上述 smoke 均为合成数据，不替代正式 pilot。
