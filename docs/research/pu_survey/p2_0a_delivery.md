# P2.0a 交付：共享规格、执行矩阵与比较边界

状态日期：2026-09-16。工程实现与 CPU/GPU 技术验证完成，**待 shuidisjtu 复核**；
不代表正式 pilot 已启动。上游要求见 [执行计划](survey_execution_plan.md) P2.0a，
实验要求仍以 [协议](pu_survey_protocol.md) 为准。

## 1. 交付物与程序权威

- `pu_toolbox/experiment/survey_protocol_v1.json`：版本化预算定义与 25 条执行单元，
  覆盖三个数据集、七个 PU 方法、oracle；其中 CIFAR native-CNN oracle 显式不可运行。
- `survey_protocol.py`：加载/校验矩阵、锁定参数检查、runner 侧再次消费、比较门禁。
- `survey_execution.py`：共享 MLP、图像预处理与 encoder 装配、冻结特征缓存、torch MLP oracle。
- `scripts/run_survey_experiment.py --protocol survey-v1`：执行绑定的矩阵单元。
  无 `--protocol` 的旧入口仍可用于技术 smoke，但 manifest 必须标记非正式结果。

配置的数值真相源只有上述 JSON；本文件解释选择及边界，不再维护另一份参数表。
`review_status` 当前为 `pending_collaborator_review`，不得将工程默认值解释为两位负责人已签字。
后续版本 `survey-v1.1` 已补齐逐 epoch 权重保存与独立恢复，预算/backbone 未改变；
审查决定见 [复核包](p2_0a_review.md)，实现与证据见 [checkpoint 交付](epoch_checkpoint_delivery.md)。

## 2. 共享规格与预算决策

### 2.1 表格、文本与深度 score 模型

nnPU、Dist-PU、Self-PU 与新版 torch MLP oracle 使用相同的单隐层 ReLU score blueprint。
nnPU/Self-PU 经既有 model/backbone 参数注入；Dist-PU 经已有 hidden_dim 参数对齐。
IMDB 必须为固定 SBERT 的 384 维特征，Spambase 使用调用方准备的 train-only z-score 特征。
各运行保留真实输入、split 摘要与完整构造参数，不能仅凭模型名称推断一致性。
Self-PU 当前运行不注入 clean validation，因此是去掉 meta-reweighting 的显式消融，
不是原论文完整 Self-PU；`method_variant` 和专用正式阻断字段记录这一边界。
其 OA clean-validation 训练接入必须单独实现，不能通过向 PA 训练泄露 clean labels 解决。

预算使用算法的**真实单位**，并非所有方法都叫 epoch：

- nnPU/Self-PU/oracle 是 mini-batch 训练；nnPU 禁止在固定预算内提前停止。
- Dist-PU 当前实现每 epoch 使用完整 train；其 batch_size 参数不控制实际训练。
  因而单列 full-batch 组，不声称与 mini-batch oracle 预算相同。
- uPU squared 是闭式求解；KLDCE 是外层/内层迭代；LBE 是 EM 与内层求解。
- PUSB 使用 `pusb_kernel`。其内置 CV 有额外的参数对/折数成本，已锁定并明确记入预算；
  一个外层候选不等于只有一次内层模型拟合。

经典方法没有 MLP backbone，不能借一个注释将它们变成统一 MLP 方法。
它们只作描述性工程对照；比较门禁还检查预算，预算不一致时即使路径相同也拒绝混排。
方法私有训练流程、shared-backbone 改装与当前 checkpoint 边界均标记 `benchmark-adapted`，
不自动升级为 `source-faithful` 或论文数值复现。

### 2.2 CIFAR 路径

图像统一使用 train-only 通道统计、随机初始化的小图 ResNet-18，不下载预训练权重。
本次工程默认关闭增强：现有 nnPU 接口没有可直接注入的增强 hook，不能写了配置却未执行。
是否引入训练增强需要负责人复核并在下一协议版本实现，不能静默覆盖当前规格。

- **native_cnn**：nnPU 注入 ResNet encoder 与 score head，端到端训练；初始化状态入 manifest。
- **cnn_feature_adapter**：encoder 只随机初始化、不训练，冻结后在 eval/no-grad 下提取
  train/pu_val/clean_val/test 的二维特征。明确是工程级 random-feature 基线，不是已训练 CNN。
  `fixed_external` 是既有 adapter 的“无本 split 拟合”接口分支，绝不表示用了外部预训练权重；
  补充的 `encoder_training` 字段明确实际行为。
- 两路径强制分组；同一路径的不同训练预算也不能混排。

缓存绑定 dataset 的四路输入、索引、seed/预处理配置、实际 encoder 状态、提取设备与 batch size。
不绑定 method、c、标记机制或真实标签，因此同 split/seed 的 adapter 方法与 oracle 可复用。
读取时检查权重和特征摘要；损坏会报错，不会静默重算。新条目先写临时目录再原子发布。
不同设备或提取 batch size 不共用同一缓存条目，避免把浮点实现差异隐藏为同一表征。

SAR 的 posterior 与标记始终在**原始输入空间**生成，再将同一标签视图用于适配特征。
否则 native 与 adapter 会因 posterior 输入不同而看到不同 P/U 标记，破坏协议公平性。

### 2.3 Oracle 的对齐决策

- MLP oracle 与 mini-batch neural 组共享 score blueprint、优化器类别和声明预算；
  同一个 dataset/seed 只执行一次，按 `c_independent` 广播。
- CIFAR adapter oracle 使用相同冻结 encoder、相同缓存特征与 MLP head。
- Dist-PU 的 full-batch 组目前没有预算匹配的 oracle；经典组没有 backbone 匹配的 MLP oracle。
  这些对照不能借其他组的 oracle 冒充协议内同规格上界。
- CIFAR native-CNN oracle 的 clean-val checkpoint trainer 按既有两阶段决策留在 Phase 2。
  当前请求该行必须在读 split/训练前失败，不能退化为展平图像的 MLP。

上述缺少匹配 oracle 的单元只允许技术结果单列。完成相应 oracle 后必须更新协议版本再重跑，
不能把历史临时结果直接并入正式榜。

## 3. 执行约束与 manifest

Runner 自己读取矩阵，不信任脚本传来的“已验证”结论。它核对实际 estimator 身份、预算参数、
model blueprint、seed、architecture、生成器/选择协议以及图像/adapter 的实际摘要。
model/backbone/encoder、随机种子和预算锁不能从构造参数或候选的嵌套参数覆盖。
设备由 `--device` 指定，不属于可更改训练规格的模型参数。

允许 c、seed、split-ref、候选池调整，但子集/偏离必须留下 `protocol_deviation`，
候选池/非先验构造参数改变、非规范矩阵也同样标记，不能进入正式榜。
矩阵的 canonical JSON 摘要忽略空白与键顺序，但不忽略内容变化。
数据目录支持 `split_{seed}` 模板或含 `split_0` 等目录的数据集根目录，实际 split manifest
必须与请求的 dataset/seed 对应，不能对同一 split 改写 seed 跑五次。

成功和候选全部失败的 manifest 都包含 `protocol_version`、`protocol_sha256`、执行单元、
backbone、budget、representation、training_path、comparability_group、偏离及正式阻断原因。
协议预检失败另写 `rejected_versioned_pilot`、请求信息和错误，不能产生成功的协议制品。
adapter 完整 manifest 合并在 representation 中，供后续聚合审计。

聚合前用 `validate_comparable_manifests` 检查同 dataset/seed/c 单元：正式资格、协议版本、
路径、组、预算、backbone、split/特征与 P/U 标记摘要。默认拒绝非正式结果；
技术诊断显式使用 `require_formal=False`，但仍不能跳过公平性检查。

## 4. 命令与验证记录

下面是运行入口，不意味着本机已有公开数据。原始数据与 splits 不进入 Git。

```bash
# 路径为数据集根目录，内部含 split_0 ... split_4。
uv run python scripts/run_survey_experiment.py data/splits/spambase \
  --protocol survey-v1 --dataset spambase --method nnpu \
  --class-prior 0.4 --c 0.1,0.3,0.5 --seeds 0,1,2,3,4 --device cuda

# CIFAR adapter 行，class-prior 数字仅为示例，须用已审计的数据池先验替换。
uv run python scripts/run_survey_experiment.py data/splits/cifar10 \
  --protocol survey-v1 --dataset cifar10 --method upu \
  --class-prior 0.4 --c 0.1,0.3,0.5 --seeds 0,1,2,3,4 \
  --adapter-cache data/cache/survey_adapter --device cuda

# 明确选 adapter oracle；native_cnn oracle 当前会拒绝。
uv run python scripts/run_survey_experiment.py data/splits/cifar10 \
  --protocol survey-v1 --dataset cifar10 --oracle \
  --training-path cnn_feature_adapter --c 0.1,0.3,0.5 --seeds 0,1,2,3,4

# 可重复的小规模验证；均使用合成数据，不是公开协议 benchmark。
uv run pytest tests/unit/experiment/test_survey_protocol.py \
  tests/unit/experiment/test_survey_execution.py \
  tests/unit/experiment/test_survey_script_protocol.py \
  tests/unit/experiment/test_survey_protocol_cnn.py -q
```

2026-09-16 本机记录：Python 3.12.2、torch 2.6.0+cu124、torchvision 0.21.0+cu124；
首次 CPU 验证在 Codex 沙箱内执行，该沙箱看不到 GPU 设备；这不代表服务器没有 CUDA。
新增 46 项 CPU 测试通过，包含真实 ResNet-18 CPU 一 epoch 训练、
二维 uPU/MLP oracle 链路和 CIFAR adapter 脚本二次缓存命中。
测试 fixture 的缩减预算使用独立矩阵，输出明确含非规范矩阵偏离，不改正式 JSON。
额外补充源文件结构扫描测试：未暂存的新 Python 文件也纳入结构门禁，忽略的缓存不纳入；
无需提前 git add/commit 才能检查交付文档与新文件的一致性。
每个测试在 pytest 临时目录生成制品；它们是工程验证，不替代真实数据正式实验。

同日经授权在沙箱外补做 GPU smoke：服务器有 8 张 RTX A6000（每张约 48 GB），
驱动 550.54.14、CUDA 12.4；上述 PyTorch 环境能正常初始化 GPU。
考虑其他任务占用，仅暴露物理 0 号卡（`CUDA_VISIBLE_DEVICES=0`），CPU 线程设为 1，
并在启动 pytest 前调用 `torch.cuda.set_per_process_memory_fraction(0.04, 0)`，
限制本进程 PyTorch allocator 约 1947 MiB；这不是整卡独占预约，也不限制 CUDA 上下文等开销。
测试明确断言 CUDA 可用，不允许靠跳过测试冒充验收。

- `test_nnpu_gpu.py`：CNN13 nnPU 一 epoch 训练与预测。
- `test_survey_protocol_cnn.py`：P2.0a 原生 ResNet-18 一 epoch 训练、PA/OA 评测、manifest 落盘。
- `test_survey_execution.py`：真实 ResNet 冻结特征 GPU 提取、512 维特征与二次缓存命中、
  encoder 状态不变；MLP oracle 两 epoch 训练、有限评分和同 seed 重跑一致。

三文件以 `-m gpu` 运行：**4 passed、18 deselected，4.64 秒**，无跳过。
全部训练模型/提取 encoder 均断言参数实际位于 CUDA。
进程 PyTorch 峰值 allocated 277.36 MiB、reserved 304.00 MiB（不含 CUDA 上下文等开销）；
JUnit 报告位于 `/tmp/pu-toolbox-p2a-gpu-tests.xml`，该临时报告不会进入 Git。
随后四个 P2.0a 专项文件以 `-m 'not gpu'` 回归：**46 passed、3 deselected，7.18 秒**。
GPU 验证仍使用合成数据和缩减预算，不代表完整 CIFAR/IMDB/Spambase benchmark、
多 seed 正式跑批或 GPU 独占调度已完成。

系统 Python 快层回归命令 `python -m pytest tests/ -q -m 'not slow and not e2e and not gpu'`
首次记录为 1506 passed、2 skipped、37 deselected；后补两个预算/失败留痕测试与三项 GPU 测试后，
提交前完整快层再次验证：**1508 passed、2 skipped、40 deselected，113.40 秒**。
格式、结构、文档链接、API 覆盖、元数据、数学渲染、技能同步与基线配置检查通过。
文档链接检查保留一个既有 README 未列 `check_api_docs.py` 的非阻断提示。
离线 sdist/wheel 构建通过，并确认 wheel 包含执行矩阵 JSON；版本未提升，构建仅供验证，未发布。

## 5. 验收边界与后续

P2.0a 工程交付覆盖矩阵、runner 消费、manifest、CIFAR 接线及 oracle 对齐的书面决定。
负责人尚需复核默认规格、随机冻结 encoder 的报告范围、各路径预算与上述 GPU 执行证据。

正式跑批仍须 P2.0b、P2.0c 与负责人复核完成。
首次交付时 `RunTrajectory` 只有最终/估计器内部最优模型，缺少全 epoch 独立选择。
后续 `survey-v1.1` 已通过回调、持久化权重、双 teacher 记录与 PA/OA 独立恢复补齐该链路；
只有实际训练满预算且快照覆盖完整，才能解除该单元的 checkpoint 阻断。
新增确认的差距是 PA 仍用 PU 分离度代理，不是协议中的 Accuracy/阈值选择准则，
已注册专用正式阻断；需要合作者锁定准则后单独补齐，不能仅删除字段升级结果。

因此当前所有产物 `formal_eligible=false` 是有意的安全边界。
未实现 CNN oracle、full-batch oracle、完整 Self-PU OA 或正式 PA 准则，不得宣称完成。
