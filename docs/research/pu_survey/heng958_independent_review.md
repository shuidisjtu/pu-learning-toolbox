# HENG958 独立复核记录

状态日期：2026-09-16。本记录只覆盖执行计划中明确交给 HENG958、且不需要
shuidisjtu 决策或签署的 P1 复核项；不替代 P2.0a/P2.0b/P2.0c 的协作验收。

## 1. P1.1 执行环境复核

### 结论

- GPU 能力 smoke **通过**：服务器可见 8 张 NVIDIA RTX A6000；在物理 GPU 2 上执行
  PyTorch CUDA 张量计算得到 `sum([1, 2]²) = 5.0`。
- 本次实际解释器为 Python 3.12.2、PyTorch 2.6.0+cu124、CUDA runtime 12.4；
  `torch.cuda.is_available()` 为 true，隔离后可见设备数为 1。
- `uv.lock` SHA-256 为
  `7a7adf38905461b4fe0eeb73e613545cbde499a829155b7a793369a5bf22e0ed`。
- `uv lock --check` 与隔离路径下的 `uv sync --frozen --extra research --extra dev --dry-run`
  均通过；解析为 86 个包（Linux PyTorch 2.13.0、torchvision 0.28.0），未在本次复核中
  下载或改写工作区环境。
- **环境一致性限制**：当前服务器全局 Python 不是 frozen lock 环境；`uv.lock` 的 Linux
  解析项为 PyTorch 2.13.0，并依赖 CUDA 13 组件；前次记录的驱动 550.54.14 低于
  CUDA 13.0 GA 所需的 580.65.06。上述结果只证明旧全局 CUDA 12.4 技术路径可用，
  不能作为正式 pilot 的依赖复现证明。当前沙箱内 `nvidia-smi` 无法连接驱动，未完成
  frozen-lock GPU 复验；须先共同决定升级驱动、调整 lock 或保留阻断，再在隔离环境
  运行并把实际版本、设备、lock digest 和 JUnit 证据写入运行记录。

### 可追溯命令

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv
CUDA_VISIBLE_DEVICES=2 python -c '<CUDA tensor smoke + version/lock digest>'
```

此前 P2.0a 深路径 GPU 测试覆盖 nnPU CNN、ResNet adapter、MLP oracle 和逐 epoch
checkpoint，详见 [P2.0a 交付记录](p2_0a_delivery.md#7-gpu-smoke)。本节新增的是负责人侧
环境复核，不重复把同一 smoke 计作正式 pilot 结果。

## 2. P1.3a nnPU/Self-PU 方法台账复核

### nnPU

- 原论文给出 two-sample P/U 设定，类别先验 `pi_p` 在论文中假定已知；因此台账
  `native_sampling_assumption=os`、`prior_semantics=population pi` 与原文一致。
- 论文 Algorithm 1 在负风险 `r < -beta` 时沿 `-grad(r)` 更新并用 `gamma` 折扣步长。
  本项目 `_nnpu_train_step` 的 correction branch 使用 `-gamma * r`，不是对
  `max(0, r)` 直接反向传播；旧台账中的“候选实现待确认”已过时并已移除。
- registry 与 estimator 均声明 PyTorch/GPU、MLP/CNN、2D/4D、encoder 注入并训练
  encoder；台账的能力块与代码契约测试一致。
- 保留边界：官方实现是 Chainer，本项目是 PyTorch clean-room 实现；`source_status`
  表示官方来源可追溯，不等于逐行源码移植。

权威来源：[NeurIPS 2017 论文](https://proceedings.neurips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html)、
[官方实现](https://github.com/kiryor/nnPUlearning)。

### Self-PU

- 原论文的三个核心块为 self-paced、self-calibrated 和 teacher/student
  self-distillation；self-calibration 使用带真实正负标签的 clean validation batch。
  因此本项目在缺少 `validation_data` 时明确标为 no-meta ablation，台账不得称为完整论文复现。
- estimator 的 `validation_data` 与 `pu_validation_data` 分离，双 student/双 EMA teacher、
  逐 epoch PU risk 和 clean-validation teacher selection 均有独立路径；与方法卡边界一致。
- registry/estimator 明确为 MLP-only 原生架构；4D 输入只是 flatten 后的接口支持，不能
  解释成 native CNN。Survey 图像行仍必须标为 feature-adapter 路径。
- 官方仓库的当前规范地址是 `VITA-Group/Self-PU`；旧 `TAMU-VITA/Self-PU` 组织链接
  指向同一项目，不再作为未决来源问题。
- 保留边界：完整 self-calibration、原生 CNN、官方图像增强与 paper-like benchmark
  均未完成，继续留在台账 `uncertainty` 和 P2.0a formal blockers 中。

权威来源：[PMLR 论文页](https://proceedings.mlr.press/v119/chen20b.html)、
[官方实现](https://github.com/VITA-Group/Self-PU)。

### 机器可检查结果

`method_ledger.json` 升至 schema 1.1，为 nnPU/Self-PU 写入 `owner_review`；合同测试要求
两项状态为 completed、日期合法、证据文档存在，并继续校验六槽所依赖的 registry 能力、
source status、先验语义和 JSON 类型。

## 3. P1.4 可执行性复核状态

当前工作树不存在 `data/splits/<dataset>/split_<seed>/split_manifest.json` 或对应 NPZ；
因此无法重新验证 CIFAR-10、IMDB、Spambase 共 15 个 split 的四路数据、预处理摘要和
脚本端到端执行。P1.4 的 shuidisjtu 侧完成声明保持不变，但 HENG958 侧复核不得标成完成。

合作者审核另发现 CIFAR 旧产物的通道统计值不符（[issue #52](https://github.com/shuidisjtu/pu-learning-toolbox/issues/52)）；
即使代码修复后，本项仍须重生成并核验真实产物，见 [P2.0a 复核包](p2_0a_review.md#审核发现-a3cifar-split-manifest-来源信息)。

解除条件：同步或重新生成三数据集的 5-seed split 产物后，运行合同测试和至少每种模态一个
`run_survey_experiment.py` smoke，并归档 split manifest digest 与命令记录。

## 4. P2.0d SAR-OA 执行路径复核

### 结论

负责人侧技术复核 **通过**，不改变 P2.0a/P2.0b/P2.0c 的正式 pilot 门禁：

- `--labeling-mechanism` 与 `--method` 正交，LBE-A/LBE-B 生成器都可配任意 Survey 方法；
- SAR 请求只接受规范 token `0.05`、`0.5`，拒绝同值异拼写和非协议值；
- SAR 路径显式传入非空 `[ProtocolOA()]`，不会因 runner 的默认值回退到 PA+OA；
- oracle+SAR 的语义冲突在读取 split/创建输出前 fail-loud；
- manifest 记录 mechanism、请求/实际标记率、夹紧前后数量、seed、label-view digest、
  posterior 来源和 `c_requested_token`；
- 同 seed 的标记摘要与 OA 指标确定性一致，SCAR 默认路径仍保留 PA+OA；
- 新增 `survey-v1.1` 端到端回归，锁住版本化入口的 SAR-LBE-B 生成器、OA-only、
  protocol version、审计字段及子集 c-grid deviation。

验证命令：

```bash
python -m pytest tests/unit/experiment/test_survey_script_sar.py \
  tests/unit/experiment/test_survey_protocol.py \
  tests/unit/experiment/test_survey_protocol_cnn.py -q
```

复核范围只证明执行路径与留痕正确；标签语义门禁仍属于 P2.0b，正式跑批仍须
P2.0a/P2.0b/P2.0c 全部验收。
