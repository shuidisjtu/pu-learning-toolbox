# VPU 图像编码器与设备端分批工程交付（2026-10-15）

状态：P3.2 独立工程补齐，shuidisjtu 技术实施；HENG958 接口/方法学复核待办。
**不是作者图像实验复现，不是正式准入，也不是 P3.2 总体验收。**

## 对齐的要求与来源

[执行计划 P3.2](../survey_execution_plan.md)要求图像/共享特征接通、训练预测分批、
checkpoint 及适配差异可追溯。本次补 VPU 尚缺的注入编码器与设备端数据容量边界。
公式、TS/OS 口径仍沿用[采样审计](../admission/vpu_sampling_audit.md)、D21/D22；
上游 commit `603bcc1e628795f57a5ac87e5b0b977273b7cf91` 的源调度证据见
[源复核记录](../admission/vpu_schedule_review_20261015.md)。本次没有新论文数值读数。

## 已实现的接口与语义

- `encoder=None` 为兼容默认；二维特征仍可直接训练。
- 四维 NCHW 必须提供 `torch.nn.Module` encoder；深拷贝后对全部参数执行
  `requires_grad_(True)` 并参与 Adam（会覆盖模板中的冻结设置），
  输出有限的 `(N, feature_dim)` 特征，再接工具箱 MLP/LogSigmoid 头。
  原用户模板不被训练或修改，图像不自动 flatten 成二维输入。
- `data_device_policy="auto"`：无 encoder 保留旧 resident；有 encoder 使用 cpu_stream。
  可显式选择 cpu_stream；encoder + resident 明确拒绝，避免图像池全量进入设备。
- cpu_stream 的完整 P、边缘池及 PU 验证数据驻留 CPU；训练 batch、MixUp、
  归一化扫描、验证和预测的单次设备输入均不超过 `batch_size`。
  这是**设备端分批**，不是磁盘流式读取：完整输入及角色池仍占主机内存。
- 验证以 float64 CPU logaddexp 合并各批 log-sum-exp，再减全局正例均值；
  不是各批风险平均，尾批不等长仍覆盖所有行。批次算子及精度变化可有浮点末位差异。
  不注入 encoder 的默认 resident 路径保留旧 float32 全量验证归约。
- TS 边缘池仍为训练 P∪U、OS 为 U-only 消融；P 池不变。验证始终用验证 P∪U，
  **只是 D22 source diagnostic，不得据此 PA/OA 选模**；不读取 clean 标签。
- 每轮最大 log-Phi 归一化仍来自当前训练边缘池，不使用验证/测试最大值。
  快照包含 encoder、头和冻结的归一化层；新快照携带推理批大小。
  旧引用缺此字段时仍走原 checkpoint 兼容规则，不重写历史制品。
- 完整 epoch 后 callback 可预测；任何 fit 异常（含后续 callback）均撤销 fitted 标志。
- registry 能力、UI CNN 候选、API、方法卡、方法台账与准入草稿同步。
  `PUPipeline` CNN 入口沿用共享 encoder 构造，不接受 classifier_params 越权注入。

## 验证范围

新增 `tests/unit/estimators/test_vpu_encoder_stream.py`：独立模板/参数更新、
四维批次上限、OS/TS 池、全验证诊断、极端值/不等长尾批的全局归约、
resident/stream 训练一致性、clone/pickle、快照及推理批大小、输入拒绝、
晚期 callback 失败和公共 pipeline CNN 接线。

这些测试使用小型合成输入和 CPU；不声称真实数据、CUDA 峰值显存、吞吐或多 seed
资源验收。默认兼容性另与修改前提交 `771a1a2` 的 VPU 隔离运行对照。

验证结果（Python 3.11 / CPU）：新增 **14 项测试通过**；VPU/能力/台账/准入/UI
定向组合 **84 passed / 1 skipped**；完整 fast suite **3779 passed / 59 skipped /
0 failures / 0 errors**，3838 项，267.080 秒，未执行 slow/e2e。
JUnit 本机记录：`/tmp/pu-vpu-encoder-target-20261015.xml` 与
`/tmp/pu-vpu-encoder-full-20261015.xml`；临时日志不是永久实验制品。

默认兼容对照：NumPy seed31、29×5、9P/20U、MLP7/depth2、epochs3/batch6/
random_state5，OS 与 TS 各自与 `771a1a2` 的 scores、max_log_phi、完整 history 和
全部 state_dict 张量逐位相等。该有限样例不外推所有平台/数据。
另用**真实共享 CNN13**（未 monkeypatch）跑公共 pipeline：8×1×16×16 合成图像、
2 折加最终重训、epochs1/iterations1/batch2，得到有限分数并保持 CPU-stream。
这是工程 smoke，不是正式 backbone/预处理规格批准。

格式、测试质量、文档链接/API、元数据、Skills 同步、数学、baseline、注释、
目录/依赖层级及 materialized recipe draft 门禁通过；离线 sdist/wheel 构建通过，
wheel 含新实现，源码包不含用户 RAR、私有 Excel 或 outgoing 数据包；未发布/推送。
冻结 protocol/comparison 规范摘要分别仍为
`287c2f45387f02714e5925b35dcb04f3f64e1e3faf8f740cab861fc5747dd84a`、
`340e18ce8fda933ccf3582e9d522aee7aac7aadfdf467737c7cb15a25dee1f4c`。

## 仍未完成及下一步

1. HENG958 复核注入 encoder 契约、CPU-stream 范围及方法学适配声明。
2. 获批共享 backbone/增强/预算、候选协议与配方注册后，才能做 P3.3 正式资源验证。
3. D22 的 PA 准则仍须另行裁决；本次不借用验证变分风险绕过阻断。
4. 作者 CIFAR 网络/增强/完整训练配方和公开数值对照仍缺。
5. 冻结 Survey v1 协议、对照矩阵、645-run 结果、签署状态均不变；
   VPU `admitted=false`，候选协议/预算/候选池引用保持 null。
