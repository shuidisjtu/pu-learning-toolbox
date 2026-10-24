# 给合作者：PULDA / GradPU 图像配方待确认清单

> 2026-10-23 时间截点补记：以下保留2026-10-18原始交接状态，不代表当前 issue 状态。
> #119/#120/#121 已按[20261021任务复核](../reviews/heng958_task_review_20261022.md)处置，
> 不因旧表重开。独立图像 runner/完整120/200小子集快照已见
> [本日交付](../delivery/candidate_image_runner_20261023.md)；正式 v3/预算边界仍保留。

状态：**待合作者确认，未签署、未正式准入**。
2026-10-18 再次读取远程：[#119](https://github.com/shuidisjtu/pu-learning-toolbox/issues/119)
和 [#120](https://github.com/shuidisjtu/pu-learning-toolbox/issues/120) 均 OPEN、0评论；
正文最后更新时间仍分别是 01:45:44Z / 03:24:54Z。
本文件是仓库内的交接清单，不表示已向 issue 发评论或关闭 issue。

## 已定方向，不需要重复选择

- PULDA：端到端 ResNet-18，benchmark-adapted，不是 feature-adapter/作者 CNN 复现。
- PULDA：`π_population` + known/oracle-prior；TS 风险角色逐批 U∪P；验证/测试保持 OS。
- GradPU：方案 B，`cnn13_no_bn`、端到端、method-private/no-BN CNN 单列，保留 BN 拒绝。
- GradPU：paper-only，不冒充作者源码或论文数值复现，不进入标准 ResNet-18 主榜。
- 所有实现、单元测试和短测均不能替代完整资源 profile、版本化配方与正式准入。

## 请按编号回复的决策

| 编号 | 需确认的内容 | 当前工程行为（不是正式批准） | 回复选项与影响 |
|---|---|---|---|
| D119-INIT | 各端到端方法初始化如何对齐 | 新入口在构造前用同 seed 独立构造，再各自 deepcopy；有模板 state hash | 接受“每个 fold/run 同 seed 重建并核 hash”，或要求共享模板；两者必须保证同 fold 的初始 encoder 权重相同 |
| D119-BN | 端到端组是否统一训练 BN | PULDA train 时更新 BN、eval 不更新 | 训练 BN；或统一冻结 BN。若选冻结，需新增显式接线与测试，不静默修改 |
| D119-AUG | 是否启用共享增强 | 默认关闭；已支持两阶段逐训练 batch 的 SimAugment/RandAugment，增强后做输入 MixUp | A：共享增强关闭、仅方法 MixUp；B：SimAugment + MixUp；C：RandAugment + MixUp。共享组内需统一，验证/测试均不增强 |
| D119-MLP | 表格模型及图像分类头的公平性边界 | 表格 MLP64×2、ResNet 后 MLP 头均为 method-private | 暂单列不混排；或给出共享层数/宽度/激活/头结构后统一接线。只统一 encoder 不代表完整模型相同 |
| D119-WARMUP | 是否保留 LDA warmup | 保留 LDA，与当前方法/源默认一致 | 保留 LDA；若另选按数据集 BCE，应明确数据集并独立登记变体，不混报或静默替换 |

可直接回复：

> INIT=同seed重建；BN=训练/冻结；AUG=A/B/C；MLP=暂单列/共享规格链接；
> WARMUP=保留LDA/另立变体。确认人、日期及适用数据集：……

这只是回复模板，空选项没有默认批准效力。签署应明确范围；不能用一条“同意”
把资源、公开数值与正式准入一并通过。GradPU 无 BN 方向不在此重新投票。

## 不用等待上述决定的工程工作

- [x] seeded ResNet-18 / CNN13-no-BN 技术构造、初始化/归一化记录。
- [x] PULDA 两阶段逐批增强、MixUp 顺序、BN/模板隔离与拒绝边界测试。
- [x] 新增只读快照空间预检，按当前源码默认120/200个快照计算，不启动训练。
- [ ] 正式 runner 的分层、实际参数与 manifest 强制核对：需先绑定版本化配方，不能直接改冻结入口。
- [ ] 完整120/200快照的真实存储、CPU/GPU峰值与多 seed资源记录：独立资源验收，预检不能代替。

## 资源预检的使用方法与边界

以下 P/U 数量仅为命令演示，实际应使用对应 split 的观测训练 P、原始 U 数量：

```bash
python -m scripts.prepare_candidate_image_resource_plan \
  --method pulda --target-dir . --n-positive 1000 --n-unlabeled 9000

python -m scripts.prepare_candidate_image_resource_plan \
  --method gradpu --target-dir . --n-positive 1000 --n-unlabeled 9000
```

脚本仅用 AST 读取当前构造默认值、计算更新数、查询指定文件系统剩余空间，输出严格 JSON；
不导入 torch、不初始化 CUDA、不读取数据集、不加载 checkpoint、不创建目录、不跑训练。
默认不提供快照样本时，空间估算/是否放得下均为 null，不能因空间大就放行。

如已有**相同方法、实际骨干/头及 checkpoint writer**生成的可信样本，可增加
`--sample-checkpoint /实际路径/epoch.pt`。脚本只读取文件大小和 SHA，不反序列化，
**不能验证样本模型/来源/格式匹配**；该匹配仍需操作者提供证据。

权重文件估算为：样本字节 × 全部快照 × candidates × retained-attempts × concurrent-runs，
再向上取整加安全余量。默认1候选、2个保留尝试（一次重试）、1个并行run、20%余量。
`--retained-attempts 1` 仅适用于已经核实不同时保留重试制品的调度；不得凭愿望减半。
选模后回收不抵扣训练峰值。实际候选并发/attempt留存策略改变时必须重新计算。

这是**权重文件空间规划**，不覆盖模型模板、optimizer、CPU/GPU内存、数据缓存、日志
或其他进程消耗；“文件空间足够”也不是资源验收通过。输出始终保留
`formal_resource_profile_complete=false`、`ready_for_formal_run=false` 和
`approved_representation_ref=null`，也不预占磁盘或 GPU。

完整实现及前一轮验收见[工程交接](pulda_gradpu_image_review_20261018.md)。
按此清单回复后，下一批先更新版本化规格和接线，再约定完整资源 profile；
GradPU 学习率等 #121 问题另列，不随本清单自动接受。

## 本轮独立工程验收

资源预检新增17项测试通过；与现有预算规格、存储探针、PULDA 增强和两方法
图像入口合计 **96 passed / 0 skipped / 0 failures / 0 errors**，23.271秒。
JUnit：`/tmp/image-resource-plan-regression.xml`。12项本地静态门禁通过。
测试覆盖重试/候选/并发乘数、整数向上取整、空间临界值、缺样本拒绝放行、
非法参数与路径、只读/不创建目录及阻止 torch 导入的子进程。
两条 CLI 仅以演示 P/U 数量完成只读预检，未读取真实数据或进行训练。

本轮未重跑全仓快层或核验远程多平台 CI；前一轮全仓结果不冒充本轮结果。
完整资源 profile、GPU实测和正式跑批均未执行，待确认项仍保持未签署。
