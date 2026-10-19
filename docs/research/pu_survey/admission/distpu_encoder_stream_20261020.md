# Dist-PU 显式 encoder 与 CPU-stream 接入（2026-10-20）

承接[双架构计划](../../../dev/dual_architecture_plan.md)阶段 2 的 Dist-PU 后续评估与
[执行计划](../survey_execution_plan.md) P3.2。来源训练损失/阶段仍以
[锁定源码审计](dist_pu_source_behavior_review_20261019.md)为准。
本轮是本地独立工程接入；正式图像规格、预算、资源与方法学复核保持原有前置。

## 接口与训练语义

`DistPUClassifier` 新增可选 `encoder=None`，能力声明与台账同步为 MLP/CNN、二维/四维。
条件是显式 `training_recipe="author_staged"` 且 `distribution_loss="author_histogram"`：
默认 clean-room 是全量目标，不能为了接收图像就静默换损失或假称已分批。因此 encoder
与默认 clean-room 组合会提前报错；普通二维 MLP 和冻结 Pilot 参数保持原路径。

- encoder 必须是 `nn.Module`，接受二维特征或四维 NCHW 输入，输出有限二维特征；fit
  deepcopy 模板、解冻训练、以 eval 单样本 probe 得到宽度，然后接原结构 MLP head。
  原模板权重、buffers、requires_grad 和模式不被拟合改写，各 CV fold 副本独立。
- 训练图像/特征、标签和持久伪标签保留在 CPU；warm-up/PU 每批只搬当前行到指定设备。
  两阶段的 loss、伪标签 identity/更新时序、Adam 重建、warm-up epoch / PU batch
  scheduler cadence 与现有 staged 实现相同，encoder 参数参与两个阶段的优化。
- 每轮保留所有尾批。若训练 batch 只有一行，BatchNorm 用运行统计，Dropout/其他模块
  保留原训练模式，前向结束或异常后恢复逐层状态。这是工具箱工程适配，不是作者 BN 规则。
- 初始伪标签扫描的 batch 为 `min(128,batch_size)`，普通 MLP 仍用原来的 128；公开推理
  按 `batch_size` 分批。包含 probe、扫描与评分在内的单次 encoder 前向行数不超过该预算。
  它是行数约束，不是显存/存储或总计算量上界。
- probe 用独立行副本，encoder 前向使用当前 batch 的 clone，隔离 in-place encoder 对
  原始训练数据、mixup 输入和调用方评分数组的改写。
- 推理检查 fitted 状态、完整样本 shape 与有限输入；恢复每个子模块原始模式，空输入不
  访问网络。旧 MLP estimator pickle 通过新增参数缺省迁移保持预测/get_params 可读。

TS 校准继续只扩大 histogram/entropy 的 marginal 风险角色，P 项、总体 prior、实际批次、
MixUp 池和伪标签训练行 identity 保持原定义。新增结构回归读取实际 histogram 调用的观测
标签/校准标志及 encoder 前向轨迹，确认 OS/TS 使用相同物理批次与 RNG 调用序列。

## Pipeline 与 UI

Registry 能力镜像后，`PUPipeline(architecture="cnn")` 可构造并注入 encoder；必须同时显式
传上述 staged/histogram 配置。下面只演示小样本 CPU 工程路径，不是正式配方：

```python
from pu_toolbox import PUPipeline

pipeline = PUPipeline(
    classifier="dist_pu", architecture="cnn", backbone="cnn13", device="cpu",
    classifier_params={
        "training_recipe": "author_staged", "distribution_loss": "author_histogram",
        "warmup_epochs": 1, "pu_epochs": 1, "batch_size": 16,
    },
    cv=2, random_state=0,
)
# pipeline.fit_evaluate(X_images, s_observed, class_prior=pi_population)
```

UI 的 CNN 候选由同一能力声明推导；手动模式提示两个必须的配置项及阶段预算。当前 UI
图像比较模式没有逐模型固定参数编辑，选择 Dist-PU 时会提示移除它或改手动配置，默认
clean-room 图像 trial 会明确拒绝；没有自动将比较中的默认方法切成另一个训练 recipe。
完整的逐模型比较配置不是本轮交付。

## 冻结兼容与验收范围

冻结协议的 `dist_pu` profile 仍是 fullbatch / mlp128；CIFAR 执行单元明确使用
`cnn_feature_adapter`，不会因 estimator 新增能力而改为原生 CNN。没有修改冻结 JSON、
历史运行 manifest、结果或比较矩阵；新 encoder 路径须在新版本协议/registry 中另行登记。

新增单元回归覆盖两个阶段真实 encoder 更新、原模板/输入隔离、单行尾批与 BN 计数、
分批上界、OS/TS 物理角色/RNG、参数/shape 失败、失败重训、pickle、逐阶段 checkpoint
推断回放。Pipeline 回归通过真正的微型卷积 encoder 和两折 CV+全量 refit 核对初始化共享、
训练副本隔离、provenance 及模型回放；未将其描述为官方 CIFAR CNN 或资源 profile。

连带补齐共享 `SnapshotPredictor` 的输入隔离：原 CPU tensor 可与调用方数组共享内存，
in-place 网络会在评分甚至异常退出时改写验证/测试输入。现每个评分 chunk 先复制为私有
数组再交给网络；二维/四维与异常回归确认原数组不变、分批大小与数值规则保持。
没有改写任何已有 checkpoint、权重或正式选择准则。

只刷新 live selection/budget 草稿的 checkpoint writer 摘要。2026-10-07 的 CPU 测量与
其他历史资源收据保留原源码摘要和测量值，不能据此声称输入复制后的当前版本峰值已测，
也不将旧探针当作新增 Dist-PU encoder 的资源上界。

## 本地验证

- encoder/Pipeline、旧 Dist-PU、能力声明、台账及来源专项 106 项通过；共享 checkpoint
  隔离与回放专项 53 项、live writer 绑定专项 48 项通过，均无失败或跳过。
- 对本项目旧提交 `19d75c0` 的 MLP 实现做 18 组短预算逐位对照：3 个 seed × OS/TS ×
  clean-room expectation / clean-room histogram / staged histogram。损失、模型权重、
  概率、优化步数和 staged 持久伪标签一致，概率最大差为 0。这不是作者源码复现或 Pilot 重跑。
- 全仓快速回归首轮 5,315 项仅发现一条旧的“Dist-PU 是 MLP-only”断言不再成立；该
  拒绝测试改为仍不支持 CNN 的 uPU，Dist-PU 的条件式拒绝由新增专项独立覆盖。
  这条历史首轮记录不代表全仓已全绿，后续完整回归单独记录。
- 随 P3MIX 输入隔离后续一并完成最终全仓快速回归：5,333 项中 5,274 项通过、59 项
  条件跳过，0 失败/错误；JUnit 为 `/tmp/p3mix-final-fast-20261020.xml`。排除 slow/e2e，
  不包含 GPU 或正式资源验收，也不代表远程多平台 CI 已确认。
- 格式、文档链接、API、测试质量、目录结构、层依赖与 baseline 配置门禁通过。

剩余前置：作者图像模型/增强/预处理未移植；正式 backbone、输入规范、BN/增强选择、
候选与阶段资格、预算/存储 profile、公开数据对照、多 seed/GPU 运行和 RV6 方法学确认仍待。
技术可训练能力不构成正式 Survey 准入。
