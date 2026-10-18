# PULDA / GradPU 图像接线：#119、#120 的工程补齐

核对时间：2026-10-18。公开网页读取成功；GitHub CLI 未认证、公开 API 返回403，
没有因此跳过 issue 正文。两条 issue 均为 OPEN、0 条评论；正文最后更新分别为
01:45:44Z、03:24:54Z。后续复核须再次读取远程，本文不是关闭 issue 或代签。

依据：[#119](https://github.com/shuidisjtu/pu-learning-toolbox/issues/119)
含 PU-Bench 补充与已定 prior 口径；
[#120](https://github.com/shuidisjtu/pu-learning-toolbox/issues/120) 采用方案 B。

## 本轮实现

`pu_toolbox.experiment.candidate_image.build_candidate_image_classifier` 是独立、
显式 opt-in 的技术构造入口，返回 `(estimator, image_setup)`。不注册正式配方，
不修改冻结协议、对照矩阵、方法台账、历史 feature cache 或正式准入位。

- PULDA：调用 `build_survey_image_encoder`，每次在隔离的 CPU RNG 下用同 seed
  构造随机 ResNet-18。初始化不依赖前一个方法消耗的 RNG，记录模板 state hash；
  fit 再 deepcopy 模板，全部参数可训练，BN 随训练更新，评估不更新。
  这确认编码器同初始权重，不代表各方法分类头也相同。
- PULDA 新增可选 `train_transform`，默认 None，旧默认路径不变。仅允许四维图像
  和 `nn.Module`；fit deepcopy 并冻结 transform 参数，保留 CPU 原始数据。
  两阶段的每个训练 P/U batch 都重新采样增强，不是在 fit 前仅增强一次。
  顺序是：缩放至 `[0,1]` → 训练增强 → PULDA 输入 MixUp → encoder 内 train-only
  通道统计归一化。每张图像独立采样由共享增强组件负责。
  预测、验证/测试及预热后的初始伪标签扫描不调用增强；持久伪标签索引与
  TS 风险角色不变。输出 shape/dtype/device/有限性不符即拒绝，不静默退化。
- PULDA 不改变 LDA/two-way margin、总体 `π_population`、逐 batch TS 风险角色、
  两阶段预算、阈值或尾批处理。没有照搬 PU-Bench 的 `π_U` 或非 CIFAR BCE warmup。
  增强是否正式启用仍须决定；构造时以 preprocessing 的显式选项为准并记录，
  不把技术支持当成统一协议已批准。
- GradPU：固定默认宽度的 `cnn13_no_bn`，使用同一 train-only 缩放/通道统计，
  原始图像上做 GP 插值，再进入 encoder 归一化；encoder 端到端训练。
  保留原 estimator 的 BN fail-closed，不自动删 BN 或换 ResNet。
  该构造入口要求 augmentation=none；对未批准的共享增强明确报错。
- image_setup 记录实际 backbone、初始化 seed/hash、输入尺寸/首层/归一化、
  trainable、BN、增强/MixUp 顺序、方法身份和未实测资源状态。
  PULDA 分层为 `end-to-end CNN`；GradPU 为 `method-private/no-BN CNN`。
  两者均未获得标准 ResNet-18 主榜准入。PULDA 头和表格 MLP64 仍是
  method-private，不因共享 encoder 自动变成共享 MLP。

## 如何使用与保存记录

仅用于独立工程试验，不应添加到现有冻结 runner 的方法列表。示意：

```python
from pu_toolbox.experiment.candidate_image import build_candidate_image_classifier
from pu_toolbox.experiment.image import fit_survey_image_preprocessing

X_train_scaled, spec = fit_survey_image_preprocessing(
    X_train, dataset="cifar10", train_augmentation="none"
)
model, image_setup = build_candidate_image_classifier(
    "pulda", spec, seed=0, class_prior=pi_population, device="cpu"
)
model.fit(X_train_scaled, s_train, os_or_ts="ts")
```

其他分区必须用同一个 spec 调用 `transform_survey_images(..., role=...)`，不重新拟合
统计。GradPU 改 method 为 `gradpu`，不传 class_prior；正式 batch/epoch 不在此示例缩短。
共享增强可在 PULDA 技术试验中显式改为 simaugment/randaugment，但须单独标注。

image_setup 是**构造记录**，不是完整实验 manifest。调用者必须将它作为独立字段
保存到该次运行的 manifest，同时保存实际 model 参数、split/indices/hash、数据集、
prior 来源/数值、`training_view_` / `calibration_applied_`、PA/OA 策略、候选/快照预算
及资源记录；不能把构造记录覆盖到 runner 的顶层正式协议字段。若 `set_params`
改了预算或模型，必须记录实际值，不能沿用默认配方声明。本轮没有把该入口接进
冻结 Survey runner，也没有批准阶段/阈值选模，不能直接启动正式跑批。

## 验收与仍待办

新增测试：`test_pulda_augmentation.py`、`test_candidate_image.py`。
第一轮16项通过：增强调用角色与次数、两阶段/MixUp 顺序、非法输出与失败重训、
clone/pickle/模板隔离、同 seed 与 RNG 隔离、真实共享 ResNet-18 的 BN/权重更新、
真实默认宽度 CNN13-no-BN 的 TS 端到端拟合和元数据。
既有 `test_grad_pu_cnn.py` 继续覆盖 raw-image 梯度范数、二阶反传、TS 插值、
checkpoint 回放和无 BN 拒绝边界；pipeline 集成测试另核 fold 隔离。

最终本地验收：全量快层 **4572 passed / 58 skipped / 0 failures / 0 errors**，
299.906秒，JUnit：`/tmp/issues119120-fast.xml`；12项静态门禁均通过。
源码目录内 wheel/sdist 构建通过，包含新模块且未混入六个私有根目录制品。
对比修改前自有 HEAD，3个 seed × MLP/CNN 的无增强 TS 路径，损失轨迹、
伪标签、更新数和预测逐 bit 一致。远程多平台 CI 未核验，不声称全仓零 warning。

| 条目 | 本轮状态 / 后续前置 |
|---|---|
| #119 编码器初始化、BN 现状、增强接口、归一化/MixUp 顺序 | 技术实现与测试已补；不代表整个端到端方法组已统一 |
| #119 统一 BN、共享增强启用与否 | 待共同决定；不新增冻结 BN 开关，不替团队选默认增强 |
| #119 表格 MLP | 维持 method-private；共享规格未定，不混排 |
| #119 warmup loss | 保持原 LDA，不因参考 benchmark 的推荐改成 BCE |
| #120 无 BN/private 路径与 paper-only 身份 | 技术构造与短测已补；不宣称作者源码或论文数值复现 |
| 正式 recipe/runner/manifest 强制分组 | 尚未接入；需版本化规格，现有构造记录不是准入凭证 |
| PULDA 120 / GradPU 200 快照资源 profile | 未实测，不沿用缩短 smoke 或旧 backbone 外推 |
| approved_representation_ref / 正式准入 | 仍 null / false，不代填、不中断既有审阅边界 |

后续顺序：先共同确认 #119 的增强/BN/MLP 选择，再登记版本化图像配方、
runner 分层与预算接线，最后执行完整快照 CPU/GPU 存储/峰值实测与负责人验收。
GradPU 的固定学习率等后续问题仍以 #121 为独立任务，不随 #120 自动解除。
