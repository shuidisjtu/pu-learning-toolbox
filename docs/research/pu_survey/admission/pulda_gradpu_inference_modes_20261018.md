# PULDA / GradPU 推理模式恢复修复

这是独立工程缺陷修复，不是 #119 的训练 BN 策略裁决，也不是 #121 正式配方接受。
训练目标、随机数/默认预算、候选/校准、冻结协议/矩阵/台账均未修改。

## 复现与原因

两个 estimator 的预测此前只保存根模块 `training`，调用 `eval()` 后使用
`model_.train(was_training)` 恢复。这会递归覆盖全部子模块，丢失调用者的混合模式：
例如根模块train但某Dropout/BN为eval，预测结束会把后者也切回train。
异常路径同样存在；而eval切换自身抛异常时，原代码还没有进入finally保护。

新增回归初始12项中10项失败、2项空输入/形状路径通过。不是根据源码样式猜测缺陷。
修复保存每个子模块的mode，eval切换与前向均放入try，在finally逐模块恢复原标志。
不调用递归train来恢复，避免又覆盖子模块。原异常对象原样抛出，不重试训练/预测。

## 验证边界

`test_pulda_gradpu_inference_modes.py` 覆盖：根train/eval与混合子模式、
首批/后批前向异常、eval切换异常、空输入/非法shape、pickle和确定性。
正常推理必须全部eval，预测结果与标准eval基线相同；torch RNG、参数/BN buffers、
requires_grad均保持。训练自身的 `model_.train()` 行为不变，不新增BN冻结选项。

源码摘要在既有预算证据草案同步；旧资源probe不改、不冒充修复后的实测。
未跑真实数据或GPU，不代表正式来源/配方/数值或资源验收。

末次专项 **56 passed / 2 skipped / 0 failures / 0 errors**，2.082秒，
JUnit `/tmp/pu-modes-final-targeted.xml`，包含14项新增模式/异常回归。
两项跳过为CUDA；12项本地静态门禁通过。
相对前一提交，两方法fit的AST逐字结构相同；本批只改变推断模式保护及配套证据。
最终全量快层另行记录，不将修改前运行冒充末次源码验收。
