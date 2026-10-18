# LaGAM 两种预测路径的模式保护（2026-10-19）

承接 P3.2，不依赖新的论文或合作者数据。原预测直接调用 model.eval，结束后不恢复
调用者模式；异常和空输入也会改变模型状态。新增7项 CPU-stream 模式契约在修复前
全部失败，实际拟合已成功，不把支持集缺失等预检失败当成问题复现。

现仅修改 `_decision_function`：输入验证后空输入直接返回，非空保存全部模块模式，
eval/设备获取/原resident或CPU-stream评分均在try内，finally逐层恢复，不递归覆盖。
分批大小、设备策略、原评分/阈值、fit/虚拟元更新、目标/采样/校准、支持集契约保持。

共享模式回归核对根train/eval和混合BN/Dropout、前向全部eval、首批/后批异常、
eval转换异常、空输入、错误shape、pickle；权重/BN buffers、requires_grad及RNG保持。
另有3项直接覆盖无encoder resident路径：成功、前向异常、eval异常，及每种情况下
空输入不改状态。异常对象原样传播，不重试。

专项共99通过、1项CUDA跳过，零失败/错误，3.826秒；
JUnit `/tmp/lagam-modes-final.xml`，八方法共享模式契约及resident补测共59项。
只在合成数据上短拟合，支持集独立生成；不使用外部test/clean_val，不证明真实来源/
组身份或标签预算，不放宽PA禁入。台账和来源补记引用本证据，不重写历史资源。
未跑真实数据/GPU、不执行作者代码、不代签正式配方/数值准入或关闭#88。

与修改前自有实现对照3 seed × resident/CNN-stream × 有/无预测回调共12条短训练：
history、optimizer步数、伪标签、全部权重和预测逐bit一致。对照不比较预测后原本有缺陷
的mode副作用，不是作者源码重放或完整预算复验。训练fit本身不变。
