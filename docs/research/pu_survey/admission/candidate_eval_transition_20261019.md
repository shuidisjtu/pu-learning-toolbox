# 三候选方法 eval 转换异常保护（2026-10-19）

P3.2 独立工程收尾：GEN-PU、Holistic-PU、PULNS 的评分函数已有逐层 mode 保存恢复，
但 eval 位于 try 外。自定义 encoder 的 train/eval 覆写在部分模式切换后抛异常时，
finally 根本不会执行，调用者捕获异常后看到模型停留在被部分改写的状态。

新增21个真实短拟合后的共享模式回归：修复前18通过、3失败，失败均为这三方法
的 eval 转换注入异常，而非训练失败或公式差异。每个方法仅将原 eval 调用移入
既有 try，恢复机制、默认参数、正常前向和训练目标全部保持。

共享 `test_pulda_gradpu_inference_modes.py` 现覆盖七方法；除 eval 异常外还检查
混合 BN/Dropout、首批/后批前向异常、pickle、空输入和形状错误。
必须保持分数、参数/BN buffers、requires_grad及torch RNG，不隐藏/重试异常。
PULNS 使用独立生成的合成支持集标签；不从外部验证或test取支持标签，也不把
这个单元测试登记为真实支持集身份/标签预算验收。

本改动不裁决 GEN-PU #126 的正式网络/输出域/六网络预算，不改变 Holistic-PU
已决定的论文趋势/初始化分支，不放宽 PULNS 的额外监督/PA 禁入或正式资源阻断。
方法台账保留原 calibration/prior 事实，新增本补记；历史测量与源审阅记录不回填。
未执行作者代码、真实数据/GPU或公开数值复现，不关闭 issue、不代签。

工程验收：三方法 fit AST 均与前一提交相同；七方法共享模式契约49项通过。
连同 GEN-PU/CNN、Holistic-PU/CNN/LZO/初始化、PULNS/CNN专项共168通过、7跳过，
零失败/错误，5.931秒，JUnit `/tmp/candidate-eval-final.xml`。
跳过不计作CUDA或完整论文预算验证；本轮结束另做最终全量快层回归。
