# Split-PU 分批评分模式恢复（2026-10-19）

P3.2 独立工程收尾；不触碰 #122 仍待确认的 JSD 方向/归一化、TS 总体比例和 prior。
不改论文目标、采样、校准、阶段预算或冻结结果，不代签正式准入。

## 缺陷与修复

共享 `_batched_score` 既用于公开预测，也用于冻结 teacher 目标、splitter 评分与
后续 distillation 目标。此前只存根 training，用递归 train 恢复，覆盖了混合子层模式；
eval 转换自身抛异常也不在 try 内。新增7项 Split-PU 参数化回归在修复前6失败，
只有空输入/非法形状路径通过。现保存全部模块标志，eval 与评分均由 finally 保护，
逐模块恢复，不递归覆盖，不隐藏或重试异常。

原 teacher/splitter/student 的优化和阶段编排不改；但这个 helper 也被 fit 调用，
不能仅凭 fit 的AST不变就声称训练等价，必须验证实际行为。

## 验证与边界

- 模式/异常、原 Split-PU、CNN 和 #122 表征专项：64通过、2项CUDA跳过，
  零失败/错误，2.670秒，JUnit `/tmp/split-modes-final.xml`。
- 共享推理模式契约现覆盖 PULDA、GradPU、Robust-PU、Split-PU；断言实际前向全eval、
  正常分数不变、首批/后批异常恢复、eval 转换异常恢复、空输入/形状和pickle。
  参数/BN buffers、requires_grad及torch RNG不变。
- 当前源码摘要在预算草稿刷新，台账指向本补记。历史资源读数保留原身份，
  没有真实数据/GPU或公开数值重放，不把路由测试升级为校准方法学复核。
- 与修改前自有实现对照3 seed × MLP/CNN × OS/TS共12条短训练路径：
  history、更新数、teacher/splitter/最终student全部权重、easy/hard数量及预测逐bit一致。
  非完整默认预算，不执行作者代码；只能证明这些正常路径保持，不证明所有输入等价。
