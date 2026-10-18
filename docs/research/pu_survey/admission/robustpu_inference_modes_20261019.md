# Robust-PU 推理模式恢复收尾（2026-10-19）

承接 P3.2 的独立工程收尾，跳过等待用户补充全文的 P3MIX。
推进前 fetch 后 main 与 origin/main 同为 `ec2202d`；公开 issue 列表没有新增项。
#126 要求保留 GEN-PU 已采用论文目标的决定和正式 recipe/预算阻断，本轮不重开目标选择。
不触碰暂停的 P2.2 review，不代签任何方法准入。

## 缺陷与修复

Robust-PU 原预测仅保存根模块 `training`，用递归 `train(was_training)` 恢复。
调用者设置的混合子层 BN/Dropout 模式因此丢失；`eval()` 本身抛异常时还未进入保护。
不是仅凭代码样式判断：新增7项 Robust-PU 参数化用例在修复前6项失败，
空输入/非法形状用例通过。失败覆盖根 train/eval、首批/后批前向异常、
eval 切换异常和 pickle 后预测。

修复保存全部模块的 mode，把 eval 与设备读取一起放入 try，finally 逐模块恢复标志，
避免递归 train 再次覆盖。原异常对象继续传播，不重试。
测试同时要求实际前向全为 eval、正常分数与标准 eval 基线一致，
参数/BN buffers、requires_grad、torch RNG 不变。

本轮复用 `test_pulda_gradpu_inference_modes.py` 的共享推理契约，原文件名保留；
现覆盖 PULDA、GradPU、Robust-PU。训练 fit、采样、目标、先验、校准和预算不改。
方法台账与来源记录新增本补记，预算草稿只刷新当前源码/绑定摘要；
来源记录中的历史验收摘要及原资源探测仍对应原测量，不回填成新源码实测。

## 边界与后续

这不是正式配方、来源数值复现或 CPU/GPU 资源验收，不关闭 #88。
正式共享图像规格、阶段候选资格、资源预算和合作者复核前置保持。
P3MIX 留待用户补充论文后再集成，不以工程测试替代来源核查。

## 验证记录

- 专项70项：68通过、2项CUDA测试跳过，零失败/错误，1.856秒；
  `/tmp/robust-modes-final.xml`。共享模式契约21项均通过。
- 与本轮修改前自有实现逐项比较3 seed × MLP/CNN × OS/TS共12条短训练路径：
  history、optimizer步数、全部state_dict及预测逐bit一致。不是作者源码重放，
  也不是默认完整预算或真实数据精度对照。
- `fit` 的AST与前一提交完全一致；15项静态及证据一致性检查通过。
  没有执行真实数据/GPU训练，历史probe不更新；远程门禁以推送后实际运行结果为准。
- 最终完整快层 `python -m pytest tests/ -m 'not slow and not e2e'`：
  **4673通过、58跳过、零失败/错误**，共4731项，318.847秒；
  本机JUnit `/tmp/robust-modes-release-fast.xml`，报告不纳入仓库。
  跳过以及未执行的slow/e2e不计作通过，不替代正式实验或CUDA验收。
