# VPU / 神经LBE评分模式保护（2026-10-19）

P3.2 独立收尾，不涉及原线性LBE、论文目标裁决、正式结果或资源验收。
VPU此前预测直接eval不恢复；神经LBE内部/公开评分只存根状态再递归train恢复，
且eval在try外。两个方法新增14项模式契约在修复前全部失败，均先完成实际短拟合。

修复对非空输入保存全部模块标志，eval与评分置于try内、finally逐层恢复；
空输入返回CPU float32空向量、不触发模式切换。VPU原预测归一化层、数值和概率
映射不变，神经LBE的class/propensity两个网络结构、E-step与更新规则不改。

共享模式契约现覆盖11方法，加LaGAM resident与神经LBE标记概率补测共83项。
神经LBE专门检查 `predict_label_proba` 的两个独立网络：成功、propensity前向异常、
propensity eval异常，以及相应空输入均恢复两边进入时状态；不会用类别分支成功
掩盖propensity失败。原异常对象不隐藏、不重试。共享测试要求实际前向全eval，
权重/BN buffers、requires_grad与RNG不变，pickle和错误形状路径不破坏模式。

连同VPU/encoder/调度/TS、神经LBE/数学专项共181通过、1项CUDA跳过，
零失败/错误，4.616秒；JUnit `/tmp/vpu-lbe-modes-final.xml`。
均为合成CPU工程证据，不是正式共享配方、真实数据数值复现或CPU/GPU资源上界。

台账增加本证据引用，先验/校准和extra-label预算保持原值。历史源码/资源读数
继续对应原测量，不回填为当前实测。未执行作者代码、正式跑批或负责人签署。

与修改前自有实现逐bit对照：VPU的3 seed × MLP/CNN × OS/TS共12路径，
神经LBE的3 seed × MLP/CNN × sample_fraction=0.5/1共12路径，均为短预算。
history、更新数、全部网络权重和预测一致；神经LBE最终E-step后验及标记概率也一致。
这不外推真实数据表现、共享正式配方或完整预算结果。
