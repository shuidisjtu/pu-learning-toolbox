# Method Card: P3MIX（组件阶段，未注册）

Li et al., ICLR 2022，[官方报告页](https://iclr.cc/virtual/2022/poster/5904)、
[作者 slides](https://iclr.cc/media/iclr-2022/Slides/5904.pdf)，第 10–12 页。
完整论文 OpenReview `NH2992OYEmj` 的下载在 2026-10-05 返回 403；作者代码尚未核实。
不以其它 benchmark 的改写代替作者来源，也不将名称相似的 mixup 当完整方法。

## 本阶段交付

`estimators/deep/p3mix.py` 有三个可核查的 batch 数学组件及训练候选池身份门禁，不是 estimator：

- 正例预测熵的 top-k candidate pool，熵端点有限，等熵按原行顺序。
- U 的 marginal 条件为闭区间 `1−gamma <= f(x) <= gamma`；marginal 从候选 P
  均匀选伙伴，其它行从当前 P/U minibatch 均匀选伙伴。
- `lambda'=max(lambda,1−lambda)` 混合特征与软标签；目标为独立 `mean_P + beta*mean_U`，
  不按 P/U 行数合并成一个均值，软目标停止梯度。

候选数组必须由训练已标记 P 生成；`p3mix_training_candidate_pool` 从明确的训练 PU 标签中
仅选 P，保留唯一整数样本身份，不选高熵 U；这是工程防泄漏门禁，不是新增论文公式。
纯 batch 函数仍不能核实候选调用者的身份声明，完整训练接口还需把分区门禁接入；调用者
不得把隐藏 PN 真值传作观察标签。`tests/unit/estimators/test_p3mix.py` 检查公式、闭区间边界、来源选择、可重复性、
不改输入、分组损失和“不注册未完成分类器”。

## 未完成与准入边界

完整 epoch/pool 更新时序、论文 P/U 构造、公开训练预算和源码版本仍待完整原文核查。
P3Mix-E 的 mean-teacher/early-learning、P3Mix-C 的高置信修正均未实现；不把基础 mixup 冒充它们。
无 registry 或方法台账“完成”条目、无 TS-OS 校准、无 GPU/正式跑批声明。
后续先补来源，再实现完整 estimator、身份/标签预算门禁、台账与资源检查。
