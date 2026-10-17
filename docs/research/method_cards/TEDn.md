# TEDⁿ：Transform–Estimate–Discard

Garg et al., NeurIPS 2021，*Mixture Proportion Estimation and PU Learning:
A Modern Approach*。[Algorithm 3 印刷页7与页8说明](https://proceedings.neurips.cc/paper_files/paper/2021/file/47b4f1bfdf6d298682e610ad74b37dca-Paper.pdf)。

新增 `pu_toolbox.estimators.deep.TEDnClassifier`，包含完整 warm-start、逐轮 BBE、
排序丢弃、神经网络更新与停止过程。它不是固定先验 CVIR，也不是仅有公式组件。
这是独立原生 TS 工程路径，尚未注册 CLI/UI/OS pipeline，未进入冻结矩阵或正式准入。

## 数据与时序

1. 从输入观察角色 P/U 各自随机留出20%（向上取整），两侧都必须非空；
   训练与留出原始位置可查，唯一字符串 sample_ids 检查可选。
2. 仅用训练 P 与训练 U 做显式 warm-start（默认5轮）。
3. 每个 TEDⁿ round 的更新前，用当前模型在同一内部留出 P/U 上评分，运行
   [BBE](BBE.md)；不读取 U 真值，不借用外部 PA/OA/test。
4. 只对**训练 U**排序，丢弃正类分数最高的 `floor(alpha_U*n_U_train)` 行，
   再用训练 P 与保留 U 更新一轮。各轮重新估计，不冻结第一次估计。
5. 在更新后以 eval 模式计算训练正类错分率加当前保留 U 的负类错分率。
   连续5次绝对变化不超过1e-4时停止，否则最多20个 TEDⁿ rounds。

**目标不是 α 加权 CVIR**：论文页8明确不对 TEDⁿ 两个角色损失按估计先验重加权，
Algorithm 3 更新两角色平均 BCE 的未加权和。发布代码是这两个均值之和再除2；
这里保留论文目标，不声称两者在有限预算/Adam/参数化下完全等价。

BBE 返回0时保留全部 U；返回1时负类角色为空，明确报错并使拟合状态无效，
不会夹紧先验、强留一行或把仅 warm-start 结果冒充完整 TEDⁿ 成功。
报错后可检查样本量/评分器信息量或显式调整 warm-start 预算，但不做静默 retry。

## 网络、隔离与成本

默认 MLP64；可注入可训练 `torch.nn.Module` encoder（输出二维特征），
支持二维特征或四维 NCHW 图像。encoder 深拷贝后联合训练，原对象不改动。
不额外拟合预处理器；调用方仍负责外部 train-only 标准化/特征生成来源，
不能让预处理、组身份、重复观测或原始 test 绕过内部位置隔离。

原始数组留在 CPU；训练 P/N 合并 forward 至多 `2*batch_size` 行，
留出、排序、误差和预测 forward 至多 `batch_size` 行。两角色各自切批，
短角色批列表循环以覆盖长角色；这是工程批策略，不是作者 zip/mask loader 配方，
尤其不能声称 BatchNorm 看到的分布完全相同。不是磁盘流式，仍需完整 CPU 数组。

`optimizer_steps_` 累计两阶段全部优化步；`epoch_provenance_` 与 snapshot
记录阶段/round/累计步；`checkpoint_epoch_count` 是 warm+round **上限**，
实际数为 `epochs_completed_`。snapshot 仅用于推理，不支持训练续跑。
`bbe_history_` 只保留每轮被选阈值、尾部计数、α、选择上界及更新前步数，
不累计保存全部候选数组；`alpha_u_` 是最后一轮**更新前**估计，不是最终网络新估计。

## 使用

```python
from pu_toolbox.estimators.deep import TEDnClassifier

# 角色 1=P、0=U，独立 P 与边缘 U；没有 U 的真实标签。
model = TEDnClassifier(input_sampling="ts", random_state=42, device="cpu")
model.fit(X, y_pu)
scores = model.decision_function(X_new)
alpha_used_last_round = model.estimate_unlabeled_prior()
```

输出高分表示正类，`predict` 在 logit 0 分割；不提供校准的 predict_proba。
`class_prior` 非空、sample_weight、外部干净验证标签均不接受。
完整参数/返回结构见 [API](../../user/reference/api.md)。

## 来源差异与验收边界

锁定作者仓库 `acmi-lab/PU_learning@5e5e350dc0588de95a36eb952e3cf5382e786aec`；
逐项文件摘要与位置见[来源记录](../pu_survey/data/tedn_source_review_20261017.json)。
论文估计时点、未加权目标与内部 split 均已核；发布代码的 BBE 尾部计数/候选、
两 logit/单 logit、半和/和、warm0 初值、loader 和 clean-target 诊断差异保留披露。

Adam、MLP、拆分比例、warm/round上限和误差容差/耐心是明确工程选择，
不代表原配方复现、正式选模资格或获批资源预算。重复使用同一留出并经 α
反馈影响后续模型，不额外保证逐轮置信区间有效；选阈值上界仍不是任意数据的真先验置信区间。
`input_sampling="ts"` 是调用方声明，不是采样设计已获验证；结果仍为 α_U，
不自动替代一般 OS 总体 π。OS→TS 校准与现有 CVIR 行为不变。
真实公开数值、GPU 多 seed、正式准入和负责人签署仍未完成。
