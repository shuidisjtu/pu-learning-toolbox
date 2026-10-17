# BBE：显式 α_U 的独立混合比例估计

Garg et al., NeurIPS 2021，*Mixture Proportion Estimation and PU Learning:
A Modern Approach*。[论文 Algorithm 1，印刷页4](https://proceedings.neurips.cc/paper_files/paper/2021/file/47b4f1bfdf6d298682e610ad74b37dca-Paper.pdf)。

本次补齐统计量和端到端二维工程入口，不是作者深度配方复现，不是 TEDⁿ，
不注册到消费总体 π 的 CLI/UI/PUPipeline 先验入口，也不改变现有 CVIR。

## 算法与来源选择

固定评分器在独立留出 P、边缘 U 上产生高分表示正类的 [0,1] 分数。
按含等号的上尾计数计算两个角色的经验尾部概率；以论文的上界准则选阈值，
返回该阈值的 U/P 尾部概率比，而非上界本身。遍历两组分数并集和零阈值；
排除正类尾部质量为零的阈值。准则相同取最低阈值；不建样本×阈值矩阵，
排序主导复杂度。完整候选、计数、比值、准则和选择位置均留在不可变结果中。

作者仓库锁定 `acmi-lab/PU_learning@5e5e350dc0588de95a36eb952e3cf5382e786aec`，
`estimator.py:64–126`。发布代码仅枚举部分 U 间隙、使用零基 i 计 U 尾部，
并可能无候选返回0；这里遵循论文完整经验上尾定义，**不称源码重放**。
源码虽接受 U 真值供一个未使用计数器读取，但该计数器不影响返回估计；
本实现不接受真值，不把该差异描述成源码估计依赖真值。
逐项摘要与位置见[机器可读核查](../pu_survey/data/bbe_source_review_20261017.json)。

## 使用与数据隔离

```python
from pu_toolbox.prior import BBEMixtureEstimator

# X 是独立采样的 P 和边缘 U；y_pu=1 表示 P，0 表示 U，不是真负标签。
estimator = BBEMixtureEstimator(input_sampling="ts", random_state=42)
estimator.fit(X, y_pu)
alpha_u = estimator.estimate_unlabeled_prior()
receipt = estimator.result_
```

默认每个观察角色随机留出20%（向上取整），只在余下训练行拟合
StandardScaler 和 balanced LogisticRegression（max_iter=1000）；留出行仅用于
评分和 BBE，不用于拟合预处理或评分器。可以传入可 clone 的概率评分器；
按 classes_ 定位正类列，不假设固定列序。只支持密集二维输入；没有图像编码器、
深度网络训练、收敛停机或 TEDⁿ 的逐轮估计/丢弃过程。

`train_indices_`、`holdout_indices_` 和 `provenance_` 记录拆分边界与角色计数。
可传 `sample_ids` 检查输入原始身份不重复，但位置隔离和唯一 ID 都不能证明
跨数据来源、病人/组、外部验证或 test 的隔离，调用方仍需独立审计。
声明 `input_sampling="ts"` 也不能自动证明采样设计；OS 输入不提供转换入口。

如果已有**未用于训练该固定评分器**的 P/U 分数，可直接调用
`best_bin_estimate(positive_scores, unlabeled_scores, delta=0.1, gamma=0.01)`。
统计函数无法从分数证明留出独立性，责任在调用方；不得传训练内拟合分数冒充留出。

## 解释与验收边界

输出目标是未标记池正例比例 α_U，不是一般 OS 下的总体 π。只有在 U 确实来自
总体边缘分布等条件成立时才有相应等价，不能自动回填公共先验参数。
有限样本可能选出0或1，保持原样、不夹紧为内点，不为下游 CVIR 隐式改值。
小样本/常数分数可能选全体尾部并返回1，不解释为总体全正。

尾部比估计与真实 α_U 的关系仍需论文的可辨识性/纯正类区间等条件；
`selection_upper_bounds` 是选阈值准则，不是任意输入上获保证的 α_U 置信区间。
默认分类器、拆分比例和有限优化预算都是明确工程选择，未获正式实验预算批准。
合成 CPU 测试不代表公开数据数值复现、真实先验校准、GPU 验收或负责人签署。
另有独立的 [TEDⁿ 分类器](TEDn.md)，复用此统计量实现 warm-start、逐轮估计/丢弃、
未加权训练与训练误差停止；本 BBE 估计器仍不包含交替分类器，也不改变固定 α 的 CVIR。
