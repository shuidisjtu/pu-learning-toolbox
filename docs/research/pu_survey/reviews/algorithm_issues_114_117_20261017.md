# 新增算法复核整改（2026-10-17）

基线 `92efb4d`，承接 GitHub #114–#117。工程整改不改变旧冻结协议、矩阵、
645-run、历史签署或正式准入；各项明确区分已验证行为和保留限制。

## #114：PUET TS bootstrap 风险归一化

TS 的 U 风险池包含原 U 与 P 的独立角色副本。原代码按该池大小取每行权重，
但 bootstrap 仅抽原 U 数量，导致根节点 U 总质量不足。现在从实际风险池
有放回抽取该池大小的行，P 仍独立抽 n_P 行，故每棵树 W_U=1、W_P=prior。
OS 抽样数量/RNG 调用和非 bootstrap 分支不变，默认 bootstrap=false 不受影响。

新增 2 views × 2 bootstrap × 3 P/U 数量 × 3 seeds 的36个组合，逐树检查
实际角色数量、风险质量、根风险0.96及恒定叶负票。修复前9个 TS-bootstrap
组合失败，修复后通过；不是只比较相同seed的重复结果。

另外只读加载基线 `92efb4d` 的原 PUET，在 seeds0/7/23 对照 OS 非bootstrap、
OS bootstrap、TS 非bootstrap，共9组成对拟合；分数、重要度、叶数、深度逐值完全相同。

## #115：LaGAM 发布采样定位

锁定作者 commit `362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8` 的
`utils/utils_data.py:182-193` 明确从 U 移除验证和已标记 P，因此将当前类与
registry 派生 scenario 统一为 SINGLE_TRAINING_SET、台账与准入草案快照为 os。
旧来源收据的初始 TS/OS 冲突保留，另补 resolution；不代签负责人。

默认 OS、calibration_applied=false，显式 TS 拒绝。这个判断只描述作者发布
实验，不断言算法理论只能 OS；独立 clean support、额外标签预算与 PA 限制不变。

## #117：CVIR 先验与运行路径

- 原生 TS：组件 `input_sampling=ts`，由调用者提供独立 P/边缘 U，显式 α_U 与
  来源；α_U 是分布参数，不是从 U 真标签计算的有限样本比例，不声称来源已统计验证。
- OS 工程适配：默认组件保持旧优化行为；独立实验注入 `CVIRKnownPriorTrainer`，
  必须给 α_U 和来源，仅 SCAR。PA 的总体 π 与训练 α_U 分开，前者绝不转发给 fit。
- OS→TS 校准：未支持；显式校准请求拒绝，台账 calibration_hooked=false。
  排序、保留与损失全部针对原始 U，无 P 偷换入候选；不预判简单合并必然够或不够。

候选 manifest 登记 input_sampling、path、known_prior、α_U、source、角色数量和
calibration_applied=false；PA 的 population π/来源仍在 selection 中。
测试检查实际候选索引和保留数量、先验参数、native/OS 路由拒收、真实 epoch
checkpoint 与 manifest，不只核布尔标志。没有把该适配器放进旧冻结矩阵。

## #116：PULNS 额外监督、隔离与逐数据集适配

再次检索[正式论文](https://cdn.aaai.org/ojs/17064/17064-13-20558-1-2-20210518.pdf)、
[作者机构页面](https://www.microsoft.com/en-us/research/publication/pulns-positive-unlabeled-learning-with-effective-negative-sample-selector/)
及源码关键词，仍未确认作者仓库/发布包；不是不存在的证明，不冒称官方源码划分。
[机器适配表](../data/pulns_dataset_adaptation_20261017.json)逐一记录八集映射、
作者覆盖、支持集准备规则及限制：MNIST/CIFAR 映射对应论文；Spambase 原文的
正负计数一致但不推断未确认的 raw 编码；其余五集明确为本项目扩展。
保留 Survey 标签映射与旧划分，不把 γ 当 c，不自动套论文支持集样本量。

新增 opt-in `PULNSExtraSupportTrainer`，只供独立 SCAR OS、终点 OA 实验：

- 调用者显式提供 clean support、来源、采样规则/seed、原训练量和是否扣除训练行。
  类别不足直接拒绝，不补采、不借 clean_val/test。适配器本身不重新划分或随机挑 support。
- 训练前对实际 bundle 四角色、prior_estimation、support 核全局稳定 IDs、互斥及摘要；
  有受试者/重复组时核所有角色组互斥，ADNI 无组证据直接阻断。
- support 只用于训练奖励与内部最优分类器，不进入 classifier 拟合行，但明确属于额外监督。
  顶层 extra_supervision、逐候选 provenance 登记额外正负标签、训练标签总量、
  原/剩余训练数、用途及 `comparison_group=extra_clean_support`，不得并入 PU-only 排名。
- 只有独立 clean_val 做外部 OA；PA、校准 TS、SAR 和旧 frozen Survey 明确拒绝。
  普通 Pipeline/CLI/UI 不因此开放，直接组件仍可用，但缺全角色证据时不宣称隔离已验证。

工程证据来自合成 CPU 输入，覆盖样本副本/跨角色 ID 重叠、重复组、来源/预算缺失、
support 中途改变、PA 拒绝、test 真值/特征变化不影响训练与选择及保存加载。
没有新增八集正式 support 制品；实际标签来源、预算/准备与正式协议仍需批准。
这项整改提供安全接口、逐集规则和明确不可执行边界，不伪造实际数据验收或人的签署。

## 独立实验调用

以下为 Python API 片段，`train/pu_val/clean_val/test` 应是已有互斥 DatasetPart。
`alpha_u` 必须由调用者依据实际 U 池给定，不能从示例捏造数据的先验。

```python
from pu_toolbox.estimators.risk.cvir import CVIRClassifier
from pu_toolbox.experiment.cvir_adapter import CVIRKnownPriorTrainer
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA

model = CVIRClassifier(unlabeled_positive_prior=alpha_u,
                       prior_source=alpha_source, input_sampling="os")
result = ExperimentRunner(
    class_prior=population_pi, protocols=[ProtocolPA(), ProtocolOA()],
    config={"c": label_frequency, "trainer": CVIRKnownPriorTrainer()},
    manifest_path="new-cvir-exploration/manifest.json",
).fit(model, train, pu_val, clean_val, test)
```

PULNS 必须另有准备好的 `support`（DatasetPart，clean，for_selection=False），
而不是上述任意验证/测试角色的别名。下面的 budget/seed/source 都取实际准备记录。

```python
from pu_toolbox.estimators.deep.pulns import PULNSClassifier
from pu_toolbox.experiment.pulns_adapter import PULNSExtraSupportTrainer

trainer = PULNSExtraSupportTrainer(
    support, dataset=dataset_name, source=support_source,
    prior_estimation_indices=prior_ids,  # 未用数据估先验时显式 []
    original_train_count=original_train_count,
    support_removed_from_train=support_removed_from_train,
    preparation_seed=support_seed, sampling_rule=support_sampling_rule,
    group_policy="not_applicable",  # 有组/ADNI 改为 grouped 并传全角色 role_groups
)
result = ExperimentRunner(
    protocols=[ProtocolOA()], config={"c": label_frequency, "trainer": trainer},
    manifest_path="new-pulns-extra-support/manifest.json",
).fit(PULNSClassifier(), train, pu_val, clean_val, test)
```

最终表/图必须显示 “PULNS + extra clean support”，并给 support 正负数与标签总预算；
`extra_clean_support` 独立分组，不与 PU-only 作不加限制的公平排名。
