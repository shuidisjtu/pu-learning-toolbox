# PUSB 线性基线来源与行为核查（2026-10-19）

范围：区分 `pusb` 线性附加基线与 `pusb_kernel` RBF 适配器，核对方法卡、estimator、registry/ledger 和作者来源登记。属于单方技术核查，不是方法负责人签署、正式准入或论文结果复现。

## 原始来源

- Kato, Teshima, Honda, “Learning from Positive and Unlabeled Data with a Selection Bias,” ICLR 2019。论文官方 OpenReview PDF：[rJzLciCqKm](https://openreview.net/pdf?id=rJzLciCqKm)。摘要说明其目标是在温和条件下学习保持真实类别后验排序的 scoring function，并通过适当阈值用于分类；这不是对观测 P/U 来源标签直接做任意分类器拟合。
- 作者代码仓库：[MasaKat0/PUlearning](https://github.com/MasaKat0/PUlearning)，工具箱锁定 commit `3401b77ccdd653d39f4f3a6258a42c7938fa9ede`（见 `benchmarks/assigned_methods/configs/official_sources.lock.json`）。当前仓库主页注明正在重制，因此复核应以锁定 commit 与本地 `official/pusb.json` 记录为准，而不是浮动分支。
- `PUSB.md` 同时记有官方 RBF 目标、中心抽样、CV、先验分位数决策及正则梯度系数差异；这些证据对应 `pusb_kernel`。不能把它们自动归给单独注册的线性 `pusb` 条目。

## 线性 `pusb` 当前行为

| 检查项 | 当前实现 | 结论 |
|---|---|---|
| 训练目标 | `PUSBClassifier.fit` 将观测 `y_pu` 直接交给 `StandardScaler + LogisticRegression`，标签为已标记 P vs U | 是通用 source-classification baseline；它没有实现论文 RBF PU 风险目标/官方中心选择与调参链。 |
| `class_prior` | 接受后只存入 `_class_prior`，不参与训练、阈值或预测 | 与“该 linear baseline 不需 π”一致；不能据此说实现了论文使用 π 的 kernel decision rule。 |
| 分数语义 | `predict_proba` 返回逻辑回归对观测来源标签的概率 | 不是理论保证的真实后验排序函数，更不是 `P(Y=1|X)` 校准概率。 |
| 阈值 | `threshold` 作用于观测 P/U 来源概率 | 仅是工程分类阈值；不能借用官方 RBF 适配器的先验分位数阈值语义。 |
| 加权 | 可将 `sample_weight` 传给 sklearn LogisticRegression | 属工具箱基线接口能力，论文的 PUSB 目标并未由此实现。 |
| 校准 | 不提供可独立替换的 U 风险角色；D20② 将线性项判为 TS-OS 校准不适用 | 保持 `calibration_hooked=false` 符合当前实现结构；Pilot 采用独立 `pusb_kernel` 项。 |

## 状态更正与测试边界

`PUSBClassifier` 先前声明 `source_status=official_exact`，但其算法实现是未声称复现论文的线性附加 baseline。现更正为 `official_related`：作者仓库与方法相关且被锁定，但该类不是该仓库算法的 exact implementation。`PUSBKernelClassifier` 的状态和 Pilot 使用方式不因这项更正而变。

基础 estimator 测试覆盖拟合、预测形状、有限概率、参数边界及缺少 U 的错误；台账契约测试绑定注册状态与来源说明。它们不验证 posterior 排序理论、不复现 ICLR 论文数据/结果，也不替代 kernel 方法的来源复核。

结论：保留 `pusb` 作为显式标注的线性 P/U baseline，不把它重新命名或纳入 Pilot；修正 source status 和文档的分数语义，补来源审计引用。冻结实验结果、比较矩阵和准入状态不变。
