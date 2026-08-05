# 数据画像与假设提示

> 前置条件：先完成 [快速开始](../quickstart.md)。
> 概念：为什么 SCAR/SAR 通常不可识别、两种证据的区分见 [concepts/scar_sar.md](../concepts/scar_sar.md)。

`profile_pu_data` 在训练前统一检查 PU 标签规模、特征质量、类先验一致性和标记机制证据。它返回结构化的 `PUDataProfile`，既可在终端阅读，也可序列化后写入实验记录。

**核心解释边界**：仅观察到 $`X`$ 与 $`S`$ 时通常无法识别 SCAR/SAR；`observed_mixture` 证据只能筛查，`audited_positives`（提供 `y_true`）才能给出可识别证据，且 `plausible` ≠ 证明 SCAR 成立。完整推导与证据表见 [concepts/scar_sar.md](../concepts/scar_sar.md)。

## 1. 基本用法

```python
from pu_toolbox.preprocessing import profile_pu_data

report = profile_pu_data(
    X,
    y_pu,
    class_prior=0.30,
    random_state=42,
)

print(report.format_text())
payload = report.to_dict()
```

公共 PU 标签会先规范化为 `{1, 0}`。`class_prior` 是用户提供或独立估计的 $`P(Y=1)`$；profiler 不会利用隐藏真值偷偷估计类先验。

若有人工复核、延迟结果或合成实验中的真实标签，可启用审计模式：

```python
audited_report = profile_pu_data(
    X,
    y_pu,
    y_true=y_true,
    class_prior=0.30,
    cv=5,
    scar_auc_threshold=0.65,
)
```

所有 `y_pu == 1` 的样本必须在 `y_true` 中也是正类，否则接口会拒绝输入。这可以及早发现“正例标签并非可信正例”的契约冲突。

## 2. 返回对象

`PUDataProfile` 包含以下字段：

| 字段 | 内容 |
|---|---|
| `summary` | 样本数、特征数、已标正例数、未标记数、PU 比例、稀疏性、类先验及隐含标记频率 |
| `feature_statistics` | 缺失/无穷值数量、常数列和低方差列索引 |
| `selection_diagnostic` | AUC、状态、证据来源、可识别性、实际折数和评估样本数 |
| `issues` | 带稳定 `code`、严重级别、解释和行动建议的问题列表 |
| `assumption_hints` | 对 SCAR/SAR、类先验依赖和结果边界的说明 |

辅助接口：

- `report.has_errors`：存在会阻止可靠训练的数据错误。
- `report.has_warnings`：存在训练前应复核的问题。
- `report.format_text()`：生成适合终端和日志的报告。
- `report.to_dict()`：生成可写入 JSON 的普通容器。

`selection_diagnostic["status"]` 有三个稳定取值：

- `plausible`：交叉验证 AUC 不高于阈值。
- `at_risk`：AUC 高于阈值，检测到特征依赖信号。
- `inconclusive`：某一组少于两个样本，或存在非有限特征值。

## 3. 问题代码与处理

| Code | 级别 | 建议 |
|---|---|---|
| `no_labeled_positives` | error | 检查标签编码或收集可信正例 |
| `no_unlabeled_samples` | error | 改用监督学习流程或补充未标记总体 |
| `missing_features` / `infinite_features` | error | 在仅由训练折拟合的 pipeline 中清理或插补 |
| `few_labeled_positives` | warning | 使用重复验证、置信区间并尽量补充正例 |
| `extreme_pu_imbalance` | warning | 使用 PU 分层切分和抗不平衡指标 |
| `constant_features` | warning | 在训练 pipeline 内删除常数列 |
| `high_dimensional_data` | warning | 使用正则化模型，避免在全数据上预处理 |
| `inconsistent_class_prior` | warning | 复核类先验、采样总体和标签定义 |
| `sar_signal` | warning | 审计正例支持 SAR；优先评估 PUSB/LBE 并做敏感性分析 |
| `observed_selection_signal` | info | 只有非识别性信号；补充审计或标记策略信息 |

错误不会被自动修复。静默插补、删除列或改写标签会改变实验数据流，并可能造成训练/验证泄漏。

## 4. 类先验一致性

若提供 `class_prior=pi`，报告计算：

```math
\widehat c = \frac{n_{\mathrm{labeled}}/n}{\pi}.
```

它是由样本已标比例和类先验推导的标记频率。若 $`\widehat c>1`$，报告 `inconsistent_class_prior`。这不自动证明类先验错误，因为总体先验与有限样本比例可能不同，但必须在实验记录中解释。

类先验应来自独立知识、训练数据内部估计或嵌套验证。不得利用测试集真实标签回填，否则会产生信息泄漏。

## 5. 阈值与复现

默认配置：

```python
profile_pu_data(
    X,
    y_pu,
    min_labeled_positives=30,
    max_unlabeled_to_positive=100.0,
    low_variance_threshold=1e-12,
    scar_auc_threshold=0.65,
    cv=5,
    random_state=42,
)
```

交叉验证折数会自动缩小到较少标记组的样本数。若最小组少于两个样本，状态为 `inconclusive`。生产报告应保存完整参数、软件版本和随机种子；不要只保存最终布尔判断。

## 6. 稀疏输入与数据泄漏

接口支持 scipy 稀疏矩阵，并在计算方差时保留隐式零。稀疏矩阵显式存储的 `NaN` 和 `inf` 也会被检测。

Profiler 可以在划分前用于只读质量检查，但任何会学习统计量的修复操作，例如插补、标准化、降维和低方差筛选，都必须在每个训练折内部拟合。审计用 `y_true` 只用于假设诊断，不得作为分类器训练特征或超参数选择捷径。

## 7. 完整示例

运行：

```bash
python examples/minimal/07_data_profiling.py
```

示例并排生成 SCAR 与线性 SAR 数据，并展示观测模式为什么是非识别性的，以及审计模式如何提供更直接的标记机制证据。
