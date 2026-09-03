# JointShift 研究评估协议

```yaml
protocol_version: 1
status: research_clean_room
paper_claim: false
scope:
  - JointShiftPUClassifier
  - DynamicJointShiftPUClassifier
  - joint_shift_baselines
```

## 1. 研究边界

本协议评价研究路径中的联合/动态分布漂移求解器，不属于稳定注册表或普通部署适配
契约。当前实现用于公开公式、训练顺序、边界条件和可重复性验证；作者源码未公开，
不得声明 `official_exact` 或论文数值复现。

公开 Wisconsin benchmark 是工具箱自定义的 concept-shift smoke 协议，不是论文原始
七个实验设置，所有产物必须保持 `paper_claim=false`。

## 2. 方法臂与消融

同一数据切分、同一网络规模和同一随机种子下，至少区分：

- `trPU`：源域 PU 训练；
- `tePU`：目标域 PU 训练；
- `fine_tune`：源域训练后目标域微调；
- `mmd`：五核 RBF-MMD 对照；
- `dynamic`：动态联合漂移求解器。

配置必须显式记录训练模式、权重修正、beta、epoch、设备、网络规模和数据集版本。

## 3. 评价指标

主结果在目标测试集上报告：

- accuracy、balanced accuracy；
- ROC-AUC、AP；
- 多 seed 均值、样本标准差和 Student-t 95% CI；
- 目标/源域类先验及其差异；
- 训练耗时、失败率和非有限值/不收敛状态。

若研究问题涉及权重估计，还必须报告权重范围、边界触及率和有效样本量；这些是训练
稳定性与覆盖诊断，不替代目标分类指标。

所有指标均按目标测试集计算。源域训练或验证结果不得冒充目标域性能。使用隐藏真值
进行最终评测时，不得将其用于超参数选择或早停。

## 4. 统计与数据审计

- 固定并公开 seed 集合、源域/目标 PU/目标测试三方切分；三者不得有样本 ID 重叠；
- 逐 trial 保存方法、seed、配置、样本数、耗时和失败原因；
- 汇总报告均值、标准差、95% CI 和配对/集合重叠审计；
- 记录数据集来源、版本、标签映射和 `resolved_config`；
- 数据、网络或训练协议偏离论文时，在报告中列出差距，不得用“paper protocol”掩盖。

## 5. 固定产物

benchmark 至少生成：

```text
config.json
trials.csv
summary.json
summary.md
split_audit
```

结果目录、配置哈希、代码 commit、依赖版本和运行设备必须可审计。模型权重和缓存可以
保留在仓库外，但不能省略 provenance。

## 6. 验收标准

1. 公式金标准、训练顺序和权重范围测试通过；
2. 固定种子下训练和预测可重复；
3. 三方数据切分无重叠；
4. 每个方法臂都有完整 trial 或明确失败记录；
5. 目标指标、置信区间和实现差距均写入汇总报告；
6. `paper_claim=false` 在配置和结果报告中一致出现。

改变方法臂、主指标、数据切分规则、统计方法或声明等级时，必须提升
`protocol_version`。
