# 数据泄露审计门禁设计

```yaml
schema_version: 1
status: design_only
scope: traditional_pu_benchmarks_and_tuning
enforcement: benchmark_preflight_and_runner_gate
```

## 1. 目的

本设计防止传统 PU 基准和调参实验把隐藏真值、测试信息或全数据统计量带入模型训练、交叉验证、
早停或参数选择。它是实验有效性的前置门禁，不改变任何估计器的数学实现。

## 2. 审计层级

### 2.1 标签与元数据隔离

模型训练和 PU-CV 只能接收 `X` 与 `y_pu`（PNU 为 `X` 与三元标签）。`y_true`、测试标签、
propensity、selection probability、数据切分标志只能进入最终审计或 oracle 评估。

trial 表不得保存原始 `y_true`、`y_pu`、测试标签或任何可重建标签的列。

### 2.2 特征列审计

默认阻断以下精确名称或大小写不敏感模式：

```text
y_true, y_pu, label, target, class_prior, prior,
propensity, selection_probability, split, fold, is_train, is_test
```

项目可在配置中增加特定字段。命中时报告列名、匹配规则和阻断原因；不能只删除列后继续运行，
除非实验配置明确声明这是一个新的、可审计的数据版本。

### 2.3 切分与重复样本审计

- 训练、验证、测试索引不得重叠；
- 相同实体 ID 或重复样本不得跨越切分；
- 官方数据集必须记录 split 索引哈希和组重叠计数；
- synthetic benchmark 必须记录 seed、生成参数和场景哈希。

### 2.4 预处理审计

插补、标准化、降维、特征选择、目标编码和任何学习统计量的变换，必须在每个训练折内拟合，
再作用于验证折/测试折。全数据预先 `fit` 的变换视为泄露，除非它被证明是与数据无关的固定变换。

## 3. 强制执行点

审计应同时提供：

1. 独立 preflight：实验开始前生成 `data_leakage_audit.json`；
2. runner gate：preflight 失败时阻断 benchmark，不生成可晋级的结果；
3. 代码路径约束：`y_true` 不得传入 estimator 或 tuning scorer；
4. 结果审计：manifest 记录审计状态、规则版本、命中项和输入哈希。

## 4. 状态与处理

| 状态 | 含义 | 处理 |
|---|---|---|
| `pass` | 所有硬性检查通过 | 允许运行 |
| `blocked` | 发现标签/切分/预处理泄露 | 阻断运行 |
| `audit_only` | 只能完成只读审计，缺少必要证明 | 不得用于性能结论 |
| `not_applicable` | 规则对当前数据类型不适用 | 必须记录理由 |

任何阻断都必须保存可复现的原因，不得静默删除问题字段或降级为成功。

## 5. 必须的负向测试

门禁测试必须覆盖：

1. 特征直接复制 `y_true`；
2. 特征包含 propensity 或 selection probability；
3. 重复样本跨 train/test；
4. 预处理在全数据上提前拟合；
5. `y_true` 被错误传入 estimator 或 tuning scorer；
6. trial 表尝试写入任意 `y_*` 原始标签列。

这些测试必须被阻断或明确拒绝；不能只验证正常数据能通过。测试对象是审计函数与 runner gate
接口本身；合成 benchmark 无折级结构，不要求以完整实验管线为测试对象。

## 6. 已有能力与当前缺口

当前项目已经具备标签用途分离、PU-CV、oracle 指标标记、官方 split 重叠审计和 trial 标签隔离
测试。本设计补充统一的特征级黑名单、重复样本门禁、预处理审计和 benchmark 启动阻断规则。

本文件是设计文档，不代表上述独立脚本和 runner gate 已经实现。实现完成前，实验报告必须注明
`audit_design_only`，不得声称已通过完整数据泄露门禁。

## 7. 实施顺序

当前 traditional_pu benchmark 为纯合成流程：无 CSV 输入、无显式切分、无预处理管线。按 YAGNI
分两阶段实施，不与尚未存在的数据流绑定：

- **阶段 A（立即可做，覆盖合成 benchmark 全流程）**：§3.3 的 `y_true` 路径约束（estimator 与
  tuning scorer 不得接收 `y_true`）、§2.1 的 trial 列写入门禁（trials.csv 禁止任何 `y_*` 原始
  标签列）、§3.4 的 manifest 审计状态记录；特征黑名单（§2.2）与重复样本检测（§2.3）先实现
  为独立审计函数，配合 §5 负向测试做单元测试。
- **阶段 B（官方数据集线进场时，契约 §7 第 6 条）**：§2.3 的切分索引/实体重复检查与 §2.4 的
  折内预处理审计挂载到真实数据流；§2.2 特征黑名单对 CSV 列名生效。

