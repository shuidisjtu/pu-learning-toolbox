# 标签语义契约：能力声明、检查点与实施计划

标签语义检查防止数值相同、含义不同的标签被送入错误的训练目标。设计依据见
[ADR-0020](../adr/0020-main-input-label-semantics.md) 和[实验层设计](experiment_layer.md)。

## 1. 问题

PU 与监督 PN 都可以用 `{0, 1}` 编码，但 `0` 分别表示未标记样本和真实负类。
取值校验无法区分两者：即使 `y_pu` 和 `y_true` 逐元素相同（例如 SCAR 标记率为 1），
也不能凭此推断标签来源。PNU 估计器另有三值输入契约，不能把三值标签直接交给二值 PU 方法。

2026-09-11 的修复前复现是：`CleanLabelGenerator`、`SupervisedTrainer` 和 nnPU
能被手工组合成一个貌似合法的 PN oracle 运行。生成器与 trainer 的视图约定一致，但 nnPU
把真实负例视作未标记样本，导致优化目标错配。检查生成器和 trainer 仍有必要，但不足以判断
估计器会怎样解释标签。

## 2. 设计依据

标签含义应由创建数据视图和消费标签的两方显式声明，而不是从数值、参数名或 batch 位置反推。
PU-Bench 的固定源码快照（`2d95a19`）提供了反例：标签角色主要靠 batch 位置分派，仅存在但
未被训练路径消费的 `label_scheme` 不能拦截误用。工具箱因此在实际训练入口读取声明，并用
错语义候选的行为测试验证检查发生在拟合之前。

## 3. 设计决策

1. **声明主输入。** `label_semantics` 是单值：`"pu"`、`"pn"` 或 `"pnu"`。
   `BasePUClassifier` 默认是 `"pu"`；内置注册方法显式声明并同步到 Registry。
   Self-PU 的主输入是 PU，其 `validation_data` 使用干净标签，不改变主输入声明。
2. **在知道预期语义的边界检查。** `core/validation.py` 的 `validate_label_semantics`
   校验声明是否合法、是否与入口预期相符；它不从 `y` 的编码推断含义。
3. **独立检查 trainer 与 estimator。** `Trainer.trains_on_real_labels` 描述 trainer
   是否消费真实标签，不能替代估计器的目标声明。`ExperimentRunner` 校验生成视图与 trainer，
   再以 clean 视图要求 `"pn"`、PU 视图要求 `"pu"`，检查模型及实际构造的候选。
   架构能力另行检查。四角色 Runner 不提供 PNU 训练视图；PNU 估计器可按自身契约直接调用。
4. **区分发现与执行。** `list-methods` 显示 Registry 中的标签语义（含别名），
   不代表方法满足某次运行的输入约束。独立 `sensitivity` 命令会在拟合前检查选用的估计器
   是否声明 `"pu"`。没有共同训练入口替所有直接 `fit` 猜测调用方的意图。
5. **不把语义检查当成防泄漏。** `guard_fit_labels` 检查对象身份和共享内存；
   `validate_label_semantics` 检查声明匹配。两者都不能验证调用方是否如实标注了标签来源。

## 4. 实施阶段与验收

| 阶段 | 当前保留的验证点 |
|---|---|
| P1：声明 | 类属性取值合法；内置方法显式声明，Registry 与实际能力一致。 |
| P2：实验 | clean 视图配 PU 估计器、PU 视图配 PN 估计器均在拟合前拒绝；构造后的候选接受同样的检查。 |
| P3：独立 CLI | `list-methods` 展示规范名及别名的实际声明；`sensitivity` 拒绝 PN、PNU 和无效声明。 |
| P4：文档 | 接入模板和 API 说明区分主标签、辅助监督、样本来源和验证角色。 |

P1/P2 是 Issue #41 阶段 A 的交付；2026-10-16 完成既有设计的实施复核。
相关测试见 `tests/contract/test_capability_declarations.py`、实验层测试和
`tests/unit/cli/test_label_semantics_entrypoints.py`。

## 5. 接入原则

增加训练入口时，先说明它从何处获得标签角色，再决定能否给共享校验器传入明确的预期语义。
调用方直接使用 estimator 时，必须主动选择匹配的数据语义并遵循方法的取值、先验和辅助监督契约。
不要为绕过检查改写既有方法的声明，也不要用真值与 PU 标签的数值比较替代来源隔离。
