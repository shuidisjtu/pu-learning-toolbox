# P2.0b 交付：标签语义门禁（阶段 A）

状态：2026-09-18 完成工程实现与回归；**HENG958 独立复核/签署待办**。
本交付不解除 `survey_protocol_v1.json` 的 `P2.0b_label_semantics_acceptance` 阻断项，
也不放行正式 P2.1 pilot。范围只含[标签语义方案](../../dev/label_semantics_plan.md)的 P1+P2；
pipeline/comparison 入口、CLI/UI 展示及 ADR 收口属阶段 B。

## 已实现

1. `BasePUClassifier.label_semantics` 默认 `"pu"`，注册的 17 个分类器全部在自身类属性显式声明；
   `pnu` 为 `"pnu"`，其余为 `"pu"`。`AlgorithmMetadata` 限定三值并由注册表同步，契约测试
   锁定“注册方法不得仅继承默认”。`self_pu` 只声明主 `fit` 输入为 PU，辅助 `validation_data`
   不借此获得 clean 视图资格。
2. `validate_label_semantics` 校验估计器主输入语义。runner 在生成视图、验证其真实性并检查
   trainer 之后、训练之前比较：PU 视图须 `"pu"`，clean 视图须 `"pn"`；每个克隆候选还会
   单独检查；候选参数改变声明时，该候选按现有重试/失败契约排除并写入 manifest。
   第三方估计器无声明时按 `"pu"` 处理，clean 视图因此 fail-closed。
3. 普通 Survey 脚本 `OracleMLP` 与版本化 pilot `PilotOracleMLP` 均显式声明 `"pn"`。
   测试覆盖 clean+SupervisedTrainer+nnPU、PU 视图+PN 估计器、未声明 sklearn 监督模型，
   均在训练前拒绝；正常 oracle 和 SCAR 路径保持通过。

标签数值本身不能证明语义：PU 的 `0` 表示未标记，PN 的 `0` 是真实负类。门禁依赖声明
真实且 runner 被调用，不能防止故意误声明，也不覆盖阶段 B 的其他入口。

## 复核交接

- 核对 17 个注册类的声明与真实主 `fit` 输入；尤其 `pnu` 三值和 `self_pu` 辅助验证标签。
- 复跑 [runner 错配回归](../../../tests/unit/experiment/test_runner_oracle.py)、
  [注册表声明契约](../../../tests/contract/test_capability_declarations.py) 和普通脚本 oracle。
- 确认 clean 视图上的第三方监督估计器必须显式加 `label_semantics="pn"` 的兼容性取舍。
- 复核通过后再移除 P2.0b 阻断项；P2.0c、R9、环境与其他 P2.1 前置仍各自阻断。
