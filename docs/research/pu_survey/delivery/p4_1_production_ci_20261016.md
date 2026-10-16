# P4.1-A：正式注册表 CI 激活接线

本轮基线为 main `397f866`。补齐正式产物的条件门禁；不生成正式注册表，不批准配方，
不修改冻结协议、比较矩阵、历史结果或任何人工复核状态。

## 行为

- draft schema 检查继续无条件执行，draft 丢失或不合法仍失败。
- 新 CI 步骤只在 `pu_toolbox/experiment/survey_recipe_registry_v1.json` 存在时执行。
  当前正式文件尚未物化，步骤应为 **skipped**，不是“正式配方通过”。
- 执行时使用 `--require-present --require-locked`：文件不存在、schema/协议/snapshot 不符、
  顶层未 locked、任一非 excluded profile 未 locked，均失败。
- 允许 schema 已合法排除的 profile 留为 excluded，不把排除项误说成可运行。
  snapshot 和原有协议/候选/预算检查均保留；仅改 lifecycle 不能绕过它们。
- `--require-locked` 单独使用也要求文件存在。普通缺省调用的历史 absent 行为不变。
- 检查只核声明及一致性，不认证签名或推断批准；不能靠改 lifecycle 代替 P4.1-B。

## 正式冻结时的剩余动作

1. HENG958 复核 runner/CLI/pilot → candidate/selection → v2 白名单审计/汇总的接口和方法参数映射。
2. 满足 P4.1-B 前置后提交真实批准的 registry 与 resolved snapshots。
3. **同次冻结提交把 CI 此步改为无条件 `--require-present --require-locked`**，
   使后续删除正式文件也失败；当前的存在性条件只用于冻结前的阶段接线。
4. 检查正式文件的真实 CI 结论并记录批准范围、摘要及代码/环境身份。

因此本轮完成的是“可激活的正式 CI 接线＋声明锁门禁”，不是 P4.1-A 全部验收或 P4.1-B 冻结。
对应运行/消费范围见[训练前绑定](p4_1_runtime_binding_20261014.md)与
[候选池身份](p4_1_pool_binding_20261014.md)。

专项回归使用临时 synthetic locked fixtures，不把它们提交成正式注册表。
覆盖非锁定顶层/子 profile、缺文件、损坏 snapshot、重复检查确定性及 CI 条件；
本轮综合测试和质量门禁记录见[推进记录](../reviews/p41_p22_progress_20261016.md)。
