# Issue #41 阶段B：公共入口标签语义门禁

## 2026-10-16 RV2 验收结论

**RV2 完成，Issue #41 可关闭。** 用户接受队友实现及本次功能复核证据，不再另要求人工签署；这是用户验收决定，不代填 HENG958 签名。下文 10-15 待审表述保留为历史状态。

验收对象：PR #110 HEAD `8b4e7519d5c5dfb8a275de8884169d464a038704`，阶段 B 实现提交 `3e65e37f91a69d641672904620048255e7a59678`。本次在独立导出的 PR 源码副本中运行，确认导入路径指向副本，复用现有环境，不是正式 frozen-lock 环境。

- 专项 **67 passed / 0 failed / 0 skipped**：两个新增语义入口测试模块、能力声明契约、标签校验、模型比较与身份防泄漏。
- 兼容性 **103 passed / 0 failed / 0 skipped**：runner oracle 行为/拒绝路径、epoch checkpoint、Self-PU、调优、CLI sensitivity、推荐器及 Pipeline/模型配置集成。
- 另用 30 行合成三值标签直接调用 PNU fit，预测分数有限；第三方 PU 默认、合法数值相等输入与 Self-PU 辅助验证边界保持。
- 代码核查确认 Pipeline 解析、运行及实际构造/clone 后消费共享守卫；比较、调优和 CLI/UI 复用，敏感性直接 fit 单独检查；推荐、配置导入、Labels 展示与 ADR/API 一致，原 runner/trainer 和身份防泄漏判据未削弱。

复现命令（在该 PR 源码检出中执行）：

```bash
uv run python -m pytest tests/unit/workflows/test_pipeline_label_semantics.py tests/unit/workflows/test_label_semantics_entrypoints.py tests/contract/test_capability_declarations.py tests/test_validation_labels.py tests/unit/model_selection/test_comparison.py tests/benchmarks/test_traditional_pu_leakage_audit.py
uv run python -m pytest tests/unit/experiment/test_runner_oracle.py tests/unit/experiment/test_runner_oracle_guards.py tests/unit/experiment/test_epoch_checkpoints.py tests/unit/estimators/test_self_pu.py tests/unit/model_selection/test_tuning.py tests/unit/cli/test_sensitivity_cmd.py tests/unit/advisor/test_recommender.py tests/integration/test_pipeline.py tests/integration/test_model_configuration.py
```

未重跑全仓测试、未做浏览器交互或正式实验。PR 的 Windows ZIP 测试夹具失败另行处理，不属于标签语义缺陷；本验收不表示整个 PR 的 CI 已通过，不改变历史结果或冻结协议。

## 任务依据与状态

2026-10-15已执行 `git fetch origin`，远程main仍为 `3158a6c`，本地包含该基线。
同时读取开放Issue正文与评论，不能只用Git提交判断合作者是否更新了任务：

- [#40](https://github.com/shuidisjtu/pu-learning-toolbox/issues/40)：P4.1-A接线与P4.1-B冻结分别验收；
  本地已有前者的工程提交，远程仍仅有阶段0，不把工程接线说成配方冻结。
- [#41](https://github.com/shuidisjtu/pu-learning-toolbox/issues/41)：阶段B公共入口收口待补，本次优先实现。
- [#57](https://github.com/shuidisjtu/pu-learning-toolbox/issues/57)：以v3为复核对象；人工签署、PA裁决及原制品限制保持。
- [#88](https://github.com/shuidisjtu/pu-learning-toolbox/issues/88)：成本分类核查延期至Survey后，不改 `robust_pu.training_cost`。
- [#69](https://github.com/shuidisjtu/pu-learning-toolbox/issues/69)：research元数据问题明确低优先，不抢占本次工作。

阶段B的工程/展示/文档已补；RV2仍须HENG958本人复核，**Issue #41不能据此关闭**。
正式协议、冻结比较矩阵、方法台账采样裁决和历史结果均未改。

## 实际消费点

| 入口 | 检查与行为 |
|---|---|
| PUPipeline显式注册名/实例 | 构造预检要求主输入PU；运行入口重查可变实例，先于先验估计 |
| 每折/最终refit | 实际clone/构造后再次检查，阻止构造参数或clone改变声明 |
| auto | 推荐器过滤非PU；实例化候选另查真实类声明，跳过并记录不兼容原因 |
| PUModelComparator | 复用构造预检，PNU候选在任何分类器训练前拒绝 |
| PUTuner | 保留trial失败隔离；不兼容trial不fit，全部失败仍返回原PipelineError契约 |
| CLI run / UI正常、比较、调优 | 经上述公共工作流消费，不因绕过ExperimentRunner失去守卫 |
| CLI sensitivity | 直接fit前调用同一core守卫，无新增训练或自动语义推断 |
| CLI list-methods | 新增Labels列，保留PNU及别名的实际声明；Auto-inst只指构造能力 |
| UI | catalog登记语义、非PU不进入PU上传选项；提示0是未标记，真值仅用于评价 |
| UI配置导入 | 按实际注册类声明预检，拒绝PN/PNU后不改widget state，手动和比较配置均覆盖 |

共享函数仍为 `validate_label_semantics`，只改善传入类时的错误名称显示。
不复制或删除runner/trainer守卫，不把身份防泄漏与语义检查合并。
PNU直接三值fit、第三方PU默认和Self-PU辅助验证语义不变。

## 验证与边界

先执行新pipeline回归，旧行为10项预期失败、1项兼容性通过，证明测试能暴露实际缺口。
新增用例用禁止fit/先验估计的spy证明提前失败，而非只检查错误字符串；覆盖PN/PNU、
实例修改、构造后修改、clone改声明、auto候选、比较/调优、CLI/UI执行及配置导入、
语义列与推荐器过滤；合法c=1式PU/真值数值相等不据此拒绝。
既有PNU比较测试改为断言更早的语义拒绝；无关优化器错误仍用合法PU方法验证失败隔离，
没有删掉该行为测试或增加质量豁免。

最终全量 `not slow and not e2e` 回归3862项通过、59项跳过，3921项、271.025秒，
无失败或错误；新增25个用例均在此范围内。CLI/UI/调参/推荐/core等首轮定向225项通过，
补充clone、配置导入及兼容性后的定向56项通过。
格式、测试质量、文档链接、API覆盖、项目元数据、公式、技能同步、基线默认配置、
注释、目录结构/依赖层及八profile draft检查通过，无新增门禁豁免。
wheel/sdist构建通过，从构建wheel直接导入后验证语义拒绝、合法PU构造及CLI Labels输出；
sdist不含raw/outgoing、Excel或RAR。工程测试不代替HENG958的RV2本人结论。
ADR见[0020](../../../adr/0020-main-input-label-semantics.md)，
用户契约见[API](../../../user/reference/api.md)、[CLI](../../../user/howto/cli.md)、
[UI](../../../user/howto/ui.md)及[调优](../../../user/howto/model_tuning.md)。
