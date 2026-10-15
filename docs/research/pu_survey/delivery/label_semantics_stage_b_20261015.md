# Issue #41 阶段B：公共入口标签语义门禁

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
