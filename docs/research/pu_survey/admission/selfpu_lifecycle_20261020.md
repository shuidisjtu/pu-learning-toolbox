# Self-PU 生命周期与评分状态补核（2026-10-20）

范围：P3.1/P3.2 的独立工程收尾，使用本地合成数组验证 toolbox 的公开状态和评分副作用。
来源论文与锁定作者代码的差异仍以[原行为审计](selfpu_source_behavior_review_20261019.md)为准。
本记录没有读取合作者数据、运行正式 benchmark、修改冻结 Pilot，或形成 RV5/RV6 签署。

## 复现的问题与修复

| 问题 | 原行为 | 当前行为 |
|---|---|---|
| 未拟合概率预测 | `predict_proba` 直接访问尚不存在的 teacher，抛 `AttributeError` | 所有公开评分路径均检查拟合状态，未拟合时抛 `NotFittedError` |
| 失败/取消重训 | 非法参数、未知 fit 参数或训练中断后，旧 fitted 标志和旧 teacher 概率仍可能可用 | 复用 `invalidate_failed_fit`，在参数绑定前及异常/取消后撤销公开拟合状态，原异常原样传播 |
| 评分状态泄漏 | 对选中 teacher 执行递归 `eval()` 后，不恢复调用方各子模块的原始模式 | 按子模块保存/恢复 training 标志；正常前向、前向异常及部分 eval 转换异常均恢复 |
| PU validation helper | `_pu_validation_risk` 同样将模型永久切至 eval | 保留同一风险计算，在退出时恢复原有逐模块模式 |
| 空输入 | 仍访问 teacher，部分 backbone 无法处理空 batch | 校验样本形状/有限性后返回空分数或 `(0,2)` 概率，不访问 teacher |

保护范围是公开 fitted 状态与评分模式，不回滚模型参数、history、optimizer、callback 外部写入
或已经保存的 checkpoint。失败对象仍可保留供诊断的属性，但不能通过公开评分和
`get_training_checkpoint` 暴露旧的成功身份。成功重训仍按既有逻辑重新完成模型。

## 成功路径的回归范围

新增 `tests/unit/estimators/test_self_pu_lifecycle.py` 覆盖：

- 五类失败输入、四类回调异常/取消、旧概率和 checkpoint 失效、后续成功重训；
- OS/TS × `untrusted_u`/`paper_population` × clean/PU/ablation 选择路径，与未包装 fit
  比较概率、history、teacher 选择以及全部两个 student/两个 teacher 的参数和 buffers；
- fit 签名、sklearn clone、pickle 回放；
- 公开概率与 PU validation helper 的 BN buffers、Dropout/RNG 和混合子模块模式不变；
- 两种评分路径的 eval/forward 异常，以及空输入/非法输入的零 teacher 访问。

已有 Self-PU TS golden 测试继续对照改造前常量，验证风险角色与默认训练轨迹；公共失败保护
没有添加新的 loss、模型选项或取代这些方法学测试。拟合时 teacher 原本均处于 eval，故
PU validation helper 的模式保护保留原训练行为。

本轮 estimator/TS golden/视图不变量、clean-support adapter/CLI、checkpoint 及台账来源/registry
专项共 212 项通过、4 项 CUDA 测试跳过，0 失败或错误（7.342 秒）；JUnit 在本机
`/tmp/selfpu-lifecycle-integration-20261020.xml`。这不是全仓重跑、正式数据或 GPU 资源验收。
既有消融与非正式 checkpoint 容量警告仍保留。

## 台账和后续边界

台账新增 `fit_failure_review_ref` / `inference_state_review_ref` 指向本记录，方法卡与执行计划同步。
候选专用的 16 方法 AST 草稿保持其原有范围；Self-PU 的生命周期由上述运行回归独立覆盖。
额外 clean support 的真实来源、标签预算、角色身份及正式准入仍须按原协议处理；Pilot
仍是无 meta 消融。原生 CNN 接入仍按 #38 的既有范围暂停，本修复不改变架构能力声明。
