# Self-PU 生命周期、参数与评分状态补核（2026-10-20）

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

## 非有限参数导致静默 NaN 的整改

进一步复现：`student_loss_weight=NaN` 可通过原有 `<0` 检查，完成拟合并将整个概率矩阵
污染为 NaN，同时保留 `_is_fitted=True`。这类输入不是合法 recipe；错误必须在训练开始前暴露。

`_validate_parameters` 现先校验所有十三项实数参数的标量类型与有限性，拒绝 bool、字符串、
数组、复数、NaN/Inf 和不能表示为有限浮点数的超大整数，再执行原有区间限制。三个正整数
预算和四个非负整数阶段位置接受 Python/NumPy 整数，拒绝 bool 和浮点预算；阶段顺序仍按
原条件约束。`require_validation` 只接受布尔值，种子为 None 或 `[0,2**32-1]` 中的整数。
全部校验在 torch seed、NumPy generator、张量/模型构造和 optimizer 之前完成。

`tests/unit/estimators/test_self_pu_parameters.py` 用异常前后 RNG 状态及禁止模型构造的探针
验证早失败；另验证 NaN 系数重训后的旧概率不可用、修正参数后可重训、Python/NumPy
标量成功路径一致，以及零权重、比例闭端点、合法 EMA 与种子边界仍可训练。该修复不改变
默认数值、损失分解、TS 角色、teacher 选择或正式 recipe 候选；有限参数本身也不证明收敛。

参数/lifecycle/TS golden/视图不变量与上述 support/checkpoint/台账关联专项共 439 项通过、
4 项 CUDA 测试跳过，0 失败或错误（7.956 秒）；JUnit 在本机
`/tmp/selfpu-parameter-integration-20261020.xml`。保留既有 warning，该轮未重跑全仓测试。

## 台账和后续边界

台账新增 `fit_failure_review_ref` / `inference_state_review_ref` 指向本记录，方法卡与执行计划同步。
参数加固另以 `parameter_validation_review_ref` 指向本记录，公开 API 说明同步参数类型与范围。
通用台账契约核对已声明的失败拟合、推理状态和参数校验记录均解析为仓库内实际文件；
来源/registry 契约另行复跑 29 项通过，避免把缺失或越界引用当作已交付工程证据。
候选专用的 16 方法 AST 草稿保持其原有范围；Self-PU 的生命周期由上述运行回归独立覆盖。
额外 clean support 的真实来源、标签预算、角色身份及正式准入仍须按原协议处理；Pilot
仍是无 meta 消融。原生 CNN 接入仍按 #38 的既有范围暂停，本修复不改变架构能力声明。
