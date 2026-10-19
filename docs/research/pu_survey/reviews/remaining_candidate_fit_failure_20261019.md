# 另五候选的失败拟合保护（2026-10-19）

补VPU、神经LBE、PAN、CVIR、RP；承接[前批五扩展候选](extended_candidate_fit_failure_20261019.md)。
各模块只新增共享 `invalidate_failed_fit` import与fit装饰器，原训练方法体不变。

原实现的参数绑定异常发生在方法体之前，不能撤销前次成功状态；VPU/CVIR的
普通异常保护还漏掉用户取消。五方法移除guard的受控复现36项中24失败、12通过；
移除包装后的原方法即本批整改前方法体，不把早期错误测试配置的失败计作缺陷证据。

包装在参数绑定前失效，任何异常/取消后再次撤销公开拟合标志并原样抛出，不重试。
不回滚旧/部分权重、历史、诊断或外部checkpoint，不把残留属性当成功制品；
公开predict/decision_function/score_samples及已实现的predict_proba均拒绝失败对象。
重新完成成功fit才能恢复，目标/优化器/采样、校准/先验、默认预算及回调可见性不变。
CVIR仍需显式α_U、走OS adaptation；VPU的D22、神经LBE/PAN/RP的OS口径不改。

## 验证

扩展生命周期文件现有9方法90项，另RP原专项与新增6项共14项，合计104项
通过、零跳过/失败/错误，4.734秒；JUnit `/tmp/remaining-fit-final.xml`。
移除保护的36项复现为 `/tmp/remaining-fit-unguarded.xml`。
覆盖未知真值/支持标签/test参数、缺失必填、RuntimeError/TypeError/KeyboardInterrupt/
SystemExit、pickle失败状态、成功重训、公开签名、clone、MLP/CNN权重与优化步数等。
原五扩展方法回归保留，PULNS/TEDn的单元测试边界仍按前批说明。

RP另验收OOF逐fold行、保留样本、最终重权及预测逐bit不变；失败后成功重训及pickle
回放一致。RP测试文件不依赖Torch；另启动真正禁止torch及其子模块导入的Python进程，
成功fit/预测、绑定失败后的拒绝预测均通过，且torch不在sys.modules。
不因共享guard或测试复用给RP新增深度学习依赖。

台账新增各方法工程引用，保留来源/校准与全部历史测量，不改变冻结协议/结果、
正式预算/准入、负责人签名，不运行真实数据/GPU或作者代码，不关闭issue。

## P3MIX 接入补记

2026-10-19 用户提供完整 P3MIX 论文并完成 classifier 接入后，将 `P3MixClassifier`
纳入 `check_candidate_fit_contracts.py` 的 AST/source binding。该方法原已使用共享
`invalidate_failed_fit` guard；专项单测现明确覆盖成功 fit 后发生训练回调中断会拒绝旧模型，
以及未知 fit 参数在函数体绑定前失败同样使对象保持未拟合。该门禁仅绑定声明与源码身份，
不改变 P3MIX 的目标、OS 校准状态、预算或未准入状态。历史 15 候选测试记录保留为旧截点，
当前 16 候选草稿和测试作为最新工程证据。

后续全仓快层发现PAN/RP来源记录的实时implementation摘要仍绑定包装前代码，
两项来源契约因此失败。保留原SHA和适用提交于implementation_history，再刷新
实时绑定字段并补本页引用；未修改上游源码摘要、历史测量、原来源结论或签署。
15候选工程草稿的审阅文档摘要随本补记刷新，不删除失败记录或放宽来源测试。
