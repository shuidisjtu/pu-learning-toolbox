# 五候选方法的失败重训状态修复（2026-10-18）

2026-10-19 后续另补[五扩展候选保护](extended_candidate_fit_failure_20261019.md)；
本页保留原五方法及原验收计数，不将旧运行改写为扩展后的全范围证据。

范围仅为 PULDA、PUET、GradPU、Robust-PU、Split-PU 的同步 `fit` 生命周期。
这是 P3.1 工程整改，不是新方法准入、论文数值复现或负责人签署。

## 缺陷与修复

在 `be798ce` 的成功拟合后传入不支持的 `sample_weight`，五方法均抛错但仍允许
公开预测；修复前定向回归 **5/5 失败**。PUET 还未在普通参数校验前撤销已拟合状态；
部分神经方法为了 epoch checkpoint 在回调前临时开放推理，回调中断可能遗留该状态。

新增无 torch 依赖的 opt-in `invalidate_failed_fit`，只装饰上述五方法：进入 `fit`
即撤销 `_is_fitted`，任何异常或用户取消后再次撤销，并原样重新抛出同一个异常，
不重试。因包装先于原方法参数绑定，误传 `y_true` 等未知角色也不会留下旧模型可用。
`functools.wraps` 保留公开签名、名称和文档；原方法仍拒绝额外标签，不吞掉未知参数。

成功训练及训练中的 checkpoint 回调可见性仍由各 estimator 原逻辑控制。
不改目标函数、梯度、随机数、默认预算、先验、OS/TS 视图、校准范围、stage 资格或
PA/OA 选模。回调正在执行时的推理能力与旧行为相同；`fit` 失败返回给调用方后，
`predict`、`decision_function`、`score_samples` 及已实现的 `predict_proba` 均拒绝调用。

**不承诺事务回滚。** 为诊断保留的旧权重、树、历史或其他属性可能仍在对象上；
回调外部副作用、已写 checkpoint 不自动删除。不得绕过公开检查直接取 `model_` 推理，
也不得把失败的训练产物视为完整成功运行。此修复不增加并发 fit/predict 安全、
断点续训或任何正式结果发布许可。恢复方式是重新完成一次成功 `fit`。

## 台账与来源绑定

- 五方法台账新增本页引用，原 `run_view` / `calibration_applied` 等不变。
- [工程证据](../data/p3_admission_evidence_20261005.json)登记失败状态及成功路径回归。
- [阶段预算草稿](../data/p3_selection_budget_spec_20261007_draft.json)刷新当前五份实现的
  SHA，并额外绑定共享 guard 的 SHA；离线 AST 检查缺失装饰器时失败。
- 历史来源复核和资源探测的旧 SHA 仍对应其注明的旧提交；本次不改写历史，也不宣称
  已在新源码上重新测量真实 CPU/GPU 峰值。Robust-PU 的历史来源补交仍对应
  `5ae320e` / 合并后的 `f3017cd`，本次只改变调用边界的状态管理。
- 所有正式预算/准入仍 pending，负责人签名仍为空；不改冻结协议/矩阵/645-run。

## 验证范围

回归位于 `tests/unit/estimators/test_candidate_fit_failures.py` 与
`tests/unit/core/test_fit_state.py`。覆盖不支持的权重、非法先验、未知真值参数、
非有限输入、八个阶段的回调抛错/取消、树构造中断、成功/失败状态序列化及失败后重训。
用相同 seed 对照包装方法与原始方法，逐位比较合成数据得分、回调序列及 optimizer
更新计数，同时检查 clone 与签名；这不是真实数据性能或论文级数值复现。

## 验收记录

- 定向测试 **206 passed**（其中五方法生命周期 80 项、共享 guard 8 项）；
  包含四条 CNN 适配路径的成功结果逐位一致及八阶段中断检查。
- 12 项本地静态门禁通过；工程证据、公开对照草稿与阶段预算校验通过，
  Robust-PU 锁定来源的六文件可选复核通过。
- 相对 `be798ce` 的五份实现 AST 差异仅为共享 guard 的 import 与装饰器，
  不改方法体。当前源码/共享依赖 SHA 以阶段预算草稿为准。
- 离线 sdist → wheel 构建成功；在 `/tmp` 仓库外从 wheel 导入五方法，
  合成拟合、pickle 得分回放与失败重训失效均通过。
  使用真正拒绝 torch 导入的 finder 单独检查 PUET，拟合/失败保护均可工作。
  包内容检查未混入上传的压缩包、Excel、raw 数组或模型权重。
- 全量快速测试 `pytest tests/ -m 'not slow and not e2e'`：
  **4409 passed / 58 skipped / 0 failures / 0 errors**，297.180 秒；
  比同步后的 `be798ce` 增加 93 项通过。JUnit：本机 `/tmp/candidate-fit-lifecycle-fast.xml`。
  不将本地验证表述为 GitHub 多平台远程 CI 通过。

未运行正式训练或占用 GPU。工程整改不解除 P3MIX 全文来源前置、Split-PU #122
负责人裁决、独立支持集/额外标签预算、正式配方及资源准入的阻断。

2026-10-19 后续更正：以上 P3MIX 阻断仅对应本页原验收时点。用户随后提供完整论文，现已
完成来源核对和 estimator 接入；最新状态见[原文交叉表](../admission/p3mix_paper_review_20261019.md)
及[执行计划](../survey_execution_plan.md)。本页原五方法测试与冻结结果未被改写。
