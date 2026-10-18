# 五扩展候选失败拟合保护（2026-10-19）

承接原五方法的[生命周期修复](candidate_fit_failure_20261018.md)，范围扩至
GEN-PU、Holistic-PU、LaGAM、PULNS及独立TEDn；不是新算法或正式准入。

## 可复现缺陷与整改

40项原始定向测试在修复前24失败、16通过，零收集错误：
五方法误传y_true/support_labels/test_data或缺少必填参数时，Python尚未进入fit方法体，
因此旧 `_is_fitted=False` 没有执行，前次成功模型仍可公开预测。
LaGAM和TEDn原有 `except Exception` 不覆盖KeyboardInterrupt/SystemExit，
回调取消会保留中途公开的已拟合状态。其余普通训练失败原已有保护，未宣称全部缺失。

五个方法仅新增 `invalidate_failed_fit` 的import及fit装饰器，共享guard实现不改。
参数绑定之前即撤销公开状态，捕获BaseException后再次撤销并原样抛出；不重试。
原fit签名、正常执行/返回值、回调可见性、目标/采样/默认预算与校准均保持。
PULNS/LaGAM仍需独立支持集，额外监督和PA禁入不变；TEDn仍为显式TS/α_U，
没有注册到总体先验、CLI、旧Survey或CVIR默认路径。

不承诺事务回滚：旧/部分权重、history、stop_reason等诊断属性可能仍在对象上；
它们不表示成功状态。回调外部副作用和既存checkpoint不删除，不支持并发fit/predict。
公开predict/decision_function/score_samples及已实现的predict_proba均拒绝失败对象；
恢复需重新完整成功fit，不能绕过检查直接访问model_推理。

## 验证与登记

新增 `test_extended_candidate_fit_failures.py` 共50项，覆盖20种参数绑定失败、
20种普通/取消异常、pickle的失败状态、成功后失败再重训、clone/签名，以及
MLP/注入CNN的包装与原方法对照。训练记录、优化步数、全部权重和预测保持，
回调可见性及更新计数不变；PULNS没有epoch_callback，改用实际训练步骤注入异常。

TEDn生命周期测试显式把独立BBE结果的alpha_U固定为0.2以隔离失败状态：
这不是估计器精度证据或正式变体，不改生产BBE/TEDn；原真实BBE测试另行保留。
支持集独立合成，不使用外部validation/test，不证明真实制品来源/预算。
本批生命周期/共享guard专项138项通过，零跳过/失败/错误，3.670秒；
JUnit `/tmp/extended-fit-final.xml`。全仓最终快层在本轮结束统一记录。

另跑五方法原单元/CNN、Holistic-PU的LZO/初始化、LaGAM虚拟步与真实BBE的TEDn
邻接回归：263通过、8跳过，零失败/错误，7.971秒；
`/tmp/extended-fit-neighbor.xml`。源码AST审计确认五份生产模块只增加guard import
与装饰器，原方法体及共享guard实现完全不变。

台账对应四方法及CVIR下独立TEDn扩展补本引用；历史源摘要/测量与签署不回填。
未执行作者代码、真实数据/GPU、正式实验或数值复现；审批及原始制品阻断保留。
