# 候选图像实际训练收据（技术入口，非正式配方）

补齐 #119/#120 的技术审计接线，不替代正式runner、四角色验证、配方冻结或准入。
仅构造时的 `image_setup` 不证明后续fit实际使用同一输入/模板/配置；新增
`pu_toolbox.experiment.candidate_image.fit_candidate_image_classifier` 负责绑定后再fit，
返回 `(estimator, training_receipt)`。没有注册到冻结CLI或旧645-run矩阵。

## 使用

```python
from pu_toolbox.experiment.candidate_image import (
    build_candidate_image_classifier, fit_candidate_image_classifier,
)

model, image_setup = build_candidate_image_classifier(
    "pulda", spec, seed=0, class_prior=pi_population, device="cpu"
)
model, training_receipt = fit_candidate_image_classifier(
    model, image_setup, X_train_scaled, s_train, os_or_ts="ts"
)
```

`spec` 和 `X_train_scaled` 必须来自同一次train-only预处理，输入为原样float32 NCHW；
`s_train` 是观测PU标签，不是隐藏PN真值。可显式传epoch_callback，接口不接收
validation/test输入。GradPU改method且不传class_prior，无BN/private分层保持。
技术短测若用set_params缩短预算，收据会保存实际入口参数，不冒充默认120/200轮配方。

## 训练前检查

- 方法类型、technical模式、native_cnn路径、比较组、未准入/未源码复现标记一致。
- seed与原构造一致，encoder模板state hash一致；不得悄悄替换骨干或先改模板权重。
- augmentation描述与构造时一致；旧记录缺描述时要求重建，不自动猜测。
- PULDA总体prior未被替换；GradPU自定义分数模型需另立描述，不在此入口偷换。
- 输入shape/dtype、像素hash与train-only预处理绑定一致；PU标签执行原有公共验证。
- 实际构造参数在fit前转成严格JSON，非有限/不支持的编码不能进入训练收据。

输入或配置失配在任何fit之前拒绝，避免“先跑完再发现描述错了”。
这是一致性检查，不是对不可信调用者的密码学来源证明。

## 收据内容与限制

记录训练输入/观测标签摘要、fit入口参数、真实训练视图/校准标志、完成epoch、
optimizer更新数、实际设备及encoder可训练状态、GradPU算法训练事实和两份源码SHA。
拟合前后再次核对输入，发现训练过程中数据发生变化不生成有效收据。
原callback/fit异常原样传播，不重试；失败收据不会回滚已经训练的模型或callback制品。

本入口会扫描CPU训练输入两次做hash，可能需要连续化临时副本，属于显式审计成本，
不是磁盘流式或常量内存训练承诺。不会保存checkpoint或创建目录，callback持久化由调用者负责。
收据中的完成epoch不等于已持久化快照数量；实际存储/内存profile仍未核验。

数值0/1不能证明调用者传来的标签确实是PU，因此标签预算、四角色隔离、环境锁、
协议/registry绑定、选模、checkpoint持久化及资源验收均未被本收据证明。
TS风险角色并集亦不自动证明恢复总体混合；相应字段仍false。
`ready_for_formal_run=false`、正式准入仍false；应把收据作为完整运行manifest的一个
独立字段保存，不覆盖正式协议身份、不用于主榜混排。

## 本轮最终验收与交接

2026-10-18 冻结本轮代码后的完整快层回归：**4666 passed / 58 skipped /
0 failures / 0 errors**，共4724项，313.528秒。
命令为 `python -m pytest tests/ -m 'not slow and not e2e'`，
本机JUnit为 `/tmp/algorithm-followup-release-fast.xml`，临时报告不提交仓库。
跳过、slow及e2e不计作通过，也不替代CUDA或正式数据实验。
本入口与关联图像、模式恢复、GradPU配方专项另有72项通过、无跳过。

格式、测试质量、文档链接、API、项目元数据、数学渲染、技能同步、baseline配置、
注释质量、目录结构、分层依赖、配方registry，以及P3准入证据、公开对照、选择预算
共15项静态/证据一致性检查均通过。这是本地验证，不冒充GitHub Actions运行结论。

本轮其他交付见[GradPU #121](gradpu_training_recipe_review_20261018.md)、
[Split-PU #122](splitpu_122_characterization_20261018.md)和
[推理模式恢复](pulda_gradpu_inference_modes_20261018.md)。
待合作者确认的配方/风险口径仍保持未签署，不修改冻结矩阵或历史结果。
P3MIX全文由用户在本轮提交推送后补充，下轮再进行来源复核和完整集成。
