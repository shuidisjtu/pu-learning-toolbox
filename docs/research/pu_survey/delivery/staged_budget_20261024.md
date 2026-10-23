# GEN-PU / PULDA 阶段预算接线（2026-10-24）

承接 [profile 身份分离](profile_identity_20261024.md)，本批把两种多阶段方法从
“统一读取 epochs”改为显式阶段预算。仅供 `survey-v3.0-draft` 工程验证，
**不是正式 v3、预算批准、13 方法全部接通或论文数值复现**。

## 本批完成

`survey_staged_budget.py` 提供 `survey-staged-budget-1`，由草案命名 budget 的
`staged_budget` 对象承载。必须显式填写算法、完整预算参数、快照轮数、逐组件字节
上界和可选阶段；不补默认预算、不接受伪造的批准字段，也不与旧 epochs 混填。

| 方法 | 显式预算参数 | 实际快照 / 可选阶段 | 优化器更新成本 |
|---|---|---|---|
| GEN-PU | GAN `max_epochs`、`classifier_epochs`、batch、hidden/latent 宽度、学习率 | 仅全部 synthetic_pn 轮；不是 GAN 轮 | 每 GAN batch 为 3D+2G 五次更新，PN 每 batch 一次 |
| PULDA | warmup/PU 轮数、P/U batch、宽度/深度、两段学习率与 weight decay | warmup + pu_mixup；零 warmup 不生成幽灵快照 | 每段每 U batch 一次更新 |

模型构造仍使用原生注册类；profile 锁定参数，unit 与 profile 的 budget 指针必须一致。
这两种草案方法不再允许套用旧单 epochs budget。原生二维 MLP 路径使用独立的
`genpu_native` / `pulda_native` model family 与对应 backbone/spec，不冒充共享 MLP。
本批明确拒绝借此接入 native CNN 或 feature-adapter；已有独立图像工程入口保留。

pilot 容量估算与 runner 预检均使用真实快照数及显式字节上界，按候选与两次尝试计峰值。
训练后、PA/OA 选择前，核对全部快照的阶段/阶段轮数/累计更新步数、实际文件体积，
GEN-PU 还核对六网络成本。缺快照、错误阶段、超界或少付 GAN 成本都拒收，失败记录
按 runner 原有重试规则保留。成功候选写 `staged_budget_receipt`，绑定预算摘要、
实际成本、快照体积和 `formal_budget_approved=false`。

旧 v1/v2 budget、默认参数、冻结协议/台账/对照矩阵与历史结果未改。
当前 planning JSON 只刷新源码摘要并纳入新预算模块，仍为22算法/23 profile/184阻断请求。

## 源码依据与验证边界

依据本仓库原生实现 `estimators/deep/gen_pu.py` 的
`checkpoint_epoch_count`、`stage_optimizer_steps_`、GAN/PN 训练循环，以及
`estimators/risk/pulda.py` 的 `checkpoint_epoch_count`、`_train_epoch`、
`_record_epoch`；不是重新解释论文或改变已锁定目标。

GEN-PU 的 OS 步数按原 U，TS 按风险池 U∪P；PULDA 的两种视图均按原 U 分批，
TS 在损失内部校准期望，不能把 union 长度错当成额外训练 batch。
校验器重用 `build_training_view` 的角色定义，不从 clean_val/test 推断成本。
保留全部原生可部署阶段供外部 PA/OA 选择，未实现任意阶段子集裁剪。

专项使用临时合成四角色数据：两方法 OS/TS 实际 CLI、PA/OA manifest、磁盘预检，
两个 seed 的原生 fit 与记录 fit 权重/分数逐位一致，错误参数/身份/指针/架构与
实际轨迹拒绝，零 warmup 边界。它们是工程验证，不是正式数据训练、GPU 资源
上界或论文精度复现；没有运行 PN-CNN 5 seed。

专项 **29 passed / 0 skipped**，收据
`/tmp/survey-staged-budget-final-target-20261024.xml`；与身份链路合跑的65项也通过。
完整快层 **5,669 passed / 59 skipped / 0 failures/errors**，共5,728项、378.610秒，
收据 `/tmp/survey-staged-budget-full-fast-20261024.xml`。
该快层收集后补入的零 warmup 边界已在上述29项专项独立通过，未加进5,728的计数。
既有环境/成本估算/收敛告警保留，条件跳过不算 GPU 通过。

静态格式（另含暂存交付独立快照的完整格式门禁）、测试/注释质量、文档/API、
元数据/数学渲染、skills、目录/层依赖、baseline、候选来源契约、旧 draft registry
和当前 planning 来源检查通过。离线 wheel/sdist 构建、模块入包及仓库外安装后的
预算 schema 校验通过；发行包不含根目录用户上传或数据/模型 payload。
文档中原有的 `data/split_artifacts_index.json` 继续随 sdist 保留，不与同名根上传混淆。
未修改或提交用户原有 `benchmarks/deep_pu/official_data.py`，未声称远程 CI 已通过。
本批按用户要求收尾，不继续扩展其他方法族。

## 后续未完成

- 其他方法的多阶段/episode/CPU 树/OOF 预算与适配；不能把本批说成通用完成。
- 两方法正式预算/候选/seed/阶段资格登记及 native CNN 的版本化架构接线。
- 五角色 support 消费、新 registry/comparison 和正式 v3 单元、相应资源及数值验收。
- 待合作者的协议/方法学审批仍保留，PN-CNN 五 seed 继续暂缓。
