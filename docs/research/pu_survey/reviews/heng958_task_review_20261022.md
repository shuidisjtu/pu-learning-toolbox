# 20261021 任务交接：复核技术意见（2026-10-22）

依据用户新增的 `heng958_tasks_20261021.md`（SHA256
`065c652df4466fc0912e39f00c864b0b5cce60e95336f11b0c0e9014aed423dc`），
与本机 `1ab7c77` 基线的协议、源码、交付收据对账。
本记录是 Codex 的技术复核回复，不是 HENG958 本人签署；不代填接受日期、
不修改冻结 JSON、方法台账或历史结果。后续任务排序只维护在[执行计划](../survey_execution_plan.md)。
原始上传文档保留原样，不随本记录提交。

## 1. 五项 review 回复

### 1.1 `v2_p4_clean_support_policy`

**技术意见：accept。** [协议 §2.4](../pu_survey_protocol.md)与
[v2 review_items](../../../../pu_toolbox/experiment/survey_protocol_v2.json)登记一致：
support 从 train 均匀抽出并移除，与剩余 train / pu_val / clean_val / test 互斥；
Spambase 100，其余 500；正负 support 标签均计入训练期真实标签预算。
覆盖 Spambase、Connect-4、MNIST、Fashion-MNIST、CIFAR-10，不扩张到另外三数据集。
`PA+clean-support` 须显式标注并与纯 PU PA 区分；正式路径属于 P3.2。

[25 份真实制品及三方法 Spambase CPU 交付](../delivery/clean_support_materialization_20261022.md)
已验证这些数据约束；不再把“support 尚无产物”当作本机当前事实。
但独立工程入口不接受 versioned Survey，不能据此宣称正式 support 单元已完成。
本意见供本人确认后回复；`v2_p4` 仍保持 pending，由 shuidisjtu 按任务约定更新。

### 1.2 `v2_p10_self_pu_scope`

**技术意见：accept。** v2 `method_profiles.self_pu.params.require_validation=false`，
保留无 clean-meta 的 Pilot 消融身份，不替换旧 profile 或旧 execution_units。
完整 Self-PU 必须另立 clean-support profile/unit，并绑定实际 support、额外标签预算、
适用训练视图和双 teacher checkpoint 选择。

[已有 full-meta 工程入口](../delivery/clean_support_checkpoints_20261022.md)显式要求
`require_validation=true`、`distillation_scope=paper_population`，并已有真实 Spambase
短 CPU 双轨回放；它只支持 SCAR OS，不能代替完整版的正式 TS-compatible 接线。
`v2_p10` 仍 pending，不能因工程测试通过自行改成 accepted。

### 1.3 #122 Split-PU

下列是可直接回复的**建议**，不是已获负责人裁决或已修改实现：

- **JSD：建议本阶段保持源码 teacher=0.7，明确登记论文/代码分歧。**
  当前 `splitpu_js_loss` 与锁定作者源码方向一致；论文式则应交换 target/pred 的角色，
  同时保留 `-0.3 log(0.3)` 分母。只改 `js_teacher_weight=0.3` 会改变归一化，
  不能称为论文式修复。若负责人选择论文式，应另做一次实现/测试/方法卡与版本登记的
  联动修改；本记录不暗改默认训练目标，也不声称源码对齐等于论文式复现。
- **batch：建议按当前 v2 §2.3 登记 benchmark 近似，不在本轮悄悄加权。**
  v2 已统一披露均衡 mini-batch 的并集不精确恢复总体比例；#122 的合成表征得到约
  0.673913 而非总体 0.4。接受操作定义不等于证明无偏；若改比例或加权，应另立并验证
  变体，同时更新预算/配方及比较说明，而不是覆盖历史结果。
- **指针：建议在下一次获授权的版本化台账更新中修正。**
  当前构造器先验赋值为 `split_pu.py:221`；优先用构造器/参数符号定位补充脆弱的行号。
  本轮不把“建议修正”伪写为已授权，不改冻结台账。

证据：[现状与独立标量/实际 teacher 批次表征](../admission/splitpu_122_characterization_20261018.md)、
`pu_toolbox/estimators/deep/split_pu.py::splitpu_js_loss` 及对应专项测试。
本轮重跑其 11 项表征通过；它们证明现状与分歧，不构成负责人裁决，#122 不据此自动关闭。

### 1.4 RV5：Self-PU

**技术结论：Pilot 必须标为 no-meta 消融，不是完整 Self-PU。**
clean-meta 的真实标签只能来自单独登记的 support，不得取外部 `clean_val` 或 test 来
反向更新训练。无 support 的冻结消融可沿原协议报告 PA，但须披露消融身份；
完整方法的 PA 只能标为 `PA+clean-support`，不能宣称 pure-PU PA 资格。
OA 的 clean validation 只负责外部选模，不因此成为训练期 support。

证据：[Self-PU 方法卡 §5–6](../../method_cards/Self-PU.md)、v2 profile 与上述 checkpoint
交付。这里确认的是角色和呈报边界，不重新签署旧数值、论文复现或完整 TS 路径。

### 1.5 RV6：五方法 TS 接线

**技术结论：在 v2 的操作性校准定义及其已披露近似下，可保留现有接线；
不能推广成严格独立 TS 抽样、总体风险无偏或论文实验复现。**

| 方法 | 本轮复核结论与必须保留的说明 |
|---|---|
| nnPU | P 正风险/先验保持；P batch 加入 U 风险。双 loader 循环及均衡 batch 只满足操作定义，不保证总体混合比例；TS 与 sample_weight 的未定义组合继续拒绝。 |
| uPU | 完整训练矩阵承担 U 风险；P 正项和先验保持，RBF 基数仍受校准前 n_U 上限约束。中心候选池变化随风险角色披露。 |
| PUSBKernel | TS 训练折/refit 使用并集；内层 CV 仍用原 OS 风险选择超参，明确是跨目标选参，不称作 TS 目标内最优。它与外部 PA/OA 选择不是同一层。 |
| Dist-PU | alignment/histogram 与 entropy 扩张边缘角色；正例损失/先验保持，MixUp 池不复制 P。entropy 扩展是协议选择，不是 population identity 必然结论。 |
| Self-PU | 负边缘风险按 U 权重质量与 P 行数混合；非均匀 meta 权重下不等于普通并集均值，不对 P 学习 meta 权重；trusted/pace/consistency/验证角色不因视图改变。 |

依据[原 RV6 技术复核](rv6_ts_calibration_technical_review_20261019.md)逐项查看当前实现，
并重跑下节专项。该结论是技术意见，RV6 的 HENG958 方法学签署仍不代填。

## 2. Issue 收尾边界

- **#126**：四个 GEN-PU blocker 仍待逐项冻结：先验加权 Du/合成 PN、六网络训练及
  阶段候选/存储、像素输出域与四 CNN encoder 配方、G 步 D 的权重/BN/dropout 模式及
  原生图像资源 profile。已有公式或 mode 回归不等于四项均已冻结；本轮不填 recipe 数值或关单。
- **#88**：[RV8 已签](rv8_method_admission_20261021.md)可作为 Robust-PU/Split-PU/LaGAM
  准入证据被有条件接受的依据；Split-PU #122、LaGAM support 正式路径及共通配方前置
  仍须保留。需能在 issue 中明确移交残余项后再收尾，不能写“算法已全部正式准入”。
- **#122**：三项建议见 §1.3；裁决/授权尚未落地，不以表征测试全绿代替关闭条件。
- **#119/#120/#121/#123/#125**：按任务交接和仓库的 20261021 处置记录视为已关闭，
  无新反证不重开；本轮未联网核实 GitHub issue 实时状态，也未评论或关闭任何 issue。

## 3. 本轮验证与用户覆盖指令

本轮重跑 RV6 的八个 estimator/view 专项文件、两个公共视图/路由文件、Split-PU 表征与
support 收据契约，共 **158 passed、0 skipped、0 failures/errors**。
本机 JUnit：`/tmp/task-intake-review-20261022.xml`。收据契约是元数据检查，
不冒充本轮重读全部 raw；25 份逐 raw 验收仍引用已有交付。未启动 GPU 或正式跑批。

用户最新指令优先于原任务表：**P3.2 中 PN-CNN 的 5 seed 实验暂缓，等协议定稿后再安排**。
独立真实 CIFAR GPU smoke 仍是未交付项，不在本轮执行；不是把两个任务一起标成完成。
其他方法或 support 的 seeds 0–4 不受影响，冻结协议及历史结果不改。
