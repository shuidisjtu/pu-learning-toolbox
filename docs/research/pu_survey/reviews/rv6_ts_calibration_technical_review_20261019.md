# RV6：五方法 TS 校准技术核验（2026-10-19）

本记录核对 `nnpu`、`upu`、`pusb_kernel`、`dist_pu`、`self_pu` 的代码接线与现有结构测试，关注 TS 视图中 P 是否只进入预期的无标签风险角色、P 风险/先验/容量/训练辅助流程是否意外变化。它是工程技术建议，不是 HENG958 的方法学签署，也不证明 TS/OS 估计量等价或正式 Survey 准入。

## 核验结果

| 方法 | TS 视图实际作用 | 保持不变的量 / 注意点 | 技术判断 |
|---|---|---|---|
| nnPU | 每步把同一正例 batch 追加到 U 风险输入；P 正例风险仍单独计算，类别先验不变 | TS 与 `sample_weight` 明确互斥，因为追加的 P 行没有 U 权重定义；DataLoader 仍按既有较短 loader 循环策略生成随机批次 | 接线符合“只扩展 U 风险角色”的实现约束。按 mini-batch 形成的 TS 经验风险仍须由方法负责人确认是否满足项目采用的校准口径 |
| uPU | TS 使用完整训练矩阵作为 U 风险输入；P 正风险和先验不变 | RBF 中心数按校准前的 `n_U` 限制，避免 TS 视图同时改变模型容量；中心候选集合随 U 风险角色扩展 | 接线与台账的 TS 标记一致；中心候选集合变化是目标角色变化的一部分，中心数量未随之变化 |
| PUSBKernel | TS 训练目标的 U 风险角色为 P∪U，正项/先验不变 | RBF 中心、prior-quantile 阈值池和 CV fold 身份不因视图改变；**TS 拟合的超参仍由 OS-view 验证目标选择**，属于明确的跨目标选择口径，不是“TS 目标内选模” | 训练风险接线结构一致；OS 验证目标是重要协议差异，需负责人确认其与预注册分析意图相符 |
| Dist-PU | expectation 路径扩展 alignment/entropy 的边缘角色；histogram 路径扩展 U histogram 与 entropy 角色；P 正损失仍单列，先验不变 | 基础 MixUp 池仍是原训练矩阵，未追加第二份 P；clean-room 与 `author_staged` 两路径都传递校准标志 | 两种分布损失和两条训练路径均有明确视图入口，未发现 P 被重复计入的接线问题 |
| Self-PU | 仅在负边缘风险组合时，将正例负损失按 P/U 角色数并入原 U 负项；不可信 U 的 meta 权重仍只覆盖不可信 U | trusted-set 人口、排程及目标仍只来自原始 U；P 正项单列；校准不扩张验证、teacher 选择或 MixUp 池；样本权重在两个视图均拒绝 | 角色范围受结构测试约束；当前加权 U 负项是按原 U 权重质量与新增 P 样本数做质量守恒组合，并非对 P 学习 meta 权重 |

## 复核证据

- 公共角色构造与 ledger gate：`pu_toolbox/experiment/training_views.py::calibrate_ts_os_batch`、`tests/unit/experiment/test_training_views.py`、`tests/unit/experiment/test_training_view_routing.py`。
- nnPU：`pu_toolbox/estimators/risk/nnpu.py::NonNegativePUClassifier.fit`；`tests/unit/estimators/test_nnpu_ts_view.py`。实现保持 P 风险单独计算，并拒绝 TS 与 sample weights 的未定义组合。
- uPU：`pu_toolbox/estimators/risk/upu.py::UPUClassifier.fit`；`tests/unit/estimators/test_upu_ts_view.py`。实现使用全部训练行承担 TS U 风险，保留原先验和预校准 U 数量所限的 RBF 基数。
- PUSBKernel：`pu_toolbox/estimators/bias_aware/pusb_kernel.py::PUSBKernelClassifier.fit`；`tests/unit/estimators/test_pusb_kernel_ts_view.py`、`test_pusb_kernel_view_invariants.py`。尤其应在解释 TS 数值时披露其内层超参验证仍按 OS 风险打分。
- Dist-PU：`pu_toolbox/estimators/risk/dist_pu.py::DistPUClassifier.fit`、`_fit_author_staged`；`tests/unit/estimators/test_dist_pu_ts_view.py`、`test_dist_pu_view_invariants.py`。结构 spy 锁定 U 角色恰为完整矩阵、没有把 P 再拼一次，并检查 MixUp/RNG 结构不随视图扩张。
- Self-PU：`pu_toolbox/estimators/deep/self_pu.py::_marginal_negative_risk`、`SelfPUClassifier.fit`；`tests/unit/estimators/test_self_pu_ts_view.py`、`test_self_pu_view_invariants.py`。结构断言锁定负风险只覆盖不可信 U 及校准后的角色数，trusted set 不纳入已标记 P。

2026-10-19 本次在 CPU 上运行以上八个 estimator/view 专项测试文件及两个公共视图/路由测试文件，结果 `All Pass`。这只是当前代码与回归断言一致的证据，不是实际数据效果、数值正确性证明或负责人接受记录。

## 负责人仍需裁决

1. 确认五种方法都适合采用本项目定义的 mini-batch/全矩阵 TS 风险角色校准，尤其 nnPU 的双 loader 循环以及 PUSBKernel 的“TS 训练、OS 验证”选模口径。
2. 确认 Dist-PU 两个分布损失与 Self-PU 的质量守恒式负风险组合，作为协议校准变体是否保留；它们不应描述成原论文实验直接复现。
3. 如结论接受或要求整改，在 RV6 复核记录中由 HENG958 给出逐方法意见；在此之前 RV6 状态仍为待复核。

本记录不更改 `method_ledger.json`、冻结协议、历史 Pilot/正式跑批结果或任何准入字段。
