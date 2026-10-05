# P3 缺失方法技术预集成交接（2026-09-28）

2026-10-05 P1-2 补记：[公开对照草稿与可比性](p3_public_comparison_20261005.md)
补12条待审原表读数与15格范围，不把校准/公式 smoke 升级为论文数值复现。
Split-PU 论文teacher50/代码20、Robust-PU混合单位与非oracle PN均单列；
本项目尚无这五方法正式结果，不消除原交接的数值对照及签署待办。

2026-10-05 P1-1 补记：[五方法准入证据包](p3_admission_evidence_20261005.md)
已补来源行为与独立公式测试、预算/选择角色和只读机器核验，引用已进入台账与原八方法草稿。
特别纠正 Split-PU：锁定上游 teacher/student 返回末轮权重，测试指标是监控，
不把未使用的 best_acc 变量解释为 test-best 选模；easy 损失归一化另列差异。
本补记只更新工程准备，不覆盖历史数值/来源摘要或代填签署。

2026-10-05 补记：PAN、RP、PULNS、GenPU、Holistic-PU 已补独立组件、方法卡、台账、共享特征与单元/契约测试；四个新增深度方法 CUDA smoke 已通过，P3MIX 仅完成 batch 组件，未注册 estimator，详见[独立推进记录](independent_progress_20261005.md)。PULNS 有额外干净 support 监督，不具备标准 PU-only PA 资格；本次不更改下文的历史证据或正式准入状态。

2026-10-04 补记：8 方法的准备顺序和逐项准入 blocker 已整理为 [机器清单](data/p3_candidate_admission_v1_draft.json)，后续排序与 P4.1 草稿交付见 [推进计划](post_pilot_priority_plan.md)。这不改变本交接的正式准入边界。

本文件只汇总**正式跑批之前**可以独立核查的工程证据。它不是 `survey_protocol_v1.json` 的修订、不是合作者签署，也不改变 P2.1/P3.1/P3.2 的正式验收状态。冻结矩阵仍只有 7 个 PU 方法和 PN oracle；新增条目仅在 `pu_toolbox/experiment/method_ledger.json` 中登记。

## 方法—视图—准入矩阵

| 方法 | 本轮代码/台账状态 | 校准实际作用 | 候选矩阵前置裁决 |
|---|---|---|---|
| VPU | 既有实现、台账和 D21/D23 视图接线；本轮修正过时方法卡，补拟合视图诊断字段 | `ts` 边缘池为 `U∪P`；验证变分风险仅是 source diagnostic，**不得**作 PA/OA 选模（D22） | 先裁决 PA 选模准则；OS 为消融，不可称原生结果 |
| PULDA | 两阶段实现、台账/角色测试；现支持注入可训练 CNN 与分批图像处理 | LDA 和 two-way margin 的 U 期望及 EMA 用逐批 `U∪P`；MixUp/伪标签原池不变 | 共享预算/正式图像规格、公开数值对照与选模准则预注册 |
| PUET | 既有 nnPU/quadratic CPU 树；本轮补台账、`os_or_ts` 与节点风险 golden | 复制 P 为 U 风险行，原 P 角色保留；U 权重分母变为 `n_U+n_P` | CPU 树预算及 P3.2 GPU 条款豁免；不能借用神经 checkpoint 配额 |
| Grad-PU | 既有式 (5)–(7) 实现；本轮补台账、`os_or_ts` 与梯度池测试 | U 风险和梯度插值的边缘池逐批取 `U∪P`；不使用类别先验 | 官方实现未核实（`source_status=not_found`）；不能声称代码逐行复现 |
| Robust-PU | nnPU warm-up + self-paced 组件；本轮补台账、风险视图 | 仅 warm-up 的 nnPU U 风险用 `U∪P`；后续伪负例仍仅为原始 U；跳过 warm-up 时拒绝 `ts` | 非完整官方 scheduler/CNN/图像增强，须单列 benchmark-adapted |
| Split-PU | nnPU teacher + easy/hard student 组件；本轮补台账、风险视图 | 仅 teacher nnPU U 风险用 `U∪P`；hard/easy 始终只在原始 U 划分 | Gaussian 特征扰动 ≠ 作者图像增强/SimSiam；不得直接裁决论文 CIFAR 数值 |
| CVIR | 本轮新增固定 `α_U` 的表格实现、台账与方法卡 | **未接线**：原论文需 U 内混合比例 `α_U`，公共总体 `π` 不能无条件代入；显式 `ts` fail-loud | 必须先登记可核的 `α_U` 来源、有限样本转换及视图协议；当前不具备 Survey runner 准入条件 |
| LaGAM | 既有 clean-support-gated 组件；本轮补台账和拒绝路径测试 | **不适用当前 TS-OS 替换**：额外干净 support set 是元训练输入，校准不消除标签预算 | 标准 PU-only PA-ineligible；OA 也要独立 support/selection 切分，禁止复用 clean_val |

上述“已校准”指训练中的对应**风险角色**已替换且可检测，不表示整条方法的每个 U 物理池都被并集替换。任何 run 的真实视图以 manifest 为准；台账是方法默认值。

## 可复核的非正式证据

1. **来源行为/公式级对照**：VPU 的变分目标、PULDA 的 LDA/two-way margin、PUET 的 Proposition 2(a) 节点风险、Grad-PU 的式 (5)–(7)、Robust-PU 的 SPL 权重、Split-PU 的加权 JS 与 LaGAM 的元标签方向，均有同名方法卡的论文/作者代码位置与 `tests/unit/estimators/` 下对应方法的测试 中的数学或行为测试；CVIR 的负例保留顺序和数量由 [原文 Algorithm 2](https://proceedings.neurips.cc/paper_files/paper/2021/file/47b4f1bfdf6d298682e610ad74b37dca-Paper.pdf)及 `test_cvir.py` 锁定。这是**来源规则对照**，不是作者仓库端到端重放或论文表格数值对齐。
2. **校准 golden/门禁**：`test_puet.py` 检查 OS→TS 的精确节点风险/分裂增益；`test_grad_pu.py` 检查 U 风险和插值行数；`test_robust_pu.py`、`test_split_pu.py` 检查 `U∪P` 均值恒等式；`test_pulda.py` 检查两阶段与角色切换；`test_lagam.py` 和 `test_cvir.py` 检查不适用的 `ts` fail-loud。`test_ledger_registry_consistency.py` 锁定注册名、来源、能力、先验口径和校准 hook，防止只改 JSON 未改训练接口。
3. **共享特征路径**：`test_p3_feature_adapter_smoke.py` 用同一个冻结图像 encoder 的四路适配制品，使 8 个方法都接受相同 2-D 特征并完成小样本拟合/预测；测试同时检查适配 manifest、索引和非有限输出。这仅证明 `cnn_feature_adapter` 接口兼容，**不**把任一 MLP/树方法说成具备原生 CIFAR CNN 或与原论文图像 backbone 可数值比较。正式候选仍需逐方法预算/存储 profile、PA/OA 选模、representation hash 与比较组预注册。
4. **短时 CUDA 技术 smoke**：2026-09-28，0 号 NVIDIA RTX A6000，驱动 550.54.14，Python 3.12.2、PyTorch 2.6.0+cu124/CUDA 12.4；`CUDA_VISIBLE_DEVICES=0 pytest -q -m gpu tests/unit/estimators/{test_vpu,test_pulda,test_grad_pu,test_robust_pu,test_split_pu,test_lagam}.py`：6 passed；`CUDA_VISIBLE_DEVICES=0 pytest -q -m gpu tests/unit/estimators/test_cvir.py tests/unit/experiment/test_p3_feature_adapter_smoke.py`：6 passed。测试后 `nvidia-smi` 显示 0 号卡占用 1 MiB。这是全局环境单卡 smoke，**不是** `uv.lock` Linux frozen-lock 复验、多 seed 资源验收或正式 P3.3 调度证据；PUET 纯 CPU，不应计算进 GPU 覆盖率。

## 待合作者裁决或正式阶段执行

- **候选执行矩阵**：本文件提供准入字段和阻断条件，但**不修改**已冻结并被比较矩阵摘要绑定的 `survey_protocol_v1.json`。若获准扩展，应新版本预注册每方法预算/参数候选、数据集/视图、PA/OA 选模、backbone/representation、存储 profile 与结果比较组，再由 runner 消费；不能只在旧矩阵追加名字。优先可讨论 PULDA、PUET、Grad-PU、Robust-PU、Split-PU；VPU 先解决 D22，CVIR 先解决 `α_U`，LaGAM 先解决 clean-support 标签预算。
- **公开数值对照**：当前只有来源公式/行为级证据，没有官方仓库与工具箱在同一数据、split、backbone、训练预算下的逐数结果。因为现有方法多为表格/特征适配，论文 CIFAR 数字只能做协议差异下的量级参考，不能伪称实现通过。需要预注册可比较锚点、协议差异及阈值后再做独立验证。
- **环境与外部审阅**：上述 GPU smoke 使用 PyTorch 2.6.0+cu124；Linux `uv.lock` 冻结依赖/驱动兼容问题见 `p2_0a_review.md`，仍需另一次隔离环境复验。方法负责人/合作者尚未签署本轮新增七条台账和 TS-OS 口径；本交接不代签。
