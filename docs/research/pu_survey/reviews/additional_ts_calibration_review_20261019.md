# PULDA、PUET、GradPU 的 TS 校准技术补核（2026-10-19）

本记录补充 RV6 的五方法核验，针对 P3.1/P3.2 中已经接入 TS 风险角色替换的 PULDA、PUET、GradPU。核对生产代码与结构/数学回归的对应关系；不代替方法负责人签署、不证明论文数值复现，也不调整正式配方、准入、冻结矩阵或历史结果。

| 方法 | 当前 TS 作用范围 | 不变量与协议边界 | 技术判断/待确认 |
|---|---|---|---|
| PULDA | 训练 batch 的 P logits 仍作原正例角色，同时以额外 U 角色副本进入 LDA 与 two-way margin 的边缘矩；EMA 校正随同一扩展角色计算 | 原始观测标签与持久伪标签池不变；MixUp 仍只作用于物理 P/U 训练行；总体先验不重估；逐 run 记录实际视图 | 代码与角色回归一致。每个 U batch 重新抽取的 P batch（有放回）组成 TS 边缘角色，是 mini-batch 适配口径，仍须与批准的实际配方一致 |
| PUET | TS 时将训练 P 显式复制为 U 风险角色行；原 P 行仍保留在正例角色 | P 行质量为 `π/n_P`，U 风险池每行质量为 `1/(n_U+n_P)`；bootstrap 时 U 抽样数同步扩大至校准池大小；先验、树结构和其他训练配置不因视图重估 | 节点根质量/36 组 bootstrap 断言与实现一致；这是已披露的 nnPU 二次节点风险适配，不代表完整作者实验重放 |
| GradPU | TS 时当前 P batch 追加到 U 风险输入；同一扩展 U 角色也作为输入梯度惩罚的插值端点 | 正例风险单列；class prior 不进入目标；插值行数按 `max(n_P_batch,n_U_batch+n_P_batch)` 生成；β 进度与优化器/训练预算没有视图分支 | 结构测试直接捕获风险 logits 与插值维数，支持接线一致性判断。论文实验采样、网络/调度和 toolkit TS 角色视图的差异仍须按 #121/正式协议裁决 |

## 代码和测试证据

- PULDA：`pu_toolbox/estimators/risk/pulda.py::PULDAClassifier._train_epoch`；`tests/unit/estimators/test_pulda.py::test_ts_calibrates_distribution_roles_without_relabeling_pseudo_pool` 与 `test_ts_distribution_input_reuses_positive_logits_as_u_role`。后者观察到原 P/U logits 后，确认 TS 风险矩仅把已有 P logits 作为零标签 U 角色重用。
- PUET：`pu_toolbox/estimators/risk/puet.py::PUExtraTreesClassifier.fit`；`tests/unit/estimators/test_puet.py::test_ts_node_risk_uses_p_rows_in_both_roles`、`test_edge_bootstrap_root_masses_and_constant_leaf`；来源与适配限制见 [`puet_adaptation_completion_20261017.md`](puet_adaptation_completion_20261017.md)。测试覆盖 OS/TS、bootstrap 开关、3 种 P/U 不平衡比例及 3 个 seed 的根质量。
- GradPU：`pu_toolbox/estimators/deep/grad_pu.py::GradPUClassifier.fit`；`tests/unit/estimators/test_grad_pu.py::test_ts_calibrates_unlabeled_risk_and_gradient_interpolation`；训练调度和 batch 语义见 [`gradpu_training_recipe_review_20261018.md`](../admission/gradpu_training_recipe_review_20261018.md)。测试直接拦截 objective 输入，确认 U 风险和插值池都包含正例角色副本。

2026-10-19 运行 `pytest -q -rA tests/contract/test_ledger_registry_consistency.py tests/unit/estimators/test_pulda.py tests/unit/estimators/test_puet.py tests/unit/estimators/test_grad_pu.py`，退出码 0，无失败；PULDA 和 GradPU 各有 1 项 CUDA-only 测试因 CUDA 不可用而跳过。保留 3 条既有高成本 pipeline warning。这些 CPU/小型合成用例不证明 CUDA 资源、公开数据效果、总体风险有限样本无偏性或正式准入。

## 尚需裁决

1. HENG958 确认 PULDA mini-batch P∪U 角色及其 EMA 矩是否为本项目接受的校准变体。
2. HENG958 确认 PUET 根节点 bootstrap 质量合同与论文第 4 节的采样口径相符；已有来源审计保留它是作者 nnPU/quadratic 分支适配，而非完整复现。
3. 按 #121 的负责人口径和版本化配方审阅 GradPU 的 TS risk/gradient interpolation；不得把 TS 校准视图等同于论文报告的具体数据抽样流程。

三个方法仍保持原 ledger sampling/calibration 字段和 `admitted` 状态；本记录只补技术审计可追溯性，不将待审字段自动改成已批准。
