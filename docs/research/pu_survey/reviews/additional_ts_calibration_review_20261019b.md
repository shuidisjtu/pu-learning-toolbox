# Robust-PU、Split-PU、GEN-PU 的 TS 校准技术补核（2026-10-19）

本记录补充已完成的 RV6 与 PULDA/PUET/GradPU 校准范围审计，检查 Robust-PU、Split-PU、GEN-PU 中 TS 视图实际影响哪些训练角色。依据本地实现和专项结构测试，并与已登记来源适配边界对读；不代表 HENG958 的方法学签署，也不替代方法负责人复核，不声明论文实验重放、方法效果或正式准入。

| 方法 | TS 校准实际作用范围 | 被测试锁定的不变量 | 技术结论与保留事项 |
|---|---|---|---|
| Robust-PU | 仅 nnPU 预训练阶段的 U 风险输入采用当前 U batch 与 P batch 的并集 | 正例项和人口先验不变；episode 的伪标签/自步权重仍按原始训练行身份计算，不把已知 P 复制进伪负例池；`pretrain_epochs=0` 时拒绝 TS | 实现、逐 batch 角色捕获测试符合“仅 warm-up 风险校准”合同。论文作者的 clean-validation 选模、较长预算和源码训练流程仍未在该适配中实现 |
| Split-PU | 仅 nnPU teacher 训练阶段把每步 P batch 同时作为边缘 U 风险角色 | teacher 的正例项独立保留；splitter 的目标、easy/hard 划分及 student distillation 后续只使用原始 U；先验只用于 teacher nnPU 风险 | 原始 P∪U 风险池与后续 U-only 分割由结构测试单独约束；JSD 方向/归一化和 TS 总体比例等 #122 裁决项仍待处理，不能把“校准仅限 teacher”理解为 #122 已解决 |
| GEN-PU | TS 将真实 `D_u` 风险池扩展为 U∪P，作用于 unlabeled discriminator 接收的真实样本；P 判别器/负例判别器的真实正例仍来自原 P | prior-weighted GAN objective、正负生成分布及后续 synthetic PN 阶段不因 OS/TS 改变；不新增任何真值标签；实际训练更新步数随 U 风险池大小变化并计入预算 | 测试确认只有 Du 的真实输入池随视图变大，且两阶段更新步数与池大小一致。论文/官方 demo 的目标差异、超参、模型和预算仍按 GEN-PU 来源审计保留，不声称源码复现 |

## 直接证据

- Robust-PU：`pu_toolbox/estimators/deep/robust_pu.py::RobustPUClassifier.fit`；`tests/unit/estimators/test_robust_pu.py::test_ts_calibrates_only_warmup_and_routes_from_ledger`、`test_ts_warmup_marginal_is_exact_p_u_union`；CNN 角色/预训练边界见 `tests/unit/estimators/test_robust_pu_cnn.py`。论文/代码适配收据为 `docs/research/pu_survey/data/robustpu_source_review_20261018.json`。
- Split-PU：`pu_toolbox/estimators/deep/split_pu.py::SplitPUClassifier.fit`；`tests/unit/estimators/test_split_pu.py::test_ts_calibrates_teacher_but_preserves_original_u_split`、`test_ts_teacher_marginal_is_exact_p_u_union`；CNN/CPU pool/checkpoint 边界见 `tests/unit/estimators/test_split_pu_cnn.py`。来源身份及实现差异见 `docs/research/method_cards/Split-PU.md` 和 #122 characterization。
- GEN-PU：`pu_toolbox/estimators/deep/gen_pu.py::GenPUClassifier.fit`；`tests/unit/estimators/test_gen_pu.py::test_fit_calibration_counts_both_stages_seed_clone_pickle_and_no_input_mutation`、`test_calibration_changes_only_du_real_pool`；像素/CNN 两阶段角色见 `tests/unit/estimators/test_gen_pu_cnn.py::test_basic_calibration_changes_only_real_du_pool_and_actual_pool_step_count`。来源和论文目标适配边界见 `docs/research/pu_survey/genpu_source_cnn_review_20261007.md`。

2026-10-19 执行 `pytest -q -rA tests/contract/test_ledger_registry_consistency.py tests/unit/estimators/test_robust_pu.py tests/unit/estimators/test_robust_pu_source_contract.py tests/unit/estimators/test_robust_pu_cnn.py tests/unit/estimators/test_split_pu.py tests/unit/estimators/test_split_pu_cnn.py tests/unit/estimators/test_gen_pu.py tests/unit/estimators/test_gen_pu_cnn.py`，退出码 0、无失败；6 项 CUDA-only 测试因 CUDA 不可用跳过，保留 1 条既有高成本 pipeline warning。`check_p3_admission_evidence.py` 与 `check_candidate_fit_contracts.py --check ...candidate_fit_contracts_20261019_draft.json` 也通过，均报告 `formal_admission=false`。这些回归只证明实现调用点符合结构断言，不验证优化器收敛、总体有限样本无偏性或外部数据效果。

## 待复核，不因本记录变更

1. HENG958 按方法确认三种适配各自的校准对象，特别是 Robust-PU 的 nnPU warm-up 与后续自步阶段分离、Split-PU 的 teacher-only TS，以及 GEN-PU 仅替换真实 Du 风险池。
2. Robust-PU 上游许可证未确认，保持 `needs_review`；三种方法的 source/adaptation level 仅按既有台账与方法卡，不因校准测试升级。
3. Split-PU #122 的 JSD 方向/分母与 TS 总体比例/prior 口径仍按 issue 待确认；GEN-PU 的目标与正式 recipe/预算仍需按计划审阅。
4. 不改 `calibration_applied`、原生采样假设、正式候选、冻结矩阵或历史实验；本记录本身不授予协议批准。
