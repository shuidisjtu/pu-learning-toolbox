# 传统 PU 基线 benchmark

七类传统 PU 分类器（Elkan--Noto、uPU、nnPU、PNU、LDCE、KLDCE、LLSVM）的合成单域
性能基线，协议与统计规则见 `docs/research/traditional_pu_metric_contract.md`。
所有报告固定 `paper_claim=false`：该基线用于工具箱自身性能评估与改进，不构成
论文复现，也不得与论文表格数值直接比较。

## 网格结构（契约 §2.1/§2.2）

- **SCAR 主网格**：六个二元 PU 方法（`elkan_noto`/`upu`/`nnpu`/`ldce`/`kldce`/
  `llsvm`）在类先验 π ∈ {0.1, 0.3, 0.5} × scale {small 400, mid 2000} 上运行，
  是算法主排名来源。PNU 不进入该网格（契约 §2.2），它运行自己的三元标签网格。
- **linear SAR 诊断线**：同一 π × scale 网格，只包含 `elkan_noto`/`upu`/`nnpu`/
  `llsvm`。LDCE/KLDCE 被排除：它们的 `flip_probability=h` 语义是翻转标签，不是
  倾向性（propensity）机制（契约 §2.3），与 SAR 数据机制不匹配。SAR 单元只作
  鲁棒性与失效边界诊断（PU-only 指标保留但不作最终优劣结论），从不参与排名。
- **PNU 三元网格**（`configs/pnu_baseline_v1.json`）：P:N:U 比例 1:1:4 / 1:2:4 /
  1:1:8 × scale，观测标签为 `{+1, -1, 0}`；PNU 不参与纯 PU 方法横向排名；二元 PU
  指标（`pu_zero_one_risk`/`pu_estimated_precision`/`pu_recall`/`pu_negative_rate`）
  在该网格中标记为不可用并记录原因。契约 §2.2 不要求 PNU 协议提供 π，但
  PNUClassifier 必填 `class_prior`：runner 取 `data.class_priors` 首项（0.3）作为
  类先验传入每个 PNU 单元，0.3 为记录级选择（非数据真实先验）。
- **PNU 计数派生**：每单元的 P/N/U 绝对数量由 `pnu_counts(ratio, n_samples)` 按
  比例 × 场景规模推导（largest-remainder 取整），因此 small/mid 两档产生不同数据；
  实际计数记为 tri-lab 行的 `n_p`/`n_n`/`n_u` 列（契约 §2.2"显式记录数量与比例"）。

## 命令

开发期（5 seeds 0..4，用于协议修复与验证）：

```bash
uv run python -m benchmarks.traditional_pu.run \
  --config benchmarks/traditional_pu/configs/seven_methods_pu_baseline_v1.json \
  --seed-set development \
  --results-dir benchmarks/traditional_pu/results/dev_baseline
```

确认期（20 seeds 100..119，v1 正式基线——修复前默认参数的历史快照，勿在当前源码下重跑）：

```bash
uv run python -m benchmarks.traditional_pu.run \
  --config benchmarks/traditional_pu/configs/seven_methods_pu_baseline_v1.json \
  --seed-set confirmation \
  --results-dir benchmarks/traditional_pu/results/confirmation_v1
```

当前工具箱基线（v2，显式锁定默认参数，契约 §5 演进记录）：

```bash
uv run python -m benchmarks.traditional_pu.run \
  --config benchmarks/traditional_pu/configs/seven_methods_pu_baseline_v2.json \
  --seed-set confirmation \
  --results-dir benchmarks/traditional_pu/results/baseline_v2
```

PNU 网格：

```bash
uv run python -m benchmarks.traditional_pu.run \
  --config benchmarks/traditional_pu/configs/pnu_baseline_v1.json \
  --results-dir benchmarks/traditional_pu/results/pnu_baseline
```

`--seed-set` 默认 `development`；默认按 `(algorithm, scenario, seed)` 断点恢复
（已有 `trials.csv` 时跳过已完成单元），`--no-resume` 强制重跑。

`trials.csv` 在每个 trial 完成后**增量落盘**（而不是网格结束时一次性写入），
因此运行中途被中断（ctrl+c / 主机 kill）只会丢失正在跑的单元，已完成部分全部
保留，再次运行即可无缝续跑（保留 `trials.csv` 即是续跑依据）。

## 已知限制（跟踪中）

- 已解决：`configs/pnu_baseline_v1.json` 曾缺类先验字段使 runner
  `_iter_scenario_specs` 触发 `KeyError`；现配置显式声明 `class_priors: [0.3]`，
  即该跟踪项关闭（先验来源见 PNU 三元网格节）。
- 已解决(a7b9bc1)：`label_frequency` 改为 scar/sar 分支内惰性读取，pnu-only
  配置不再 `KeyError`；`--timeout-profile` 对任意 config（含 pnu-only）优雅降级，
  不再 `StopIteration`。PNU 只在 `pnu_baseline_v1.json` 出现（契约 §2.2），
  `seven_methods_pu_baseline_v1.json` 不再包含 `pnu`。

## 产物契约（契约 §5）

每个 `results/<run-name>/` 下：

- `trials.csv`：逐 trial 行——`algorithm`/`scenario`/`seed`/`status`/
  `elapsed_seconds`/`warning_count`/`failure_reason` + 全量指标列 +
  `pred_positive_rate`/`degenerate_prediction`（预测阳性率与全正/全负退化标记，
  调优方案 §3.1/§3.3；非 success 行为 NaN）；
- `summary.csv`：成功率与均值/样本标准差/95% 置信区间；
- `resolved_config.json`：解析后配置（含 `resolved_at`）；
- `run_manifest.json`：`schema_version`/`created_at_utc`/`protocol`/
  `paper_claim=false`/`config_sha256`/`git_commit`/`git_worktree_dirty`/
  `runner_sha256`/`seed_set`/`seeds`/`n_trials`/`environment`/`limitations`/
  `data_leakage_audit`（preflight 报告整段嵌入；未提供报告时为 `audit_only`）；
- `data_leakage_audit.json`：泄露审计 preflight 产物（规则版本、状态、命中项、
  场景哈希；见下方"泄露审计门禁"）；
- `report.md`：人读摘要。

## 泄露审计门禁（审计设计 §7 阶段 A）

每次运行（含 resume）在网格执行前先跑 preflight，生成 `data_leakage_audit.json`：

- **y_true 路径约束**：estimator `fit` 调用点守卫（identity + 内存共享检查，不做数值
  相等——`label_frequency=1.0` 时 `y_pu == y_true` 是合法极端）；泄露命中即阻断整次运行。
- **trial 列写入门禁**：任何 `y_*` 原始标签列在写入 `trials.csv` 前被阻断（新行与
  resume 读取两处检查点）。
- **特征黑名单 / 重复样本检测**：独立审计函数（全名、大小写不敏感匹配），合成流程无
  持久特征集与切分，preflight 记录 `not_applicable` 及阶段 B 理由。
- **阻断行为**：preflight 或 runner 门禁命中时，运行立即中止，CLI 返回码 **1**（与
  0 成功 / 2 全失败区分），`data_leakage_audit.json` 保留可复现原因，不生成可晋级产物。
- **边界**：切分索引/实体重复检查与折内预处理审计属阶段 B（官方数据集线进场时）。
  `--timeout-profile` 是诊断工具，不产生晋级产物，不走 preflight。

## 超时与冻结流程（契约 §6）

当前 `configs/*.json` 中的 `timeouts` 是**未冻结的宽松保护上限**（统一 120s）：
冻结前以此宽松上限保护运行，超时单元记录 `status="timeout"` 并
完整保存超时原因，不静默丢弃。

超时是**每单元的守卫**而非墙钟预算：超时单元的 worker 线程无法在 CPython 中
终止，会被放弃（abandoned）并继续在后台运行完真实耗时，阻塞后续单元直到其
真正结束。因此 `timeout` 行的 `elapsed_seconds` 记录的是守卫时刻，实际占用是
该 trial 的真实运行时长。进程级隔离（把每个 trial 放进独立进程以便可终止）是
已跟踪的后续改进项。

冻结流程：

1. 用 `--timeout-profile <out.csv>` 记录每个算法单次最小规模 trial（seed=0）
   的实际耗时分布（`algorithm`/`elapsed_seconds`/`n_features`/`n_samples`），
   随后退出，不进入网格；
2. 汇总为每算法耗时分布，冻结算法特异的超时阈值；
3. 将冻结后的 `timeouts` 写回 `configs/*.json` 并记录冻结依据。

```bash
uv run python -m benchmarks.traditional_pu.run \
  --config benchmarks/traditional_pu/configs/seven_methods_pu_baseline_v1.json \
  --results-dir /tmp/profile \
  --timeout-profile /tmp/timeout_profile.csv
```

## 配对判定（契约 §6，M6 候选变体）

候选变体与基线的比较基于**同一 seed 的配对差值及其 95% 置信区间**（`paired_diff_ci`，
t 分布），而非仅比较两个均值：

```bash
uv run python -m benchmarks.traditional_pu.compare \
  --baseline-dir benchmarks/traditional_pu/results/confirmation_v1 \
  --candidate-dir benchmarks/traditional_pu/results/confirmation_v1_candidates \
  --metric pu_zero_one_risk \
  --oracle-metric pu_auc_roc \
  --budget-seconds 120 \
  --out benchmarks/traditional_pu/results/m6_candidate_verdict.csv
```

输出按 `(algorithm, scenario)` 逐单元列 `n_paired`（双方 success 的配对种子数）、
`diff_mean`/`diff_ci95_low`/`diff_ci95_high`（候选−基线）、两侧成功率与 P95 耗时，
以及四条件判定（`cond1_improves`：配对 CI 在指标方向上支持改善；
`cond2_success_ok`：成功率未恶化（5pp 容差，记录在列）；
`cond3_budget_ok`：候选 P95 耗时 ≤ `/--budget-seconds`；
`confirmed_improvement`：cond1∧cond2∧cond3 且配对 `n_paired>0`；契约条件 4
"协议一致"由工具级校验保证——(scenario, seed) 键不同则拒绝配对并返回码 2）。

`--oracle-metric`（默认 `pu_auc_roc`）对隐藏真值指标再做一次同 seed 配对 CI，
输出 `oracle_cond1_improves` 与 `oracle_only_improvement`（oracle 指标改善而 PU
主指标未改善的单元，调优方案 §5——不得宣称 PU 调优成功）；trials 缺该指标列时
跳过该分类并在 stderr 警告，列值如实为 False。

基线侧无 success 值的单元（如开发基线中 LDCE/KLDCE 全域 nonconverged）无法配对，
记录 `paired_available=false`：此时以成功率/未收敛率对比作为主证据，不构造 CI。

## M6 当前结论（候选判定结果）

- **LDCE**：`max_iter=100→10000` 为确认的收敛修复
  （`results/confirmation_v1_candidates_ldce/findings.md`）。确认期 120/120
  单元 success（基线 1/120）、最慢 28.9s（预算 120s）、可配对单元指标与基线
  完全一致（diff=0,修复不改变解语义）。契约 §6 条件 1（配对 CI）在基线无
  性能值的 5/6 cell 不可评估，故工具列 `confirmed_improvement=False`，
  实质判定为"收敛修复推荐"（按契约字面条件 1 不标 confirmed）。
  **已回写工具箱**：`LDCEClassifier` 默认 `max_iter` 已从 100 改为 10000
  （v1 基线仍以 100 运行，见契约 §5 演进记录）。
- **KLDCE**：参数放宽候选被证伪（F5：ACS 停滞 + mid 单轮 QP > 300s），
  不进入任何候选网格；修复需实现层工作，单独跟踪。
- 后续（2026-08-24）：KLDCE 判据修复（见 `results/development_v1_kldce_fixed/`
  与 findings 修订段），small 档
  从"全域 nonconverged"变为全 success；mid 档 QP 效率仍超出预算（实现层
  后续）。
- 后续（2026-08-25）：KLDCE 实现层修复完成（原生 SMO + 单调 ACS 回滚 +
  `inner_tol=1e-6`，见 `results/confirmation_v1_candidates_kldce/findings.md`）：
  修复后默认参数下 development 30/30、confirmation 120/120 success，
  mid 档 P95 46.0s < 120s 预算。判定为**确认的收敛修复推荐**
  （paired CI 不可评估，基线侧 0 success；compare.py 列如实为 False，
  与 LDCE 同款路径）。
- **v2 正式基线（2026-08-25，`results/baseline_v2`）**：LDCE/KLDCE 修复写回
  工具箱默认参数后，以显式锁定参数的 v2 配置重跑 20-seed 确认期，7 方法
  全量 success，成为代表当前工具箱的正式基线。v2 配置带
  `locks_source_defaults` 标记，由 `scripts/check_baseline_configs.py` 门禁校验
  与源码构造器默认值一致——未来默认参数演进在门禁报警，而非基线静默漂移
  （契约 §5 演进记录）。`confirmation_v1` 保留为修复前历史快照。

## 开发期与正式基线的差异

- 开发期 5 seeds（0..4）：先运行并修复协议问题、验证网格可执行、暴露数据/指标
  缺陷；其结果不是正式基线。
- 确认期 20 seeds（100..119）：正式基线；两个集合不得重叠。
- 正式基线只接受确认种子集产物，且 `run_manifest.json` 必须记录代码 commit、
  依赖版本、种子集合、配置哈希与数据来源；`trials.csv` 原始表与摘要一并提交。
- v2 基线（`baseline_v2`）代表当前工具箱：7 方法显式锁定默认参数（LDCE
  `max_iter=10000`、KLDCE 修复后默认），与 `confirmation_v1` 同种子、同网格。
  v1 与 v2 的差异即 LDCE/KLDCE 的默认参数演进（契约 §5）；其余方法无演进，
  两基线数字应一致。

## 声明边界

- 仅合成数据：SCAR 主网格 + linear SAR 诊断线；固定 `n_features=5`、
  `label_frequency=0.5`、`separation=2.0`；不扫描特征维度。
- 默认构造器参数（源码构造器默认值）；uPU 的既有调参变体
  （`loss=double_hinge, reg_lambda=0.01`）只作 `existing_tuned_reference` 附加
  变体，不替代默认参数基线。
- `y_true` 只用于最终 oracle 评测列，禁止任何参数/阈值/早停选择触碰它。
- PNU 不参与纯 PU 方法横向排名；SAR 单元从不参与排名；LDCE/KLDCE 不进入 SAR 线。
- 超时阈值冻结前，所有 `timeouts` 均为宽松保护上限，不构成性能预算声明。
