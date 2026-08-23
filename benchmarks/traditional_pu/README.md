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

## 命令

开发期（5 seeds 0..4，用于协议修复与验证）：

```bash
uv run python -m benchmarks.traditional_pu.run \
  --config benchmarks/traditional_pu/configs/seven_methods_pu_baseline_v1.json \
  --seed-set development \
  --results-dir benchmarks/traditional_pu/results/dev_baseline
```

确认期（20 seeds 100..119，正式基线）：

```bash
uv run python -m benchmarks.traditional_pu.run \
  --config benchmarks/traditional_pu/configs/seven_methods_pu_baseline_v1.json \
  --seed-set confirmation \
  --results-dir benchmarks/traditional_pu/results/baseline_v1
```

PNU 网格：

```bash
uv run python -m benchmarks.traditional_pu.run \
  --config benchmarks/traditional_pu/configs/pnu_baseline_v1.json \
  --results-dir benchmarks/traditional_pu/results/pnu_baseline
```

`--seed-set` 默认 `development`；默认按 `(algorithm, scenario, seed)` 断点恢复
（已有 `trials.csv` 时跳过已完成单元），`--no-resume` 强制重跑。

## 已知限制（跟踪中）

- 已解决：`configs/pnu_baseline_v1.json` 曾缺类先验字段使 runner
  `_iter_scenario_specs` 触发 `KeyError`；现配置显式声明 `class_priors: [0.3]`，
  即该跟踪项关闭（先验来源见 PNU 三元网格节）。
- 跟踪中：`_trial_body` 无条件读取 `data_cfg["label_frequency"]`
  （runner.py:217），pnu-only 配置（契约 §2.2 不要求该字段）会触发
  `KeyError: 'label_frequency'`；PNU 分支不使用该值。需 runner 侧对 pnu 单元
  惰性读取（Task 7/8 处理）。
- `--timeout-profile` 对没有 scar-small 单元的方法（pnu-only 配置；以及
  `seven_methods_pu_baseline_v1.json` 中的 `pnu`）会 `StopIteration`（跟踪项
  M5，最终评审 triage）；超时冻结（Task 8）在其修复后执行。

## 产物契约（契约 §5）

每个 `results/<run-name>/` 下：

- `trials.csv`：逐 trial 行——`algorithm`/`scenario`/`seed`/`status`/
  `elapsed_seconds`/`warning_count`/`failure_reason` + 全量指标列；
- `summary.csv`：成功率与均值/样本标准差/95% 置信区间；
- `resolved_config.json`：解析后配置（含 `resolved_at`）；
- `run_manifest.json`：`schema_version`/`created_at_utc`/`protocol`/
  `paper_claim=false`/`config_sha256`/`git_commit`/`git_worktree_dirty`/
  `runner_sha256`/`seed_set`/`seeds`/`n_trials`/`environment`/`limitations`；
- `report.md`：人读摘要。

## 超时与冻结流程（契约 §6）

当前 `configs/*.json` 中的 `timeouts` 是**未冻结的宽松保护上限**（统一 600s，
LLSVM 1800s）：冻结前以此宽松上限保护运行，超时单元记录 `status="timeout"` 并
完整保存超时原因，不静默丢弃。

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

## 开发期与正式基线的差异

- 开发期 5 seeds（0..4）：先运行并修复协议问题、验证网格可执行、暴露数据/指标
  缺陷；其结果不是正式基线。
- 确认期 20 seeds（100..119）：正式基线；两个集合不得重叠。
- 正式基线只接受确认种子集产物，且 `run_manifest.json` 必须记录代码 commit、
  依赖版本、种子集合、配置哈希与数据来源；`trials.csv` 原始表与摘要一并提交。

## 声明边界

- 仅合成数据：SCAR 主网格 + linear SAR 诊断线；固定 `n_features=5`、
  `label_frequency=0.5`、`separation=2.0`；不扫描特征维度。
- 默认构造器参数（源码构造器默认值）；uPU 的既有调参变体
  （`loss=double_hinge, reg_lambda=0.01`）只作 `existing_tuned_reference` 附加
  变体，不替代默认参数基线。
- `y_true` 只用于最终 oracle 评测列，禁止任何参数/阈值/早停选择触碰它。
- PNU 不参与纯 PU 方法横向排名；SAR 单元从不参与排名；LDCE/KLDCE 不进入 SAR 线。
- 超时阈值冻结前，所有 `timeouts` 均为宽松保护上限，不构成性能预算声明。
