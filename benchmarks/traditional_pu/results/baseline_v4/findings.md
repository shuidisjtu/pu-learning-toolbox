# v4 正式基线:第 6 步写回第 1 轮(LDCE 组合默认值)

20-seed confirmation(种子 100..119),配置 `configs/seven_methods_pu_baseline_v4.json`。
ADR-0016 第 6 步写回第 1 轮:LDCE 默认值由 `centroid_radius=1.0 ×
covariance_ridge=1e-4` 调优为已确认组合 `centroid_radius=0.1 ×
covariance_ridge=1e-2`(ldce_tuning_r2 轮,6/6 配对 CI confirmed,risk
0.436→0.150,组合增益为单参数 2 倍以上)。其余五方法参数不变。

## 写回正确性论证(源码级 + 逐单元)

- **源码默认已更新**(ldce.py 构造器):`centroid_radius=0.1`、
  `covariance_ridge=1e-2`,docstring 注明调优依据;默认值契约测试
  (TestWritebackDefaults,3 方法)锁定新默认,防止静默漂移。
- **重锁**:v2/v3 配置摘除 `locks_source_defaults`(冻结历史快照,v1 先例),
  v4 配置 `locks_source_defaults=True` 钉住新默认;`check_baseline_configs`
  门禁通过,门禁测试基准 v2→v4。
- **companion 审计**(写回核心证据):`ldce_baseline_v4` 确认重跑 120/120
  success、0 退化,与 `ldce_tuning_r2/conf/r4_radius0p1_ridge1em2` 候选行
  逐单元 **max|diff|=0**(25 数值列,status/params/degenerate 全等)——新默认
  的行为与已验证候选完全相同,6/6 confirmed 结论直接迁移到新默认。

## 结果(20 seeds,1200/1200 success)

| 方法 | 行数 | 来源 |
|---|---|---|
| ldce | 120 | baseline_v4_ldce 确认重跑(risk 0.116..0.196,mean 0.150) |
| upu / nnpu / llsvm / elkan_noto / kldce | 各 240/120 | 参数未变、数据/种子未变 → 确定性继承 v3 行(零重跑) |

- 退化:ldce 0/120;其余方法继承 v3(elkan_noto 0/240,ldce/kldce 修复后 0/120)。
- 非 ldce 行与 v3 逐单元一致(参数、数据、种子、协议全同,确定性运行)。

## 与 v3 的关系

- v3 config/results 保留为 frozen history snapshot(写回前基线,与 v1/v2
  先例一致);其 `locks_source_defaults` 摘除,不再钉住当前源码默认。
- 后续第 6 步写回轮(uPU loss='squared'、elkan_noto mode='weighted_retraining')
  在 v4 基础上继续,每轮升一个版本并重锁。

## 遗留

- 写回后的最终全量网格报告(README 结果表)待三个写回候选全部落地后统一
  更新(避免每轮重写);
- 第 6 步写回第 2 轮(uPU)紧随本轮,流程同此轮。
