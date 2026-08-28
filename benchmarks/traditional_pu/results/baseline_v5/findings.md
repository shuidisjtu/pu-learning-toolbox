# v5 正式基线:第 6 步写回第 2 轮(uPU loss='squared')

20-seed confirmation(种子 100..119),配置 `configs/seven_methods_pu_baseline_v5.json`。
ADR-0016 第 6 步写回第 2 轮:uPU 默认 `loss` 由 `"double_hinge"` 调优为
`"squared"`(upu_tuning_r1 轮,12/12 配对 CI confirmed:6 SCAR + 6 SAR,diff
−0.035..−0.163,CI 上界全 < 0;闭式解,耗时 ~4× 更低)。LDCE 保留 v4 写回
组合。其余方法参数不变。

## 写回正确性论证(源码级 + 逐单元)

- **源码默认已更新**(upu.py 构造器):`loss="squared"`,docstring 注明调优
  依据;默认值契约测试(TestWritebackDefaults,3 方法)锁定新默认;既有
  元数据测试同步更新。
- **重锁**:v4 摘除 `locks_source_defaults`(冻结历史快照,先例链
  v1/v2/v3),v5 配置 `locks_source_defaults=True` 钉住新默认;门禁测试
  基准 v4→v5;`check_baseline_configs` 通过。
- **companion 审计**(写回核心证据):`upu_baseline_v5` 确认重跑 240/240
  success、0 退化,与 `upu_tuning_r1/conf/r4_loss_squared` 候选行逐单元
  **max|diff|=0**(26 数值列,status/params 全等)——新默认的行为与已验证
  候选完全相同,12/12 confirmed 结论直接迁移到新默认。

## 结果(20 seeds,1200/1200 success)

| 方法 | 行数 | 来源 |
|---|---|---|
| upu | 240 | baseline_v5_upu 确认重跑(risk −0.217..+0.053,mean −0.098) |
| 其余 5 方法 | 960 | 参数未变、数据/种子未变 → 确定性继承 v4 行(零重跑) |

- 退化:upu 0/240;其余方法继承 v4(全 0)。
- 非 upu 行与 v4 逐单元一致(参数、数据、种子、协议全同,确定性运行);
  其中 ldce 行 = v4 写回组合,已由上一轮审计锁定。

## 与 v4 的关系

- v4 config/results 保留为 frozen history snapshot(写回前基线,先例链
  v1/v2/v3),其 `locks_source_defaults` 摘除。
- 第 6 步写回第 3 轮(elkan_noto mode='weighted_retraining')在 v5 基础上
  继续,升 v6 并重锁。

## 遗留

- 写回后的最终全量网格报告(README 结果表)待 elkan_noto 写回完成后统一
  更新;
- 第 6 步写回第 3 轮(elkan_noto)紧随本轮,流程同前两轮。
