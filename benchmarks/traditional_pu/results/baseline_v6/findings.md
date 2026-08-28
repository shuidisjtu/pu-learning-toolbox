# v6 正式基线:第 6 步写回第 3 轮(elkan_noto mode='weighted_retraining')

20-seed confirmation(种子 100..119),配置 `configs/seven_methods_pu_baseline_v6.json`。
ADR-0016 第 6 步写回第 3 轮(末轮):elkan_noto 默认 `mode` 由
`"probability_correction"` 调优为 `"weighted_retraining"`(elkan_noto_tuning_r2
轮,12/12 配对 CI confirmed:6 SCAR + 6 SAR,diff −0.009..−0.067,CI 上界全 <
0;改善归因 mode 本身——weighted 模式下 calibration_method 不进入最终预测)。
LDCE 保留 v4 组合、uPU 保留 v5 squared。其余方法参数不变。

## 写回正确性论证(源码级 + 逐单元)

- **源码默认已更新**(elkan_noto.py 构造器):`mode="weighted_retraining"`,
  docstring 注明调优依据;默认值契约测试(TestWritebackDefaults,3 方法)
  锁定新默认;4 个隐式依赖默认 probability_correction 行为的测试显式传
  mode 保持原意(weighted 下 `predict_label_proba` 返回 None、非校准估计器
  的重复采样权重可为负,均属 mode 语义差异)。
- **重锁**:v5 摘除 `locks_source_defaults`(冻结历史快照,先例链 v1..v5),
  v6 配置 `locks_source_defaults=True` 钉住新默认;门禁测试基准 v5→v6;
  `check_baseline_configs` 通过。
- **companion 审计**(写回核心证据):`elkan_noto_baseline_v6` 确认重跑
  240/240 success、0 退化,与 `elkan_noto_tuning_r2/conf/r2_weighted` 候选
  行逐单元 **max|diff|=0**(25 数值列,params/status 全等)——新默认的行为与
  已验证候选完全相同,12/12 confirmed 结论直接迁移到新默认。附带验证:
  r2_isotonic_weighted 数值亦全一致(params 仅 calibration_method 不同),
  实证复现 r2 结构性发现(weighted 下校准方法不进最终预测)。

## 结果(20 seeds,1200/1200 success)

| 方法 | 行数 | 来源 |
|---|---|---|
| elkan_noto | 240 | baseline_v6_elkan_noto 确认重跑(risk −0.218..+0.096,mean −0.105) |
| 其余 5 方法 | 960 | 参数未变、数据/种子未变 → 确定性继承 v5 行(零重跑) |

- 退化:elkan_noto 0/240;其余方法继承 v5(全 0)。
- 非 elkan_noto 行与 v5 逐单元一致(确定性运行);其中 upu 行 = v5 写回
  squared、ldce 行 = v4 写回组合,均由前两轮审计锁定。

## 与 v5 的关系与第 6 步收尾

- v5 config/results 保留为 frozen history snapshot(先例链 v1..v5)。
- **三个写回候选全部落地**:LDCE 组合(v4)、uPU squared(v5)、elkan_noto
  weighted(v6)。第 6 步写回完成,ADR-0016 跟进状态、README 结果表已统一
  更新,后续调优/审计以 v6 为对齐基线。

## 遗留

- PNU 三元协议独立(`pnu_baseline_v2`),不参与本轮;
- 后续任何默认值变更仍按 §7 协议:重锁升版 + 确认重跑 + companion 审计。
