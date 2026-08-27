# KLDCE 调优第 1 轮重跑(修复后,b₀ 类对称):参数簇无可调增益

调优方案 §8 第 5 步第一轮的重跑。第 1 轮(r1)在旧 b₀ 恢复语义下被低先验全负
预测毁掉:10 候选全过 success 但零候选通过 §3.3 退化筛选,参数簇被否定。根因
修复(2026-08-27,类对称 b₀)后,退化现象消失,本轮在修复后的默认语义
(kldce_baseline_v3,与 baseline_v3 的 KLDCE 行一致)下重跑**同一 10 候选网格**。

## 网格设计(与 r1 相同)

参数簇 = {`covariance_ridge`, `reg_strength`, `centroid_radius`}(默认 0.0 / 1.0 /
1.0)。

| 候选 | covariance_ridge | reg_strength | centroid_radius |
|---|---|---|---|
| r3_default | 0.0 | 1.0 | 1.0 |
| r3_ridge_0p01 / r3_ridge_0p1 | 0.01 / 0.1 | 1.0 | 1.0 |
| r3_reg_0p1 / r3_reg_10 | 0.0 | 0.1 / 10.0 | 1.0 |
| r3_radius_0p1 / r3_radius_10 | 0.0 | 1.0 | 0.1 / 10.0 |
| r3_ridge0p1_reg0p1 | 0.1 | 0.1 | 1.0 |
| r3_ridge0p1_radius0p1 | 0.1 | 1.0 | 0.1 |
| r3_reg0p1_radius10 | 0.0 | 0.1 | 10.0 |

## dev 筛选:退化消失,参数效应消失

全部 10 候选 30/30 success、**零退化**(r1 为全部退化,§3.3 全灭)。risk 恢复
判别力但候选间差异极小(0.3174..0.3207,30-seed 均值的波动级别):
`reg_strength` 是唯一有微弱效应的参数(10 → 0.3174、0.1 → 0.3187,默认
0.3207);`covariance_ridge`、`centroid_radius` 及组合全部与默认 risk 相同。
前 3 入选:`r3_reg_10`、`r3_reg0p1_radius10`、`r3_reg_0p1`。

## conf + verdicts(20 seeds 100..119)

companion r3_default 120/120,与 baseline_v3 的 KLDCE 行逐单元 max|diff|=0
(24 数值列,唯一差异耗时列)——修复后基线可复现性审计通过。退化率 0/120 =
companion 0/120。预算:r3_default p95 49.9s,候选 16.9–39.2s,均 ≪ 120s。

| 候选 | verdict | 细节 |
|---|---|---|
| r3_reg_0p1 | 0/6 confirmed,1 oracle_only | 全部 diff −0.006..+0.009,CI 跨 0;pi=0.5 两单元 diff 精确 0 |
| r3_reg_10 | 0/6 confirmed,0 oracle_only | 全部 diff −0.012..+0.011,CI 跨 0;pi=0.5 两单元 diff 精确 0 |
| r3_reg0p1_radius10 | 0/6 confirmed,1 oracle_only | 与 r3_reg_0p1 逐 cell 相同(radius 无效应) |

## 结论

修复后 KLDCE 默认参数即为(本合成网格上的)有效工作点:退化消除、risk 0.126..0.458
随先验单调、recall 0.59..0.79(pi=0.1/0.3),且该参数簇在确认种子上无可调增益
(0/6 confirmed)。r1 的"参数簇无法清除低先验全负"结论已被修复本身取代——问题
不在参数簇而在 b₀ 语义;修复已写回实现,默认参数无需变更。本轮无写回候选
(§5 条件 1 未满足,oracle_only 不构成 PU 调优成功)。

## 遗留

- `centroid_radius` 在质心凸起不可达时完全无效应(pi=0.5 边界),实现层面可保留;
- baseline_v3 的 KLDCE 行已由 kldce_baseline_v3 重跑替换(240→120 行,
  SCAR 6 场景,退化 80/960→0),见 baseline_v3 findings 更新;
- 第 6 步写回候选名单不变(本参数簇无候选),ADR-0016 跟进状态已登记。
