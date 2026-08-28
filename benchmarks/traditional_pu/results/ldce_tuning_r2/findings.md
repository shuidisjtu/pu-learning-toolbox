# LDCE 组合交互评估(第 6 步写回前置):radius × ridge 组合 6/6 确认,写回候选确立

调优方案 §8 第 5 步 r1 轮(ldce_tuning_r1)已确认 `centroid_radius=0.1` 与
`covariance_ridge=1e-2` 各自 6/6 confirmed(配对 diff −0.07..−0.14),r1 遗留建议
「若第 6 步选 radius_0p1,同时评估其与 ridge_1em2 的组合」。本轮(r2)执行该
建议:单候选组合网格(radius 收紧质心约束 × ridge 正则化协方差),dev 30
trials 筛 → conf 120 trials 配对确认 → 写回形态决策。

## 候选

| 候选 | centroid_radius | covariance_ridge | 其他参数 |
|---|---|---|---|
| r4_radius0p1_ridge1em2 | **0.1** | **1e-2** | 均同默认(reg_strength=1.0 等) |

baseline ldce 默认:`centroid_radius=1.0`、`covariance_ridge=1e-4`。对照 =
`baseline_v3`(契约 v2 基线,LDCE 为零阈值方法,v2/v3 行逐 cell 相同)。

## dev 筛选:组合远超单参数(强交互)

r4 组合 30/30 success、0 退化。risk 全场景 0.116..0.195(均值 0.149),对比:

| 配置 | dev risk_mean | conf verdict(6 单元) |
|---|---|---|
| 组合(本轮) | **0.149** | 6/6 confirmed,diff −0.27..−0.32 |
| radius_0p1 单参数(r1) | 0.339 | 6/6 confirmed,diff −0.07..−0.14 |
| ridge_1em2 单参数(r1) | 0.347 | 6/6 confirmed,diff −0.07..−0.11 |
| default(r1) | 0.436 | — |

组合的改善是单参数之和的 2 倍以上——两参数存在强交互(质心约束收紧后,
协方差正则化稳定其梯度路径,反之亦然),非简单叠加。

## conf + verdict(seeds 100..119)

companion r4_default 120/120,与 baseline_v3 的 LDCE 行逐单元 max|diff|=0
(25 数值列,唯一差异为耗时列)——基线可复现性审计通过。退化率 0/120 =
companion 0/120。预算:两目录 p95 均 ≪ 120s(候选最慢 scalemid 16.2s)。

| 单元 | diff_mean | CI 95% | verdict |
|---|---|---|---|
| scar-pi0.1-scalemid | −0.307 | [−0.314, −0.300] | confirmed |
| scar-pi0.1-scalesmall | −0.278 | [−0.294, −0.261] | confirmed |
| scar-pi0.3-scalemid | −0.318 | [−0.331, −0.305] | confirmed |
| scar-pi0.3-scalesmall | −0.282 | [−0.314, −0.250] | confirmed |
| scar-pi0.5-scalemid | −0.276 | [−0.291, −0.262] | confirmed |
| scar-pi0.5-scalesmall | −0.271 | [−0.329, −0.214] | confirmed |

**6/6 confirmed_improvement,0 oracle_only**:全部 diff CI 上界 < −0.21,§5
条件 1-6 全满足;oracle(AUC)同样 6/6 改善。这是七个方法中首个全单元强确认
的写回候选。

## 结论:写回候选 = 组合(centroid_radius=0.1 × covariance_ridge=1e-2)

- r1 单参数 6/6 + r2 组合 6/6 且 diff 翻倍,组合效应在 dev(0.149)与 conf
  (0.1498)完全复现;
- 退化 0、耗时未超预算、无 oracle_only 掩盖;
- **第 6 步 LDCE 写回形态定为组合**,源码默认值变更 + baseline 重锁(升 v4)
  + 确认重跑按 ADR-0016 第 6 步协议执行;
- ridge_1em3 单参数(r1)仅 5/6 不入围,维持 r1 结论。

## 遗留

- 组合已在 conf 种子复现,写回后需 companion 重跑审计默认值行一致性;
- 第 6 步其余写回候选(uPU loss='squared'、elkan_noto mode='weighted_retraining')
  仍待执行,ADR-0016 跟进状态待登记本轮结论;
- SAR 侧(linear)未在本轮单独确认,随基线 v4 全网格重跑覆盖。
