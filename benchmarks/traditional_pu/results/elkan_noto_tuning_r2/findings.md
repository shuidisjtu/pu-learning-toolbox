# Elkan-Noto 调优第 1 轮重跑(契约 v2 口径):mode=weighted_retraining 12/12 晋级

调优方案 §8 第 5 步第六轮的重跑。第 6 轮(r1)因主指标失真在 dev 阶段中止:
`pu_zero_one_risk` 旧口径对原始 score 用固定零阈值,而 Elkan-Noto 的
`_decision_function` 返回概率尺度 g/c(恒 >0),导致 risk ≡ 1−π。指标契约升 v2
(def1544)后 risk 跟随各分类器原生 `predict()` 阈值(Elkan-Noto 为 0.5),本轮在
新口径 + 新基线(baseline_v3)下重跑同一 10 候选参数表。r1 产物保留为旧口径
证据,不覆盖。

## 网格设计(与 r1 相同)

参数簇 = {`calibration_method`, `n_cv_folds`, `eps`, `mode`}(默认 sigmoid / 3 /
1e-12 / probability_correction)。

| 候选 | calibration_method | n_cv_folds | eps | mode |
|---|---|---|---|---|
| r2_default | sigmoid | 3 | 1e-12 | probability_correction |
| r2_isotonic | isotonic | 3 | 1e-12 | probability_correction |
| r2_cv5 / r2_cv10 | sigmoid | 5 / 10 | 1e-12 | probability_correction |
| r2_eps_1em9 / r2_eps_1em6 | sigmoid | 3 | 1e-9 / 1e-6 | probability_correction |
| r2_weighted | sigmoid | 3 | 1e-12 | weighted_retraining |
| r2_isotonic_weighted | isotonic | 3 | 1e-12 | weighted_retraining |
| r2_cv10_isotonic | isotonic | 10 | 1e-12 | probability_correction |
| r2_weighted_cv5 | sigmoid | 5 | 1e-12 | weighted_retraining |

## dev 筛选:新口径恢复判别力

全部 10 候选 60/60 success、零退化。scar risk_mean 不再常数,恢复判别力:

| 排名 | 候选 | scar risk_mean | 备注 |
|---|---|---|---|
| 1 | r2_isotonic_weighted | −0.0827 | weighted × isotonic |
| 2 | r2_weighted | −0.0827 | weighted(sigmoid) |
| 3 | r2_weighted_cv5 | −0.0809 | weighted × cv5 |
| 4 | r2_default | −0.0331 | 默认锚点 |
| 5-7 | r2_eps_* / r2_isotonic | −0.0331 | eps 与标定方法单换零效应 |
| 8-10 | r2_cv5 / r2_cv10 / r2_cv10_isotonic | −0.0278 / −0.0227 / −0.0227 | CV 折数增大轻微劣化 |

信号方向清晰:改善完全来自 `mode=weighted_retraining`;calibration_method 与
eps 在 probability_correction 模式零效应,CV 折数增大有害。

## conf + verdicts(20 seeds 100..119)

companion r2_default 240/240,与 baseline_v3 的 elkan_noto 行逐单元 max|diff|=0
(24 个数值列,唯一差异为耗时列硬件噪声)——新口径基线可复现性审计通过。
退化率 0/240 = companion 0/240。

| 候选 | verdict | 细节 |
|---|---|---|
| r2_weighted | **12/12 confirmed_improvement** | 6 SCAR + 6 SAR 全部严格改善;diff −0.009..−0.067(SCAR −0.022..−0.067,随先验递增;SAR −0.009..−0.024) |
| r2_isotonic_weighted | **12/12 confirmed_improvement** | 与 r2_weighted 数字逐 cell 完全相同 |
| r2_weighted_cv5 | 10/12(SCAR 6/6) | sar-pi0.1-small / sar-pi0.5-small 两单元 CI 上界略超 0(+0.0012/+0.0034) |

§5 条件核对(r2_weighted):①12/12 配对 CI 严格改善;②成功率 100% = 100%;
③退化率 0 = companion 0;④P95 耗时 0.093s ≪ 120s 预算;⑤协议与口径一致
(同 config 派生、同 seed、契约 v2);⑥确认种子复现。全部满足。

结构性发现:`r2_isotonic_weighted` 与 `r2_weighted` 数字逐 cell 完全相同——
weighted_retraining 模式下 calibration_method 不参与最终预测(加权重训路径
不经过 sigmoid/isotonic 标定),改善完全归因于 mode 本身。写回候选应取
`mode=weighted_retraining`(其余默认),不带 isotonic。

## 结论

第 6 轮在修复后的指标口径下产出**正面 verdict**:`mode=weighted_retraining`
12/12 全单元确认改善——Elkan-Noto 参数簇本身未被证伪,且找到比默认
probability_correction 更优的结构模式。写回默认值属第 6 步决策(§7:须重锁
基线 + 确认种子重跑),本轮不写回;r1 保留为旧口径证据,ADR-0016 跟进状态
已登记口径修复与重跑关系。

## 遗留

- weighted_retraining 模式下校准方法的零效应值得实现级复核(是否 weight 路径
  确实绕过标定),但不影响 verdict(两候选逐 cell 一致)。
- 若第 6 步决定写回 `mode=weighted_retraining`,baseline_v3 需重锁并重跑
  elkan_noto 行(§7 前置条件)。
