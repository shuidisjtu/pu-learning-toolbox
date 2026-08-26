# Elkan-Noto 调优第 1 轮:calibration_method / n_cv_folds / eps / mode 参数簇

调优方案 §8 第 5 步第六轮(§4 表第 6 行)。dev 种子 0..4;预算 120s。**本轮在 dev
阶段即中止:主指标 `pu_zero_one_risk` 对 Elkan-Noto 失真,§4 筛选链无判别力,conf
确认不会产生任何有意义的结果(与 KLDCE 轮同款"dev 即终止"路径)。不写回默认值。**

## 网格设计

参数簇 = {`calibration_method`, `n_cv_folds`, `eps`, `mode`}(默认 sigmoid / 3 /
1e-12 / probability_correction;`base_estimator` 为 JSON 不可表达参数,本轮起加入
`generate_round_configs` 的非可调拒绝集)。基线预检已见异常:risk 恒等于 1−π
(0.9/0.7/0.5 × π=0.1/0.3/0.5)——全正或全负预测的签名之一,但基线无退化列
(step-4 前产物)无法确认。

| 候选 | calibration_method | n_cv_folds | eps | mode | 设计意图 |
|---|---|---|---|---|---|
| r6_default | sigmoid | 3 | 1e-12 | probability_correction | 参照锚点 |
| r6_isotonic | isotonic | 3 | 1e-12 | probability_correction | 标定方法换 |
| r6_cv5 / r6_cv10 | sigmoid | 5 / 10 | 1e-12 | probability_correction | CV 折数 |
| r6_eps_1em9 / r6_eps_1em6 | sigmoid | 3 | 1e-9 / 1e-6 | probability_correction | c 估计数值稳定 |
| r6_weighted | sigmoid | 3 | 1e-12 | weighted_retraining | 加权重训(结构变化) |
| r6_isotonic_weighted | isotonic | 3 | 1e-12 | weighted_retraining | 组合 |
| r6_cv10_isotonic | isotonic | 10 | 1e-12 | probability_correction | 组合 |
| r6_weighted_cv5 | sigmoid | 5 | 1e-12 | weighted_retraining | 组合 |

## dev 筛选结果:主指标失真,筛选链失效

全部 10 候选 60/60 success、零退化;但 **scar risk_mean 全部 ≈ 0.7 常数**
(0.7000 ×9,weighted 系 0.6916/0.6995)——0.7 = (1−π) 按 π∈{0.1,0.3,0.5} 六单元
的精确平均。逐 cell 复现确认:risk 精确等于 1−π(0.9/0.7/0.5),与参数变化无关。
rank 的"入选"是常数上的伪差异(weighted 与默认差 0.0084),无判别力。

## 根因调查(复现脚本,seed=0/pi0.1/small)

```
propensity_ (c): 0.6715
scores: min=0.0001 max=1.4345 frac>0=1.000
pred==1 frac: 0.055
FNR_P=0.000  FPR_U=1.000  →  risk_manual = 0.9000 = 2π·0 + 1 − π
```

三层机制:

1. `ElkanNotoClassifier._decision_function` 返回 `g(x)/c`(标签概率估计 ÷ 倾向性
   c)。g = sigmoid 输出 ∈ (0,1),c ∈ (0,1),故 **scores 恒 > 0**;
2. `pu_zero_one_risk`(du Plessis Eq. 2)假设决策阈值 0(scores>0 ⇔ 判正),对
   elkan_noto 得 FNR=0、FPR=1 → risk ≡ 1−π,与实际分类行为无关;
3. `ElkanNotoClassifier._predict` 使用 **0.5 阈值**(f≥0.5 判正,实际 5.5% 判正),
   与指标的 0 阈值脱钩——预测行为本身正常。

对照:oracle 指标完全正常(pu_auc_roc 0.997~1.0、AP 0.97~1.0)——分类器质量极好,
`g/c` 保序,排序不受影响。**问题在协议主指标的阈值语义与 elkan_noto 决策函数
语义不匹配,不在分类器本身。**

## 结论

**否定 verdict:主指标失真,调优不可进行。** 本轮在 dev 阶段终止,跳过 conf
(companion + compare 只会复制 1−π 常数)。参数簇本身未被证伪——没有任何证据说
calibration/CV/eps/mode 不可调,而是**没有可信的 PU 主指标来衡量它们**。

**协议跟进项(登记,不在本轮修复)**:

- `pu_zero_one_risk` 对 elkan_noto 的阈值语义不匹配。候选修复方向:①对
  elkan_noto 用 0.5 阈值变体(FNR=mean(scores<0.5)、FPR=mean(scores≥0.5)),
  与 `_predict` 一致;②或改用 predict 标签口径的 zero-one 估计;③或把
  `_decision_function` 改为 0 阈值语义的量(如 logit(f)−logit(c))。任一方案
  都改变协议指标定义,必须经 TDD + 契约评审,并重跑受影响轮次;
- 关联:baseline_v2 的 elkan_noto risk 同样为 1−π(基线报告中的 0.9/0.7/0.5 即
  该失真的产物,非真实分类质量),后续基线报告须注明或先修复指标;
- 该跟进与 KLDCE 轮"低先验全负"(实现级)不同:这是协议指标级问题,优先级更高
  (影响所有使用 pu_zero_one_risk 对 elkan_noto 的结论)。

## 遗留

- dev 10 候选(60 trials/候选)全量提交;无 conf 产物、无 verdict CSV(按计划
  纪律:筛选链失效即终止)。
- oracle 指标列(pu_auc_roc/AP)在全量 dev trials 中可查,证明分类质量正常,供
  跟进项引用。
- 下一轮(第 7 轮)= PNU(§4 表第 7 行:eta、reg_lambda、basis、kernel_width;
  三元网格,rank 口径需适配)。
