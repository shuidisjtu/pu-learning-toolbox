# LLSVM 调优第 1 轮:learning_rate / reg_lambda / gamma / patience+min_epochs 参数簇

调优方案 §8 第 5 步第五轮(§4 表第 5 行)。dev 种子 0..4、conf 种子 100..119;
预算 120s;对照 = `r5_default` companion(冻结基线参数的 llsvm-only 视图)。本轮只
产出 verdict,不写回源码默认值(§7)。

LLSVM 进 SAR 诊断线,dev/conf 网格 60/240 trials(rank 筛选链只用 30 个 SCAR 行,
SAR 行作诊断)。构造器全部参数为 JSON 可表达数值(学习率 `learning_rate` 是普通
float,与 nnPU 的 optimizer 不同,可直接进候选网格)。

## 网格设计

参数簇 = {`learning_rate`, `reg_lambda`, `gamma`, `patience`/`min_epochs`}(默认
5e-6 / 1.0 / 10.0 / 100+200;非凸 SGD,重点风险 = 种子方差与训练成本)。基线弱点
定位:scar-pi0.1-small 是唯一 risk 为正的 cell(+0.0061,其余全部负),lr/reg 探针
按"低先验拟合不足"设计。

| 候选 | learning_rate | reg_lambda | gamma | patience/min_epochs | 设计意图 |
|---|---|---|---|---|---|
| r5_default | 5e-6 | 1.0 | 10 | 100/200 | 参照锚点 + companion 来源 |
| r5_lr_1em5 | 1e-5 | 1.0 | 10 | 100/200 | lr ×2 |
| r5_lr_2em5 | 2e-5 | 1.0 | 10 | 100/200 | lr ×4(主探针) |
| r5_lr_2em6 | 2e-6 | 1.0 | 10 | 100/200 | lr ÷2.5(反方向对照) |
| r5_reg_0p1 | 5e-6 | 0.1 | 10 | 100/200 | 正则收紧 ×10 |
| r5_reg_10 | 5e-6 | 10.0 | 10 | 100/200 | 正则放大 ×10 |
| r5_gamma_5 | 5e-6 | 1.0 | 5 | 100/200 | 损失尺度减半 |
| r5_gamma_20 | 5e-6 | 1.0 | 20 | 100/200 | 损失尺度翻倍 |
| r5_pat200_min400 | 5e-6 | 1.0 | 10 | 200/400 | 训练预算加大 |
| r5_lr1em5_reg0p1 | 1e-5 | 0.1 | 10 | 100/200 | lr×2 + 正则收紧组合 |

## dev 筛选结果(§4 筛选链,SCAR 行口径)

全部 10 候选 60/60 success(30 SCAR + 30 SAR),零退化,淘汰链无命中。

| 候选 | scar risk_mean | P95 耗时(s) | 筛选结论 |
|---|---|---|---|
| r5_lr_2em5 | −0.060948 | 0.8 | 入选(第 1) |
| r5_lr1em5_reg0p1 | −0.059949 | 3.2 | 入选(第 2) |
| r5_pat200_min400 | −0.059349 | 4.1 | 入选(第 3) |
| r5_lr_1em5 | −0.055887 | 1.5 | 第 4,未入选 |
| r5_default | −0.055214 | 2.3 | 第 5 |
| r5_reg_10 | −0.054781 | 1.1 | 第 6 |
| r5_reg_0p1 | −0.048461 | 2.8 | 第 7 |
| r5_gamma_5 | −0.025578 | 2.8 | 第 8 |
| r5_lr_2em6 | −0.009241 | 2.8 | 第 9 |
| r5_gamma_20 | −0.008549 | 2.2 | 第 10 |

观察:①lr 方向单调(2e-5 > 1e-5 > 5e-6 > 2e-6),lr 放大同时更快(0.8s vs 2.3s);
②γ 减小/增大均显著有害(−0.026/−0.009);③reg 单独影响小,但 lr1em5×reg0p1 组合
优于 lr1em5 单独;④前三名 dev 差距仅 ~0.006——5-seed 筛选对高方差方法分辨率不足。

## companion 一致性审计

`r5_default`(冻结基线参数的 llsvm-only 视图)conf 240/240 success,与 `baseline_v2`
的 llsvm 行在 **13 个公共数值指标列上 max|diff| = 0.0**——runner 确定性成立,
companion 可作为 compare 的对齐基线。

## conf verdict(compare.py,严格配对 CI 口径)

全部 240/240 success;P95 ≤ 2.3s(预算 120s 宽裕);退化率 0/240 = 基线 0/240
(§5 条件 2/3/4 通过)。SCAR 主网格 6 cells 是晋级依据:

| cell | r5_lr_2em5 | r5_lr1em5_reg0p1 | r5_pat200_min400 |
|---|---|---|---|
| scar-pi0.1-small | CI 含 0(−0.0078) | CI 含 0(−0.0088) | CI 含 0(−0.0078) |
| scar-pi0.1-mid | **变差 +0.0473(CI 高 +0.147)** | **confirmed**(−0.0010) | CI 含 0(−0.0001) |
| scar-pi0.3-small | CI 含 0(−0.0054) | CI 含 0(−0.0070) | CI 含 0(−0.0051) |
| scar-pi0.3-mid | **confirmed**(−0.0047) | **confirmed**(−0.0031) | CI 含 0(−0.0021) |
| scar-pi0.5-small | **confirmed**(−0.0212) | CI 含 0(+0.0013) | CI 含 0(−0.0049) |
| scar-pi0.5-mid | **confirmed**(−0.0092) | **confirmed**(−0.0042) | **confirmed**(−0.0042) |

SAR 诊断线:lr_2em5 为 3 confirmed + 3 CI 含 0;lr1em5_reg0p1 为 2 confirmed +
4 CI 含 0;pat200_min400 为 1 confirmed + **sar-pi0.3-mid 显著变差(+0.039,CI 高
+0.127)**。

- **r5_lr_2em5**:3/6 SCAR confirmed,但 scar-pi0.1-mid 显著变差(+0.047,种子方差
  大,CI 宽至 +0.147)——lr 放大在低先验 mid 放大了 SGD 的种子方差,dev 第 1 的
  排名在 20-seed conf 下未站稳,不推荐;
- **r5_lr1em5_reg0p1**:3/6 SCAR confirmed、零变差 cell,是最稳的候选,但改善幅度
  小(−0.001 至 −0.004),未达全单元;
- **r5_pat200_min400**:1/6 SCAR + sar-pi0.3-mid 显著变差,训练预算加大无价值。

## 结论

**无全单元晋级候选(§5 条件 1 未满足)——部分改善 verdict,不写回默认值。**

- 三个入选者最高仅 3/6 SCAR cells `confirmed_improvement`,不构成晋级;
- **dev 筛选的乐观性偏差首次被 conf 证伪**:r5_lr_2em5 在 5-seed dev 排第 1
  (risk −0.0609),但 20-seed conf 中低先验 mid 显著变差(+0.047)——LLSVM 非凸
  SGD 的种子方差使 30-trial dev 的分辨率不足(前三名 dev 差距 <0.006 即已提示);
  该偏差是协议设计内预期的(dev 只是筛选,晋级必须过 conf 严格 CI),但 LLSVM 轮
  首次实际触发;
- lr 方向(放大)有 dev 级信号但与 conf 稳定性冲突;γ 减小有害;reg 组合小幅改善;
- 唯一 oracle_only cells 已如实分类,未宣称 PU 调优成功。

**写回讨论(§7,第 6 步)**:无晋级候选,不写回。若未来重开 LLSVM 轮,建议先解决
种子方差(如 dev 种子数增加或 lr 更细网格 1e-5~2e-5),再评估 lr1em5_reg0p1 型
组合。

## 遗留

- dev 10 候选(60 trials/候选)+ conf 4 目录(companion + 3 入选,240 trials/
  目录)全量提交;P95 全部 < 5s。
- verdict CSV ×3 全量提交;oracle_only cells 已如实分类。
- 下一轮(第 6 轮)= Elkan-Noto(§4 表第 6 行:calibration method、CV 折数、
  eps)。
