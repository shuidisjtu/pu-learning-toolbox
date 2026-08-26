# nnPU 调优第 1 轮:beta/gamma、batch_size、max_epochs/patience 参数簇

调优方案 §8 第 5 步第三轮(§4 表第 3 行)。dev 种子 0..4、conf 种子 100..119;
预算 120s;对照 = `r3_default` companion(冻结基线参数的 nnpu-only 视图)。本轮只产出
verdict,不写回源码默认值(§7)。

nnPU 进入 SAR 诊断线(契约 §2.3),故 dev/conf 网格为 **60/240 trials**(12 cells:
6 SCAR + 6 SAR),为七轮循环中首个 60-trial dev 网格。`rank_candidates` 相应扩展:
SCAR 行参与 §4 筛选链,SAR 行只作诊断统计(`sar_n_total`/`sar_n_success` 列),与
"SAR 单元从不参与排名"的契约口径一致(本轮 TDD 完成,测试 15→20)。

## 网格设计

参数簇 = {`beta`, `gamma`, `batch_size`, `max_epochs`, `patience`}(默认 0.0 / 1.0 /
256 / 200 / 20;`model`/`optimizer`/`device` 为 JSON 不可表达参数,本轮起
`generate_round_configs` 显式拒绝对其 override 并排除出构造器默认覆盖校验)。
该实现的 nnPU 为 Kiryo et al. 2017 Algorithm 1:修正分支激活条件为
`r = R_u⁻ − π·R_p⁻ < −β`,γ 为修正分支步长折扣(γ=0 时 opt_loss=0,不取)。

关键机制约束:**r ≥ −π 恒成立**(R_u⁻ ≥ 0、R_p⁻ ≤ 1 ⇒ r ≥ −π),因此 β ≥ π 的
单元修正分支数学上永不激活。π=0.1 cells 中 β ≥ 0.1 等价于纯 uPU——β=0.05 是
π=0.1 cells 内唯一真正改变修正行为的探针,β=0.1/0.25 在该 cells 预期与彼此一致。

| 候选 | beta | gamma | batch_size | max_epochs | patience | 设计意图 |
|---|---|---|---|---|---|---|
| r3_default | 0.0 | 1.0 | 256 | 200 | 20 | 参照锚点 + companion 来源 |
| r3_beta_0p05 | 0.05 | 1.0 | 256 | 200 | 20 | 修正阈值微抬(π=0.1 安全线内) |
| r3_beta_0p1 | 0.1 | 1.0 | 256 | 200 | 20 | 修正阈值 = π=0.1 边界(该 cells 即纯 uPU) |
| r3_beta_0p25 | 0.25 | 1.0 | 256 | 200 | 20 | 阈值抬升(π=0.1/0.3 cells 均纯 uPU 化) |
| r3_gamma_0p5 | 0.0 | 0.5 | 256 | 200 | 20 | 修正步长半力度 |
| r3_gamma_0p1 | 0.0 | 0.1 | 256 | 200 | 20 | 修正弱(近似 uPU) |
| r3_bs_64 | 0.0 | 1.0 | 64 | 200 | 20 | 小批噪声(P 批受 n_P 截断) |
| r3_bs_1024 | 0.0 | 1.0 | 1024 | 200 | 20 | 大批(mid U 批近全量) |
| r3_ep500_pat50 | 0.0 | 1.0 | 256 | 500 | 50 | 训练预算加大(基线 2.5s 已早停,验证余量) |
| r3_beta0p05_gamma0p5 | 0.05 | 0.5 | 256 | 200 | 20 | 组合:阈值微抬 + 半力度 |

## dev 筛选结果(§4 筛选链,SCAR 行口径)

全部 10 候选 60/60 success(30 SCAR + 30 SAR),零退化,淘汰链无命中。

| 候选 | scar risk_mean | P95 耗时(s) | 筛选结论 |
|---|---|---|---|
| r3_ep500_pat50 | −0.078612 | 8.9 | 入选(第 1) |
| r3_beta_0p25 | −0.078176 | 4.1 | 入选(第 2) |
| r3_beta_0p1 | −0.077771 | 4.8 | 入选(第 3) |
| r3_beta0p05_gamma0p5 | −0.076369 | 1.9 | 第 4,未入选 |
| r3_bs_64 | −0.075870 | 9.1 | 第 5,未入选 |
| r3_beta_0p05 | −0.075414 | 3.5 | 第 6,未入选 |
| r3_gamma_0p1 | −0.074002 | 3.8 | 第 7 |
| r3_gamma_0p5 | −0.071476 | 3.6 | 第 8 |
| r3_default | −0.070162 | 3.6 | 第 9 |
| r3_bs_1024 | +0.031869 | 2.5 | 第 10(显著变差) |

观察:①β 增大方向单调改善(0 → 0.05 → 0.1 → 0.25 对应 −0.0702 → −0.0754 →
−0.0778 → −0.0782),修正阈值抬升对 pi0.3/pi0.5 cells 有利;②`ep500_pat50`
与 β=0.25 并列第一——基线 2.5s 早停虽已收敛,加大训练预算仍带来微小增益;
③`bs_1024` 显著变差(+0.10 量级,大批次失效);④γ 单独影响微弱(0.5/0.1 仅
±0.004);⑤bs_64 略好于默认但 P95 耗时 2.5×(9.1s)。dev 差距总体很小
(±0.008),严格配对 CI 能否确认是 conf 阶段的关键看点。

## companion 一致性审计

`r3_default`(冻结基线参数的 nnpu-only 视图)conf 240/240 success,与 `baseline_v2`
的 nnpu 行在 **13 个公共数值指标列上 max|diff| = 0.0**——当前 runner 确定性成立,
companion 可作为 compare 的对齐基线(§5 条件 5 协议一致由工具级键对齐保证)。

## conf verdict(compare.py,严格配对 CI 口径)

全部 240/240 success、P95 < 10s(预算 120s 宽裕)、退化率 0/240 = 基线 0/240
(§5 条件 2/3/4 通过)。SCAR 主网格 6 cells 是晋级依据(契约:SAR 从不参与排名):

| cell | r3_ep500_pat50 | r3_beta_0p25 | r3_beta_0p1 |
|---|---|---|---|
| scar-pi0.1-small | **confirmed**(−0.0166) | 无变化(diff=0) | 无变化(diff=0) |
| scar-pi0.1-mid | oracle_only(−0.0005,CI 含 0) | oracle_only(−0.0007,CI 含 0) | oracle_only(−0.0007,CI 含 0) |
| scar-pi0.3-small | **confirmed**(−0.0262) | oracle_only(−0.0012,CI 含 0) | oracle_only(−0.0012,CI 含 0) |
| scar-pi0.3-mid | oracle_only(**+0.0089 变差**) | **confirmed**(−0.0101) | **confirmed**(−0.0099) |
| scar-pi0.5-small | oracle_only(−0.0060,CI 含 0) | **confirmed**(−0.0044) | **confirmed**(−0.0042) |
| scar-pi0.5-mid | oracle_only(**+0.0191 变差**) | **confirmed**(−0.0313) | **confirmed**(−0.0285) |

SAR 诊断线(不参与排名):ep500_pat50 为 2 confirmed + 4 变差(small 改善、mid
全面变差,与 SCAR 同格局);β=0.25/0.1 为 4 confirmed + 2 无变化,**零变差 cell**。

- **r3_ep500_pat50**:SCAR 2/6 严格改善(均为 small),mid 3 cells PU 变差
  (oracle 指标改善,按 §5 属 oracle_only,不得宣称 PU 调优成功)——加大训练预算
  在 mid 出现过拟合信号,方向混杂,不推荐。
- **r3_beta_0p25 / r3_beta_0p1**:SCAR 3/6 严格改善(pi0.3-mid、pi0.5-mid、
  pi0.5-small),2 cells oracle_only(pi0.1-mid、pi0.3-small,方向改善但 CI 含 0),
  pi0.1-small 与默认**精确无差异**(diff=0.000000)——β ≥ π 时修正分支永不激活的
  机制预测被数据验证(β=0.1 与 β=0.25 在该 cell 训练路径完全一致)。β=0.1 与
  β=0.25 幅度接近,0.1 已是饱和点。

## 结论

**无全单元晋级候选(§5 条件 1 未满足),但参数簇存在方向性信号——部分改善 verdict,
不写回默认值。**

- 三个入选者在 SCAR 主网格均未达 6/6 `confirmed_improvement`,不构成
  `hyperparameter_tuning` 晋级(对照 LDCE 轮 radius_0p1/ridge_1em2 的全 6/6);
- **β 增大是正方向**:0.1/0.25 在 3/6 SCAR + 4/6 SAR cells 严格改善,零变差 cell;
  但低先验 cells 机制性无响应(β ≥ π 永不修正),这是该实现的固有边界而非可调
  超参问题——低先验单元若要改善须 β ∈ (0, π) 内取值(本轮 β=0.05 dev 已显示
  介于默认与 β=0.1 之间的单调位置,未进 conf);
- **ep500_pat50 方向混杂**(small 改善、mid 变差),不推荐;
- γ、batch_size 无方向性价值(dev 已排除,bs_1024 显著变差为边界记录)。

**写回讨论(§7,第 6 步)**:无全单元晋级,不写回。若未来在低先验修复后重开 nnPU
轮,网格建议以 β ∈ (0.05, 0.25) 细分为主线,并考虑 β 随 π 自适应(实现层改动,
非 JSON 可表达)。

## 遗留

- dev 10 候选(60 trials/候选)+ conf 4 目录(companion + 3 入选,240 trials/
  目录)全量提交;P95 全部 < 10s。
- verdict CSV ×3 全量提交;oracle_only cells 已如实分类,未宣称 PU 调优成功。
- 环境备注:本轮会话 Bash 工具固定 2 分钟超时,dev 循环通过断点续跑滚动完成,
  无数据丢失(增量落盘协议生效);conf 由用户在独立终端一次跑完。
- 下一轮(第 4 轮)= uPU(§4 表第 4 行:loss、reg_lambda、basis、kernel width)。
