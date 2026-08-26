# KLDCE 调优第 1 轮:covariance_ridge / reg_strength / centroid_radius 参数簇

调优方案 §8 第 5 步首轮(§4 表第 1 行)。dev 种子 0..4(30 trials/候选),conf 种子
100..119(120 trials/候选);预算 120s;对照 = `baseline_v2`(冻结基线)。本轮只产出
verdict,不写回源码默认值(§7)。

## 网格设计

参数簇 = {`covariance_ridge`, `reg_strength`, `centroid_radius`}(默认 0.0 / 1.0 / 1.0)。
`inner_tol`/`max_inner_iter` 已随实现修复(commit `cd53c17`)确定为 1e-6 / 2000,不参与搜索。
`sigma`/`mom_groups`/`max_acs_iter` 不在首轮簇内(§4)。单参数扰动 + 三组两两组合,不跑全笛卡尔积
(§4:避免不可解释的交互与过度搜索)。

| 候选 | covariance_ridge | reg_strength | centroid_radius | 设计意图 |
|---|---|---|---|---|
| r1_default | 0.0 | 1.0 | 1.0 | 参照锚点(冻结基线参数)+ companion 来源 |
| r1_ridge_0p01 | 0.01 | 1.0 | 1.0 | 协方差岭正则微扰 |
| r1_ridge_0p1 | 0.1 | 1.0 | 1.0 | 岭正则主探针(不取 ≥1.0:ridge 只连续改变质心方向,大值掩盖良态效应) |
| r1_reg_0p1 | 0.0 | 0.1 | 1.0 | 正则放大向(对抗全负预测的主方向) |
| r1_reg_10 | 0.0 | 10.0 | 1.0 | 平滑向对照格(预期加深退化) |
| r1_radius_0p1 | 0.0 | 1.0 | 0.1 | 质心收紧 |
| r1_radius_10 | 0.0 | 1.0 | 10.0 | 质心放开 |
| r1_ridge0p1_reg0p1 | 0.1 | 0.1 | 1.0 | ridge × reg |
| r1_ridge0p1_radius0p1 | 0.1 | 1.0 | 0.1 | ridge × radius |
| r1_reg0p1_radius10 | 0.0 | 0.1 | 10.0 | 低先验针对性(质心凸起 10× + 位移最远,正驱动最强组合;替换原 (0, 0.1, 0.1)——后者与两个单参数格同向,交互信息量低) |

## 关键基线观测(网格设计的依据)

`baseline_v2` 的 KLDCE 行(20 seeds × 6 cells 聚合):pi=0.1 与 pi=0.3 的全部 4 个单元
`pu_recall=0.000`、`pu_negative_rate=1.000`(全负预测);pi=0.5 两个单元 recall 仅
0.075–0.099。按 §3.3 定义,冻结基线参数在 4/6 单元产生退化预测——**参照候选
r1_default 本身就会被 dev 筛选链淘汰**,本轮的高概率结果是"参数簇无法清除低先验
全负预测"的否定 verdict。这正是 §4 表为 KLDCE 标注的重点风险(低先验全负预测)。

## dev 筛选结果(§4 筛选链)

10 候选全部 30/30 success,无超时/非收敛;筛选链唯一拦路的是退化规则:

| 候选 | n_success | n_degenerate | 退化单元 | risk_mean | P95 耗时(s) | 筛选结论 |
|---|---|---|---|---|---|---|
| r1_default | 30 | 20 | pi0.1×2 + pi0.3×2 | 0.282771 | 29.9 | 淘汰(退化) |
| r1_ridge_0p01 | 30 | 20 | 同上 | 0.282771 | 23.9 | 淘汰(退化) |
| r1_ridge_0p1 | 30 | 20 | 同上 | 0.282771 | 15.0 | 淘汰(退化) |
| r1_reg_0p1 | 30 | 12 | pi0.1×2 + pi0.3-small | 0.281641 | 38.8 | 淘汰(退化) |
| r1_reg_10 | 30 | 20 | 同默认 | 0.282771 | 16.9 | 淘汰(退化) |
| r1_radius_0p1 | 30 | 20 | 同默认 | 0.282771 | 29.3 | 淘汰(退化) |
| r1_radius_10 | 30 | 20 | 同默认 | 0.282771 | 30.4 | 淘汰(退化) |
| r1_ridge0p1_reg0p1 | 30 | 12 | 同 reg_0p1 | 0.281641 | 26.5 | 淘汰(退化) |
| r1_ridge0p1_radius0p1 | 30 | 20 | 同默认 | 0.282771 | 16.7 | 淘汰(退化) |
| r1_reg0p1_radius10 | 30 | 12 | 同 reg_0p1 | 0.281641 | 40.3 | 淘汰(退化) |

**合格候选:0**。按 §4 筛选链(任一成功单元退化即淘汰),无候选进入 conf 确认阶段,
conf/compare/companion 步骤不再执行(方案执行规则:selected 为空 → 跳过确认直接记录结论)。

### 逐单元观察(唯一的方向性信号)

| 单元 | r1_default recall | r1_reg_0p1 recall | 备注 |
|---|---|---|---|
| scar-pi0.1-scalemid | 0.000(5/5 退化) | 0.0019(4/5 退化) | λ 下调使 1 个种子出现正预测 |
| scar-pi0.1-scalesmall | 0.000(5/5) | 0.0125(4/5) | 同上 |
| scar-pi0.3-scalemid | 0.000(5/5) | 0.0047(0/5 退化) | **该单元退化被完全清除** |
| scar-pi0.3-scalesmall | 0.000(5/5) | 0.0038(4/5) | 部分缓解 |
| pi0.5 两单元 | 0.075–0.105(无退化) | 0.075–0.105(不变) | λ 不改变该区域 |

`reg_strength=0.1`(正则放大)是唯一有方向性效果的旋钮:退化 20→12/30,低先验 recall 从
全 0 升到 0.002–0.013——但距可用分类质量差两个数量级,且 `centroid_radius` 叠加无
增量(`r1_reg0p1_radius10` 与 `r1_reg_0p1` 退化分布完全一致)。`covariance_ridge`
在所有组合中零效果(risk 与退化均不变,合成数据协方差良态,ridge 分支本就不应激活)。

## 结论

**否定 verdict:首轮参数簇无法清除 KLDCE 低先验全负预测,无候选晋级,无默认值写回。**

- 结果类型:本轮不构成 `hyperparameter_tuning` 晋级;低先验全负问题(§4 表 KLDCE
  重点风险)在该参数簇内不可解,归入 `implementation_fix` 跟踪——与历史 F5(ACS 停滞,
  `results/confirmation_v1_candidates_kldce/findings.md`)同款路径:超参数探针穷尽后,
  问题指向实现层(决策函数 `b₀` 先验恢复 / 低先验下质心核凸起项失活)。
- 全负预测下 `pu_zero_one_risk` 对参数几乎不变(0.2828 vs 0.2816),该指标本身不能
  区分全负退化与真实分类质量——退化标记(§3.3)在本轮承担了全部筛选功能,验证了
  §8 第 4 步前置扩展的必要性。
- 下一动作建议(优先级序):①实现级诊断(低先验决策路径的 b₀/凸起项数值检查);
  ②若走超参数路线,`reg_strength` 继续下调(0.01/0.03)与 `sigma`/`mom_groups`
  探针可在实现诊断后作为第二轮候选簇——但按当前信号,单靠超参数到达可用质量的
  概率低。

## 遗留

- 无 conf 阶段产物(0 候选晋级);dev 10 目录 + trials 全量提交。
- 全部 10 候选 30/30 success、P95 15–40s < 120s 预算:耗时维度无问题,问题纯粹在
  预测质量。
- `data_leakage_audit.json` 已由 preflight 在各 dev 目录落盘(status=pass,
  y_true_path / trial_label_columns 为 runner_gate)。
- 本轮未触碰 `baseline_v2` 与源码默认值;`check_baseline_configs` 门禁不受影响。
