# LDCE 调优第 1 轮:reg_strength / covariance_ridge / centroid_radius / max_iter 参数簇

调优方案 §8 第 5 步第二轮(§4 表第 2 行)。dev 种子 0..4(30 trials/候选),conf 种子
100..119(120 trials/候选);预算 120s;对照 = `baseline_v2`(冻结基线)。本轮只产出
verdict,不写回源码默认值(§7)。

## 网格设计

参数簇 = {`reg_strength`, `covariance_ridge`, `centroid_radius`, `max_iter`}(默认
1.0 / 1e-4 / 1.0 / 10000;`max_iter=10000` 为已写回的收敛修复值)。`learning_rate`/
`n_inner_iter`/`mom_groups`/`tol` 不在首轮簇内(§4)。本轮全部为**单参数探针**,不设
两两组合:LDCE 基线健康(6/6 单元无退化,见下),无 KLDCE 轮那种"低先验全负"定向
需求,单参数扰动即是最干净的小范围筛选(§4:避免不可解释的交互)。

| 候选 | covariance_ridge | reg_strength | centroid_radius | max_iter | 设计意图 |
|---|---|---|---|---|---|
| r2_default | 1e-4 | 1.0 | 1.0 | 10000 | 参照锚点 + companion 来源 |
| r2_ridge_0 | 0.0 | 1.0 | 1.0 | 10000 | 病态风险探针(LDCE 默认 1e-4 非零,与 KLDCE 的 0.0 不同) |
| r2_ridge_1em3 | 1e-3 | 1.0 | 1.0 | 10000 | 岭正则增强 ×10 |
| r2_ridge_1em2 | 1e-2 | 1.0 | 1.0 | 10000 | 岭正则增强 ×100 |
| r2_reg_0p1 | 1e-4 | 0.1 | 1.0 | 10000 | 正则放大(梯度法步长上界放大,稳定性风险探针) |
| r2_reg_10 | 1e-4 | 10.0 | 1.0 | 10000 | 正则收紧 |
| r2_radius_0p1 | 1e-4 | 1.0 | 0.1 | 10000 | 质心收紧 |
| r2_radius_10 | 1e-4 | 1.0 | 10.0 | 10000 | 质心放开 |
| r2_maxiter_5000 | 1e-4 | 1.0 | 1.0 | 5000 | 收敛预算半减(验证 10000 是否必要) |
| r2_maxiter_20000 | 1e-4 | 1.0 | 1.0 | 20000 | 收敛预算翻倍 |

## dev 筛选结果(§4 筛选链)

| 候选 | n_success | n_degenerate | risk_mean | P95 耗时(s) | 筛选结论 |
|---|---|---|---|---|---|
| r2_radius_0p1 | 30 | 0 | 0.338699 | 11.6 | 入选(第 1) |
| r2_ridge_1em2 | 30 | 0 | 0.346716 | 38.1 | 入选(第 2) |
| r2_ridge_1em3 | 30 | 0 | 0.421210 | 42.5 | 入选(第 3) |
| r2_reg_10 | 30 | 0 | 0.423347 | 1.2 | 第 4,未入选 |
| r2_default | 30 | 0 | 0.435772 | 8.4 | 第 5 |
| r2_maxiter_20000 | 30 | 0 | 0.435772 | 8.3 | 与默认一致 |
| r2_maxiter_5000 | 30 | 0 | 0.435772 | 8.0 | 与默认一致(5000 已够收敛) |
| r2_ridge_0 | 30 | 0 | 0.438522 | 14.1 | 变差(默认 1e-4 正则确有作用) |
| r2_radius_10 | 30 | 0 | 0.480764 | 15.4 | 变差 |
| r2_reg_0p1 | 21 | 0 | — | 75.9 | **淘汰**(success 21/30,λ=0.1 使梯度法不稳定) |

观察:①`max_iter` 在 5000–20000 间对风险零影响,收敛修复(100→10000)之后的余量充足,
5000 已够;②`centroid_radius=0.1`(质心收紧)是最大收益方向(risk −0.097);
③`covariance_ridge` 增强(1e-3/1e-2)次之(risk −0.015/−0.089),但代价是 P95 耗时
从 8.4s 升到 38–42s(更稳的正则使梯度路径更长);④`reg_strength=0.1` 直接破坏
稳定性(9 单元失败)——放大方向在此实现中不可行。

## companion 一致性审计

`r2_default`(冻结基线参数的 ldce-only 视图)conf 120/120 success,与 `baseline_v2`
的 ldce 行在 11 个公共指标列上 **max|diff| = 0.0**——当前 runner 确定性成立,companion
可作为 compare 的对齐基线(§5 条件 5 协议一致由工具级键对齐保证)。

## conf verdict(compare.py,严格配对 CI 口径)

| 候选 | confirmed | oracle_only | 配对 diff 范围(每单元) | 退化率(基线 0/120) | 判定 |
|---|---|---|---|---|---|
| r2_radius_0p1 | **6/6** | 0 | −0.072 至 −0.137(全部 CI 上界 < 0) | 0/120 | **confirmed_improvement** |
| r2_ridge_1em2 | **6/6** | 0 | −0.073 至 −0.110(全部 CI 上界 < 0) | 0/120 | **confirmed_improvement** |
| r2_ridge_1em3 | 5/6 | 1(pi0.5-small,PU 指标 CI 含 0) | −0.005 至 −0.028 | 0/120 | 部分改善,未全单元通过 |

cond2(成功率 120/120)与 cond3(P95 ≤ 120s)在全部单元通过;无退化预测
(§5 条件 3:候选 0 = 基线 0)。晋级规则 §5 全条件对 r2_radius_0p1 与
r2_ridge_1em2 成立。

## 结论

**正面 verdict:两个候选达到 `confirmed_improvement`(hyperparameter_tuning 类型)。**

- **r2_radius_0p1**(`centroid_radius=0.1`)为首选:全 6 单元严格改善,risk 均值
  0.339(dev)vs 默认 0.436,配对 diff −0.072 至 −0.137,P95 11.6s(默认 8.4s,
  略慢但预算充裕);
- **r2_ridge_1em2**(`covariance_ridge=1e-2`)次选:全 6 单元严格改善,risk 0.347,
  P95 38.1s(代价是 ~4.5× 耗时,正则增强使梯度路径更长);
- r2_ridge_1em3 改善幅度最小且 1 单元仅 oracle 改善,不构成 PU 调优成功。

**写回讨论(§7,第 6 步)**:是否将 `centroid_radius=0.1`(或组合)写回源码默认值,
需另行决策——写回必须同步重锁基线(更新 v2 锁定值或升 v3)并在确认种子重跑,
否则 `check_baseline_configs` 门禁报警。本轮不写回。两个候选的 conf 产物与
verdict CSV 全量提交,供第 6 步引用。

## 遗留

- dev 10 候选 + conf 4 目录(companion + 3 入选)全量提交;P95 全部 < 120s。
- r2_reg_0p1(λ=0.1)的 9 单元失败已记录(梯度稳定性边界),不再复测。
- 若第 6 步选 radius_0p1:建议同时评估 radius_0p1 与 ridge_1em2 的组合
  (第 3 轮网格可含 `centroid_radius=0.1 × covariance_ridge=1e-2/1e-3`),
  以及 `centroid_radius` 在 0.1–1.0 之间更细的取值(0.3/0.5)。
