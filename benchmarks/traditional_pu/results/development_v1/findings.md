# M4 开发基线:协议/实现审查发现

5-seed 开发基线(种子 0..4)运行后审查记录。协议问题已按契约 §6 处理:
非成功 trial 不入主性能均值、计入 `success_rate`;状态与原因完整保留在
`trials.csv`。

## F1. LDCE/KLDCE 默认参数不收敛(非 bug,默认值偏保守)

- 现象:全部 30 个 LDCE 单元(3π × 2 档 × 5 种子)与全部 15 个 small 档
  KLDCE 单元为 `nonconverged`,失败原因 `explicit non-convergence`。
- 诊断(seed=0, small, π=0.1):
  - LDCE 目标函数单调下降(rel_change 从 2.9 降到 7.5e-5),最后
    rel_change = 2.7e-6 仍 > tol=1e-6,只差一步;将 `max_iter` 提到 3000 后
    **146 轮收敛**(`n_iter_=146, converged=True`)。
  - 结论:**LDCE 侧:实现无 bug,默认 `max_iter=100` 对合成数据(400/2000
    样本)偏紧,为参数问题**(→ M6 变体,已确认修复,见
    `results/confirmation_v1_candidates_ldce/findings.md`)。
  - **KLDCE 侧已更正**:放大 `max_acs_iter` 无效果,升级为 F5 的诊断
    (实现层停滞,非参数问题)。
- 处理:M6 针对 LDCE(`max_iter > 100`)的变体评估,默认参数基线不动(契约 §5)。

## F2. KLDCE mid 档结构性失败(`max_dual_variables` 上限)

- 现象:全部 15 个 mid 档 KLDCE 单元 `failed`,原因
  `ValueError('Number of dual variables (~3900) exceeds max_dual_variables (1000).')`,
  与种子无关(构造器 fit 前硬校验)。
- 原因:KLDCE 双变量数 = n + n_U;mid 档 n=2000 → ~3901 > 源码默认上限 1000。
- 处理:**已证实放宽上限不可行**(放宽容许后单轮 QP > 300s,五轮未完成一
  轮,超所有合理预算),与 F5 的停滞诊断一同归入实现层改进;基线记录为诊断列
  (`diag_n_acs_iter` 等),该单元在 SCAR mid 档无性能数据。

## F3. ElkanNoto Brier/ECE 不可用(`predict_proba` 越界 [0,1])

- 现象:ElkanNoto 的 `predict_proba` 已文档化支持越界;实测(seed=0, small)
  输出范围 [9.4e-05, 1.43],8/400 个样本 >1。契约 §1 要求 Brier/ECE 只接受
  真 `predict_proba` 且严格 ∈ [0,1] → 指标记为 NaN +
  `invalid probability output (not in [0,1])`(unavailable,非失败)。
- 处理:契约 §1 语义;ElkanNoto 的校准指标在基线上整体标记不可用,不影响
  `success_rate` 与主指标(`pu_zero_one_risk`)比较。

## F4. PNU 计数派生缺陷(已修复并重跑)

- 现象:开发基线初版中 PNU 的 small/mid 两档产出**完全相同**的数据(原实现用
  固定绝对计数 25/25/100 等,从不消费 `n_samples`),违背契约 §2.1/§2.2。
- 修复:`data.py` 新增 `pnu_counts(ratio, n_samples)`(比例 × 规模,
  largest-remainder 取整);runner 每单元记录 `n_p`/`n_n`/`n_u` 列;补 4 个测试。
  重跑后 30/30 success、两档数据不同(如 1:1:4: small 67/66/267, mid 334/333/1333)。
- 说明:旧产物 `development_v1_pnu`(P1)已由修正后版本覆盖。

## F5. KLDCE ACS 停滞:参数放宽无效(实现层问题,非参数问题)

M6 曾按 F1/F2 设计 KLDCE 变体(`max_acs_iter=50→500`,
`max_dual_variables=1000→5000`)。5-seed 网格中 small 档单元全部
`timeout`(120s 预算内跑不完),单单元定向探针(seed=0, small, π=0.1)
暴露根本原因:

- **下层 QP 第 1 轮即最优**:`dual_obj` 自第 2 轮起恒定(1.384962),
  `eq_residual≈1e-11`、`box_violation=0`;z 不动、QP KKT 已满足。
- **上层 ACS 卡死**:centroid 约束残差(src `ACS history`)恒为 **1.00**
  永不下降;`mu` 每轮更新但从不满足自己的约束,而收敛判据
  `max(rel_obj_change, mu_change, kkt_residual) < tol` 盯住的是
  `mu_change`/`kkt`,不含 centroid 残差 → 判据永不触发。
- 量化:300 轮仍 `converged=False`(每轮 ~0.7s,共 208.5s);150 轮与
  300 轮轨迹完全相同(停滞,而非慢收敛)。
- mid 档另有限制:放宽 `max_dual_variables` 后首轮 QP > 300s(未完成),
  单轮成本超过任何合理预算。

指向两类实现级修复(独立跟踪,不属于本轮参数调优):
1. 复核重心更新(附录 Eq. 35)实现与论文的一致性——mu 可能在投影边界
   震荡,导致约束残差恒 1.0;
2. 收敛判据应纳入(或替换为)centroid 约束残差/目标值结合判据;
3. mid 档 QP 效率(双变量 ~3901 的稀疏化或预算内近似解)。

KLDCE 已从 M6 候选网格移除(`configs/seven_methods_pu_candidates_v1.json`
limitations 记录与此处证据一致)。
