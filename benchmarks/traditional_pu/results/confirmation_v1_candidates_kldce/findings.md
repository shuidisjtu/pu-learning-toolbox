# M6 候选判定记录:KLDCE 实现层修复(原生 SMO + 单调 ACS 回滚)

20-seed confirmation(种子 100..119)配对判定,契约 §6。配置:
`configs/seven_methods_pu_candidates_v1.json`(KLDCE 默认参数,2026-08-25
修复后)。判定表:`results/m6_kldce_verdict.csv`(compare.py 产出)。

## 结论

**KLDCE 实现层修复确认为收敛修复**。修复内容(提交 `cd53c17`):
原生配对 SMO 内层求解器 + 质心单调回滚(消除 period-2 极限环)+
`inner_tol` 1e-8→1e-6。基线(默认参数,修复前)KLDCE 全域 0 success
(60 structural failed + 59 nonconverged + 1 timeout);候选(默认参数,修复后)
**120/120 success**。

| 判定 | 基线(修复前默认参数) | 候选(修复后默认参数) |
|---|---|---|
| KLDCE SCAR 单元 success | 0/120 | **120/120** |
| 成功率(条件 2) | 不存在恶化:6 档全 0→1 | ✓ |
| P95 耗时(条件 3,预算 120s) | —(失败单元 ~ms) | 最慢 46.0s(scar-pi0.1-scalemid);P95 < 47s | ✓ |
| 配对 95% CI(条件 1) | 不可评估:基线侧无 success 值(n_paired=0) | 记录,非改善证据缺失 |

契约 §6 条件 1(同种子配对 CI)在基线无性能值处数学上不可满足:6 档全部
`n_paired=0`。该修复的改进证据是**可用性**:基线侧无指标可出,候选侧全量可出。
因此按契约字面四条件不标 `confirmed_improvement`(compare.py 列如实为
False),但作为**确认的收敛修复推荐**记录。

## 根因与修复(2026-08-25)

- F2(mid 档结构性失败)的根因是 `max_dual_variables=1000` 硬上限;该参数
  已随原生 SMO 废弃(提交 `87c8f5d`),修复后 mid 档正常求解。
- F5(ACS 停滞)的根因是质心更新(附录式 35)的 period-2 极限环:式(35)
  在 μ=0 处 Taylor 展开且假设 Σ̂ 正定,真实数据 Σ̂ 带负特征值使 `q<0`
  或大步长 Taylor 失真,μ 更新降低对偶目标,ACS 在边界点与 m_hat 间
  交替、永不满足收敛判据。修复:单调接受 + 回滚冻结(方法卡 §3.2/§6)。
- `inner_tol=1e-6`:1e-6→1e-8 是亚线性逼近尾巴,O(n) 对更新无消费方,
  且曾使 mid 档内层打满 `max_inner_iter=2000`(方法卡 §8 探针)。

## 收敛分布(候选,20 seeds)

- development(5 seeds,30 cells):30/30 success,mean 9.7s,max 29.6s
- confirmation(20 seeds,120 cells):120/120 success,mid 档
  mean 24.8-31.0s / P95 46.0s(π=0.1 最慢),small 档 ≤2.5s

## 协议一致性(条件 4)

- 与确认基线相同的 (scenario, seed) 键、同数据网格、同指标口径;
  compare.py 的对齐校验通过。
- 其余五个方法(elkan_noto/upu/nnpu/ldce/llsvm)与基线同参数重跑,
  全部 success,无误伤。LDCE 的 max_iter=10000 写回后与候选配置显式
  参数一致,重跑结果与 `confirmation_v1_candidates_ldce` 一致。

## 遗留

- `development_v1_kldce_fixed`(2026-08-24 判据修复)与
  `development_v1_kldce_smo_fix`(本次)两个 development 目录并存;
  正式基线仍以 `confirmation_v1` 为准,本次为候选判定不替换基线。
- F2/F5 在 `development_v1/findings.md` 中的历史记录保留(标注为
  修复前事实),不再构成当前限制。
