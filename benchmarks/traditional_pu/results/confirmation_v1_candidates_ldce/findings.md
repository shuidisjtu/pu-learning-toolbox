# M6 候选判定记录:LDCE 收敛修复(max_iter=10000)

20-seed confirmation(种子 100..119)配对判定,契约 §6。配置:
`configs/seven_methods_pu_candidates_v1.json`。判定表:
`results/m6_ldce_verdict.csv`(compare.py 产出)。

## 结论

**LDCE `max_iter` 100 → 10000 确认为收敛修复**。修复只放宽收敛上限,不改变
目标函数与解语义:所有基线侧能成功的单元(评估上唯一可用点,scar-pi0.5-scalesmall
seed=114)与候选的 `pu_zero_one_risk` 完全一致(diff=0),即新配置在该单元是
同一解;扩大上限仅让其余单元从"未跑完"变为"收敛后输出结果"。

| 判定 | 基线(默认参数) | 候选(max_iter=10000) |
|---|---|---|
| LDCE SCAR 单元 success | 1/120(非收敛 119) | **120/120** |
| 成功率(条件 2) | 不存在恶化:1.00 ≥ 0.05 + 9 档全 0→1 | ✓ |
| P95 耗时(条件 3,预算 120s) | —(非收敛单元 1-2s) | 最慢 28.9s;P95 < 40s | ✓ |
| 配对 95% CI(条件 1) | 不可评估:基线侧 5/6 cell 无 success 值 | 记录,非改善证据缺失 |

契约 §6 条件 1(同种子配对 CI)在基线无性能值处数学上不可满足:5/6 cell
`n_paired=0`(基线全域非收敛),唯一可配对 cell(seed=114,diff=0.0)不支持
"改善"判断。该修复的改进证据是**可用性**:基线侧无指标可出,候选侧全量可出
且数值与基线一致。因此按契约字面四条件,该变体不标 `confirmed_improvement`
(compare.py 列如实为 False),但作为**确认的收敛修复推荐**记录:
新签名用户应从 `max_iter=10000` 起步(等价于"跑满到收敛",不再在 100 轮
上限内截断)。

## 收敛分布(候选,20 seeds)

- 收敛轮数 mean 1056 / max 4971(max_iter=10000 余量 2x)
- 迭代单调耗时 ~4.6ms/轮(n=2000);最慢单元 28.9s ≪ 120s 预算

## 协议一致性(条件 4)

- 与确认基线相同的 (scenario, seed) 键、同数据网格、同指标口径;
  compare.py 的对齐校验通过。
- 其余四个方法(elkan_noto/upu/nnpu/llsvm)为空参数变体,配对差 0.0,
  无误伤(条件 4 对照)。

## 被排除的 KLDCE 候选(见 development_v1/findings.md F5)

KLDCE `max_acs_iter`/`max_dual_variables` 放宽经 5-seed 定向探针与单单元
诊断证明:**不是参数问题而是实现层的 ACS 停滞**,故 KLDCE 不进本候选网格
(config 的 limitations 已记录)。其单轮 QP 成本(mid 档 >300s)也超出所有
合理预算;修复需实现层工作(单独跟踪)。
