# v2 正式基线:当前工具箱状态(显式锁定默认参数)

20-seed confirmation(种子 100..119),配置
`configs/seven_methods_pu_baseline_v2.json`。v2 基线的目的是让正式基线重新代表
**当前工具箱**——LDCE/KLDCE 的修复写回源码默认参数后,`confirmation_v1`
(v1 基线)仍是修复前快照,二者脱钩。

## 一致性审计(v1 基线 commit d34fd345 之后的代码演进)

| 方法 | 演进 | v1 基线是否仍代表当前工具箱 |
|---|---|---|
| elkan_noto / upu / nnpu / llsvm | 无 | 是 |
| pnu | 无 | 是(`confirmation_v1_pnu` 继续有效,未重跑) |
| ldce | `max_iter` 100→10000(28edfd6) | 否,v1 为 1/120 success |
| kldce | 内层 SMO 重写 + 单调 ACS 回滚 + `inner_tol` 1e-6(cd53c17) | 否,v1 为 0/120 success |

## v2 配置的锁定机制

v1 配置的 `methods` 为空 `{}` = "跟随源码构造器默认值",默认参数演进后配置
与基线数字静默脱钩。v2 配置将 7 个方法的全部构造器默认参数**显式钉死**
(runner 注入的 `random_state`/`class_prior`/`flip_probability` 除外),并带
`locks_source_defaults: true` 标记,由 `scripts/check_baseline_configs.py`
门禁持续校验与源码构造器默认值一致:未来默认参数再演进时门禁报警,
基线不再静默漂移(契约 §5 演进记录)。

v1 配置保持历史形态,仅作快照,不得在当前源码下重跑(README 命令节已标注)。

## 结果(20 seeds,1200/1200 success)

`trials.csv`(原始 1200 行,增量落盘)与 `summary.csv` 完整提交。各方法:

| 方法 | v2 success | v1 基线 success | 备注 |
|---|---|---|---|
| elkan_noto / upu / nnpu / llsvm | 各 240/240 | 各 240/240 | 无演进,逐单元指标与 v1 **完全一致**(max\|diff\|=0) |
| ldce | 120/120 | 1/120 | 与 `confirmation_v1_candidates_ldce` 逐单元一致(diff=0) |
| kldce | 120/120 | 0/120 | 与 `confirmation_v1_candidates_kldce` 逐单元一致(diff=0) |

LDCE/KLDCE 修复写回工具箱默认参数后收敛性验证:small 档全部 ≤2.5s,mid 档
均值 24.8–31.0s、P95 46.0s(KLDCE),全部在 120s 预算内。与候选判定
(`confirmation_v1_candidates_ldce`/`_kldce`)生效参数相同,数字逐单元一致——
v2 基线即候选结果的**正式转正**。

manifest:git_commit `2a732d7`(运行起点)、git_worktree_dirty True(运行时有
未提交的锁参改动,配置身份以 config_sha256 为准)、n_trials 1200、seed_set
confirmation。

## 与候选判定的关系

v2 基线使用的默认参数与 M6 候选网格
(`confirmation_v1_candidates_ldce`/`confirmation_v1_candidates_kldce`)生效参数
一致(候选配置 `ldce: {max_iter: 10000}`、`kldce: {}` = 修复后默认),因此 v2
基线的 LDCE/KLDCE 数字应与候选一致,是候选结果的正式转正;其余 5 个方法应与
v1 基线一致(无演进)。v1/candidates 目录全部保留。

## 遗留

- `confirmation_v1`(v1)与 `baseline_v2` 并存:v1 为修复前历史快照,v2 为
  当前工具箱正式基线。横向对比当前方法请以 v2 为准。
- PNU 无默认参数演进,`confirmation_v1_pnu` 继续有效,未随 v2 重跑。
- timeouts 仍为未冻结的宽松保护上限(统一 120s),冻结流程见 README。
