# ADR-0016:传统 PU 七轮调优 verdict 与第 6 步写回决策框架

- 状态:已接受(2026-08-26)
- 触发复审:第 6 步写回执行时,或任一跟进项修复完成时

## 背景

调优方案 §8 第 5 步七算法调优主循环(每轮:dev 30/60 trials 筛选 → 前 3 conf
120/240 trials → compare.py 严格配对 CI verdict)全部完成。七轮结论分布:

| 轮 | 算法 | Verdict | 核心证据 |
|---|---|---|---|
| 1 | KLDCE | 否定 | 10 候选全过 success 但零候选通过 §3.3 退化筛选(低先验全负) |
| 2 | LDCE | 正面 ×2 | `centroid_radius=0.1`、`covariance_ridge=1e-2` 均 6/6 confirmed |
| 3 | nnPU | 部分改善 | β=0.25/0.1 为 3/6 SCAR + 4/6 SAR confirmed、零变差,无全单元晋级 |
| 4 | uPU | 全单元晋级 | `loss='squared'` 12/12 confirmed(diff −0.035~−0.149 随先验递增) |
| 5 | LLSVM | 部分改善 | lr_2em5 dev 第 1 但 conf pi0.1-mid 变差(+0.047,dev 乐观性偏差) |
| 6 | Elkan-Noto | dev 中止 | 主指标失真(decision_function=g/c 恒 >0 → risk≡1−π) |
| 7 | PNU | 饱和 | 基线 oracle 全 1.0,conf diff 全 0,默认参数已达上限 |

companion 一致性审计 7 轮全部 max|diff|=0.0,配对 CI 结论可信。

## 决策

1. **写回候选名单(第 6 步)定为三个,逐个独立决策**:
   - uPU `loss='squared'`(12/12 全单元晋级,首选);
   - LDCE `centroid_radius=0.1`(6/6,risk 0.436→0.339);
   - LDCE `covariance_ridge=1e-2`(6/6,risk 0.347,代价 P95 8.4s→38s)。
2. **写回的前置条件(§7,不可缩减)**:任一写回必须同步执行 ①基线重锁
   (v2 更新或升 v3,`check_baseline_configs` 门禁同步)②确认种子
   (100..119)重跑 ③companion 一致性审计复验。写回按最小面执行:一次只
   改一个算法的默认值,每改一个重锁一次。
3. **LDCE 双候选的组合先评估**:radius 与 ridge 同簇,写回前先跑
   `centroid_radius=0.1 × covariance_ridge=1e-2` 组合候选确认无交互劣化;
   组合若无增益,只写回 radius_0p1(收益大、耗时代价小)。
4. **两个跟进项立档,优先级定序**:
   - **Elkan-Noto 指标跟进项(协议级,优先)**:`pu_zero_one_risk` 对
     elkan_noto 的阈值语义失配(0 阈值 vs `_predict` 的 0.5 阈值)。修复
     方向:0.5 阈值变体或 predict 标签口径,须 TDD + 契约评审;顺带修正
     baseline_v2 的 elkan_noto risk 行(0.9/0.7/0.5 为失真产物)。完成后
     重跑 Elkan-Noto 调优轮。
   - **KLDCE 实现跟进项(实现级)**:低先验全负预测(参数簇无法清除,已由
     第 1 轮证伪"调参可救"路径)。方向:决策函数 b₀/质心凸起诊断;完成后
     重跑 KLDCE 调优轮。
5. **非晋级轮一律不写回**:否定/部分改善/饱和轮次的 verdict 留档为证据
   (findings.md),不作默认值变更;nnPU 的 β 方向性信号(零变差)仅在实现
   变更后按新基线重评。
6. **调优轮工具链沉淀为长期资产**:`tuning_round.py`(7 方法注册表、
   scar/sar/pnu 网格口径、higher_is_better 方向)、`compare.py` verdict、
   `pnu_baseline_v2.json` 冻结基线随工具箱保留,后续算法或协议变更时
   复用同一流水线。

## 备选方案

- **立即全量写回三个候选**:违反 §7 最小面纪律(一次改三个默认值无法
  归因任何回归),否决。
- **只写回 uPU,放弃 LDCE**:LDCE 双候选 6/6 confirmed 证据充分,放弃
  无理由;折中方案(只写 radius 不写 ridge)由决策 3 的组合评估结果定。
- **先修跟进项再写回**:两个跟进项均为算法/协议级缺陷,与写回候选
  (已确认改善)无依赖关系,串行等待会推迟已验证的收益,不采用。

## 后果

- 第 6 步写回工作包明确:3 候选 + 组合评估 + 基线重锁流程,可独立执行;
- 两个跟进项进入跟踪清单,修复后触发对应调优轮重跑(Elkan-Noto 需协议
  评审,KLDCE 需实现诊断);
- 七轮 verdict 全量留档(findings.md ×7),任何未来基线报告可直接引用
  逐轮证据,不依赖口头结论。

## 跟进状态

- 2026-08-27：Elkan--Noto 协议级问题已按“predict 标签口径”修复并由工作流/benchmark
  回归测试锁定；指标契约升至 schema v2。历史第 6 轮结果仍保留为旧口径证据，修复后的
  Elkan--Noto 调优轮和受影响基线重跑另行生成新结果，不覆盖历史产物。
- 2026-08-27：受影响基线重跑完成（baseline_v3，契约 v2 口径，1200/1200；
  五个零阈值方法指标列与 v2 逐单元 max|diff|=0，elkan_noto risk 失真消除）。
- 2026-08-27：Elkan--Noto 调优轮重跑完成（r2）：`mode=weighted_retraining`
  12/12 全单元 confirmed（r2_weighted 与 r2_isotonic_weighted 数字逐 cell 相同，
  改善归因 mode 本身）——第 6 步写回候选名单新增 elkan_noto `mode=
  weighted_retraining`（写回须重锁 baseline_v3 的 elkan_noto 行 + 确认种子
  重跑）。
- 2026-08-27：KLDCE 实现跟进项根因修复完成（`fix/kldce-bias-class-symmetric`）：
  b₀ 恢复的全体自由 SV 中位数在低先验下坍缩到多数类（负类）簇 → b₀≈−1 → 全负
  预测（排序判别力 AUC 完好，问题仅在 0 阈值）。修复为类对称中位数（每类中位数
  再平均，等价附录式 37–40 四项平均的类平衡意图）+ 单类回退 bounded-interval；
  pi=0.1 seed0 预测阳性率 0→0.278、recall 0→0.65（train AUC 不变）。TDD 回归
  锁定（TestBiasRecoveryClassSymmetry + math 数值用例）。**待办**：baseline_v3
  的 KLDCE 行重跑（implementation_fix §7）+ KLDCE 调优轮重跑（新 b₀ 语义）。
- 2026-08-27：KLDCE 基线行重跑完成——baseline_v3 的 KLDCE 行由
  kldce_baseline_v3 重跑替换（退化 80/960→0；pi=0.1/0.3 recall 0→0.59..0.79，
  pi=0.5 不变），其余 5 方法行不受影响。KLDCE 调优轮重跑（r3）完成：
  退化消失、risk 恢复 0.32 量级，三候选均 0/6 confirmed（diff 全在 ±0.02 内、
  CI 跨 0、pi=0.5 单元 diff 精确 0）——参数簇无可调增益，默认参数即有效
  工作点，r1 否定结论被修复取代，无新写回候选。至此两个跟进项均已闭环。
- 2026-08-28：LDCE 组合评估（ldce_tuning_r2）完成——`centroid_radius=0.1 ×
  covariance_ridge=1e-2` 组合 conf 6/6 confirmed（diff −0.27..−0.32，为单参数
  −0.07..−0.14 的 2 倍以上，强交互确认；dev 0.149 / conf 0.1498 复现）。
- 2026-08-28：**第 6 步写回第 1 轮（LDCE）完成**——源码默认值改为组合
  （ldce.py，默认值契约测试锁定）；基线重锁升 **v4**（v2/v3 摘
  `locks_source_defaults` 冻结为历史快照，v4 钉住新默认，门禁通过）；
  `ldce_baseline_v4` 确认重跑 120/120、0 退化，与新默认逐单元 max|diff|=0
  （companion 审计通过，6/6 confirmed 直接迁移）；非 ldce 行确定性继承 v3。
  剩余写回：uPU `loss='squared'`（12/12）、elkan_noto `mode=
  weighted_retraining`（12/12），各升一个基线版本并确认重跑。
