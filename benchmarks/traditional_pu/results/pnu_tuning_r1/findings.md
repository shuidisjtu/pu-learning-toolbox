# PNU 调优第 1 轮:eta / reg_lambda / basis / kernel_width 参数簇

调优方案 §8 第 5 步第七轮(§4 表第 7 行)——七轮循环的最后一轮。dev 种子 0..4、
conf 种子 100..119;预算 120s;对照 = `r7_default` companion(PNU 冻结基线 v2 的
pnu-only 视图)。本轮只产出 verdict,不写回源码默认值(§7)。

## 协议适配(本轮新增的两处口径)

1. **筛选指标 = `pu_auc_roc`(oracle)**:PNU 三元网格中二元 PU 指标
   (`pu_zero_one_risk` 等)全部不可用(契约 §2.2:requires binary PU labels),
   可用指标均为 oracle/supervised 族。§4 的固定筛选准则无法字面执行,
   本轮以 `pu_auc_roc` 为筛选与 verdict 指标并明示口径:PNU 轮的 verdict 语义
   是"三元网格上的分类质量判定",不是 PU 风险改善判定。
2. **rank 主网格 = pnu- 行**:`rank_candidates` 扩展为 scar-/pnu- 双主网格
   (混合拒绝,恰 30 行),并新增 `higher_is_better` 方向参数(pu_auc_roc 是
   higher-is-better,固定升序会反向排序——TDD 修复)。
3. **PNU 冻结基线 v2**:`pnu_baseline_v1.json` 的 methods.pnu 为空(构造器默认
   隐式),不满足候选生成的"构造器默认全覆盖"校验;新建
   `configs/pnu_baseline_v2.json`(显式锁定构造器默认 + `locks_source_defaults`,
   与 seven_methods v2 同风格),门禁 `check_baseline_configs` 通过。

## 网格设计

参数簇 = {`eta`, `reg_lambda`, `basis`, `kernel_width`}(默认 0.0 / 0.001 /
linear / None;eta ∈ [−1,1],0=PN 监督、+1=PU、−1=NU;class_prior 为必填构造器
参数,runner 注入不可调)。基线预检:confirmation_v1_pnu 的 oracle 指标全
1.0(分离度 2.0 的合成数据上默认参数已完美分类)——**预期饱和轮**。

| 候选 | eta | reg_lambda | basis | kernel_width | 设计意图 |
|---|---|---|---|---|---|
| r7_default | 0.0 | 0.001 | linear | — | 参照锚点 + companion 来源 |
| r7_eta_0p1 / r7_eta_0p5 | +0.1 / +0.5 | 0.001 | linear | — | PU 方向权重(P 内拟合减弱) |
| r7_eta_m0p1 / r7_eta_m0p5 | −0.1 / −0.5 | 0.001 | linear | — | NU 方向权重 |
| r7_reg_1em4 / r7_reg_0p01 | 0.0 | 1e-4 / 0.01 | linear | — | 正则扰动 |
| r7_rbf_w1 / r7_rbf_w2 | 0.0 | 0.001 | rbf | 1.0 / 2.0 | 非线性基 |
| r7_eta0p5_rbf_w1 | +0.5 | 0.001 | rbf | 1.0 | 极端组合(边界探针) |

## dev 筛选结果(§4 筛选链,pnu 主网格 30 行,metric=pu_auc_roc,higher-is-better)

全部 10 候选 30/30 success、零退化、P95 ≤ 0.05s;无淘汰。

| 候选 | auc mean | 筛选结论 |
|---|---|---|
| r7_default | 1.000000 | 入选(第 1,并列) |
| r7_eta_0p1 | 1.000000 | 入选(第 2,并列) |
| r7_eta_m0p1 | 1.000000 | 入选(第 3,并列) |
| r7_rbf_w1 / r7_rbf_w2 / r7_reg_0p01 / r7_reg_1em4 | 1.000000 | 并列 4-7 |
| r7_eta_m0p5 | 0.999999 | 第 8(微扰) |
| r7_eta_0p5 | 0.999974 | 第 9(微扰) |
| r7_eta0p5_rbf_w1 | 0.993880 | 第 10(最大扰动) |

观察:①7/10 候选 dev 精确 1.0——**饱和确认**,参数簇在 dev 无改善空间;②eta
极端值(±0.5)产生 1e-4~1e-5 量级的微扰,组合(eta0.5×rbf_w1)扰动最大
(0.994)——均为边界行为,方向为劣化。

## companion 一致性审计

`r7_default`(PNU 冻结基线 v2 参数)conf 120/120 success,与 `confirmation_v1_pnu`
在 5 个公共数值指标列上 **max|diff| = 0.0**——runner 确定性成立,companion 可作
compare 的对齐基线。

## conf verdict(compare.py,metric=pu_auc_roc,严格配对 CI 口径)

| 候选 | confirmed | oracle_only | 配对 diff | 退化率 | 判定 |
|---|---|---|---|---|---|
| r7_eta_0p1 | 0/6 | 0 | 全 0 | 0/120 = 基线 | **饱和:无改善** |
| r7_eta_m0p1 | 0/6 | 0 | 全 0 | 0/120 = 基线 | **饱和:无改善** |

两个入选者的 6 cells 配对 diff 全部精确为 0(conf 20 seeds 下与 companion 完全
一致)——dev 的并列入选在 conf 得到确认:±0.1 的 eta 扰动对已完美的分类器零影响。
compare exit=1(0 confirmed,正常结果)。

## 结论

**饱和 verdict:参数簇无改善空间,默认参数已最优,无写回。**

- dev 7/10 候选精确 1.0、conf 两个入选者 diff 全 0——PNU 默认参数(eta=0,
  linear basis,λ=0.001)在当前合成数据协议上已达分类质量上限(1.0),该参数簇
  无可调空间;
- eta 极端值(±0.5)与 rbf 组合仅产生劣化方向的微扰(≤0.006),构成边界记录,
  不构成改善证据;
- 该结论是诚实的饱和判定:七轮循环中唯一"调优的正确答案就是不动"的算法;
- §5 条件 1(严格配对 CI 改善)对任何候选都不可满足(天花板 1.0),按规则不得
  宣称 confirmed_improvement——本轮 verdict 即"无改善",与 KLDCE/Elkan-Noto
  轮(不可调)不同,这里**没有不可调的问题,只是没有改善空间**。

**写回讨论(§7,第 6 步)**:不涉及——无候选改善,不写回。

## 遗留

- dev 10 候选(30 trials/候选)+ conf 3 目录(companion + 2 入选,120 trials/
  目录)全量提交;verdict CSV ×2 全量提交;P95 全部 < 0.05s。
- `configs/pnu_baseline_v2.json` 为 PNU 首个显式锁定基线(门禁通过);v1 保留
  为未锁定历史配置。
- **七轮调优循环(§8 第 5 步)至此全部完成**:KLDCE(否定/实现级跟进)、
  LDCE(正面 ×2)、nnPU(部分改善)、uPU(全单元晋级)、LLSVM(部分改善/乐观性
  偏差)、Elkan-Noto(dev 中止/指标跟进项)、PNU(饱和)。第 6 步(写回决策:
  LDCE centroid_radius=0.1、LDCE covariance_ridge=1e-2、uPU loss='squared')
  与两个跟进项(KLDCE 低先验全负、Elkan-Noto 指标阈值语义)转入后续工作。
