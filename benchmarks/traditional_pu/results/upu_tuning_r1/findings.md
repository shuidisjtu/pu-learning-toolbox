# uPU 调优第 1 轮:loss / reg_lambda / basis / kernel_width 参数簇

调优方案 §8 第 5 步第四轮(§4 表第 4 行)。dev 种子 0..4、conf 种子 100..119;
预算 120s;对照 = `r4_default` companion(冻结基线参数的 upu-only 视图)。本轮只产出
verdict,不写回源码默认值(§7)。

uPU 进 SAR 诊断线,dev/conf 网格 60/240 trials(与 nnPU 轮同款 scar/sar 拆分口径,
rank 筛选链只用 30 个 SCAR 行)。

## 网格设计

参数簇 = {`loss`, `reg_lambda`, `basis`, `kernel_width`}(默认 double_hinge / 0.001 /
linear / None;`basis='rbf'` 时 `kernel_width` 必填且 >0,`n_centers` 默认
min(200, n_U),两者均属簇内数值参数,无 JSON 不可表达项)。基线 risk 已近零
(0.0001~0.0008,double_hinge + linear 在可分离合成数据上几乎完美),本轮预期
调优空间极小——squared loss 是该先验下唯一结构性不同的解路径(闭式解)。

| 候选 | loss | reg_lambda | basis | kernel_width | 设计意图 |
|---|---|---|---|---|---|
| r4_default | double_hinge | 0.001 | linear | — | 参照锚点 + companion 来源 |
| r4_loss_logistic | logistic | 0.001 | linear | — | 损失函数换(迭代解) |
| r4_loss_squared | squared | 0.001 | linear | — | 平方损失闭式解 |
| r4_reg_1em4 | double_hinge | 1e-4 | linear | — | 正则放宽 ×10 |
| r4_reg_1em5 | double_hinge | 1e-5 | linear | — | 正则放宽 ×100 |
| r4_reg_0p01 | double_hinge | 0.01 | linear | — | 正则收紧 ×10 |
| r4_rbf_w0p5 | double_hinge | 0.001 | rbf | 0.5 | RBF 窄核 |
| r4_rbf_w1 | double_hinge | 0.001 | rbf | 1.0 | RBF 中核 |
| r4_rbf_w2 | double_hinge | 0.001 | rbf | 2.0 | RBF 宽核 |
| r4_rbf_w1_reg0p01 | double_hinge | 0.01 | rbf | 1.0 | rbf + 正则收紧组合 |

## dev 筛选结果(§4 筛选链,SCAR 行口径)

全部 10 候选 60/60 success(30 SCAR + 30 SAR);**r4_rbf_w0p5 因 5/30 SCAR 单元
退化预测被 §3.3 淘汰链剔除**(窄核过拟合——七轮循环首个 dev 阶段退化淘汰案例)。

| 候选 | scar risk_mean | P95 耗时(s) | 筛选结论 |
|---|---|---|---|
| r4_loss_squared | −0.095044 | 0.02 | 入选(第 1) |
| r4_rbf_w1 | −0.005079 | 1.8 | 入选(第 2) |
| r4_reg_0p01 | −0.000124 | 0.03 | 入选(第 3) |
| r4_reg_1em4 | −0.000103 | 0.08 | 第 4,未入选 |
| r4_loss_logistic | −0.000069 | 0.03 | 第 5,未入选 |
| r4_default | −0.000049 | 0.05 | 第 6 |
| r4_reg_1em5 | +0.000307 | 0.08 | 第 7 |
| r4_rbf_w2 | +0.000858 | 2.4 | 第 8 |
| r4_rbf_w1_reg0p01 | +0.001423 | 1.3 | 第 9 |
| r4_rbf_w0p5 | +0.087794 | 2.3 | **淘汰(5/30 退化)** |

观察:①`squared` loss 是唯一结构性信号(−0.095 vs 默认 −0.00005),闭式解在
可分离数据上找到明显更优的决策面;②logistic 与 double_hinge 几乎等价(±0.00002);
③reg_lambda 在 1e-5~0.01 区间影响微弱;④rbf 仅在 w=1 且低先验 cells 有改善信号,
w=0.5 直接退化。

## companion 一致性审计

`r4_default`(冻结基线参数的 upu-only 视图)conf 240/240 success,与 `baseline_v2`
的 upu 行在 **12 个公共数值指标列上 max|diff| = 0.0**——runner 确定性成立,
companion 可作为 compare 的对齐基线。

## conf verdict(compare.py,严格配对 CI 口径)

全部 240/240 success;P95 ≤ 1.8s(预算 120s 宽裕);退化率 0/240 = 基线 0/240
(§5 条件 2/3/4 通过)。

| 候选 | SCAR confirmed | SAR confirmed | 配对 diff 范围(SCAR) | 判定 |
|---|---|---|---|---|
| **r4_loss_squared** | **6/6** | **6/6** | −0.035 至 −0.149(全部 CI 上界 < 0) | **confirmed_improvement** |
| r4_rbf_w1 | 2/6(仅 pi0.1) | 2/6(仅 pi0.1) | pi0.1:−0.018/−0.019;pi0.3/0.5 变差或 CI 含 0 | 部分改善(仅低先验) |
| r4_reg_0p01 | 0/6 | 1/6 | ±0.001 量级,CI 均含 0 | 无改善 |

**r4_loss_squared 逐单元明细**(全部 12 cells 严格改善,CI 上界 < 0):

| 单元 | diff_mean | 单元 | diff_mean |
|---|---|---|---|
| scar-pi0.1-small | −0.0345 | sar-pi0.1-small | −0.0369 |
| scar-pi0.1-mid | −0.0352 | sar-pi0.1-mid | −0.0354 |
| scar-pi0.3-small | −0.0952 | sar-pi0.3-small | −0.1147 |
| scar-pi0.3-mid | −0.1092 | sar-pi0.3-mid | −0.1152 |
| scar-pi0.5-small | −0.1329 | sar-pi0.5-small | −0.1621 |
| scar-pi0.5-mid | −0.1495 | sar-pi0.5-mid | −0.1631 |

改善幅度随类先验单调递增(pi0.1 −0.035 → pi0.3 −0.10 → pi0.5 −0.15),SAR 线
同格局略强。

## 结论

**正面 verdict:r4_loss_squared(`loss='squared'`)达到 `confirmed_improvement`,
七轮循环首个全单元晋级候选(12/12 cells,含全部 6 个 SCAR 主网格单元)。**

§5 晋级条件逐条核对:
1. 逐单元严格配对 CI 改善:12/12(全部 CI 上界 < 0)✓
2. 成功率 240/240 = 基线 ✓(未恶化)
3. 退化率 0/240 = 基线 0/240 ✓
4. P95 ≤ 1.8s ≪ 120s 预算 ✓
5. 协议一致:companion max|diff|=0.0 + 工具级键对齐 ✓
6. 确认种子(100..119)未参与筛选 ✓

**写回讨论(§7,第 6 步)**:`loss='squared'` 是否写回源码默认值属第 6 步决策——
写回必须同步重锁基线(更新 v2 锁定值或升 v3)并在确认种子重跑,否则
`check_baseline_configs` 门禁报警。本轮不写回,verdict CSV 与 conf 产物全量提交
供第 6 步引用。

## 遗留

- dev 10 候选(60 trials/候选)+ conf 4 目录(companion + 3 入选,240 trials/
  目录)全量提交;P95 全部 < 3s。
- r4_rbf_w0p5 的 5 单元退化已按 §3.3 淘汰并记录,不进入 conf。
- r4_rbf_w1 的 pi0.1-only 改善与 r4_reg_0p01 的零改善如实记录,不宣称成功。
- 下一轮(第 5 轮)= LLSVM(§4 表第 5 行:学习率、正则、gamma、patience/min
  epochs)。
