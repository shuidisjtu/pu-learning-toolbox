# v3 正式基线:契约 v2 口径(risk 跟随原生 predict 阈值)

20-seed confirmation(种子 100..119),配置
`configs/seven_methods_pu_baseline_v3.json`。指标契约升 v2(def1544):
`pu_zero_one_risk` 不再对原始 decision score 使用固定零阈值,而是跟随各分类器
`predict()` 的原生二元决策。v3 基线的目的:让正式基线的 risk 行符合契约 v2,
修复 elkan_noto 概率尺度(原生阈值 0.5)与零阈值不相容导致的 risk ≡ 1−π 失真。

## 受影响性论证(源码级)

| 方法 | `_predict` 阈值 | 受口径变化影响 |
|---|---|---|
| elkan_noto | `f >= 0.5`(概率尺度) | **是** |
| upu / nnpu / llsvm / ldce / kldce | `>= 0.0` | 否(pred 与旧 scores 口径数值等价) |
| pnu | 三元协议无二元 risk(独立 config `pnu_baseline_v2`) | 否,未重跑 |

## 一致性审计(v3 vs v2,逐 (algorithm, scenario, seed) 单元)

- 五个零阈值方法(upu/nnpu/llsvm/ldce/kldce,960 行 × 23 个数值指标列):
  **所有指标列 max|diff|=0**(完全一致);唯一差异是 `elapsed_seconds`
  硬件计时自然波动(0.03–13.7s 量级),不属指标。
- elkan_noto(240 行):risk 由 v2 的常数 0.9/0.7/0.5(≡1−π)变为真实风险
  -0.218..0.096(237 个不同值);`pred_positive_rate` 随先验单调上升
  (π=0.1: 0.06,π=0.3: 0.20,π=0.5: 0.42),符合 Elkan-Noto 概率校正语义;
  `degenerate_prediction` 0/240。

## 结果(20 seeds,1200/1200 success)

| 方法 | v3 success | 与 v2 逐单元一致性 |
|---|---|---|
| elkan_noto | 240/240 | risk 口径修复(见上);oracle 指标(ROC-AUC 等)不受口径影响,仍逐单元一致 |
| upu / nnpu / llsvm / ldce / kldce | 各 240/240 | 指标列 max\|diff\|=0 |

新 runner 同时产出 `pred_positive_rate` / `degenerate_prediction` 列(v2 为
step-4 前产物,无此二列):elkan_noto 0 退化,其余五方法退化 80/960
(KLDCE/LDCE 低先验全负,第 1/2 轮既有发现,与口径变化无关)。

manifest:git_commit `4ae9311`(运行起点)、git_worktree_dirty True(运行时
v3 config 尚未提交,配置身份以 config_sha256 `6a18238d` 为准)、n_trials 1200、
seed_set confirmation。

## 与 v2 的关系

- v2 config/results 保留为 frozen history snapshot(v1 先例),不得在当前源码
  口径下重跑,README 已标注。
- 后续传统 PU 调优轮(KLDCE 重跑、第 6 步写回基线重锁)以 **v3 为对齐基线**;
  Elkan-Noto 调优轮重跑在契约 v2 口径下进行。
- PNU 三元协议未受影响,`pnu_baseline_v2` 继续有效。

## 遗留

- `elapsed_seconds` 重跑波动属硬件计时,无指标意义;冻结耗时预算时以 v3 为准。
- baseline_v2 的 elkan_noto risk 行(0.9/0.7/0.5)是旧口径产物,历史调优轮
  (elkan_noto_tuning_r1)的证据保留为旧口径记录,新口径证据以本基线与重跑轮为准。
