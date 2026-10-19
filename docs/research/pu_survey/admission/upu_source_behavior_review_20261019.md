# uPU 来源与行为核查（2026-10-19）

范围：核对 du Plessis et al. ICML 2015 论文、官方关联 `pywsl` 包说明、工具箱 uPU loss/estimator、TS 校准与测试范围。属于单方技术审计，不是方法负责人签署、正式准入或完整实验复现。

## 来源及版本边界

- 原论文：[PMLR v37 paper page/PDF](https://proceedings.mlr.press/v37/plessis15.html)，提出通过正例与 U 使用不同损失的凸 PU 风险，并特别提出 double-hinge（C-DH）；摘要明确 naive P/U separation 有偏，本文不能把任意 P/U 分类器都称为 uPU。
- 官方关联代码：[t-sakai-kure/pywsl](https://github.com/t-sakai-kure/pywsl)，于 2026-10-19 只读解析 `master` 并检出 `79a5bf104a53cb6d3498918e435bd88ac5331543`（提交日期 2020-04-03；MIT，`setup.cfg` 版本 0.1.4）；快照已写入 `official_sources.lock.json`。锁定的是仓库身份，不代表 toolbox 的所有 loss 分支或实验与其等价。该快照 README 列出 ICML 2015 uPU，相关分类器位于 `pywsl/pul/pu_mr.py::PU_SL`；其 `fit` 通过 Gaussian/linear basis 上的闭式线性系统求解，没有 loss 参数或 double-hinge 分支。代码片段及默认参数只用于来源行为比较，未执行、复制或引入上游实现。

## 论文目标与 toolbox 路径

| 项目 | 论文/官方范围 | toolbox 实现与核查 |
|---|---|---|
| 风险结构 | SCAR、总体先验 π 已知；正例项为 `−π E_P[g]`，U 项为 `E_U[ℓ(−g)]`；凸性要求论文指定的 linear-odd/convex 损失关系 | `UPUClassifier` 与 `UPULoss` 分别对 P/U 风险按各自样本数取均值；先验必填。数学 golden/梯度测试覆盖 squared/logistic。 |
| 平方型闭式目标 | `PU_SL.fit` 解 `(E_U[φφᵀ] + λ_src I)w = 2πE_P[φ] − E_U[φ]`（`pu_mr.py:42-47`） | `_fit_squared` 解 `(.5 E_U[φφᵀ] + λ_tool I)w = πE_P[φ] − .5E_U[φ]`；两边同乘 2 可知**同一设计矩阵下** `λ_tool=λ_src/2` 时系数相同。新增有限维代数回归锁定此对应关系，不运行上游代码。 |
| 损失分支 | 论文重点为 double hinge；锁定 `PU_SL` 只有固定平方型闭式路径（`pu_mr.py:12-20,26-47`） | toolbox 另实现 C-DH 的 SLSQP QP 及 C-LL 的 L-BFGS，属论文推导 clean-room 扩展，不是该源码路径的直接复刻。默认 `loss="squared"`，方法卡与台账记录其与论文主推 C-DH 不同，pilot recipe 必须显式选型。 |
| basis、截距与默认值 | `PU_SL` 默认 Gaussian、`sigma=.1`、`lam=1`、`n_basis=200`；`basis='lm'` 用 `homo_coord` 附加常数列，正则矩阵为 `λI`（含常数项）（`pu_mr.py:14-20,33-47`; `comcalc.py::homo_coord`） | toolbox 默认 linear、`reg_lambda=1e-3`；平方路径不拟合 intercept，且只正则化系数。RBF 核函数指数形式一致，但中心池按训练视图构造，抽样 RNG/顺序不承诺重放作者实现。函数/目标代数等价不等于默认配方相同。 |
| 正则与偏置 | 正则作用于模型系数；intercept 不惩罚 | 代码对 `α` 正则、不惩罚 intercept；方法卡列出的线性/RBF basis、solver 和数值参数是工程规格，不应读作论文全部固定。 |
| 视图校准 | 原文 case-control U 为总体边缘；项目 TS-OS 规则把承担 U 风险角色的数据由 `D_U` 扩为 `D_U∪D_P` | `fit(os_or_ts="ts")` 只替换 U 风险角色；π 与 P 项不变。RBF center pool 跟随 view，但中心数按校准前 `n_U` 截断；sample_weight 被忽略，因此不需互斥门。TS 是项目协议层适配，不是原文独立算法结论。 |
| 选模与验证 | 原文实验用论文指定的 PU zero-one CV 和其设置 | 本 estimator 无内置 validation/test 接口；项目正式候选/选模流程由 protocol 控制。方法卡记录 loss 默认及超参方案与论文原始实验并不完全一致。 |

## 验证证据与缺口

- `tests/unit/losses/test_upu_loss.py` 覆盖风险手算 golden、先验作用、logistic/squared 梯度及有限差分。
- `tests/unit/estimators/test_upu.py` 与 `test_upu_ts_view.py` 覆盖拟合、solver/参数、视图角色、RBF centers 与校准前后容量约束。
- 测试证明本实现与登记公式及工程不变量一致，不证明 `pywsl` 某个特定源码版本的逐行一致性，也不证明完整 ICML 2015 数据/调参/结果已复现。
- 来源治理缺口已补为可复核的 Git commit pin；该旧快照含 MIT 许可并声明 PyPI 版本 0.1.4，但来源锁没有把浮动 PyPI 最新包等同此提交。逐文件公式核对后，只在共用设计矩阵且 `lambda_tool=lambda_source/2` 时确认平方解代数一致；默认 basis、截距/正则语义、RBF 中心抽样不一致。`pu_sl` 调参 helper 第 98 行把 `np.empty` 以多个位置参数调用、第 114/119 行含变量名不一致、第 128 行引用未定义 `tmp`，静态上有明显执行缺陷；它不是 demo 使用的 `PU_SL` sklearn estimator，未执行该 helper。

结论：`official_bundle` 应理解为官方关联软件中有 uPU 实现，不等于 toolbox 三种 loss 都有官方源码对应。Squared 分支存在有限的闭式公式对应，但 basis/default/参数差异仍阻止整体复现声明；C-DH/C-LL 适配、TS calibration 需分层披露。该核查不改算法参数、冻结结果、准入或正式 recipe。
