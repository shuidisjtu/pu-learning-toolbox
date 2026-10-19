# KLDCE 来源与行为核查（2026-10-19）

范围：核对 KLDCE 方法卡、生产求解路径、SMO 实现及现有测试与 Gong et al. 原论文/在线附录的对应关系。属于单方技术来源审计，不是方法负责人签署、正式准入或论文结果复现。

## 来源核对

- 原论文为 Gong et al., “Loss Decomposition and Centroid Estimation for Positive and Unlabeled Learning,” IEEE TPAMI 43(3), 2021。作者公开稿可在[作者主页 PDF](https://gcatnjust.github.io/ChenGong/paper/gong_pami2019.pdf)核对；出版信息与 DOI `10.1109/TPAMI.2019.2941684` 可在[PubMed 书目记录](https://pubmed.ncbi.nlm.nih.gov/31535983/)核对。
- 论文 §2–3 定义 censoring PU：单一 i.i.d. 总体样本，正例以概率 `η` 被隐去，负例不会变成观测正例；式 (1)–(3) 给出噪声与类先验关系。作者稿 PDF 第 2–3 页（PDF 页序）对应上述定义。
- 核化 KLDCE 与 appendix SMO 的逐式说明以当前[方法卡](../../method_cards/KLDCE.md)记录的正文 §6.2、在线补充附录 Algorithm 1 和式 (21)–(40) 为来源索引。公开稿正文能核对方法设定，但不能单独代替在线附录中的 SMO 规格。

2026-10-19 来源身份更正：仓库此前曾把作者页 `CEGE_PAMI20.rar` 写成 KLDCE 的 `upstream_url`，与“未找到 KLDCE 官方代码”的方法卡结论相矛盾。该压缩包实际是 CEGE 会议早期版本，属于相关线性 LDCE 家族的历史材料，不是 TPAMI 终稿 KLDCE 实现；因此 KLDCE registry/ledger 的来源状态已改为 `not_found`、`upstream_url=null`。线性 `centroid_pu` 的相关 archive 注记不变。`not_found` 只表示本次审阅未找到可确认的 KLDCE 作者实现，不证明其他位置不存在实现。

2026-10-19 补查作者维护的[出版物目录](https://gcatnjust.github.io/ChenGong/publications.html)：TPAMI 2019 条目列出该论文及 paper 链接，没有 code 链接；同一目录对若干其他论文会明确列出 `[code]`。这支持“作者目录当前未提供 KLDCE 官方实现入口”的有限结论，但不能证明作者从未在其他位置发布代码，也不改变 `not_found` 的非绝对含义。目录把论文版权归属出版社，并注明页面所列代码仅供研究、商业使用需许可；该声明不是 KLDCE 软件许可证，不能据此推定存在或缺少具体代码授权。

## 当前实现行为

| 项目 | 代码证据 | 核查结论 |
|---|---|---|
| `fit` 生产内层求解器 | `pu_toolbox/estimators/risk/kldce.py` 中 ACS 循环调用 `_solve_dual_smo`；`_solve_qp_oracle` 没有被 `fit` 调用 | 生产路径是本地 SMO；SLSQP 仅作测试/诊断 oracle。模块与类 docstring 原先仍称生产 QP oracle，现已更正。 |
| 目标与约束 | `kldce_smo.py::_solve_dual_smo` 对 `_build_dual_qp` 给出的同一个 `Q,d,Aeq,lb,ub` 做解析可行 pair step，并逐步保持线性等式 | 与固定质心联合对偶 QP 相符；更新具有约束可行性。 |
| 配对选择 | `_solve_dual_smo` 将完整 `z=[α;γ]` 交给 `_smo_pair_select`，在拼接变量上全局选择一对 | **不是附录 Algorithm 1 中 α/γ 两组分别选择并更新的字面行为**。它是对联合 QP 的广义 pair-coordinate 求解策略；不可称作逐行或逐组复现。 |
| 外层更新顺序 | `KLDCEClassifier.fit` 先固定 μ 求联合 QP，再更新 μ | 与附录 Algorithm 1 先更新 μ、再运行 SMO 的顺序不同；属于显式 ACS 顺序适配。 |
| 核与质心 | 生产实现限定 RBF，按 Gaussian Taylor 质心公式更新；退化时可单调回滚/锁定 μ | 论文推导边界及数值保护均须按方法卡中的实现决策披露；回滚不是论文原算法。 |

## 测试证据与限制

- `tests/estimators/risk/test_kldce_smo.py` 覆盖解析 pair step、等式/盒约束、KKT 终止、退化停滞不误报收敛、SMO 与 SLSQP 的 dual objective 一致性，以及 fit-level 收敛 smoke。
- 这些测试支持“实现解同一个固定 μ 的盒约束等式 QP，并与数值 oracle 相符”；它们**不证明**与附录按 α/γ 分组的更新次序逐步等价，也不证明论文全部实验协议已复现。
- 论文未定义 KKT 违反度的具体数值选择、所有边界/退化处理及工程停止细节；方法卡中对应选择仍属实现约定。

## 结论 / 后续

结论：原生 SMO 已进入生产 `fit`，过期的“SMO deferred”及“fit 使用 SLSQP”说明已修正。联合 QP 目标、oracle 对照和收敛行为有现存测试支撑。实现相对论文仍有两项明确差异：全体 `[α,γ]` 联合选对（非两组分开更新）以及 ACS 块更新顺序不同；论文未定义的选择/容差仍是实现自由度。

不在本次自行改为分组 SMO：这会改变求解策略和既有数值行为，应由方法负责人依据附录原文复核后决定。KLDCE 的来源/实现状态继续待 RV8 负责人复核；该审计不改变冻结矩阵、历史结果或准入状态。
