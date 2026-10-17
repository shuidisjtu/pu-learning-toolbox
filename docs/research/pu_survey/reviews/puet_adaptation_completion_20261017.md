# PUET 审核补项：归一化证据与适配回归（2026-10-17）

承接合作者审核分支 `adb00c4` 的 P3.1 记录中三项非阻断补充：#114 风险池质量证据
同步、独立适配回归、max_features 越界差异登记。本轮不合并该分支、不代签，不改变
PUET 的训练代码、默认配置、历史冻结结果或正式准入状态。

## 来源核对

- 作者 checkout：`jonathanwilton/PUExtraTrees` @
  `5cb15d7e021a6e24d4cd300278500c38729e2775`；重新核对 `tree.py`、`trees.py`、
  `LICENSE` 的 SHA-256，见既有五方法机器证据包。源码只读，不运行作者训练代码。
- [论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/98257285340854262185500e59bc0f28-Paper-Conference.pdf)
  §3 Proposition 2(a) 是所实现的 nnPU 二次节点风险；§4 描述分别对 P/U bootstrap。
  不因此推导已重放论文全部模型或实验流程。
- [锁定作者源码](https://github.com/jonathanwilton/PUExtraTrees/blob/5cb15d7e021a6e24d4cd300278500c38729e2775/tree.py)：
  199–207行处理 max_features；319行只检查父节点大小；347–348行抽裁边阈值；
  353–363行选择最小子风险，不要求其小于父风险。

max_features 大于输入维数时，作者打印提示并返回 None，**不是 raise ValueError**。
工具箱目前对正整数截断到输入维数，保留原构造参数。审核允许 fail-loud 或登记，
本轮选择明确登记并测试既有语义，不为了源码对齐而改变已存在的配置行为。

## #114：归一化证据补齐

每棵树的正例 bootstrap 仍取 n_P 行；U bootstrap 取**实际 U 风险池的全部行数**：
OS 为 n_U，TS-compatible 为 n_U+n_P（含 P 角色副本）。对应每行权重分母与抽样数
一致，根节点总质量 P=总体π、U=1。这里是树根质量，不是把每个子节点各自重归一化。
原总体先验不重新估计，原 P 行的观测标签不变，不引入验证或测试真值。

既有 `test_edge_bootstrap_root_masses_and_constant_leaf` 的36组参数回归现纳入
`source_behavior_tests`。结构化 bootstrap_risk_contract 同步到方法台账、来源证据
calibration 和预算草稿 calibration_fact；检查器及其变异测试阻止缺字段、台账/证据
共同错误改值、把 bool 当质量数、使用原 n_U 作为 TS 抽样数、删除关键回归指针。
签署、正式候选/预算、资源批准均不变；未改真实执行代码或冒称新算法校准完成。

## 新增独立适配回归

文件 `tests/unit/estimators/test_puet_adaptation_contract.py`：

| 要求 | 独立算例 | 证明范围 |
|---|---|---|
| 子节点最小样本 | 4P/4U；唯一切分为1P和3P/4U；根风险0.96，子风险和0.84 | 父节点大小通过且 gain=0.12>0；min_samples_leaf=1允许、2拒绝，不能以父节点或负增益测试替代 |
| 完整阈值区间 | 值0/1/9/10，固定随机分位0.01/0.99 | 工具箱阈值0.1/9.9落在作者裁边区间[0.4,9.6]之外，且实际被选为有效分裂 |
| 严格负增益拒绝 | 10P/10U，切为9P/1U和1P/9U，先验0.95或0.99 | 先验0.95时根风险0.19，子风险和约0.339889；负增益真实存在，保持单叶与零重要度，不冒用零增益算例 |
| 越界特征数语义 | 三维输入，max_features=4或100，与all比较 | 同 seed 下边际、重要度、叶数逐值相同；构造参数仍为请求值；不宣称与作者拒绝拟合等价 |

这些测试锁定**工具箱已披露的适配**，并不把它们认证为作者实现或论文数值重放。
bootstrap 的36组回归与本次8组适配回归作用不同，互不替代。

临时内存变异检查逐项移除子叶约束、改为裁边区间、允许首个负增益、改写构造特征预算，
四项均被对应新回归检出；未修改生产源码文件。七组检查器变异另覆盖机器契约漂移。
证据包更新触发的 draft 绑定漂移也已同步到公开对照/预算草稿：只刷新非冻结证据摘要与
calibration_fact，不改旧协议/矩阵摘要、读数、numeric_eligible 或正式批准字段。

## 重复核验

```bash
python scripts/check_p3_admission_evidence.py
python scripts/check_p3_admission_evidence.py --source-dir puet=PATH_TO_PINNED_CHECKOUT
python scripts/prepare_p3_selection_budget_spec.py \
  --check docs/research/pu_survey/data/p3_selection_budget_spec_20261007_draft.json
pytest tests/unit/estimators/test_puet.py \
  tests/unit/estimators/test_puet_adaptation_contract.py \
  tests/unit/estimators/test_p3_source_behavior.py \
  tests/unit/estimators/test_p3_budget_selection_evidence.py \
  tests/unit/scripts/test_check_p3_admission_evidence.py \
  tests/unit/scripts/test_prepare_p3_selection_budget_spec.py \
  tests/contract/test_p3_candidate_admission.py
```

仍待：正式候选/CPU预算与存储、方法特定 PA/OA 资格、正式共享规格与必要公开数值、
负责人审阅。GPU 条款对 CPU 树方法的适用性仍按既有任务边界处理，不代为批准。
P3MIX 完整原始来源与 estimator 缺口没有被本轮 PUET 补项替代或宣布完成。
