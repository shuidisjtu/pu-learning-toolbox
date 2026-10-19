# 16候选生命周期机器门禁（2026-10-19；P3MIX补记）

补P3.1/P3.2工程证据与P4.1-A接口核验的衔接，不批准正式配置或算法准入。
前批修复虽已在台账留痕，原机器预算草稿只核五方法；新增只读入口
`scripts/check_candidate_fit_contracts.py`，统一检查16候选的声明与当前身份。

## 覆盖与命令

PULDA、PUET、GradPU、Robust-PU、Split-PU、GEN-PU、Holistic-PU、LaGAM、PULNS、
独立TEDn、VPU、神经LBE、PAN、CVIR、RP及P3MIX。TEDn仍在台账CVIR的独立扩展下，
没有新增正式注册。2026-10-19 用户提供完整论文后，P3MIX 已有独立完整 estimator；其静态
fit 声明现纳入本门禁，具体失败重训回归见专项测试及[补充记录](remaining_candidate_fit_failure_20261019.md)。

```bash
python scripts/check_candidate_fit_contracts.py
python scripts/check_candidate_fit_contracts.py --check \
  docs/research/pu_survey/data/candidate_fit_contracts_20261019_draft.json
```

默认只向stdout输出严格JSON；核验模式只读记录，无写文件、无训练、无制品反序列化、
无Torch/estimator导入。静态CI增加对该工程草稿的必需检查，不把草稿称为正式lock。
后续相关源码或工程审阅文档变化须显式重新生成并审阅身份，不抹去历史记录。

## 核验内容与边界

- 每个声明类必须直接声明唯一fit，恰有一次共享guard装饰及直接import；
  拒绝可吞未知标签的可变参数、y_true/test_data/clean_val入口。
  PULNS/LaGAM的显式support_data保留，不伪称PU-only。
- 绑定实现、共享guard、检查器及工程审阅文档的规范化文本SHA；
  Windows/Linux统一换行，不以CRLF差异伪造源码变化。
- 保存参数名称/位置/关键字类别、必填或默认值，以及台账view/calibration事实；
  不自动换算prior、改风险角色、增加标签预算或放行阶段。
- 拒绝缺失/越出仓库的证据文件和假批准字段；formal_admission始终false、签署为空。

这是AST声明及身份核对，不是任意Python的安全验证或训练行为证明。
BaseException传播、无重试、签名透明、失效后拒绝公开预测等运行语义仍由
`test_fit_state.py`、原候选/扩展生命周期与RP无Torch测试验收。
不证明事务回滚、外部制品清理、并发安全、资源上界或论文数值复现。

## 工程验收

检查器专项27项通过、零跳过/失败/错误，5.825秒（旧15候选截点）；更新为16候选后的
专项结果以本轮测试运行记录为准。包括逐方法删guard的反例、
guard import/额外clean角色/kwargs/重复装饰/越界引用拒收、源码变化与CRLF、
陈旧/伪准入记录和CLI，以及禁止Torch导入的独立进程。
原协议、矩阵、历史资源/结果和负责人签署保持不动；全仓最终验收在本轮结束记录。
