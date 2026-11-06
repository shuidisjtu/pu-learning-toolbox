# PU Survey 当前状态与能力验收索引

更新：2026-11-06。本文记录当前交付边界，不再维护历史任务流水和服务器排程。
实验规则见[协议](pu_survey_protocol.md)，操作入口见[运行指南](runbook.md)。

## 1. 当前状态

- 训练任务已结束；算法结果已整理为表格和文档，正式训练产物已保存。
- 原合作者 360 fit 由用户负责，仍待 HENG958 审核与整理；代码重构不替代结果审核。
- GradPU、KLDCE 的问题暂时搁置；本轮不重写 estimator、不恢复训练、不补判结果有效。
- 新运行使用 `survey-v3.2` 和 `survey-recipes-v1.2`，文献事实由报告调用方按需提供。
  `runnable` 表示实现可执行，配方锁定表示项目配置已明确，均不等于论文数值复现或结果验收。
- 实验层保留结果报告生成能力及公共实验 API；历史队列、交接审批和阶段凭证已退出运行依赖。

## 2. 交付能力与验收

| 能力 | 入口与验证对象 | 验收要求 |
|---|---|---|
| 四角色实验 | `ExperimentRunner`、`DatasetBundle`、`DatasetPart` | train / pu_val / clean_val / test 隔离，自定义生成器与 trainer 可注入 |
| 独立选模 | `ProtocolPA`、`ProtocolOA`、checkpoint trainer | PA/OA 分别选择配置、阶段/快照与阈值；test 仅评测 |
| 官方执行示例 | `scripts/run_survey_experiment.py` | 已准备数据可运行；支持当前项目配方及 PN oracle |
| 范围规划与续跑 | `scripts/run_survey_pilot.py` | 数据集/方法/路径交集；续跑核视图、profile、配方与 split 身份 |
| 结果统计 | `scripts/summarize_survey_results.py` | 同配置五 seed 的均值和样本标准差；缺 seed、失败、缺指标明确展示 |
| 工作簿与结论 | `scripts/export_survey_results.py` | SCAR–PA、SCAR–OA、SAR–OA、PN oracle 分组；保留诊断卡和结论依据 |
| 最小复现信息 | manifest、selection、所选权重引用 | seed、split 身份、配置、视图/先验、选择记录、代码与环境身份可解释 |

公共接口的设计边界见[实验层设计](../../dev/experiment_layer.md)与
[ADR-0018](../../adr/0018-experiment-layer-public-api.md)。
配置的五 seed 完整性、统计有效性与是否审核完成分别表达，不能相互替代。

## 3. 科学约束与结果解释

- test 不参与先验、候选、阈值或 checkpoint 选择；PA 只读取 PU 验证标签，OA 读取独立干净验证集。
- PULNS、LaGAM、Self-PU full-meta 的独立 clean support 属于额外训练监督。
  从 train 划出 support 时必须移除原行，并记录正负标签数及角色身份；不能复用 clean_val/test。
  `PA+clean-support` 与纯 PU PA 分列，SAR 的 LBE-A / LBE-B 也分别展示。
- OS/TS 按实际运行视图与先验语义分组，不给旧结果换标签；PN oracle 每 dataset/path/seed 一次，与 c 无关。
- 训练成本按实际 fit 记账，PA/OA 不重复收费；树、fold、GAN 阶段、episode 与神经 epoch 不混称。
- checkpoint 是推理权重和选择依据，不自动构成 optimizer/RNG 训练续跑能力。
- 文献事实、实验观测和推断分别表达。文献锚点可选；没有来源或足够观测时不生成支持性结论。
  方法差异与来源边界见[方法卡](../method_cards/)，数据身份与许可见[数据说明](data/README.md)。

## 4. 验证入口与剩余工作

- 通用 API、隔离与选模：`tests/unit/experiment/test_runner.py`、`test_experiment_dependencies.py`。
- 当前配置与续跑：`test_survey_protocol.py`、`test_pilot_recipe_resume.py`、`test_pilot_resume_view.py`。
- 统计与报告：`test_results.py`、`test_survey_workbook.py`、`test_survey_conclusions.py`。
- 数据准备：`scripts/prepare_survey_splits.py`、`scripts/prepare_clean_support.py`；来源事实保存在本地
  数据 manifest 和[数据说明](data/README.md)所列机器记录中。

软件验收以实际执行的检查为准；结果审核、GradPU/KLDCE 后续决定和对外发布范围仍由研究团队完成。
历史提交可以用于追溯实现变化，工具箱不再保存整套实验过程凭证。
