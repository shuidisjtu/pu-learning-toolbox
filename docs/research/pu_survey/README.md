# PU Survey

工具箱提供可复用的实验运行与结果报告能力。科学设置见[实验协议](pu_survey_protocol.md)，实际操作见[运行指南](runbook.md)，公共 API 见[实验层设计](../../dev/experiment_layer.md)。

## 交付物与完成定义

**A. 结果文件生成能力**

- 从已有 v3 manifest 或 summary 生成一个含四组结果页的工作簿：SCAR–PA、SCAR–OA、SAR–OA（LBE-A/B 分开）、PN oracle。训练路径、数据集、视图和 support 分开，不生成跨数据集总排名。
- Accuracy 按五个 seed 计算均值与样本标准差；AUC、F1、Precision、Recall 是附加指标。缺 seed、无定义指标、失败、数值风险与排除项明确展示，不补零或补造结果。
- 单配置成本、完整调参成本和峰值显存保留；PA/OA 引用同一份 fit 成本，避免重复计费。
- 输出 `survey_results.xlsx`、`diagnostic_cards.md/json`、`conclusions.md/json`。结论区分文献事实、本实验观测和推断，说明所需证据缺失的诊断信号。

**B. 代码功能**

- `ExperimentRunner.fit(model, train, pu_val, clean_val, test)`：四角色隔离、PA/OA 独立选模与独立测试。
- 可注入数据生成器、训练器和选模策略；DIY 实验无需读取内置 Survey 配置。
- 官方入口读取用户准备的四路数据，记录候选轨迹、选择记录、所选权重与指标。
- PN oracle 与 PU 对照共享 split、表征和预算，仅使用 OA。
- 保存 seed、实际配置、split/视图/先验语义、环境与代码/依赖身份，支持从数据到报告的完整命令链。

## 配置与结果读取

新运行只维护 `survey-v3.2`（[执行配置](../../../pu_toolbox/experiment/survey_protocol_v3_2.json)）和 `survey-recipes-v1.2`（[配方](../../../pu_toolbox/experiment/survey_recipe_registry_v1_2.json)）。
配方锁定与可运行标识表示工程执行能力，不代表论文数值复现、结果审核完成或正式准入。

已有 v3.0/v3.1/v3.2 manifest 通过只读结果链使用，保留原版本、blockers、失败和选模身份。已有 summary 可直接导出，无需旧运行配置、比较矩阵或审批文档。读取不会改写训练产物，也不会自动清除正式资格限制。

方法能力与来源见[算法注册表](../../../pu_toolbox/registry/builtin_methods.py)和[方法卡](../method_cards/)；[方法元信息](../../../pu_toolbox/experiment/method_ledger.json)仅保存视图、校准与先验说明。文献事实可通过 `--literature` 显式提供；原生 OS/TS 诊断分组可通过 `--native-views` 提供，缺少时报告缺失原因。

## 当前实验状态

训练任务已经结束，训练产物与已有结果另行保存。用户负责的 360 fit 待 HENG958 审核整理；GradPU/KLDCE 暂时搁置，本次重构不恢复训练、不重写 estimator，也不追认数值风险结果。实验软件是否可用与已有结果是否完成审核分别判断。

工具箱不再保存队列、服务器排程、交接、修复任务单和全过程审批流水。历史可从清理前提交 `7d11f2c4` 查阅，已保存 manifest 不改标为新版本。数据来源、许可与 ADNI 替代数据事实继续保留，见[数据说明](data/README.md)。
