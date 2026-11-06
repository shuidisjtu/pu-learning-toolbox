# 用户文档

本目录面向**库使用者**（含第三方仓库中调用本工具箱 API 的开发者），
是 PU Learning Toolbox 的使用文档，按任务类型分三层：concepts（原理）/
howto（操作）/ reference（契约），按阅读路径组织：

1. **[快速开始](quickstart.md)**（5 分钟）——安装、CLI 三命令跑通演示数据、Python 最小片段。**从这里开始。**
2. **概念**（`concepts/`，想理解原理时读）：
   - [PU 问题设定](concepts/pu_problem.md)——标记机制、符号与 π 的角色
   - [PU 输入采样与训练风险视图](concepts/pu_sampling.md)——OS/TS、有限池、原生 U 与实际损失池
   - [SCAR/SAR 机制与识别边界](concepts/scar_sar.md)
   - [类先验的目标、来源与换算](concepts/class_prior_semantics.md)——总体/U/风险池/拼接表
   - [PU 选模](concepts/pu_model_selection.md)——PA/OA 条件、独立选择与监督预算
   - [表征与训练路径](concepts/representations.md)——encoder、端到端 CNN 与冻结特征
   - [选择 PU 方法](concepts/method_selection.md)——选型决策原理
3. **操作指南**（`howto/`，做具体任务时读，按实验流程排序）：

   | 任务 | 文档 |
   |---|---|
   | 生成 SCAR/SAR 数据 | [howto/sar_simulation.md](howto/sar_simulation.md) |
   | 构造同池 OS 与独立 TS 数据 | [howto/pu_sampling.md](howto/pu_sampling.md) |
   | 数据画像与假设提示 | [howto/data_profiling.md](howto/data_profiling.md) |
   | PUPipeline 端到端训练评估 | [howto/pipeline.md](howto/pipeline.md) |
   | 使用命令行接口 | [howto/cli.md](howto/cli.md) |
   | 调整模型与搜索超参数 | [howto/model_tuning.md](howto/model_tuning.md) |
   | 定制四角色实验与扩展策略 | [howto/custom_experiments.md](howto/custom_experiments.md) |
   | 使用图形界面 | [howto/ui.md](howto/ui.md) |
   | 生成诊断报告 | [howto/diagnostic_reports.md](howto/diagnostic_reports.md) |
   | 类先验与标记倾向敏感性分析 | [howto/sensitivity_analysis.md](howto/sensitivity_analysis.md) |
   | 审计并处理源域到目标域的分布漂移 | [howto/distribution_shift.md](howto/distribution_shift.md) |
   | 训练 Self-PU 分类器 | [howto/self_pu.md](howto/self_pu.md) |

4. **API 参考**（[reference/api.md](reference/api.md)，查精确参数时读）

推荐顺序：快速开始 →（可选概念）→ 按任务的 howto。每篇 howto 开头有前置条件、结尾有下一步。

已有真实 PU 数据：先确认[采样来源](concepts/pu_sampling.md)，再检查
[标记假设](concepts/scar_sar.md)与[先验目标](concepts/class_prior_semantics.md)，
然后使用[数据画像](howto/data_profiling.md)和 [Pipeline](howto/pipeline.md)。

合成/对照研究：从[问题维度](concepts/pu_problem.md#6-用独立维度描述研究设置)出发，
依次读采样、标记政策、先验、[选模](concepts/pu_model_selection.md)与
[表征](concepts/representations.md)，再运行[四角色 DIY 示例](howto/custom_experiments.md)。
采样操作见[OS/TS 教程](howto/pu_sampling.md)与[示例 15](../../examples/minimal/15_pu_sampling.py)。
先验传递见[示例 16](../../examples/minimal/16_prior_targets.py)与[Pipeline 指南](howto/pipeline.md)。
encoder 完整教程按[实施进度](../dev/process_checklist.md)继续推进。

开发者与贡献者文档见 [docs/README.md](../README.md) 的开发者栏。
