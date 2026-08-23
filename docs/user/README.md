# 用户文档

本目录是 PU Learning Toolbox 的使用文档，按阅读路径组织：

1. **[快速开始](quickstart.md)**（5 分钟）——安装、CLI 三命令跑通演示数据、Python 最小片段。**从这里开始。**
2. **概念**（`concepts/`，想理解原理时读）：
   - [PU 问题设定](concepts/pu_problem.md)——标记机制、符号与 π 的角色
   - [SCAR/SAR 机制与识别边界](concepts/scar_sar.md)
   - [选择 PU 方法](concepts/method_selection.md)——选型决策原理
3. **操作指南**（`howto/`，做具体任务时读，按实验流程排序）：

   | 任务 | 文档 |
   |---|---|
   | 生成 SCAR/SAR 数据 | [howto/sar_simulation.md](howto/sar_simulation.md) |
   | 数据画像与假设提示 | [howto/data_profiling.md](howto/data_profiling.md) |
   | PUPipeline 端到端训练评估 | [howto/pipeline.md](howto/pipeline.md) |
   | 使用命令行接口 | [howto/cli.md](howto/cli.md) |
   | 调整模型与搜索超参数 | [howto/model_tuning.md](howto/model_tuning.md) |
   | 使用图形界面 | [howto/ui.md](howto/ui.md) |
   | 生成诊断报告 | [howto/diagnostic_reports.md](howto/diagnostic_reports.md) |
   | 类先验与标记倾向敏感性分析 | [howto/sensitivity_analysis.md](howto/sensitivity_analysis.md) |
   | 审计并处理源域到目标域的分布漂移 | [howto/distribution_shift.md](howto/distribution_shift.md) |
   | 训练 Self-PU 分类器 | [howto/self_pu.md](howto/self_pu.md) |

4. **API 参考**（[reference/api.md](reference/api.md)，查精确参数时读）

推荐顺序：快速开始 →（可选概念）→ 按任务的 howto。每篇 howto 开头有前置条件、结尾有下一步。

开发者与贡献者文档见 [docs/README.md](../README.md) 的开发者栏。
