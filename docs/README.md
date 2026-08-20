# PU Learning Toolbox 文档

## 用户文档（docs/user/）

从 [快速开始](user/quickstart.md) 开始，5 分钟跑通首个实验。

| 类别 | 文档 | 用途 |
|---|---|---|
| 入口 | [user/README.md](user/README.md) | 用户旅程图（从哪篇开始、按什么顺序读） |
| 快速开始 | [user/quickstart.md](user/quickstart.md) | 安装 + CLI 三命令 + Python 最小片段 |
| 概念 | [user/concepts/pu_problem.md](user/concepts/pu_problem.md) | PU 问题设定、符号表、π 的角色 |
| 概念 | [user/concepts/scar_sar.md](user/concepts/scar_sar.md) | SCAR/SAR 机制与识别边界 |
| 概念 | [user/concepts/method_selection.md](user/concepts/method_selection.md) | 选型决策原理（推荐器 + 决策表） |
| 操作 | [user/howto/pipeline.md](user/howto/pipeline.md) | PUPipeline 端到端训练评估 |
| 操作 | [user/howto/cli.md](user/howto/cli.md) | 命令行接口 |
| 操作 | [user/howto/model_tuning.md](user/howto/model_tuning.md) | 模型参数与 PU-aware 网格搜索 |
| 操作 | [user/howto/ui.md](user/howto/ui.md) | 图形界面安装与使用 |
| 操作 | [user/howto/data_profiling.md](user/howto/data_profiling.md) | 数据画像与假设提示 |
| 操作 | [user/howto/diagnostic_reports.md](user/howto/diagnostic_reports.md) | 生成诊断报告 |
| 操作 | [user/howto/sensitivity_analysis.md](user/howto/sensitivity_analysis.md) | 类先验与标记倾向敏感性分析 |
| 操作 | [user/howto/distribution_shift.md](user/howto/distribution_shift.md) | 分布漂移审计与协变量加权适配 |
| 操作 | [user/howto/sar_simulation.md](user/howto/sar_simulation.md) | 生成 SCAR/SAR 数据 |
| 操作 | [user/howto/self_pu.md](user/howto/self_pu.md) | 训练 Self-PU 分类器 |
| 操作 | [user/howto/using_skill.md](user/howto/using_skill.md) | 启用与使用 pu-workflow Skill |
| 参考 | [user/reference/api.md](user/reference/api.md) | 核心 API 精确契约 |

## 开发者文档（docs/dev/）

贡献前必读；文档间的权威顺序见 [CONTRIBUTING.md](../CONTRIBUTING.md) 第 1 节。

| 文档 | 用途 |
|---|---|
| [dev/architecture.md](dev/architecture.md) | 当前架构:模块分层、数据流、注册表 |
| [dev/project_structure.md](dev/project_structure.md) | 目录结构（权威来源） |
| [dev/compatibility.md](dev/compatibility.md) | Python/依赖支持矩阵、CI 职责与构建策略 |
| [dev/architecture_audit.md](dev/architecture_audit.md) | 审计发现快照、复跑指南与治理机制（ADR-0001） |
| [dev/process_checklist.md](dev/process_checklist.md) | 进度清单与发布状态（权威来源） |
| [dev/distribution_shift_aware_pu.md](dev/distribution_shift_aware_pu.md) | 分布漂移感知 PU 的假设、实现边界与验收标准 |
| [dev/distribution_shift_aware_pu_checklist.md](dev/distribution_shift_aware_pu_checklist.md) | 分布漂移功能补充任务清单与完成证据 |
| [dev/release_process.md](dev/release_process.md) | 发布流程（版本策略、预检、上传、回滚、维护） |
| [research/method_cards/](research/method_cards/) | 17 篇论文方法卡（公式、复现状态、实现边界） |

## 架构决策(docs/adr/)

| 文档 | 用途 |
|---|---|
| [adr/](adr/) | ADR 索引(14 篇:架构治理/解耦/复现分级/测试 CI/流程惯例/发布策略/目录合并/方法卡清洗等) |

> ADR 记决策,版本/进度状态见 dev/process_checklist.md 与 dev/release_process.md。

## 其他

- [../README.md](../README.md)：项目门面（英文）；[../README.zh-CN.md](../README.zh-CN.md)：中文版
- [../CONTRIBUTING.md](../CONTRIBUTING.md)：代码贡献、论文复现状态与 benchmark 产物管理
- [../examples/minimal/](../examples/minimal/)：12 个最小可运行示例脚本
- [../benchmarks/deep_pu/README.md](../benchmarks/deep_pu/README.md)：深度 PU benchmark（InfoMax PU、WConPU、DGPU runner 与多 seed 结果）
