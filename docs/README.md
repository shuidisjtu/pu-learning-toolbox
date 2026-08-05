# PU Learning Toolbox

## 1. 项目定位

在只有"已标记正样本 + 无标记样本"的条件下，提供数据校验、类先验估计、模型训练、评估、算法选择与论文方法复现。
- **兼容 sklearn**，非 PU 专家也能快速训练基线模型。
- **先搭框架再集成算法**，当前 16 个方法均为 clean-room 实现（NATIVE）。
- **差异化支持 SAR / Instance-Dependent PU**，不局限于 SCAR 假设。

## 2. 基本问题设定

| 符号 | 含义 |
|---|---|
| $x$ | 样本特征 |
| $y \in \{0,1\}$ | 真实类别标签（训练时不可完全观测） |
| $s \in \{0,1\}$ | 是否被标记为正类 |
| $\pi = P(y=1)$ | 类先验 |
| $c = P(s=1 \mid y=1)$ | SCAR 下正类被标记的常数概率 |
| $c(x) = P(s=1 \mid y=1, x)$ | SAR 下实例相关的标记倾向 |

数据场景与假设的详细分类见 [`method_selection.md`](method_selection.md)。

## 3. 分层依赖

| 层级 | 依赖 | 内容 |
|---|---|---|
| Core | numpy, scipy, pandas, scikit-learn | 标签处理、校验、经典算法（含 uPU）、类先验、指标、CV、registry |
| Torch Extension | torch | nnPU, Dist-PU, 深度表征学习 |
| Research Extension | torchvision, lightning | 图像 benchmark, 复杂论文复现 |

```bash
pip install pu-toolbox
pip install pu-toolbox[torch]
pip install pu-toolbox[research]
pip install pu-toolbox[all]
```

## 4. 论文覆盖

覆盖 16 篇 PU Learning 论文，按 5 个算法族组织。算法选型、场景/假设匹配见 [`method_selection.md`](method_selection.md)，论文→模块映射见 [`architecture.md`](architecture.md) 的“论文方法到模块的映射”部分，源码状态见 [`resources_optimized.md`](resources_optimized.md)。

## 5. 文档索引

| 文件 | 作用 |
|---|---|
| `README.md` | 项目总览、定位、分层依赖、文档索引 |
| `architecture.md` | 包结构、基类 API、数据流、注册表设计 |
| `project_structure.md` | 完整目录结构（权威来源） |
| `method_selection.md` | 算法分类、选型逻辑、推荐器设计 |
| `development_roadmap.md` | Phase 0–6 任务拆分、版本规划、实施优先级 |
| `development_compatibility.md` | Python/依赖支持矩阵、CI 职责和构建策略 |
| `resources_optimized.md` | 论文源码状态、URL、集成策略 |
| `user/sar_simulation.md` | SCAR/SAR propensity、PU 标签和合成 benchmark 使用指南 |
| `user/data_profiling.md` | PU 数据质量画像、SCAR/SAR 证据边界与行动建议 |
| `user/diagnostic_reports.md` | 模型输出、指标证据级别及 JSON/Markdown 诊断报告 |
| `user/sensitivity_analysis.md` | 类先验/平均标记倾向假设扫描、相容性与指标区间 |
| `user/self_pu.md` | Self-PU clean validation、三阶段训练、审计状态与复现边界 |
| `user/pipeline.md` | PUPipeline 端到端工作流：画像→先验→训练→CV→评估 |
| `user/cli.md` | pu-toolbox 命令行接口：run / list-methods / list-priors / make-demo-data |
| `project_management/decision_log.md` | 项目决策日志 |
| `project_management/process_checklist.md` | 开发流程检查清单 |
| `project_management/division.txt` | 任务分工说明 |
| `../benchmarks/deep_pu/README.md` | InfoMax PU、WConPU、DGPU runner、配置锁与多 seed 结果 |

代码贡献、分支、测试、论文复现状态和 benchmark 产物管理见项目根目录
[`CONTRIBUTING.md`](../CONTRIBUTING.md)。
