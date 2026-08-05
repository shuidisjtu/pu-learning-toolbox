# 文档体系重构设计

> 状态：已批准，待实施。实施完成后本文档蒸馏为正式记录（参考 `cli_design.md` 的先例），期间作为重写与迁移的执行依据。
> 日期：2026-08-05

## 1. 背景与现状问题

当前 22 篇 markdown 文档存在四类问题：

1. **受众混排**：`docs/README.md` 索引 22 行平铺，用户指南、架构文档、进度清单、方法卡、任务分工并列一张表；`division.txt` 已不存在仍残留在索引中（失效链接）。用户需自行分辨哪些文档属于自己。
2. **用户文档无主线**：`docs/user/` 下 7 篇各自独立成文（"这是某某 API"形态），缺少从「数据 → 画像 → 选型 → 训练 → 评估 → 敏感性」的旅程串联；`pipeline.md`（最接近主线的文档）被写成类 API 参考而非使用路径；`self_pu.md`（方法特定指南）混入通用指南。
3. **质量两极分化**：`sar_simulation` / `data_profiling` / `diagnostic_reports` / `pipeline` / `cli` / `sensitivity_analysis` 信息密度高、有数学定义与边界说明；`method_selection.md` 选型表格原因列是空话（"简单直观，SCAR baseline"），按"论文目录"而非"我的数据"组织；根 `README.md` Features 空洞罗列；`development_roadmap.md` 中英标题混排、已完成/未完成混杂、与 `process_checklist.md` 重复；`architecture.md` 核心原则宣言式。
4. **语言混乱**：README 英文（另有 zh-CN 版）、docs 主体中文、部分标题英文（"Method Selection Guide" 等）。

## 2. 目标

- 用户文档与开发者文档在**目录层面**清晰分离（scikit-learn 的受众分离原则）。
- 用户文档形成**显式旅程主线**（Claude Code 的 Quickstart 主线 + 页面串联原则），文档之间、小节之间可导航。
- 重写 4 篇低质量文档（`method_selection` / 根 `README` / `development_roadmap` / `architecture`），消除空话与重复。
- 语言统一：根 README 双语（英文 + zh-CN），docs/ 全部中文（含标题）。

## 3. 参考体系与启示

| 参考 | 启示 |
|---|---|
| scikit-learn 文档（install/User Guide/API/Examples/Development 分离） | 受众在目录层面物理分离；API 参考独立于指南 |
| Claude Code Docs（Quickstart 步骤化 + 核心概念/使用指南/参考分组 + 页面级"前置条件/请参阅/接下来呢"） | Quickstart 是主线入口；概念与操作分离；页面间织成旅程 |

## 4. 目标目录结构

```
docs/
  README.md                  ← 重写：文档首页（用户/开发者两栏导航 + 阅读路径）
  user/                      ← 用户文档
    README.md                ← 新建：旅程图（从哪篇开始 → 顺序 → 每篇用途）
    quickstart.md            ← 新建：快速开始（安装 + CLI 3 命令 + Python 最小片段）
    concepts/                ← 概念（解释原理）
      pu_problem.md          ← 新建：PU 问题设定与符号表
      scar_sar.md            ← 新建：SCAR/SAR 机制与识别边界
      method_selection.md    ← 重写：选型决策原理
    howto/                   ← How-to（目标导向操作指南）
      pipeline.md / cli.md / data_profiling.md / diagnostic_reports.md /
      sensitivity_analysis.md / sar_simulation.md / self_pu.md
    reference/
      api.md                 ← 新建：核心 API 精确契约
  dev/                       ← 开发者文档
    architecture.md          ← 重写
    project_structure.md     ← 迁移（内容不动）
    roadmap.md               ← 重写（原 development_roadmap.md）
    compatibility.md         ← 迁移（原 development_compatibility.md，标题改中文）
    resources.md             ← 迁移（原 resources_optimized.md，标题改中文）
  project_management/        ← 原位不动（process_checklist / decision_log / cli_design）
  research/
    method_cards/            ← 原位不动
```

### 完整映射表

| 现在 | 去向 | 动作 |
|---|---|---|
| `docs/README.md` | `docs/README.md` | 重写为文档首页（两栏导航） |
| `docs/method_selection.md` | `docs/user/concepts/method_selection.md` | 重写 |
| `docs/architecture.md` | `docs/dev/architecture.md` | 重写 |
| `docs/development_roadmap.md` | `docs/dev/roadmap.md` | 重写 |
| `docs/development_compatibility.md` | `docs/dev/compatibility.md` | 迁移 + 标题改中文 |
| `docs/project_structure.md` | `docs/dev/project_structure.md` | 迁移 |
| `docs/resources_optimized.md` | `docs/dev/resources.md` | 迁移 + 标题改中文 |
| `docs/user/` 下 7 篇 | `docs/user/howto/` | 迁移 + 标题改动作式 |
| `docs/project_management/*` | 原位 | 不动 |
| `docs/research/method_cards/*` | 原位 | 不动 |
| — | `docs/user/README.md` | 新建（旅程图） |
| — | `docs/user/quickstart.md` | 新建 |
| — | `docs/user/concepts/pu_problem.md` | 新建 |
| — | `docs/user/concepts/scar_sar.md` | 新建 |
| — | `docs/user/reference/api.md` | 新建 |

## 5. 用户文档旅程机制

**主线**：`quickstart` →（按需读 concepts）→ `howto/*` → 查参数时 `reference/api.md`

三套页面级机制贯穿全部用户文档：

1. **前置条件**（文档开头）：`> 前置条件：先完成 quickstart.md。相关概念：SCAR/SAR 见 concepts/scar_sar.md`
2. **下一步**（文档结尾）：每篇末尾给 1–3 个明确的旅程下一站链接（如 pipeline.md 结尾 →「分析假设敏感性：sensitivity_analysis.md」）
3. **参数不重复**：howto 只给关键参数，精确契约链到 `reference/api.md`

**概念与操作分离**：现有高质量文档中"解释原理"部分（sar_simulation 的机制定义、data_profiling 的识别边界等）抽到 `concepts/`，howto 只留操作。这是唯一的内容级重组，其余操作内容原样迁移。

**标题动作式**（howto 七篇 + method_selection 全列）：
- `sar_simulation.md` →「生成 SCAR/SAR 数据」
- `data_profiling.md` →「数据画像与假设提示」
- `diagnostic_reports.md` →「生成诊断报告」
- `pipeline.md` →「PUPipeline 端到端训练评估」
- `cli.md` →「使用命令行接口」
- `sensitivity_analysis.md` →「类先验与标记倾向敏感性分析」
- `self_pu.md` →「训练 Self-PU 分类器」
- `method_selection.md` →「选择 PU 方法」

**quickstart.md 内容来源**：根 README Quick Start 部分 + `cli.md` 三命令流程 + `pipeline.md` 最小片段，整理为步骤化叙事（Claude Code 式：开始前 → 步骤 → 接下来呢）。

## 6. 重写文档大纲

### 6.1 method_selection.md → 「选择 PU 方法」（user/concepts/）

问题：按"论文目录"组织（算法族罗列），不是按"我的数据"组织；原因列空话。

```
1. 推荐的决策路径 —— 首选推荐器（3 行代码），本文档解释它为什么这样选
2. π 在 PU 中的角色（哪些方法需要 π、π 错误的影响——决策的轴心）
3. 按数据条件决策（决策表，从数据出发）：
   有/无 π × 有无部分负样本 × 数据规模 × GPU × 怀疑 SAR
4. 算法族的设计思想（Risk Estimation / Bias-Aware / Deep PU 各自解决什么，
   不罗列方法清单）
5. 领域全景（TIcE、PU Bagging 等"扩展参考"压缩为一小节，标注不在本工具内）
```

### 6.2 根 README.md（英文）+ README.zh-CN.md 同步

```
徽章 + 一句话定位
真实 Hello World（make_sar_dataset，15 行内，可复制运行）
核心特性 4-5 条（每条约 10 词 + 代码佐证）
CLI 3 条命令
文档入口表精简为 6 个关键链接 → "完整导航见 docs/README.md"
安装 / 开发命令 / License
```

### 6.3 development_roadmap.md → 「路线图」（dev/roadmap.md）

- 纯中文；只留总体策略叙事（framework-first）+ 版本路线（0.1→1.0）
- 阶段细节与未完成项指向 `process_checklist.md`（权威进度来源）；删除与 checklist 重复的 WP 分解表

### 6.4 architecture.md → 「架构设计」（dev/architecture.md）

- 定位从"宣言"改为"设计决策与代价"（每条原则一行理由）
- 保留：模块分层与依赖方向、数据流图、registry/元数据设计（`_SYNC_FIELDS`、TrainingCost 等）
- 与 `project_structure.md`（有什么文件）严格分工：architecture 只讲"为什么这么组织"

## 7. 实施路径与验证

```
1. git mv 迁移（保留历史）：user/ 7 篇 → user/howto/；3 篇 dev 文档 → dev/
2. 新建 5 篇：user/README.md、quickstart.md、concepts/pu_problem.md、
   concepts/scar_sar.md、reference/api.md
3. 重写 4 篇（按 §6 大纲）
4. 重写 docs/README.md（两栏导航）+ 更新根 README / README.zh-CN.md 链接
5. 全局链接修复：所有交叉引用（含 method_cards 对 user/ 文档的反向引用）
6. 验证：check_doc_links.py + check_math_rendering.py + check_project_metadata.py
7. 提交：按逻辑分 2-3 个 commit（迁移 → 新建 → 重写）
```

**风险控制**：迁移与重写一次性完成后才跑门禁（避免中间态误报）；`$`...$` 公式格式迁移保持原样；文档不涉代码，无需跑测试套件。

**分支**：项目级重构，开 `docs/architecture` 分支走 PR；或按用户偏好直接在 main（参照 CLI 文档迁移先例）。实施开始时定。

## 8. 范围外

- `docs/research/method_cards/`（16 篇）：不动（研究资料，受 check_math_rendering 门禁保护）
- `docs/project_management/`：原位不动
- 代码、测试、CI：不动
- 文档中的公式（`$`...$` 与 `$$...$$`）格式：保持原样，只随迁移移动位置
