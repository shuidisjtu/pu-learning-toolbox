# 架构健康度全景审计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对照「架构腐朽」判断框架,对 pu-learning-toolbox 做全库全景审计,产出 `docs/dev/architecture_audit.md`(逐信号绿/黄/红 + 证据 + 建议 + 行动项 + 复跑指南)。

**Architecture:** 四个领域子代理并行审计(代码面/工程面/文档面/治理面),统一输出契约(信号 × 状态 × 证据 file:line × 置信度);主上下文汇总交叉验证,red 项亲自抽样复核后写入报告。

**Tech Stack:** 子代理只读审计(Glob/Grep/Read/Bash),无代码改动;报告为 Markdown 文档。

## Global Constraints

- 纯审计:不改任何代码、文档、CI、skill(红/黄项只记录,行动进 backlog)
- 所有发现必须带 `file:line` 证据;置信度 < 高 必须标注
- 不触碰队友分支(`feature/pu-sensitivity-analysis`、`zihenglin/my-change`)
- 报告遵循现有 docs 风格;双语场景写明"中文为准"
- 命令全部 `uv run` 前缀;Windows 路径 Python 中禁用裸反斜杠
- Git 提交不加 `Co-Authored-By`

---

### Task 1: 编写四代理提示词与输出契约

**Files:**
- Create: `docs/superpowers/plans/2026-08-09-architecture-audit.md`(本计划,含契约)

**Interfaces:**
- Produces: 四个代理 prompt 模板 + 统一输出 schema(信号 × 状态 × 证据 × 置信度),供 Task 2 派发

- [ ] **Step 1: 定义统一输出契约**

每代理返回 Markdown,结构:
```
## <信号> S1删除风险 / S2局部性 / S3承重bug / S4疤痕组织 或 表现 T1真相分裂 / T2概念膨胀 / T3治理腐朽
### 发现列表
- [状态: 绿|黄|红] <一句话发现> — 证据: `<file>:<line>` (置信度: 高/中/低)
### 补充观察(不入信号,但值得记录)
```

- [ ] **Step 2: 定义四代理分工边界**

| 代理 | 领域 | 扫描对象 | 必查项 |
|---|---|---|---|
| A | 代码面 | `pu_toolbox/` 全部 .py(47 文件,~15.8k 行) | 死代码/未用导出、重复实现、>500 行文件局部性、registry/契约漂移、api_only 假实现 |
| B | 工程面 | `tests/`(738 用例)+ `benchmarks/` + `scripts/` + `examples/` | skip/xfail 标记价值度、永远通过的测试、门禁脚本自身健康、benchmark 数据漂移 |
| C | 文档面 | `docs/` 40 篇 + README 双语 + CONTRIBUTING | method card vs 实现漂移、双语 README 漂移、过时 API 引用、概念膨胀(术语重复定义) |
| D | 治理面 | git 历史 + `.github/workflows/` + `.claude/skills/` + `.agents/skills/` | 演化轨迹疤痕、门禁价值度(是否永远通过)、决策记录缺失(为什么缺失)、门禁范围分裂 |

- [ ] **Step 3: Commit 本计划文档**

```bash
git add docs/superpowers/plans/2026-08-09-architecture-audit.md
git commit -m "docs: add architecture audit plan"
```

---

### Task 2: 并行派发四审计子代理

**Files:**
- 只读访问全库,不改任何文件

**Interfaces:**
- Consumes: Task 1 的输出契约与分工边界
- Produces: 四份结构化发现(每份按 4 信号 + 3 表现组织)

- [ ] **Step 1: 单消息内并行 spawn 4 个 Agent(Explore 型,只读)**

每个代理 prompt 包含:
1. 判断框架全文(4 信号 + 3 表现的操作化定义,来自 design doc)
2. 领域范围与必查项
3. 输出契约(Step 1 的 schema)
4. 硬约束:只读、发现必须带 file:line、置信度标注

- [ ] **Step 2: 等待全部完成,收集四份发现**

预期:4 份 Markdown 结构化输出;任一代理失败则补发(不重发其他代理)。

---

### Task 3: 汇总与交叉验证

**Files:**
- 无(纯分析)

**Interfaces:**
- Consumes: Task 2 四份发现
- Produces: 去重后的发现总表(按信号分组)+ red 项清单(待复核)

- [ ] **Step 1: 汇总去重**

将四份发现合并为信号分组总表;跨代理重复的发现合并(保留置信度最高者);冲突判定(同文件同信号不同结论)标记为"待裁决"。

- [ ] **Step 2: 产出 red 项清单**

总表中状态=红 的发现全部列入复核清单,附证据定位。

---

### Task 4: red 项抽样复核(主上下文亲自验证)

**Files:**
- 只读验证指定文件

**Interfaces:**
- Consumes: Task 3 的 red 项清单
- Produces: 复核后的最终 red 项(含修正/否决记录)

- [ ] **Step 1: 逐项复核**

对每个 red 项:亲自 Read/Grep 证据位置,确认是否成立;不成立则降级为黄或删除,并在报告"复核记录"中写明理由。

- [ ] **Step 2: 黄项抽检**

对置信度=高 的黄项抽 30% 复核;其余标注"未抽检"。

---

### Task 5: 撰写审计报告 `docs/dev/architecture_audit.md`

**Files:**
- Create: `docs/dev/architecture_audit.md`

**Interfaces:**
- Consumes: Task 3 总表 + Task 4 复核结果
- Produces: 最终报告(用户唯一交付物)

- [ ] **Step 1: 按大纲撰写报告**

结构:
```
# 架构健康度审计报告(2026-08-09)
## 1. 审计元信息(日期/范围/方法/复跑方式)
## 2. 总评(健康度判词 + 代谢率/腐朽率评估)
## 3. 四信号逐条(状态 + 证据 + 影响 + 建议)
## 4. 三表现逐条(同上)
## 5. 行动项清单(按优先级,red 必列,标注 backlog)
## 6. 复跑指南
## 7. 复核记录(red 项复核结果)
```

- [ ] **Step 2: 门禁验证**

```bash
uv run python scripts/check_doc_links.py    # 新文档链接合法
uv run python scripts/check_format.py       # 格式门禁(仅报告 md 不受影响,确认通过)
uv run pytest tests/ -m "not slow" -q       # 无代码改动,确认全绿
```

预期:三道命令全部退出 0。

- [ ] **Step 3: Commit**

```bash
git add docs/dev/architecture_audit.md
git commit -m "docs: add architecture health audit report"
```

---

### Task 6: 汇报

- [ ] **Step 1: 向用户汇报审计结论**

汇报格式:总评 + 每信号判词 + top 行动项;报告路径;复跑建议。

---

## Self-Review

**1. Spec coverage:**
- design doc 的四代理分工 → Task 2 ✅
- 输出契约(信号×状态×证据×置信度)→ Task 1 Step 1 ✅
- 报告大纲 6 节 → Task 5 Step 1 ✅
- red 项抽样复核 → Task 4 ✅
- 边界(不改代码/不动队友分支)→ Global Constraints + Task 2 prompt ✅
- 复跑指南 → Task 5 报告 §6 ✅

**2. Placeholder scan:** 无 TBD/TODO;每步含具体内容 ✅

**3. Type consistency:** 信号命名(S1-S4/T1-T3)在 Task 1/3/4/5 中一致 ✅
