# ADR 迁移设计:决策记录从冗长文档迁移到 docs/adr/

- 日期:2026-08-13
- 状态:已获批(用户逐节确认)
- 参考格式:`D:\ClaudeCodeProject\Multimodel_Blog_Helper\docs\adr\`

## 1. 背景与痛点

- `docs/dev/architecture_audit.md`(190 行)中三批治理的 commit 流水账持续堆积,信息与「发现了什么」混杂。
- `docs/project_management/decision_log.md` 是 24 行大表,单行极密,决策与理由不可独立检索。
- `docs/dev/architecture.md` §1 承担「设计决策与代价」,架构文档职责过载。
- 早期实现/测试相关内容占据大量篇幅,决策信息不集中。

## 2. 目标

用 ADR(编号、主题化、带状态与触发复审条件)统一管理决策记录,三个源文档各自瘦身回到单一职责:

- **ADR**:记录「决策与理由」,带状态(已接受/已取代)与触发复审条件
- **审计报告**:只记录「发现了什么」(信号基线 + 复跑指南)
- **架构文档**:只描述「现在是什么」(分层、数据流、API、注册表)

## 3. 范围决策(已与用户确认)

| # | 决策 | 选择 |
|---|---|---|
| 1 | 迁移范围 | decision_log 全部 + architecture_audit 的决策类内容 + architecture.md §1 决策段 |
| 2 | 回溯粒度 | 精选回溯 ~12 篇,合并同类、跳过琐碎;decision_log 整表删除 |
| 3 | 审计报告处置 | 瘦身为「发现快照 + 复跑指南」,批次流水账 → ADR 链接 |
| 4 | ADR 位置 | `docs/adr/` 顶层,附 README.md 索引 |
| 5 | 版本状态原则 | ADR 记决策不承载版本状态;版本/发布权威在 release_process.md + process_checklist.md |

## 4. ADR 格式

```
docs/adr/
  README.md          # 索引:编号/标题/状态/日期一览 + 新增 ADR 规则
  0001-architecture-governance.md
  ...
```

每篇结构(40-60 行,中文):

- `# ADR-NNNN:标题`
- 状态(已接受/已取代)+ 触发复审条件
- 背景
- 决策
- 备选方案
- 后果

README.md 另含两条规则:① ADR 记决策,版本/进度状态见 release_process.md 与 process_checklist.md;② 琐碎流程惯例写 CONTRIBUTING,不立 ADR。

## 5. 12 篇 ADR 清单

| # | 标题 | 吸收的决策 |
|---|---|---|
| 0001 | 架构治理机制 | 审计四信号/三表现、单源助手清单、代谢率红线(batch1/2/3 流水账) |
| 0002 | 核心包轻量 + torch 可选依赖 | architecture.md §1 行 1 |
| 0003 | 类先验/标记倾向/损失/分类器解耦 | architecture.md §1 行 2 |
| 0004 | registry 元数据驱动 | architecture.md §1 行 3 |
| 0005 | 复现可信度分级 | adapter 优先/clean-room、claim-safe、PUSB Table 2 严格子集、provenance 锁、PUSBKernel 独立注册 |
| 0006 | SAR 中长期定位 | architecture.md §1 行 5 |
| 0007 | 测试金字塔与 CI 分层 | 金字塔、PR/nightly 分层、check_format 门禁、test_quality 严格默认 |
| 0008 | 协作与文档惯例 | feature/fix 分支、docs 受众分层、论文分工、Phase 重整 |
| 0009 | 类先验估计修复与 auto 默认切换 | v1.2.1 五项修复 + recpe→pen_l1(背景记录被取代的旧默认) |
| 0010 | CLI/auto/skill 工作流 | CLI 薄封装、推荐器成本维度、Deep PU 接入、pu-workflow skill、skill install |
| 0011 | 发布体验修复 | v1.1.0/v1.1.1:device 自动检测、子命令化、`__version__` 门禁、max-epochs |
| 0012 | 依赖与发布策略 | Python 3.10-3.12、pyproject 权威、uv.lock 不入库、首版 1.0.0 理由 |

标题不含版本号;版本号只在「背景」中作上下文。

## 6. 源文档处置

### 6.1 decision_log.md → 删除

24 行决策表全部进 ADR,文件删除。引用点同步:

- `process_checklist.md` 底部「关键决策见 decision_log」→ `docs/adr/`
- `docs/README.md` 索引换条目

### 6.2 architecture_audit.md → 190 → ~110 行

| 段 | 处置 |
|---|---|
| §1 元信息 | 保留,加一行「决策已迁移至 docs/adr/0001」 |
| §2 总评 | 保留判定段落;三块批次治理 blockquote → 一行 ADR-0001 链接 |
| §3/§4 信号表 | 保留(下次审计的对照基线) |
| §5 行动项(两批 commit 流水) | 整段删除 → ADR-0001 承接,留链接 |
| §6 复跑指南 | 保留;单源助手清单行 → 指向 CONTRIBUTING §5.1(消除第三份清单副本) |
| §7 复核记录 | 保留 |

### 6.3 architecture.md → 303 → ~285 行

§1「设计决策与代价」5 行表整段删除,替换为一行:「设计决策见 `docs/adr/`(0002-0006)」。「与 project_structure.md 的分工」说明保留。

## 7. 索引与门禁同步

- `project_structure.md` §5 文档树:删 decision_log 行,新增 `docs/adr/` 子树(README + 12 篇)
- `docs/README.md` 索引、`process_checklist.md` 尾部指针、`CONTRIBUTING.md` §1 权威来源(加 `docs/adr/` 条目)
- 全局 grep `decision_log` 确认无残留死链(check_doc_links 门禁 orphan 检查兜底)
- check_doc_links 的 `_EXCLUDED_DOC_DIRS` 已含 superpowers,本 spec 目录不受门禁约束
- 本 spec 存放于 `docs/dev/specs/`(docs/superpowers/ 被 .gitignore 排除),已登记 docs/README.md 索引;实现完成后按 `deep_pipeline_design.md` 先例蒸馏删除

## 8. 执行与验证

- 分支:`feature/adr-migration`
- 提交划分(小步快跑,每步后跑 check_doc_links):
  1. 新建 12 篇 ADR + README 索引
  2. 删除 decision_log.md + 指针更新
  3. architecture_audit.md 瘦身
  4. architecture.md 拆段
  5. 索引/门禁同步收尾
- 验收:六道门禁全绿;纯文档改动,测试套件最后跑一次确认无意外
- 完成后:`dev-workflow` 流程(PR + 合并)
