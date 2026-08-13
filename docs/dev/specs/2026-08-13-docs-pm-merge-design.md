# 设计规格:docs/project_management/ 合并入 docs/dev/ + skill 测试数动态化

> 日期:2026-08-13 | 状态:已批准(头脑风暴逐块确认)

## 1. 背景与动机

- `docs/project_management/` 仅 3 份文档(281 行),读者(开发者/维护者)与
  `docs/dev/` 受众重合,单独成目录的收益已不存在。
- `cli_design.md` 参数表已静默漂移:缺 `--prior-param/--architecture/--backbone/
  --device/--max-epochs` 五个后增参数,且与 `docs/user/howto/cli.md`(完整、最新)重复;
  ADR-0010 已覆盖其大部分设计决策。
- `dev-workflow` skill「项目现状速查」写死测试数 811(实际 907),审计
  (architecture_audit.md §7)已确认「skill 数字过时」为复发问题。
- 用户确立的原则:代码 docstring/注释已清晰的行为细节,文档不再重述「是什么」,
  只记「决策与理由」。

## 2. 设计原则

- **P1 受众分层收敛**:文档内聚单位 = 「一个读者的一段旅程,在一处闭环」;
  四层受众(user/dev/research/project_management)收敛为三层。
- **P2 代码是行为真相源**:docstring/注释是行为细节的唯一真相源;
  文档复述行为细节会造成静默漂移(实测:cli_design 参数表缺 5 参数)。
- **P3 真相源带时间维度**:决策修订 = 新 ADR + 旧 ADR 引用注,不改写历史记录。

## 3. 迁移动作

```
docs/project_management/            → 目录消失
  process_checklist.md              → git mv → docs/dev/process_checklist.md(内容压缩,§5.1)
  release_process.md                → git mv → docs/dev/release_process.md(内容不变)
  cli_design.md                     → git rm(独特决策并入 ADR-0010,§5.2)
```

## 4. 引用重定向清单(全库 12 处)

| # | 文件 | 改动 |
|---|---|---|
| 1 | CONTRIBUTING.md:10,16,132 | 权威来源 #2/#7、§7 路径 → `docs/dev/` |
| 2 | docs/README.md | 「项目过程」节删除;process_checklist、release_process 行并入开发者文档表;cli_design 行删除;新增 `[dev/specs/](dev/specs/)` 与 `[dev/plans/](dev/plans/)` 目录行(覆盖本规格与实施计划,Rule 4) |
| 3 | docs/adr/README.md | 路径引用更新 + 索引新增 0013 行(13 篇) |
| 4 | docs/adr/0012:30 | 路径 → `docs/dev/release_process.md` |
| 5 | docs/dev/roadmap.md:43 | 链接 → 同目录 `process_checklist.md` |
| 6 | docs/adr/0008:16 | 决策 #2 加修订注(§6) |
| 7 | docs/dev/project_structure.md:353-356 | 树删除 `project_management/` 子树;dev 子树新增 `process_checklist.md`、`release_process.md` 两行(保留原注释) |
| 8 | docs/research/method_cards/LDCE.md:271 | backtick 路径 → `docs/dev/process_checklist.md`(Rule 1 拦截点) |
| 9 | scripts/check_doc_links.py:61 | 注释更新(提及 project_management 的豁免说明措辞) |
| 10 | .claude/skills/dev-workflow/SKILL.md:19,35 | 开发前/完成后步骤路径 → `docs/dev/` |

> architecture_audit.md:107 的 T3 历史发现是审计快照,保留原样。

## 5. 内容处置

### 5.1 process_checklist.md(94 → ~40 行)

```
# 进度清单

> 执行顺序说明 + 阶段定义以本文档为准 + Method Card 为可选

## 阶段历史(已闭环)
- Phase 0 ✅ 骨架 + Registry + 测试框架
- Phase 1 ✅ 核心 PU 风险估计(Elkan-Noto/uPU/nnPU/ReCPE/PNU/splitters/metrics/examples)
- Phase 2 ✅ 部分(penL1、推荐器;三包装器与 TIcE/AlphaMax 列 v1 范围外)
- Phase 3 ✅ 机制就绪(PUSB 全链路;全量运行依赖外部,见下)
- Phase 4 ✅ 画像/推荐/诊断/敏感性
- Phase 5 ✅ SAR 模拟与 PUSB/LBE benchmark
- Phase 6 ✅ 大部分(Self-PU/Dist-PU/InfoMax/WConPU/DGPU 全链路;剩余见下)
   详细逐条勾选史见 git log

## 未完成项
- [ ] Phase 3 官方数据/历史环境全量运行(依赖外部提供)
- [ ] Phase 6 WConPU 官方视觉 + DGPU EDM paper-like 全量(依赖 CUDA/授权数据)
- [ ] InfoMax 未公开类别分组、batch size、KM 变体核对
- ⚠️ v1 范围外:Phase 2 三经典包装器 + TIcE/AlphaMax

## 发布状态 (v1.3.0)
(原样保留:版本/17 算法/6 门禁/范围外/依赖外部/ADR 链接)
```

压缩时保留头部引言块(执行顺序调整、阶段定义权威、Method Card 可选,3 行);
Phase 1/2 的引言块信息(实际优先顺序)并入对应行摘要;其余已完成的逐条明细
归 git log。文件末尾 `../adr/` 链接移动后仍有效,不改。

### 5.2 cli_design.md → 删除,独特决策并入 ADR-0010

| 原章节 | 归宿 |
|---|---|
| 背景与目标 / 技术选型 / 命令结构 | ADR-0010 决策 #1 已覆盖,删除 |
| run 参数表(过时) | user/howto/cli.md(权威投影),删除 |
| list-methods 一致性判定 | 补入 ADR-0010:复用 `_missing_required_params`,CLI 判定与 auto 实例化必须同一逻辑 |
| make-demo-data --n/--c 对齐 | 补入 ADR-0010:--n 为每类样本数(总 2n),对齐底层 (n, c) 建模,废弃草稿 --n-positive |
| 数据流 / 错误处理 / 退出码 | 代码 + user/howto/cli.md 退出码节,删除 |
| 模块结构树 | 代码自身,删除 |
| 范围边界史 | git log + 发布状态节,删除 |

### 5.3 release_process.md → 平铺移动,内容不变

纯操作流程清单,无代码可重述、无过时内容;文中两处
`process_checklist.md 发布状态节` 为裸名引用,移动后同目录,依然准确。

## 6. ADR 处置

1. **新增 ADR-0013「docs 目录合并」**:背景(PM 仅 3 文档、cli_design 漂移、
   四层分层实际收敛)→ 决策(目录解散平铺、清单历史压缩、确立 P2 原则)
   → 备选(dev/process/ 子目录 / 原样平铺 / 全部并入 CONTRIBUTING,均否决)
   → 后果(0008 修订注、12 处重定向、索引节合并、§9 后续工作登记)。
2. **ADR-0008** 决策 #2 末尾加一行,原文不改:
   `> 2026-08-13 修订:project_management/ 层解散并入 dev/,见 ADR-0013。`
3. **ADR-0010** 补 2 条决策(§5.2 两行)。
4. **docs/adr/README.md**:索引加 0013 行,路径引用更新。

## 7. dev-workflow skill 速查改造

```markdown
- **当前阶段与算法数**: 以 docs/dev/process_checklist.md 发布状态节为准(执行前读取)
- **测试**: 不写死——执行前运行
  `uv run pytest tests/ --collect-only -q -m "not slow and not e2e"`
  读取实际收集数(PR 快层;e2e + slow = nightly 顶层)
- **质量门禁**: 6 个脚本(名单见下方「开发完成后」第 2 步,同文件内自洽)
- **CI**: GitHub Actions — Python matrix + ruff lint/format + wheel 安装冒烟
```

## 8. 门禁与验证

| 门禁 | 影响 |
|---|---|
| check_doc_links | Rule 1/4/5 是本次改动主验证器,必须全绿 |
| check_skill_sync | dev-workflow 单份拷贝,门禁只查 pu-workflow 双份,不受影响 |
| 其余 4 道 | 纯文档/skill 改动,预期零影响,CI 兜底 |
| pytest | 无代码变更,CI 矩阵兜底 |

## 9. 后续工作(本批次不做,记入 ADR-0013 后果)

- **lychee**(链接/锚点检查)与本项目的兼容性配置评估:能否在不制造
  双机制分裂的前提下覆盖 Rule 5 的锚点盲区,或由 check_doc_links 自补锚点校验。
- **mkdocstrings + Griffe**(API 参考自动生成)兼容性评估:docstring 成为
  API 文档唯一真相源,消灭手工契约表漂移;需评估 MkDocs 构建链引入成本。

## 10. 验收标准

1. `docs/project_management/` 目录不存在;两文件经 git mv 保留历史。
2. 全库无 `docs/project_management/` 旧路径引用(不含历史审计快照)。
3. process_checklist.md ≤ ~40 行,未完成项与发布状态节完整。
4. ADR-0013 入库、0008 修订注、0010 补 2 决策、adr/README 索引 13 篇。
5. skill 速查无写死测试数,收集命令可执行。
6. 六门禁全绿,CI 通过。
