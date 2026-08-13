# docs/project_management/ 合并入 docs/dev/ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 解散 docs/project_management/,两份文档 git mv 入 docs/dev/(清单历史压缩),cli_design 删除并并入 ADR-0010,新增 ADR-0013,dev-workflow skill 测试数动态化。

**Architecture:** 纯文档治理批次:git mv 保留历史 → 全库 12 处引用重定向 → ADR 处置(0013 新建/0008 修订注/0010 补丁)→ skill 速查改造。验收器是 check_doc_links 的 Rule 1/4/5,辅以其余五门禁。

**Tech Stack:** git mv/rm、Markdown、六道质量门禁(uv run)。

**Spec:** `docs/dev/specs/2026-08-13-docs-pm-merge-design.md`

## Global Constraints

- 所有 Python 命令前缀 `uv run`(唯一包管理器)。
- 修改文件前必须先 Read(hook 要求);删除文件逐个删。
- Git 提交不加 `Co-Authored-By`;分支 `feature/docs-pm-merge`,PR 合并 main。
- 中间态门禁预期:Task 1 结束后 rule-1/4/5 红(9 处,清单见 Task 1 Step 5);Task 2 结束后必须全绿,此后每个任务结束都必须全绿。
- architecture_audit.md:107 的 T3 历史发现是审计快照,不改写。
- 文档中文、代码注释英文;Windows 路径在 Python 代码中禁止裸反斜杠(本批次无 Python 代码改动)。
- 最终基线:6 道门禁全绿 + pytest 快层全绿 + CI 通过。

---

### Task 1: 文件迁移 + process_checklist 历史压缩

**Files:**
- Move: `docs/project_management/process_checklist.md` → `docs/dev/process_checklist.md`(内容重写)
- Move: `docs/project_management/release_process.md` → `docs/dev/release_process.md`(内容不变)
- Delete: `docs/project_management/cli_design.md`

**Interfaces:**
- Consumes: 无
- Produces: 新路径 `docs/dev/process_checklist.md`、`docs/dev/release_process.md`(Task 2/3/4 的引用目标);压缩后的清单结构(未完成项/发布状态节,Task 2 无依赖、ADR-0013 后果引用其角色)

- [ ] **Step 1: git mv 两份文档(逐个执行)**

```bash
git mv docs/project_management/process_checklist.md docs/dev/process_checklist.md
git mv docs/project_management/release_process.md docs/dev/release_process.md
```

- [ ] **Step 2: git rm cli_design.md(单独执行,避免批量删除)**

```bash
git rm docs/project_management/cli_design.md
```

- [ ] **Step 3: 确认 docs/project_management/ 目录已空并消失**

```bash
ls docs/project_management 2>&1
```

Expected: `No such file or directory`(git 不跟踪空目录,mv/rm 后目录自动消失)。

- [ ] **Step 4: Read 现有 process_checklist.md 后整文件重写为压缩版**

Read `docs/dev/process_checklist.md`(git mv 保留了原 94 行内容),然后用以下完整内容覆盖:

```markdown
# 进度清单

> 实际执行顺序与原始路线图有调整：优先实现 PU 特有的风险估计方法（工具箱核心差异化能力），经典分类器包装器后移。
> 阶段定义以本文档为准，`docs/dev/roadmap.md` 为高层路线图。
> **Method Card 为可选文档**，新算法接入不要求必写。

## 阶段历史（已闭环）

- Phase 0 ✅ 项目骨架：pyproject + Core 基类 + Registry（初始 15 个 api_only 占位，现已按实现状态升级）+ 测试框架
- Phase 1 ✅ 核心 PU 风险估计：Elkan-Noto / uPU / nnPU / ReCPE / PNU / PU splitters / metrics / minimal examples
- Phase 2 ✅ 部分：penL1 类先验与算法推荐器完成；三经典包装器与 TIcE/AlphaMax 列 v1 范围外
- Phase 3 ✅ 机制就绪：PUSB benchmark 全链路（来源锁/manifest/shard 聚合/断点续跑/审计器）；官方数据全量运行依赖外部，见下
- Phase 4 ✅ 推荐与诊断：Data Profiler / SCAR-SAR 提示 / 推荐器 / 诊断报告 / 敏感性分析
- Phase 5 ✅ SAR：数据模拟器、PUSB/LBE/Centroid/LLSVM 接口与 SCAR vs SAR 对比 benchmark
- Phase 6 ✅ 深度 PU 大部分：Self-PU / Dist-PU / InfoMax PU / WConPU / DGPU 全链路（clean-room 多 seed、Fashion-MNIST 3-seed smoke、InfoMax 暂定协议 20 seeds）；剩余见下

> 逐条明细与批次历史见 git log。

## 未完成项

- [ ] Phase 3 官方数据/历史环境全量运行（依赖外部官方数据与历史环境提供，非工具箱缺口）
- [ ] Phase 6 WConPU 官方视觉 + DGPU EDM paper-like 全量（依赖 CUDA/授权数据）
- [ ] InfoMax 未公开类别分组、batch size 与 KM 变体核对
- ⚠️ v1 范围外：Phase 2 三经典包装器 + TIcE/AlphaMax 类先验估计

## 发布状态 (v1.3.0)

- **版本**: `1.3.0`（2026-08-10：类先验估计用户角度体验修复——`make_sar_dataset` 默认 SAR 警示、CLI 报告估计可靠性上下文（估计器名/边界注/Assumption Notes）、demo 分离度 1.0、list-priors 别名分组、`--prior-param` 非法值前置拦截、NaN 友好报错、推荐器不再对非识别 at_risk 信号提升 SAR 方法、`is_scar_plausible` 改名 `is_observed_dependence_absent`）
- **算法**: 17 个已注册方法，全部 native 实现
- **质量门禁**: 6 道（test_quality / doc_links / project_metadata / math_rendering / skill_sync / format）
- **v1 范围外**: Phase 2 三个经典包装器与 TIcE/AlphaMax 类先验估计
- **依赖外部**: Phase 3 官方历史环境，以及 WConPU CUDA/授权数据和 DGPU EDM/CelebA
  全量运行；InfoMax 暂定 Fashion-MNIST 20-seed 协议已执行

历史执行记录见 git log；关键决策见 [`docs/adr/`](../adr/)。
```

- [ ] **Step 5: 运行 check_doc_links,确认门禁如预期变红(约 7 处)**

```bash
uv run python scripts/check_doc_links.py
```

Expected(仅允许以下错误,Task 2 修复;共计 9 处门禁错误):
- rule-1:docs/adr/README.md:21 `docs/project_management/release_process.md`
- rule-1:docs/adr/0012-dependency-release-policy.md:30 `docs/project_management/release_process.md`
- rule-1:docs/research/method_cards/LDCE.md:271 `docs/project_management/process_checklist.md`
- rule-4:docs/dev/process_checklist.md 与 docs/dev/release_process.md 未列入 docs/README.md(2 处)
- rule-5:docs/README.md 三处 project_management 链接、roadmap.md 一处链接(4 处)

另注:CONTRIBUTING.md 第 10、15-16(两处)、132 行共 4 处旧路径**不被门禁扫描**
(_find_md_files 只收 README/CLAUDE/docs/),Task 2 仍须更新(纯正确性)。
出现其他错误需停下核查;若 rule-1 报 spec/plan 文件的 `docs/dev/...` 引用,
属预期(本步已使文件存在,应转为通过)。

- [ ] **Step 6: 提交**

```bash
git add docs/
git commit -m "docs: move pm docs into dev/ and compress checklist history"
```

---

### Task 2: 全库引用重定向 + 索引合并 + 目录树更新

**Files:**
- Modify: `CONTRIBUTING.md:10,15-16,132`
- Modify: `docs/README.md`(开发者文档表 + 删除「项目过程」节)
- Modify: `docs/dev/roadmap.md:43`
- Modify: `docs/dev/project_structure.md:327-332,353-356`
- Modify: `docs/research/method_cards/LDCE.md:271`
- Modify: `docs/adr/README.md:21`(路径)、`docs/adr/0012-dependency-release-policy.md:30`(路径)
- Modify: `scripts/check_doc_links.py:60-63`(注释)

**Interfaces:**
- Consumes: Task 1 的新路径
- Produces: 全绿 check_doc_links(Task 3/4 的基线)

- [ ] **Step 1: CONTRIBUTING.md 三处路径替换(逐个 Edit)**

1. 第 10 行:

```markdown
2. `docs/dev/process_checklist.md`：当前任务完成状态。
```

2. 第 15-16 行:

```markdown
7. `docs/adr/`：架构与流程决策记录（决策的权威来源；版本/进度状态见
   `docs/dev/process_checklist.md` 与 `docs/dev/release_process.md`）。
```

3. 第 132 行:

```markdown
- `docs/dev/process_checklist.md` 的勾选与发布状态。
```

- [ ] **Step 2: docs/README.md 开发者文档表新增 4 行(在 `resources.md` 行之后、`research/method_cards/` 行之前)**

```markdown
| [dev/process_checklist.md](dev/process_checklist.md) | 进度清单与发布状态（权威来源） |
| [dev/release_process.md](dev/release_process.md) | 发布流程（版本策略、预检、上传、回滚、维护） |
| [dev/specs/](dev/specs/) | 设计规格（实施后蒸馏进 ADR 并删除） |
| [dev/plans/](dev/plans/) | 实施计划（实施后蒸馏进 ADR 并删除） |
```

- [ ] **Step 3: docs/README.md 整节删除「项目过程(docs/project_management/)」**

删除以下整节(表头「## 项目过程（docs/project_management/）」至「发布流程」行):

```markdown
## 项目过程（docs/project_management/）

| 文档 | 用途 |
|---|---|
| [project_management/process_checklist.md](project_management/process_checklist.md) | 进度清单（权威来源） |
| [project_management/cli_design.md](project_management/cli_design.md) | CLI 设计文档：命令结构、参数契约、错误处理与模块边界 |
| [project_management/release_process.md](project_management/release_process.md) | 发布流程（版本策略、预检、上传、回滚、维护） |
```

- [ ] **Step 4: docs/dev/roadmap.md:43 链接改同目录**

```markdown
任务粒度的完成状态以 [process_checklist.md](process_checklist.md) 为权威来源，本文档只保留阶段叙事与版本路线，不再逐条重复。
```

- [ ] **Step 5: docs/dev/project_structure.md 目录树更新**

1. dev/ 子树(resources.md 行后)新增 3 行(Boy Scout:补上缺失的 architecture_audit.md 行):

```text
    architecture_audit.md       # 审计发现快照、复跑指南与治理机制（ADR-0001）
    process_checklist.md        # 进度清单与发布状态（权威来源）
    release_process.md          # 发布流程：版本策略、预检清单、上传、回滚与维护
```

2. 删除 project_management/ 子树整块:

```text
  project_management/
    process_checklist.md
    cli_design.md
    release_process.md          # 发布流程：版本策略、预检清单、上传、回滚与维护
```

- [ ] **Step 6: docs/research/method_cards/LDCE.md:271 路径替换**

```markdown
> 以下为基于 `BasePUClassifier` 契约（`core/base.py`）和 `docs/dev/process_checklist.md` 的项目建议，非论文原文。
```

- [ ] **Step 7: docs/adr/README.md:21 路径更新**

```markdown
- ADR 记决策(背景/决策/备选方案/后果),**不承载版本状态**;版本/发布状态权威在
  `docs/dev/release_process.md` 与 `docs/dev/process_checklist.md` 发布状态节。
```

- [ ] **Step 8: docs/adr/0012-dependency-release-policy.md:30 路径更新**

```markdown
- 版本演进/发布状态记录于 `docs/dev/release_process.md`
  与 `docs/dev/process_checklist.md` 发布状态节;本 ADR 不承载版本状态。
```

- [ ] **Step 9: scripts/check_doc_links.py 注释更新(60-63 行)**

```python
# Docs subdirectories excluded from ALL checks. research/ (method cards)
# is in scope: it is the densest citation source and must not be
# wholesale-exempted.
```

- [ ] **Step 10: 运行 check_doc_links,必须全绿**

```bash
uv run python scripts/check_doc_links.py
```

Expected: `All checks passed.`

- [ ] **Step 11: 提交**

```bash
git add CONTRIBUTING.md docs/README.md docs/dev/roadmap.md docs/dev/project_structure.md docs/research/method_cards/LDCE.md docs/adr/README.md docs/adr/0012-dependency-release-policy.md scripts/check_doc_links.py
git commit -m "docs: redirect references to merged docs/dev/ paths"
```

---

### Task 3: ADR 处置(0013 新建 + 0008 修订注 + 0010 补丁 + 索引更新)

**Files:**
- Create: `docs/adr/0013-docs-directory-merge.md`
- Modify: `docs/adr/0008-collaboration-conventions.md`(决策 #2 后加修订注)
- Modify: `docs/adr/0010-cli-auto-skill.md`(补 2 条决策)
- Modify: `docs/adr/README.md`(索引 13 行 + 规则路径更新)

**Interfaces:**
- Consumes: Task 2 的全绿基线
- Produces: ADR-0013(后续工作登记:lychee/mkdocstrings 评估);0008/0010 修订

- [ ] **Step 1: 新建 docs/adr/0013-docs-directory-merge.md**

```markdown
# ADR-0013:docs 目录合并

- 状态:已接受
- 触发复审:文档导航再混乱,或 docs/ 新增受众层时

## 背景

docs/project_management/ 仅 3 份文档(281 行),读者(开发者/维护者)与
docs/dev/ 受众重合;cli_design.md 参数表已静默漂移(缺 5 个后增参数)
且与 docs/user/howto/cli.md 重复;ADR-0008 的 user/dev/research/
project_management 四层分层在实际演进中收敛为三层。

## 决策

1. **project_management/ 解散**:process_checklist.md、release_process.md
   git mv 平铺至 docs/dev/;cli_design.md 删除,独特决策并入 ADR-0010。
2. **进度清单历史压缩**:每 Phase 一行摘要,逐条明细归 git log;未完成项
   与发布状态节保留。
3. **文档原则确立**:代码 docstring/注释是行为细节的真相源,文档只记
   决策与理由,不重述「是什么」(cli_design 参数表漂移即反例)。

## 备选方案

- **docs/dev/process/ 子目录**:重造受众子层,与「目录即受众」的目标矛盾。否决。
- **原样平铺**:保留历史流水与漂移的参数表,违背压缩目标。否决。
- **三份全部并入 CONTRIBUTING**:流程清单混入贡献指南,受众混淆。否决。

## 后果

- ADR-0008 决策 #2 加修订注,原文不改。
- 全库 12 处引用重定向至 docs/dev/ 路径;docs/README 索引三节合并。
- 后续工作:lychee(链接/锚点检查)与 mkdocstrings+Griffe(API 参考自动
  生成)在本项目的兼容性配置与使用评估;评估结论与采用决定另立 ADR。
```

- [ ] **Step 2: ADR-0008 决策 #2 后加修订注**

在决策 #2 末尾(「根 README 双语。」之后)插入:

```markdown
   > 2026-08-13 修订:project_management/ 层解散并入 dev/,见 ADR-0013。
```

- [ ] **Step 3: ADR-0010 决策 #5 之后补 2 条**

```markdown
6. **list-methods 可实例化判定与 auto 模式一致**:复用
   `pipeline._missing_required_params`(同包内部导入),CLI 判定与 auto
   实例化判定必须同一逻辑。
7. **make-demo-data 对齐底层 (n, c) 建模**:`--n` 为每类样本数(总 2n),
   `--c` 为 SCAR 标注概率;废弃早期草稿的 `--n-positive`(底层不是
   n_positive 建模)。
```

- [ ] **Step 4: docs/adr/README.md 索引表追加一行(路径已在 Task 2 更新)**

```markdown
| 0013 | docs 目录合并 | 已接受 | 2026-08-13 |
```

- [ ] **Step 5: 运行 check_doc_links,必须全绿**

```bash
uv run python scripts/check_doc_links.py
```

Expected: `All checks passed.`(0013 文中不出现指向不存在文件的
`docs/...\.md` 反引号路径;若有,改为不指向文件的形式。)

- [ ] **Step 6: 提交**

```bash
git add docs/adr/
git commit -m "docs: record docs directory merge as ADR-0013"
```

---

### Task 4: dev-workflow skill 路径与速查改造

**Files:**
- Modify: `.claude/skills/dev-workflow/SKILL.md:19,35,62-68`

**Interfaces:**
- Consumes: Task 2 的全绿基线(与 Task 3 无顺序依赖)
- Produces: 速查动态测试数(收集命令);无写死数字

- [ ] **Step 1: Read SKILL.md 后替换开发前第 4 步(19 行)**

```markdown
4. 检查 `docs/dev/process_checklist.md` 确认当前 Phase 进度
```

- [ ] **Step 2: 替换开发完成后第 4 步(35 行)**

```markdown
4. 更新 `docs/dev/process_checklist.md`（勾选完成项、更新发布状态节）
```

- [ ] **Step 3: 整节替换「项目现状速查」(62-68 行)**

```markdown
## 项目现状速查

- **当前阶段与算法数**: 以 docs/dev/process_checklist.md 发布状态节为准（执行前读取）
- **测试**: 不写死——执行前运行
  `uv run pytest tests/ --collect-only -q -m "not slow and not e2e"`
  读取实际收集数（PR 快层；e2e + slow = nightly 顶层）
- **质量门禁**: 6 个脚本（名单见下方「开发完成后」第 2 步，同文件内自洽）
- **CI**: GitHub Actions — Python matrix + ruff lint/format + wheel 安装冒烟
```

- [ ] **Step 4: 验证收集命令可用且输出实际收集数**

```bash
uv run pytest tests/ --collect-only -q -m "not slow and not e2e" 2>&1 | tail -n 3
```

Expected: 输出 `N tests collected`(N 为当前实际收集数,不写回 skill)。

- [ ] **Step 5: 运行门禁,确认全绿**

```bash
uv run python scripts/check_skill_sync.py
uv run python scripts/check_doc_links.py
```

Expected: 均 `All checks passed.` / `ok`(check_skill_sync 只查 pu-workflow 双份,dev-workflow 单份改动不影响)。

- [ ] **Step 6: 提交**

```bash
git add .claude/skills/dev-workflow/SKILL.md
git commit -m "docs: de-hardcode test count in dev-workflow skill quickref"
```

---

### Task 5: 收尾验证与 PR

**Files:** 无新增;可能删除 `docs/dev/specs/2026-08-13-docs-pm-merge-design.md` 与 `docs/dev/plans/2026-08-13-docs-pm-merge.md`(Step 4 用户决定)。

**Interfaces:**
- Consumes: Task 1-4 全部
- Produces: 绿门禁基线 + PR

- [ ] **Step 1: 全库残留检查**

```bash
grep -rn "project_management" --include="*.md" --include="*.py" --exclude-dir=.git --exclude-dir=.venv . 2>/dev/null
```

Expected: 仅允许 `docs/dev/architecture_audit.md:107`(历史快照)与
`docs/dev/specs/`、`docs/dev/plans/`(本批次的规格与计划自身,描述旧路径)。
出现其他命中 → 停下修复。

- [ ] **Step 2: 六道门禁全跑**

```bash
uv run python scripts/check_test_quality.py
uv run python scripts/check_doc_links.py
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
uv run python scripts/check_format.py
```

Expected: 全部通过(本批次无 Python 代码改动,test_quality/format 预期不变)。

- [ ] **Step 3: pytest 快层**

```bash
uv run pytest tests/ -q -m "not slow and not e2e"
```

Expected: 全绿(通过数约等于 Task 4 Step 4 的收集数,允许环境性 skip 差异)。

- [ ] **Step 4: 询问用户是否按 ADR 迁移先例蒸馏删除 spec/plan**

向用户提问:实施已完成,是否按先例删除 `docs/dev/specs/2026-08-13-docs-pm-merge-design.md`
与 `docs/dev/plans/2026-08-13-docs-pm-merge.md`(决策已蒸馏进 ADR-0013)?
- 若删:逐个 `git rm` 两文件,从 docs/README.md 删除 `[dev/specs/]` 与
  `[dev/plans/]` 两行,若目录已空则 `rmdir docs/dev/specs docs/dev/plans`,
  重跑 `uv run python scripts/check_doc_links.py` 必须仍全绿。
- 若留:跳过。

- [ ] **Step 5: 提交收尾**

```bash
git add -A docs/
git commit -m "docs: finalize docs merge batch"
```

(若 Step 4 无改动,跳过本步。)

- [ ] **Step 6: 推送并提 PR(提醒用户开启代理 127.0.0.1:7897)**

```bash
git push -u origin feature/docs-pm-merge
gh pr create --title "docs: merge project_management into dev docs" --body "docs/project_management/ 解散(3 份文档 281 行):process_checklist 压缩平铺、release_process 平铺、cli_design 删除并入 ADR-0010;新增 ADR-0013 与 0008 修订注;dev-workflow skill 测试数改为动态收集。见 docs/dev/specs/2026-08-13-docs-pm-merge-design.md"
```

- [ ] **Step 7: CI 全绿后合并与清理**

```bash
gh pr merge --merge --delete-branch
git checkout main
git pull --ff-only origin main
git remote prune origin
```

Expected: 本地 main 与 origin 同步,分支引用清理完毕,工作树干净。
