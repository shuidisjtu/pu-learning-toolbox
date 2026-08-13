# ADR 迁移实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把决策记录从 decision_log.md、architecture_audit.md 批次流水账、architecture.md §1 迁移到 `docs/adr/`(12 篇主题化 ADR),三个源文档瘦身回到单一职责。

**Architecture:** 纯文档重构:新建 `docs/adr/`(README 索引 + 12 篇 ADR),删除 decision_log.md,审计报告瘦身为发现快照,architecture.md 拆出决策段,索引/门禁同步。无代码改动。

**Tech Stack:** Markdown、git、check_doc_links 门禁。

## Global Constraints

- 项目文档全中文(技术标识符保留原文)。
- **ADR 记决策,不承载版本状态**;版本/发布权威在 `docs/project_management/release_process.md` 与 `process_checklist.md` 发布状态节。ADR 标题不含版本号,版本号只出现在「背景」。
- ADR 格式固定:标题 → 状态(已接受/已取代)+ 触发复审 → 背景 → 决策 → 备选方案 → 后果;每篇 25-55 行。
- `docs/superpowers/` 被 .gitignore,计划/规格文件放 `docs/dev/specs/` 与 `docs/dev/plans/`。
- 分支 `feature/adr-migration`(已存在,含 spec 提交 fbe4bd9);每个任务一个提交。
- 修改文件前必须先 Read(hook 强制)。
- 每个任务结束运行 `uv run python scripts/check_doc_links.py`,必须 "All checks passed."。

---

### Task 1: 新建 docs/adr/ 目录(README 索引 + 12 篇 ADR)

**Files:**
- Create: `docs/adr/README.md`
- Create: `docs/adr/0001-architecture-governance.md` … `docs/adr/0012-dependency-release-policy.md`(共 12 篇)
- Modify: `docs/README.md`(在「项目过程」节前新增「架构决策(docs/adr/)」小节)

**Interfaces:**
- Consumes: 无(首个任务)
- Produces: `docs/adr/` 目录,后续任务链接指向它;docs/README.md 索引行

- [ ] **Step 1: Read 源文档**(执行者从 spec 与 git 历史取素材;本计划已内联全部 ADR 全文,直接按 Step 2 写文件即可)

- [ ] **Step 2: 创建 docs/adr/README.md**

```markdown
# ADR 索引

| 编号 | 标题 | 状态 | 日期 |
|---|---|---|---|
| 0001 | 架构治理机制 | 已接受 | 2026-08-13 |
| 0002 | 核心包轻量 + torch 可选依赖 | 已接受 | 2026-08-13 |
| 0003 | 类先验/标记倾向/损失/分类器解耦 | 已接受 | 2026-08-13 |
| 0004 | registry 元数据驱动 | 已接受 | 2026-08-13 |
| 0005 | 复现可信度分级 | 已接受 | 2026-08-13 |
| 0006 | SAR 中长期定位 | 已接受 | 2026-08-13 |
| 0007 | 测试金字塔与 CI 分层 | 已接受 | 2026-08-13 |
| 0008 | 协作与文档惯例 | 已接受 | 2026-08-13 |
| 0009 | 类先验估计修复与 auto 默认切换 | 已接受 | 2026-08-13 |
| 0010 | CLI/auto/skill 工作流 | 已接受 | 2026-08-13 |
| 0011 | 发布体验修复 | 已接受 | 2026-08-13 |
| 0012 | 依赖与发布策略 | 已接受 | 2026-08-13 |

## 规则

- ADR 记决策(背景/决策/备选方案/后果),**不承载版本状态**;版本/发布状态权威在
  `docs/project_management/release_process.md` 与 `process_checklist.md` 发布状态节。
- 琐碎流程惯例写 `CONTRIBUTING.md`,不立 ADR。
- 新决策发生时即写 ADR,不事后批量补记。
- 治理批次流水账(commit 号级别)不记入 ADR 正文;审计发现快照见
  `docs/dev/architecture_audit.md`。
```

- [ ] **Step 3: 创建 docs/adr/0001-architecture-governance.md**

```markdown
# ADR-0001:架构治理机制

- 状态:已接受(2026-08-13)
- 触发复审:发布 minor 版本后,或新增 >5 个文件时复跑架构审计

## 背景

2026-08-09 对全库做四代理并行审计(171 个 .py、42 篇文档),判定「黄——健康,
早期可逆腐朽」:发现 RBF 核公式 5 份独立实现、check_doc_links 三处真实空洞
(孤儿文档漏检已发生)、check_math_rendering 静默假绿、死代码与疤痕组织等信号。
第一批(14 项)与第二批(14 项)治理全部闭环;2026-08-13 第三批对 8 个新提交
做增量检查,又发现 3 处「刚治理过的模式重演」并闭环。

## 决策

1. **审计框架**:架构腐朽四信号(删除风险/局部性/承重 bug/疤痕组织)+ 三表现
   (真相分裂/概念膨胀/治理腐朽),判定绿/黄/红。
2. **单源助手机制**:跨模块共用逻辑提取为单一实现,现 8 项:canonical_hash、
   json_safe、sigmoid_stable、rbf_weights、validate_true_binary_labels、
   check_scalar_in_range、solve_prior_from_positive_fraction、git_worktree_dirty。
   新代码必须复用而非内联重写;清单维护于 `CONTRIBUTING.md` §5.1。
3. **代谢率红线**:PR 增量检查发现 >1 处单源违规 = 黄线(PR 内收敛治理);
   ≥3 处或同一概念第 3 次分裂 = 红线(触发该区域结构性重构评估)。
4. **审计复跑条件**:每发布一个 minor 版本后,或每引入 >5 个新文件时。

## 备选方案

- **一次性大重构**:一次清完所有信号,但违反小步快跑原则,回归面不可控。否决。
- **纯人工审查、不设红线**:治理结果依赖人工记忆,「机制存在但未被执行」的
  裂缝会重演。否决。

## 后果

- 治理从「一次性运动」转为「每批新代码的持续检查」;审计报告瘦身为发现快照,
  批次治理流水账由本 ADR 承接。
- 单源清单成为贡献者可见的硬约束(CONTRIBUTING.md §5.1)。
```

- [ ] **Step 4: 创建 docs/adr/0002-core-lightweight-optional-deps.md**

```markdown
# ADR-0002:核心包轻量 + torch 可选依赖

- 状态:已接受
- 触发复审:需要引入 torch 之外的重量级依赖时

## 背景

工具箱面向 PU 研究者,基础安装应保持轻量;深度学习方法依赖 torch,体积与
安装成本显著高于 numpy 技术栈。

## 决策

- Core 包零深度学习依赖;torch 方法放入 optional extras(`[torch]`/`[research]`)。
- 深度估计器默认 `device=None` 自动检测 CUDA(共享 `core/device.py` 单源,
  见 ADR-0011)。
- 可选依赖不得让基础包导入失败;缺失依赖时延迟导入并给出可行动错误。

## 备选方案

- **torch 进必装依赖**:污染轻量安装,非深度用户承担成本。否决。
- **深度方法单独成包**:注册表统一性与推荐器感知被破坏。否决。

## 后果

- `pip install pu-toolbox` 保持轻量;深度方法需显式 extras。
- 每个深度模块需自行保证「无 torch 可导入、有 torch 可用」。
```

- [ ] **Step 5: 创建 docs/adr/0003-module-decoupling.md**

```markdown
# ADR-0003:类先验/标记倾向/损失/分类器解耦

- 状态:已接受
- 触发复审:概念间出现新耦合需求时

## 背景

PU 学习涉及多个可独立演进的概念:类先验估计、标记倾向、PU 损失函数、
分类器训练。若耦合在单一类中,任一概念的替换都要动整体。

## 决策

- 四概念解耦为独立模块:类先验 `prior/`、损失 `losses/`、分类器 `estimators/`、
  标记倾向 `preprocessing/selection_bias.py`。
- 先验估计器可被任意分类器复用;分类器通过 `class_prior` 参数显式注入先验。

## 备选方案

- **一体化 Pipeline 类**:不可独立测试与替换,单概念改动影响面不可推理。否决。

## 后果

- 各概念可独立测试(如损失模块有独立 golden 测试)。
- 分类器组装需显式注入先验(组装成本换测试与替换自由)。
```

- [ ] **Step 6: 创建 docs/adr/0004-registry-metadata-driven.md**

```markdown
# ADR-0004:registry 元数据驱动

- 状态:已接受
- 触发复审:新算法需要注册表之外的发现机制时

## 背景

17 个已注册算法需要统一的发现、推荐与 CLI 展示;每新增算法不应改动
推荐器与 CLI 调用方。

## 决策

- 所有算法经 registry 管理,元数据(name/aliases/family/scenario/assumption/
  requires_class_prior/backend/maturity/source_status/implementation_status/
  training_cost)驱动发现与推荐。
- 注册即被推荐器与 CLI 感知;CLI 辅助命令(list-methods/list-priors)从
  registry 实时读取。
- 别名解析逻辑集中在 `registry/registry.py` 一处。

## 备选方案

- **硬编码算法清单**:每加算法需改多处调用方,漂移风险高。否决。

## 后果

- 元数据与实现必须同步(防漂移测试锁死);`api_only` 不得伪装为可训练实现。
```

- [ ] **Step 7: 创建 docs/adr/0005-reproduction-fidelity.md**

```markdown
# ADR-0005:复现可信度分级

- 状态:已接受
- 触发复审:新论文接入策略选择,或官方数据/源码状态变化时

## 背景

论文复现可信度差异大:官方源码可获得性与授权状态不同,官方数据可能
需单独授权;项目要求「不假装可用、不虚报复现」。

## 决策

1. **source_status 分级**:official_exact > official_bundle > official_related >
   third_party_only > not_found;有官方源码的论文优先 adapter,无源码走
   clean-room 实现。
2. **claim-safe 原则**:benchmark 产物默认 `paper_claim=false`;配置未锁定的
   维度(如论文未公开类别分组)必须声明「暂定协议」,不得宣称论文数值复现。
3. **PUSB Table 2 严格子集策略**:manifest 锁定 6 数据集 sha256/形状/类别
   计数,fidelity 降级项显式声明,可审计可复跑。
4. **provenance 锁**:磁盘配置 == resolved_config.json == manifest
   config_sha256 三重硬锁,锁测试强制执行;因此 runtime.resume_required 等
   字段不可无代价删除(保留 + 文档化「记录意图、未强制」)。
5. **PUSBKernelClassifier 独立注册**(非 LDCE 别名):与官方实现有 0.5·reg
   分歧,独立注册保证元数据诚实。

## 备选方案

- **全部 clean-room**:有官方源码时浪费最高可信度来源。否决。
- **宣称官方复现但无 provenance 锁**:数据/源码漂移不可检测,违背诚实
  记录。否决。

## 后果

- benchmark 产物可审计可复跑;配置字段变更需显式决策(锁测试会拦截)。
- 官方数据/历史环境全量运行仍依赖执行方提供(非工具箱缺口)。
```

- [ ] **Step 8: 创建 docs/adr/0006-sar-positioning.md**

```markdown
# ADR-0006:SAR 中长期定位

- 状态:已接受
- 触发复审:SCAR 方法覆盖饱和,或出现更强 SAR 工具时

## 背景

通用 PU 工具多数只支持 SCAR 假设;SAR/Instance-Dependent PU 是工具箱
可能的差异化方向,但实现成本高。

## 决策

- SAR/Instance-Dependent PU 为中长期差异化重点:selection-bias 数据模拟器
  (常数/线性/非线性 propensity)、LBE/PUSB bias-aware 估计器、SCAR vs SAR
  对比 benchmark(3 mechanisms × 10 seeds)已落地。

## 备选方案

- **专注 SCAR**:无差异化,与通用工具同质。否决。
- **立即全面 SAR**:实现成本高,且前置依赖(类先验、风险估计)未就绪时
  不可行。否决。

## 后果

- 后续阶段向 SAR 深度方法倾斜;SCAR 仍是基础面,SCAR/SAR 识别边界在
  文档与诊断报告中显式声明(非识别性筛查不伪装成识别)。
```

- [ ] **Step 9: 创建 docs/adr/0007-test-pyramid-ci.md**

```markdown
# ADR-0007:测试金字塔与 CI 分层

- 状态:已接受
- 触发复审:测试执行时间失衡,或 CI 反馈周期超时

## 背景

811+ 测试全量跑慢;慢速/子进程测试拖累 PR 反馈;2026-08-09 CI 曾因本地
漏跑 `ruff format --check` 失败。

## 决策

1. **金字塔分层**:unit/math/property/contract(算法特有/公式/不变量/契约)→
   integration(跨组件)→ e2e(真实子进程旅程)→ slow(慢速)。
2. **CI 分层**:PR 快层 `-m "not slow and not e2e"`(unit + integration +
   静态门禁);nightly 顶层 `-m "slow or e2e"`(每周一 03:23 UTC +
   workflow_dispatch)。
3. **check_format.py 为第 6 道门禁**:ruff check + format --check 全目录,
   本地与 CI 同一入口。
4. **check_test_quality 严格默认**:每文件 ≤15 测试方法、basic/param/edge/
   determ 四分类全覆盖;`--lenient` 显式退出。

## 备选方案

- **全量每次跑**:PR 反馈分钟级变小时级。否决。
- **无测试质量门禁**:分类缺口与超限文件无人发现(曾实际发生)。否决。

## 后果

- PR 反馈周期分钟级;慢速与 e2e 由 nightly 兜底。
- 测试质量(分类/上限)由门禁持续执行,而非依赖人工记忆。
```

- [ ] **Step 10: 创建 docs/adr/0008-collaboration-conventions.md**

```markdown
# ADR-0008:协作与文档惯例

- 状态:已接受
- 触发复审:协作方式变化(如外部贡献者增多),或文档导航混乱时

## 背景

双人 + agent 协作项目,需约定分支、文档组织与任务分工,避免冲突与
导航混乱。

## 决策

1. **分支规范**:代码改动开 `feature/<name>` 或 `fix/<name>`,提 PR 合并到
   main;不在 main 直接开发。
2. **docs 受众分层**:docs/ 按 user/(旅程化)/dev/(开发者)/research/(方法卡)/
   project_management/(过程)分层,参考 scikit-learn 受众分离;docs 全中文、
   根 README 双语。
3. **论文分工**:shuidisjtu 负责基础 6 篇(Elkan-Noto/uPU/nnPU/PNU/Centroid/
   LLSVM),HENG958 负责扩展 6+3 篇(penL1/ReCPE/Dist-PU/PUSB/LBE/Self-PU
   + InfoMax/WConPU/DGPU)。
4. **Phase 重整**:核心 PU 风险估计优先(Elkan-Noto → uPU → nnPU → ReCPE),
   经典分类器包装器后移;阶段定义以 `process_checklist.md` 为准。

## 备选方案

- **main 直接提交**:main 稳定性无保障。否决。
- **文档平铺**:22 篇平铺索引无法区分受众(已实际重构)。否决。

## 后果

- 协作并行无冲突;文档导航按受众可预期。
- 本 ADR 后续新增的 ADR 目录(docs/adr/)是第 3 项的补充,不推翻受众分层。
```

- [ ] **Step 11: 创建 docs/adr/0009-prior-estimation-fix.md**

```markdown
# ADR-0009:类先验估计修复与 auto 默认切换

- 状态:已接受(2026-08-10)
- 触发复审:新估计器接入,或 auto 默认再评估时

## 背景

v1.2.0 真实验证发现:常规 SCAR 数据上全部类先验估计器系统性低估
(recpe 坍缩至 0.036 vs 真值 0.5),级联导致 auto 模式 UPU 全判负
recall=0。根因:固定 σ 尺度失配、max-MMD 带宽偏选宽、OOF 折内类比例
偏移、1% 分位数 + 未校准概率坍缩。旧测试宽区间 0.25–0.75 恰好掩盖
27% 偏差。

## 决策

1. pen_l1 默认 `sigma=None` 数据自适应(0.6×标准化数据中位 pairwise
   距离,显式 σ 兼容)。
2. KM 默认 `width_selection="relative"`(0.1×中位距离;作者 mmd_grid 保留
   可选)。
3. ElkanNoto OOF 折内 sqrt 类平衡 sample_weight(几何中点插值)。
4. ReCPE 默认 base 换官方风格 KM2 + `_DensityRatioCPE` 分位数 0.01→0.25;
   边界声明:全部 base 变体仍固有低估,修复目标仅为不坍缩。
5. **auto 默认估计器 recpe → pen_l1**(跨分离度回归入带后切换;recpe 保留
   用于 irreducibility 失效场景)——取代此前的 recpe 默认。
6. prior 测试补数值准确性断言(math golden + 跨分离度/先验护栏 + 集成
   auto 带护栏)。

## 备选方案

- **仅调参、不换默认**:recpe 坍缩场景仍复发。否决。
- **全量替换 recpe**:irreducibility 失效场景下 recpe 仍有价值。否决。

## 后果

- auto 路径默认可靠;测试从「宽区间」升级为数值断言。
- 全 base 变体固有低估的边界在文档中显式声明,不假装消除。
```

- [ ] **Step 12: 创建 docs/adr/0010-cli-auto-skill.md**

```markdown
# ADR-0010:CLI/auto/skill 工作流

- 状态:已接受
- 触发复审:CLI 命令数膨胀,或 agent 生态变化时

## 背景

CLI 需零新增依赖、可扩展;agent 工作流(pu-workflow skill)需可复用、
可分发;默认 run 路径曾实测 ~30s。

## 决策

1. **CLI 采用 argparse 单命令薄封装**(run/list-methods/list-priors/
   make-demo-data/…):所有逻辑在 PUPipeline,CLI 只做参数解析/CSV IO/
   错误映射;辅助命令从 registry 实时读取。
2. **run 默认 auto 模式引入推荐器训练成本维度**(第 7 维)+ LLSVM 收敛
   早停(默认开):默认 run 实测 30s → 2s。
3. **Deep PU 接入 Pipeline/CLI**:两级参数 `architecture`(mlp/cnn)+
   `backbone`(cnn13/resnet18/resnet50);`--data` 接受 .npy 4D NCHW 图像;
   DGPU/Self-PU 不接入(无单骨架插拔概念)。
4. **pu-workflow skill 通用化**:开放规范/双目录 SKILL.md + 中文解读指南,
   check_skill_sync 门禁保证一致。
5. **skill install 子命令**:SKILL.md 随 wheel 分发,一键安装到
   `~/.claude/skills/` 与 `~/.agents/skills/`(默认跳过已存在,--force 覆盖)。

## 备选方案

- **click/typer**:引入新依赖,违反零依赖原则。否决。
- **每算法硬编码 CLI 命令**:注册即感知更省维护。否决。

## 后果

- 新算法注册后 CLI 自动可见;skill 三份副本风险由门禁与随包分发控制。
- 默认路径耗时是产品面指标,后续改动受此约束。
```

- [ ] **Step 13: 创建 docs/adr/0011-release-experience-fix.md**

```markdown
# ADR-0011:发布体验修复

- 状态:已接受(2026-08-10)
- 触发复审:用户以真实安装实测暴露新体验问题时

## 背景

用户以真实 PyPI 安装 + GPU 实测发现问题:GPU 机器默认吃 CPU;WConPU
默认 800 epoch 无早停需 ~1.5h;skill 环节脚本在包外导致 pip 用户无法
使用;发布的 wheel 中 `import pu_toolbox` 返回错误版本号。

## 决策

1. 深度估计器默认 `device=None` 自动检测 CUDA(共享 `core/device.py`
   单源;CLI `--device` 默认 auto)。
2. profile/recommend/sensitivity 收为 CLI 子命令(scripts 改兼容包装)。
3. CLI `--max-epochs` 透传;WConPU 默认 max_epochs 800→100。
4. `__version__` 漂移修复:`pu_toolbox/__init__.py` 硬编码版本改为与
   pyproject 同步;check_project_metadata 门禁新增 `__version__` 与
   `project.version` 一致性检查(负向验证通过)。

## 备选方案

- **仅文档提示**:默认路径体验是产品面,文档救不了默认行为。否决。

## 后果

- GPU 机器默认吃 GPU;pip 用户可用 skill 环节脚本;版本漂移有门禁拦截。
- device 解析逻辑单源化,后续新增后端复用同一入口。
```

- [ ] **Step 14: 创建 docs/adr/0012-dependency-release-policy.md**

```markdown
# ADR-0012:依赖与发布策略

- 状态:已接受
- 触发复审:Python 支持矩阵调整,或打包方式变更时

## 背景

项目是 library,需在 CI 持续验证声明范围内的最新可解析依赖;首版发布时
项目从未出过正式版(0.1.0.dev0),但 roadmap 0.1→0.6 功能已全部完成
(17 算法、6 门禁)。曾发生 `.python-version` 覆盖 CI matrix、未验证却
声明 3.13、源码可用但发布包缺文件的问题。

## 决策

1. **Python 支持收敛 3.10-3.12**;CI 显式锁定 matrix interpreter,分离
   测试、静态门禁和 wheel 安装冒烟。
2. **pyproject.toml 为依赖权威来源**;`uv.lock` 不入库;`requirements.txt`
   仅作开发环境快照(问题复查用,不与 pyproject 手工双向同步)。
3. **v1.0.0 首版**:功能齐全后直接升 1.0.0 首版发布。

## 备选方案

- **提交 uv.lock**:library 需要验证声明范围内最新可解析依赖,锁文件
  会掩盖漂移。否决。
- **保留 .python-version 覆盖 CI**:曾导致声明 3.13 未验证。否决。

## 后果

- 版本演进/发布状态记录于 `docs/project_management/release_process.md`
  与 `process_checklist.md` 发布状态节;本 ADR 不承载版本状态。
- CI matrix 与 extras 的一致性由 check_project_metadata 门禁维护。
```

- [ ] **Step 15: docs/README.md 新增「架构决策」小节**

Read `docs/README.md` 后,在「## 项目过程(docs/project_management/)」标题前插入:

```markdown
## 架构决策(docs/adr/)

| 文档 | 用途 |
|---|---|
| [adr/README.md](adr/README.md) | ADR 索引(12 篇:架构治理/解耦/复现分级/测试 CI/流程惯例/发布策略等) |

> ADR 记决策,版本/进度状态见 process_checklist.md 与 release_process.md。
```

- [ ] **Step 16: 运行文档门禁**

Run: `uv run python scripts/check_doc_links.py`
Expected: `All checks passed.`(adr/ 内有 README.md 索引,满足索引完备性)

- [ ] **Step 17: Commit**

```bash
git add docs/adr/ docs/README.md
git commit -m "docs: add ADR directory with 12 decision records and index"
```

---

### Task 2: 删除 decision_log.md 并更新指针

**Files:**
- Delete: `docs/project_management/decision_log.md`
- Modify: `docs/project_management/process_checklist.md`(底部指针)
- Modify: `docs/README.md`(「项目过程」节删 decision_log 行)

**Interfaces:**
- Consumes: Task 1 的 `docs/adr/` 目录
- Produces: 无 decision_log 死链;后续任务在此基础上改审计报告与架构文档

- [ ] **Step 1: 逐个删除文件(项目约束:避免批量删除触发 hooks)**

Run: `git rm docs/project_management/decision_log.md`

- [ ] **Step 2: Read process_checklist.md 底部,替换指针**

将 `历史执行记录见 git log;关键决策见 \`docs/project_management/decision_log.md\`。` 替换为:

```markdown
历史执行记录见 git log;关键决策见 [`docs/adr/`](../adr/)。
```

- [ ] **Step 3: docs/README.md「项目过程」节删除 decision_log 行**

删除行:

```markdown
| [project_management/decision_log.md](project_management/decision_log.md) | 项目决策日志（含文档体系重构决策） |
```

- [ ] **Step 4: 运行文档门禁**

Run: `uv run python scripts/check_doc_links.py`
Expected: `All checks passed.`(无 orphan 链接;project_structure.md 文档树中的纯文本路径不构成链接,Task 5 统一更新)

- [ ] **Step 5: Commit**

```bash
git add docs/project_management/ docs/README.md
git commit -m "docs: delete decision_log.md, superseded by docs/adr/"
```

---

### Task 3: architecture_audit.md 瘦身

**Files:**
- Modify: `docs/dev/architecture_audit.md`

**Interfaces:**
- Consumes: Task 1 的 ADR-0001(审计机制)
- Produces: 瘦身后的审计报告(§3/§4 信号表 + §6 复跑指南 + §7 复核记录),供下次审计复跑对照

- [ ] **Step 1: Read docs/dev/architecture_audit.md 全文**

- [ ] **Step 2: §1 元信息表加一行**

在「| 复跑方法 | ...」行后加:

```markdown
| 决策迁移 | 2026-08-13 治理批次与行动项流水账已迁移至 [ADR-0001](../../adr/0001-architecture-governance.md);本文档只保留发现快照 |
```

- [ ] **Step 3: §2 总评替换三块批次流水账**

将 §2 中「第一批治理后更新」「第二批治理后更新」「文档对齐更新」「第三批治理后更新」四个 blockquote 整块替换为一行:

```markdown
> 批次治理记录(行动项、commit、闭环状态)已迁移至
> [ADR-0001](../../adr/0001-architecture-governance.md);总评判定与
> 信号清单保留于此,作为下次复跑审计的对照基线。
```

保留 §2 首段总评与「做得好的/要警惕的」两段原文不动。

- [ ] **Step 4: §5 行动项清单整节替换**

将「## 5. 行动项清单」至「## 6. 复跑指南」之间的全部内容(含第一批/第二批全部条目与保留说明)替换为:

```markdown
## 5. 行动项清单

已迁移至 [ADR-0001](../../adr/0001-architecture-governance.md)(架构治理机制:
审计框架、单源助手、代谢率红线、复跑条件)。第一批 14 项与第二批 14 项均
已闭环,第三批 3 项已闭环(2026-08-13)。
```

- [ ] **Step 5: §6 复跑指南中单源清单行改指向 CONTRIBUTING**

将「- **第二批单源助手清单(2026-08-09,2026-08-13 更新)**:…」整行替换为:

```markdown
- **单源助手清单**:以 `CONTRIBUTING.md` §5.1 为权威(消除清单第三份副本);
  复跑时核对新代码是否复用、check_test_quality 是否保持严格默认、
  slow 套件是否随 nightly 自动执行
```

- [ ] **Step 6: 运行文档门禁 + 确认行数**

Run: `uv run python scripts/check_doc_links.py && wc -l docs/dev/architecture_audit.md`
Expected: `All checks passed.` 且行数 ≤ 120(原 190)

- [ ] **Step 7: Commit**

```bash
git add docs/dev/architecture_audit.md
git commit -m "docs: slim architecture audit to findings snapshot, batch records now in ADR-0001"
```

---

### Task 4: architecture.md 拆出「设计决策与代价」

**Files:**
- Modify: `docs/dev/architecture.md`

**Interfaces:**
- Consumes: Task 1 的 ADR-0002~0006(原 §1 五行决策的去向)
- Produces: architecture.md 只保留「现在是什么」(分层/数据流/API/注册表)

- [ ] **Step 1: Read docs/dev/architecture.md 的 §1 段**

- [ ] **Step 2: 替换 §1 决策表**

将「## 1. 设计决策与代价」标题 + 五行决策表(至「**与 \`project_structure.md\` 的分工**」行之前)整体替换为:

```markdown
## 1. 设计决策

设计决策与代价已迁移至 [docs/adr/](../../adr/README.md)(ADR-0002 核心包
轻量化、ADR-0003 概念解耦、ADR-0004 registry 元数据驱动、ADR-0005 复现
可信度分级、ADR-0006 SAR 定位)。本文档只描述当前架构。

**与 `project_structure.md` 的分工**：本文档解释"为什么这样组织"（决策、
依赖方向、数据流）；文件清单与目录结构以 [`project_structure.md`](project_structure.md) 为权威来源。
```

(「与 project_structure.md 的分工」行内容原样保留,只是随新 §1 一起出现)

- [ ] **Step 3: 运行文档门禁**

Run: `uv run python scripts/check_doc_links.py`
Expected: `All checks passed.`

- [ ] **Step 4: Commit**

```bash
git add docs/dev/architecture.md
git commit -m "docs: extract design-decision section from architecture.md into ADRs"
```

---

### Task 5: 索引/门禁同步收尾 + 全量验收

**Files:**
- Modify: `docs/dev/project_structure.md`(§5 文档树:删 decision_log 行、加 adr/ 子树)
- Modify: `CONTRIBUTING.md`(§1 权威来源加 docs/adr/ 条目)
- Modify: `docs/README.md`(如有遗漏行)

**Interfaces:**
- Consumes: Task 1-4 的全部产出
- Produces: 全库无 decision_log 引用;六门禁全绿;分支可提 PR

- [ ] **Step 1: grep 全库残留引用**

Run: `grep -rn "decision_log" docs/ CONTRIBUTING.md README.md .claude/ 2>/dev/null`
Expected: 无输出(若 .claude/ 有引用一并处理)

- [ ] **Step 2: Read project_structure.md §5,更新文档树**

删除 `decision_log.md` 行;在「## 5. 文档(docs/)」树中,`dev/` 块之后、`research/` 之前新增:

```text
  adr/                          (架构与流程决策记录:12 篇 ADR + README 索引)
```

(spec 与 plan 两个临时文件只存在于本分支,不进权威目录树,Task 6 删除)

- [ ] **Step 3: CONTRIBUTING.md §1 权威来源加一条**

在「6. `requirements.txt`…」后加:

```markdown
7. `docs/adr/`：架构与流程决策记录（决策的权威来源；版本/进度状态见
   process_checklist.md 与 release_process.md）。
```

- [ ] **Step 4: 六道门禁全跑**

Run:

```bash
uv run python scripts/check_doc_links.py
uv run python scripts/check_test_quality.py
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
uv run python scripts/check_format.py
```

Expected: 全部通过(check_doc_links "All checks passed."、test_quality "✓ All checks passed."、metadata "aligned"、math_rendering "Total issues: 0"、skill_sync "identical"、format "clean")

- [ ] **Step 5: 快速测试确认无意外**

Run: `uv run pytest tests/ -q -m "not slow and not e2e" 2>&1 | tail -2`
Expected: 全部通过(纯文档改动,907 passed 基线)

- [ ] **Step 6: Commit**

```bash
git add docs/dev/project_structure.md CONTRIBUTING.md docs/README.md
git commit -m "docs: sync indexes and gate references for ADR migration"
```

---

### Task 6: 蒸馏删除 spec 与 plan

**Files:**
- Delete: `docs/dev/specs/2026-08-13-adr-migration-design.md`
- Delete: `docs/dev/plans/2026-08-13-adr-migration.md`
- Modify: `docs/README.md`(删 spec 索引行)

**Interfaces:**
- Consumes: Task 1-5 全部完成
- Produces: 仓库不留过程文档(按 deep_pipeline_design.md 先例)

- [ ] **Step 1: 逐个删除文件(项目约束:避免批量删除触发 hooks)**

```bash
git rm docs/dev/specs/2026-08-13-adr-migration-design.md
git rm docs/dev/plans/2026-08-13-adr-migration.md
```

- [ ] **Step 2: docs/README.md 删 spec 索引行**

删除行:

```markdown
| [dev/specs/2026-08-13-adr-migration-design.md](dev/specs/2026-08-13-adr-migration-design.md) | ADR 迁移设计 spec（实现后按先例蒸馏删除） |
```

- [ ] **Step 3: 运行文档门禁**

Run: `uv run python scripts/check_doc_links.py`
Expected: `All checks passed.`

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs: distill ADR migration spec and plan after implementation"
```

---

## 完成标准

- `docs/adr/` 12 篇 ADR + README 索引存在且六门禁全绿
- 全库 grep `decision_log` 无残留
- architecture_audit.md ≤ 120 行;architecture.md §1 只有链接
- 分支 `feature/adr-migration` 共 6 个提交,PR 合并到 main(dev-workflow 流程)
