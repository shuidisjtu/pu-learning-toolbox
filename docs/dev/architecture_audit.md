# 架构健康度审计报告

> 审计日期:2026-08-09 | 审计范围:全库(pu_toolbox 代码面 + tests/benchmarks/scripts 工程面 + docs 文档面 + git/CI/skills 治理面)
> 方法:四领域子代理并行审计(代码/工程/文档/治理),统一契约输出(信号 × 绿/黄/红 × file:line 证据 × 置信度),主上下文对全部红项与关键黄项抽样复核后定稿。
> 判断框架:架构腐朽四信号(删除风险/局部性/承重bug/疤痕组织)+ 三表现(真相分裂/概念膨胀/治理腐朽)。

## 1. 审计元信息

| 项 | 值 |
|---|---|
| 审计日 | 2026-08-09(仓库 HEAD c3501e9 + 审计 spec 5dab13a/6c97cb5) |
| 规模 | 171 个 .py;pu_toolbox ~15.8k 行;tests+benchmarks+scripts ~17.2k 行;docs 40 篇 ~9.8k 行 |
| 覆盖 | 四代理共读取:代码面全量 47 文件通读、工程面 58 测试文件 + 16 benchmark 脚本 + 6 门禁脚本全文、文档面 docs/README 索引 + method cards 全量 + README 双语逐行 diff、治理面 CI/双份 skill/git 历史(206 commits) |
| 门禁基线 | 6 道门禁全绿,738 passed(收集 739,1 个环境性 skip) |
| 复跑方法 | 按 §6 重派四代理,或按行动项进度局部复核 |

## 2. 总评

**判定:黄——健康系统,处于早期可逆腐朽阶段,代谢率仍高于腐朽率,但已出现两处"治理盲区已实际漏检"与一处"代码-推导自相矛盾",需在 v1.0.0 后的稳定期优先治理。**

- **做得好的**:17 个算法全部 NATIVE(0 个 api_only 假实现);registry 有防漂移测试锁死;PUSB benchmark 有 manifest sha256 锁定(真相分裂的最小化范例);文档-实现抽查 4 篇方法卡全部一致;双语 README 逐行同步(但源于近期人工治理而非结构性保障)。
- **要警惕的**:① 提取公共逻辑的机制健全(校验助手存在)但新代码不遵循——"机制存在"与"机制被执行"之间出现裂缝;② check_doc_links 声称覆盖实则排除最多引用密集的目录,漏检已实际发生(孤儿文档入库无人发现);③ 一处与自身推导注释矛盾的数学实现未被测试锁定。

> **第二批治理后更新(2026-08-09,分支 fix/architecture-decay-batch2)**:§3/§4 全部黄/红项已闭环(14/14,2 项有意保留,见 §5 第二批治理);"机制存在但未被执行"的裂缝由 check_test_quality 严格默认与 nightly CI 转为结构性约束。判定维持黄——整体健康,残余风险集中在代谢率(新代码是否持续复用单源助手)与保留项。
>
> **文档对齐更新(2026-08-10)**:§1 中"project_structure.md §3 测试树与实际 1:1 双向对齐(commit 021e4b3)"的承诺再次失效——第二批治理新增的 `test_class_prior.py`/`test_pnu_loss.py` 与后续文件(utils/activations.py、utils/serialization.py、estimators/risk/_class_prior.py、benchmarks/_common.py、scripts/pu_workflow/、nightly.yml)共 8 处目录树漂移,已随 2026-08-10 文档检查全部修复。§4 T3 表格为审计时点快照(nightly.yml 由第二批第 12 条治理落地);复跑指南 §6 不变。
>
> **第三批治理后更新(2026-08-13,分支 fix/architecture-decay-batch3)**:对 6492e45..c4754e7(8 提交,benchmark 审计器 + InfoMax 先验矩阵)做增量代谢率检查,判定维持黄——canonical_hash 迁入 serialization.py 是正确收敛,但新代码出现 3 处信号,已闭环:① `_git_worktree_dirty` 第 5 份实现收敛到 `benchmarks/_common.py::git_worktree_dirty`(含 output-exclude 语义,4 调用点统一);② `unlabeled_class_prior` 两处内联校验改走 `check_scalar_in_range(inclusive=False)`;③ 审计器 `_check_splits` 对畸形 `dataset_splits`(非 dict)由静默放行改为报错,并补齐 6 个未测错误分支。两项经权衡不改:`_METRIC_COLUMNS` 确认为跨 runner 指标并集(裁剪会制造盲区),补维护注释;`runtime.resume_required` 保留——provenance 锁测试(config == resolved == manifest hash)不允许无代价删除,已在 runner 文档化为「记录意图、未强制」。单源清单与代谢率红线已写入 CONTRIBUTING.md §5.1。

## 3. 四信号逐条

### S1 删除风险 — 黄

| 状态 | 发现 | 证据 | 置信度 |
|---|---|---|---|
| 黄 | `PNULoss` 类完全死亡:不在 losses/__init__ 导出、无测试、无生产调用(PNUClassifier 用模块级函数) | `pu_toolbox/losses/pnu.py:110` | 高 |
| 黄 | 校验助手 `check_scalar_in_range`/`check_positive` 零调用,各分类器内联校验 | `pu_toolbox/core/validation.py:86,120` | 高 |
| 黄 | `PenL1Estimator = ClassPriorEstimator` 别名导出但全仓库零消费者 | `pu_toolbox/prior/pen_l1.py:94`、`prior/__init__.py:5` | 高 |
| 黄 | `upu.py:46-49` 标注 "Backwards-compatible private aliases" 的 3 个 `_build_*` 别名不存在旧名消费者——虚构的向后兼容 | `pu_toolbox/estimators/risk/upu.py:46-49` | 高 |
| 黄 | `DEFAULT_RANDOM_SEED`(config.py:14)、`set_global_seed`(仅 conftest 用)、`pu_validation_risk` 公共方法(仅测试用)零生产消费者 | `pu_toolbox/core/config.py:14`、`core/random.py:10`、`risk/upu.py:483` | 高 |

**影响**:删除风险初现但规模可控(均为易删除项,无"删不掉"的承重层)。**建议**:随 v1.0.0 后清理,一次性移除或标注 deprecated。

### S2 局部性丧失 — 黄

| 状态 | 发现 | 证据 | 置信度 |
|---|---|---|---|
| 黄 | `pipeline.py`(938 行)单文件多职责:`fit_evaluate` 单方法 ~275 行内联完成 prior 解析/画像/auto 推荐/deep 搭建/CV/诊断/报告 | `pu_toolbox/workflows/pipeline.py:331-605` | 高 |
| 黄 | `kldce.py`(1107 行)与 `ldce.py`(570 行)重复实现同一论文的类先验推导与分母检查(见 T1) | `risk/kldce.py:844` vs `risk/ldce.py:408-431` | 高 |
| 黄 | `losses/pnu.py`(225 行,eta≥0/eta<0 双分支风险公式)无专属测试文件,PNU 损失仅经契约冒烟间接覆盖 | `tests/unit/losses/` 缺 test_pnu_loss.py | 高 |
| 黄 | `_canonical_hash` 在两个 benchmark runner 中逐字节重复,单边改动即产生跨域哈希语义漂移 | `benchmarks/assigned_methods/runner.py:69` vs `benchmarks/deep_pu/runner.py:346` | 高 |

**影响**:改动 pipeline/kldce 的影响边界不可完整推理(大文件 + 跨文件知识重复)。**建议**:pipeline 的 `fit_evaluate` 按阶段拆内部函数;pnnu loss 补专属 golden 测试。

### S3 承重bug — 黄(一处高优先)

| 状态 | 发现 | 证据 | 置信度 |
|---|---|---|---|
| 黄(优先) | **KLDCE 偏置恢复代码与自身推导注释自相矛盾**:推导块(604-606)明确 ỹ=−1 时 γⱼ=0 → `b₀ ≤ −1−g`,执行代码两个 γ 分支均 append `1.0−g`(差 +2);且 `free_alpha_mask` 注释声称过滤 ỹ=+1 但掩码无 y_tilde 条件,U 样本自由 α 以正样本公式计入。测试只断言模式与有限性、不断言数值,偏差未被锁定。U 样本 γ 多数 free 时中位数偏置估计系统性偏 +2 | `pu_toolbox/estimators/risk/kldce.py:581-637`;测试盲区 `tests/estimators/risk/test_kldce_math.py:412-514` | 中(代码-注释矛盾为高;是否真 bug 需对照论文/官方实现复核) |
| 黄 | `list_algorithms` docstring 声称 `trainable_only=True` 排除 "api_only **and experimental**",代码只排除 api_only | `pu_toolbox/registry/registry.py:258-288` | 高 |
| 绿 | 文档固化错误行为:抽查 4 篇方法卡(KLDCE/PUSB/Self-PU + api.md 总览表 18 行)与实现逐一核对全部一致 | — | 高 |

**影响**:KLDCE 项若为真 bug,是典型的"错误行为未被测试锁定 → 潜在契约化"。**建议**:优先对照论文(Du Plessis 2017 附录)与官方源码复核 γ 分支,再决定修代码或修注释;补数值断言测试。

### S4 疤痕组织 — 黄

| 状态 | 发现 | 证据 | 置信度 |
|---|---|---|---|
| 黄 | `sample_weight` 僵尸参数七种语义家族式蔓延:LDCE/KLDCE/UPU/PNU 文档化忽略、DistPU/LBE 静默忽略、SelfPU 抛 NotImplementedError、nnPU/PUSB/ElkanNoto/DGPU/WConPU 真实支持 | `risk/dist_pu.py:66`、`risk/lbe.py:56` 等 | 高 |
| 黄 | `.gitignore` 206 提交改 14 次,积累 18 行 benchmark 结果白名单逐目录例外,无统一忽略规则 | `.gitignore:59-77` | 中 |
| 黄 | 门禁豁免名单只增不减:check_test_quality 5+5 文件手工名单,无退出机制 | `scripts/check_test_quality.py:91-97,102-108` | 中 |
| 黄 | 4 个 benchmark 测试文件模块级 pytestmark 与逐函数 marker 双重声明完全冗余,门禁 AST 访问器看不见模块级 pytestmark | `tests/benchmarks/test_assigned_benchmark_runner.py:23` 等 | 高 |
| 绿 | pusb_kernel 对官方 0.5·reg 分歧的文档化("official_compatibility")是诚实记录而非掩盖 | — | 高 |

**影响**:家族式蔓延是最典型的疤痕形态——修复时无法局部推理。**建议**:sample_weight 语义统一为三档(支持/文档化忽略/静默忽略→文档化)。

## 4. 三表现逐条

### T1 真相分裂 — 红(核心问题)

| 状态 | 发现 | 证据 | 置信度 |
|---|---|---|---|
| **红** | **RBF 核公式 5 份独立实现**,各自维护:utils/basis、kldce、pusb_kernel(含截距列)、pen_l1 内联、kernel_mean 内联 | `utils/basis.py:34`、`risk/kldce.py:59`、`bias_aware/pusb_kernel.py:45`、`prior/pen_l1.py:57-60`、`prior/kernel_mean.py:140` | 高(复核通过) |
| 黄 | class_prior 范围校验 ≥9 个分类器内联复制,校验助手存在却无人调用 | `risk/nnpu.py:182-183` 等 9 处 | 高 |
| 黄 | 类先验推导与 1−2ph 分母检查在 ldce/kldce 逐字重复 | `risk/kldce.py:844-864` vs `risk/ldce.py:408-431` | 高 |
| 黄 | PU 零一风险公式双份(`metrics.pu_zero_one_risk` vs `upu._pu_validation_risk` 逐行相同);`_sigmoid`/`_sigmoid_stable` 逐字相同;y_true 校验 4 份实现 | `metrics/classification.py:27-72`、`losses/upu.py:36`、`losses/nnpu.py:38`、`workflows/pipeline.py:893-903` | 高 |
| 黄 | `docs/dev/project_structure.md` §3 测试树漂移:未收录 `tests/unit/scripts/`、`tests/unit/workflow_scripts/`、benchmarks 8 文件缺 4,且门禁对新增文件不可见,漂移静默发生 | `docs/dev/project_structure.md:121-195` | 高(复核通过) |
| 绿 | PUSB manifest 单一真源(URL/sha256/形状/类别计数)+ loader 强制 + 伪造 manifest 拒绝测试——真相分裂的反例样板 | `benchmarks/assigned_methods/pusb_table2_data.py:43-107` | 高 |

**影响**:RBF 5 份是全局最典型的真相分裂——改一处其余四处漂移,且无任何门禁能发现公式漂移。**建议**:RBF 单源化到 `utils/basis.py`(4 处改为调用);class_prior 校验统一走 `check_scalar_in_range`。

### T2 概念膨胀 — 黄

| 状态 | 发现 | 证据 | 置信度 |
|---|---|---|---|
| 黄 | 同一算法 4 个名字:`ClassPriorEstimator`/别名 `PenL1Estimator`/注册名 `class_prior_estimation`/注册别名 `pen_l1`(其中 PenL1Estimator 零消费者) | `prior/__init__.py:5`、`registry/builtin_methods.py:51` | 高 |
| 黄 | DGPU 双重族归属:method_selection 同时列入 Bias-Aware 与 Deep 两族,registry 与 api.md 只标 deep | `docs/user/method_selection.md:74-81` vs `registry/builtin_methods.py:379` | 高 |
| 黄 | `pu_data_summary`/`pnu_data_summary` 双键 `n_features`/`n_features_out` 别名并存 | `preprocessing/profiling.py:110-112` | 高 |
| 黄 | "17 篇论文"计数含两个非论文变体(KLDCE 核化版、PUSBKernel),README 口径与"论文"语义有细微偏差;architecture.md:187 仍写 "15" | `docs/dev/architecture.md:187` | 中 |
| 绿 | 核心术语(已标记正样本/类先验/SCAR/SAR)命名统一,pu_problem.md 符号表承担术语表职责 | — | 高 |

**影响**:概念膨胀可控,主要风险是别名与零消费者并存。**建议**:移除 PenL1Estimator 别名或标注 deprecated;统一 DGPU 族归属表述。

### T3 治理腐朽 — 红(两处已实际漏检)

| 状态 | 发现 | 证据 | 置信度 |
|---|---|---|---|
| **红** | **check_doc_links 三个真实空洞,漏检已实际发生**:① Rule 1 只匹配反引号 `.py` 路径,markdown 链接与非 .py 路径不查;② `_EXCLUDED_DOC_DIRS` 整体豁免 research(方法卡——引用最密集)/project_management/superpowers/figures 四目录;③ 索引完备性只查 docs 顶层 + 3 个白名单 PM 文件,子目录新增文档不查。后果:`pu_workflow_design.md` 未进任何索引且未被门禁告警,实测运行 exit 0 | `scripts/check_doc_links.py:47,53-58,61-65` | 高(复核通过) |
| **红(黄升级)** | **check_math_rendering 静默假绿**:唯一不锚定 PROJECT_ROOT 的门禁,`glob.glob("docs/research/method_cards/*.md")` 相对 cwd,误目录运行输出 "Total issues: 0" exit 0 空扫放行(B、D 两代理独立发现) | `scripts/check_math_rendering.py:84` | 高(复核通过) |
| 黄(交叉) | `paper` marker 空转:pyproject 注册 + CLAUDE.md/CONTRIBUTING 文档化,全仓库 0 个测试使用(B、D 独立发现);文档化命令 `uv run pytest -m paper` 恒收集 0 用例 | `pyproject.toml:110`、`scripts/check_test_quality.py:37-44` | 高(复核通过) |
| 黄 | check_test_quality 默认宽松:缺 1 类覆盖放行、依赖测试名关键词启发式可绕过、CI 非 strict | `scripts/check_test_quality.py:272`、`.github/workflows/tests.yml:76` | 中 |
| 黄 | decision_log 停更 2026-08-06,其后 ≥6 条治理级决策只有"做了什么"无"为什么";自身排序非时间序 | `docs/project_management/decision_log.md:13` | 高 |
| 黄 | `slow` 测试无自动执行环境:CI 跑 `not slow`,无 nightly workflow,唯一 slow 套件只能靠人工记得跑 | `.github/workflows/tests.yml:63`、`tests/unit/estimators/test_nnpu.py:503` | 中 |
| 绿 | 6 门禁全部接入 CI、均有失败路径测试;marker 注册集与门禁集合精确一致;CI 无永远成功/冗余步骤;roadmap 如实反映阻塞项 | — | 高 |

**影响**:治理面的核心矛盾——门禁体系数量多(6 道)但覆盖面有结构性空洞,且"永远通过的检查"(paper marker、math_rendering 空扫)正是腐朽文本中"无价值的检查"。**建议**:优先补 check_doc_links 与 math_rendering 两个洞;paper marker 落地或移除二选一;decision_log 补齐。

## 5. 行动项清单

### P0 — 正确性风险,优先处理

1. ✅ 已治理(2026-08-09,commit 046af07):KLDCE 偏置恢复复核确认 `−1−g` 正确(互补松弛:free 乘子 → ỹ·f=1),代码四处分支 + docstring + 推导注释统一;新增数值断言测试(`test_edge_gamma_free_bias_uses_neg_one_minus_g`,锁定 b₀=−1.25)。**后续发现**:KLDCE 方法卡 §6.5 同含 `1−g` 错误,已随 Task 14 一并修正(commit 870f437)
2. ✅ 已治理(2026-08-09,commit 7cb73b3):check_doc_links 三洞补全——PATH_PATTERN 支持 .py+.md、新增 rule-5 markdown 链接存在性检查、索引完备性扩展全树;`_EXCLUDED_DOC_DIRS` 缩小为 {superpowers, figures}。门禁随即捕获审计事件本身(orphan 报错),测试 9 个
3. ✅ 已治理(2026-08-09,commit 5c3ffb2):check_math_rendering 锚定 PROJECT_ROOT,空扫描拒绝 exit 1;跨目录运行验证不假绿

### P1 — 真相分裂与治理盲区

4. ✅ 已治理(2026-08-09,commit f358630):RBF 单源化到 `utils/basis.py`(kldce/pen_l1/kernel_mean 三处委托;pusb_kernel 保留并注释);KLDCE MATH golden 数值 bit-identical。**后续发现**:kldce.py:375 零中心高斯为第 6 处同公式(输入形态不同,暂未合并,待 triage)
5. ✅ 已治理(2026-08-09,commit 85db12f):`paper` marker 落地——PUSB Table 2 两文件 + deep_pu_model_selection(纯本地验证),`-m paper` 收集 28 用例(原 0);pyproject 描述同步更新(commit 870f437)
6. ✅ 已治理(2026-08-09,commit ffdb6c1):class_prior 校验统一走 `check_scalar_in_range`(9 处);LDCE/KLDCE 由闭区间收紧为开区间(无调用方依赖)。**后续发现**:范围外仍有 ~11 处内联校验(metrics/losses/diagnostics/preprocessing 等),待 triage
7. ✅ 已治理(2026-08-09,commit 82718a4):check_test_quality 新增豁免复核段——每次运行打印豁免清单与理由,覆盖 ≥3/4 分类的文件提示可移出(真实运行:9 个豁免文件 7 个被标记可移出);exit-code 语义不变

### P2 — 文档与流程

8. ✅ 已治理(2026-08-09,commit 9c2283a):decision_log 补齐 5 条决策(08-06 至 08-09,审计当日估 ≥6,蒸馏定稿 5 条),并按先例将 `pu_workflow_design.md` 蒸馏后删除;check_doc_links 随即转绿
9. ✅ 已治理(本地,gitignored 不提交):CLAUDE.md:29 死链修复(旧路径 → `docs/dev/project_structure.md`)
10. ✅ 已治理(2026-08-09,commit 021e4b3):`project_structure.md` §3 测试树与实际 1:1 双向对齐(补 unit/scripts、workflow_scripts、benchmarks 8 文件,删空目录 registry,另补 6 处既有漂移)
11. ✅ 已治理(本地,gitignored 不提交):dev-workflow skill 状态速查更新(705 → 760,2026-08-09)
12. ✅ 已治理(2026-08-09,commit 13312dc):死代码清理 5 项删除(PNULoss、PenL1Estimator 别名、upu 假别名、DEFAULT_RANDOM_SEED、check_positive),set_global_seed 保留(conftest 真实使用);两处 method card 引用同 commit 修复
13. ✅ 已治理(2026-08-09,commit 3fc348b):`sample_weight` 语义文档化——DistPU/LBE 补"ignored"声明、SelfPU 补 NotImplementedError 说明(纯文档,无行为变更)
14. ✅ 已治理(2026-08-09,commit 870f437):nnpu.md §7.1 签名补 `class_prior`/`device`(与 nnpu.py 逐字符一致);architecture.md "15"→"17"+脚注;LDCE.md Connect-4 形状交叉说明(manifest 验证 67557×126);DGPU 族归属按 registry `Fam.DEEP_PU` 权威标注

### 第二批治理(2026-08-09,分支 fix/architecture-decay-batch2,14/14 闭环 + 2 项保留)

> 范围:第一批未列入行动项的全部 14 项发现(§3/§4 剩余黄项与两处"后续发现"),按"批次 A 单源化 → 批次 B 工程整洁 → 批次 C 局部性"14 任务实施,全部由既有测试锁定行为。commit 范围 9837cf5..3e629a0,共 17 个提交(2 计划文档 + 14 任务 + 1 修正 bf9fa10)。编号对应本报告 §3/§4 表格行。

1. ✅ 已治理(2026-08-09,commit 8e42b37):`list_algorithms` docstring 与代码一致——`trainable_only` 说明改为仅排除 `api_only`,移除已不存在的 `EXPERIMENTAL` 措辞(§3 S3 行 2)
2. ✅ 已治理(2026-08-09,commit 93a86e6):PU 零一风险单源化——`upu._pu_validation_risk` 删除,`pu_validation_risk` 委托 `metrics.pu_zero_one_risk`;`_sigmoid`/`_sigmoid_stable` 逐字双份提取为新 `utils/activations.py::sigmoid_stable`(行为逐位一致)(§4 T1 行 4)
3. ✅ 已治理(2026-08-09,commit 5c4bc27):`fit_evaluate`(275 行)按内聚段拆为私有 helper,`fit_evaluate` 缩短为编排层,行为零变化(§3 S2 行 1)
4. ✅ 已治理(2026-08-09,commit c9b0d18):新增 `tests/unit/losses/test_pnu_loss.py`——五个模块级函数数值锁定 + basic/param/edge/determ 四分类(§3 S2 行 3)
5. ✅ 已治理(2026-08-09,commit 40746af):`_canonical_hash` 5 份定义(4 个命名 def + pusb_official_data.py 内联 1 处)收敛到新 `benchmarks/_common.py::canonical_hash`,7 处调用点统一(§3 S2 行 4)
6. ✅ 已治理(2026-08-09,commit 1b13850):kldce/ldce 类先验推导与分母检查提取为 `estimators/risk/_class_prior.py::solve_prior_from_positive_fraction`(§3 S2 行 2 / §4 T1 行 3)
7. ✅ 已治理(2026-08-09,commit a6971d1):y_true 值域校验内联实现收敛到 `core/validation.py::validate_true_binary_labels`,6 处调用点统一(§4 T1 行 4)
8. ✅ 已治理(2026-08-09,commit da0a1b8):`.gitignore` benchmark 结果白名单 18 行压缩为 2 行(`!benchmarks/assigned_methods/results/` + `!benchmarks/assigned_methods/results/**`),忽略语义不变(§3 S4 行 2)
9. ✅ 已治理(2026-08-09,commit 1db1527):8 个 benchmark 测试文件 marker 单一来源——5 文件删逐函数 `unit`、3 文件补模块级 `pytestmark = [unit, paper]`,paper 覆盖完整(§3 S4 行 4)
10. ✅ 已治理(2026-08-09,commit 0699cc8):`n_features_out` 别名键删除,`pu_data_summary`/`pnu_data_summary` 仅 `n_features` 单键,v1.0.0 发布前完成避免破坏性变更(§4 T2 行 3)
11. ✅ 已治理(2026-08-09,commit 791ed73):check_test_quality 默认严格(`strict=True`,`--lenient` 显式退出),本地与 CI 行为对齐,摸底分类缺口文件补齐(§4 T3 行 4)
12. ✅ 已治理(2026-08-09,commit c2368bb):新增 `.github/workflows/nightly.yml`——slow 套件接入每周定时 CI(周一 03:23 UTC)+ workflow_dispatch(§4 T3 行 6)
13. ✅ 已治理(2026-08-09,commit c121cb6):RBF 第 6 处(上文 P1-4 后续发现的 kldce.py:375 零中心高斯)提取为 `utils/basis.py::rbf_weights`,6/6 处全数单源(§4 T1 行 1 后续)
14. ✅ 已治理(2026-08-09,commit 5059bd6 + bf9fa10):class_prior 范围校验剩余 ~11 处内联收敛到 `check_scalar_in_range`(metrics/losses/diagnostics/preprocessing/deep 等 10 文件,b89ae78;pipeline `_validate_prior_value` 另以独立提交 bf9fa10 收敛)(§4 T1 行 2 后续)

**保留说明(2 项,有意不改):**

- **pusb_kernel 存在性校验**:复核确认 `_validate_parameters` 已用 `check_scalar_in_range(class_prior, 0.0, 1.0, "class_prior", inclusive=False)`,与单源化目标一致,保持不动(§4 T1 行 2 的确认项)
- **losses/pnu.py 不内联校验**:纯函数公式模块(五个模块级函数),输入校验由 PNUClassifier 边界负责,模块 docstring 声明契约;本批只补测试(test_pnu_loss.py),不改校验位置

**提交清单**:`git log --oneline main..HEAD` —— 9837cf5(计划)、60cde85(任务重编号)、3ecdee9..3e629a0(14 任务 + bf9fa10 修正)。全部 17 个提交在分支 fix/architecture-decay-batch2。

## 6. 复跑指南

- **触发时机**:每发布一个 minor 版本后;或每引入 >5 个新文件时
- **复跑方法**:按本报告 §1 的四代理契约重新派发(代码/工程/文档/治理),重点复核本报告所有红项是否已治理、是否有新信号出现
- **代谢率检查**:治理腐朽的根本解是让"提取的机制被新代码遵循"——P1-6/7 与第二批单源化落地后,检查新代码是否用共享助手而非再次内联
- **第二批单源助手清单(2026-08-09,2026-08-13 更新)**:`canonical_hash`(utils/serialization.py,benchmarks/_common.py re-export)、`sigmoid_stable`(utils/activations.py)、`rbf_weights`(utils/basis.py)、`validate_true_binary_labels`(core/validation.py)、`solve_prior_from_positive_fraction`(estimators/risk/_class_prior.py)、`check_scalar_in_range`(core/validation.py)、`git_worktree_dirty`(benchmarks/_common.py);复跑时核对新代码是否复用、check_test_quality 是否保持严格默认、slow 套件是否随 nightly 自动执行

## 7. 复核记录(主上下文抽样)

| 发现 | 复核结果 |
|---|---|
| RBF 5 份实现(红) | ✅ 确认:5 处独立存在,公式数学一致但各自维护 |
| check_doc_links 三空洞(红) | ✅ 确认:47 行只匹配 .py;53-58 整体豁免 4 目录;61-65 白名单仅 3 文件 |
| check_math_rendering 假绿(黄→红) | ✅ 确认:84 行裸 glob 相对 cwd;B/D 双代理独立发现 |
| KLDCE γ 分支矛盾 | ✅ 确认:推导注释 604-606 与执行代码 629-637 矛盾,差 +2;数学结论待对照论文 |
| paper marker 空转 | ✅ 确认:tests/ 0 个 `pytest.mark.paper` |
| CLAUDE.md:29 死链 | ✅ 确认:CLAUDE.md 误写的旧路径 docs/project_structure.md 已迁移为 `docs/dev/project_structure.md` |
| project_structure.md 测试树漂移 | ✅ 确认:tests/unit/ 实际 12 目录,文档未收录 scripts/workflow_scripts |
| dev-workflow skill 数字过时 | ✅ 确认:SKILL.md:65 "705 passed(2026-08-08)" |
| nnpu.md §7.1 签名漂移 | ✅ 确认:卡签名缺 `class_prior`/`device` |

> 本文档为纯审计记录,不包含任何代码改动。行动项请按优先级排期治理,治理完成后更新本文档对应条目状态。
