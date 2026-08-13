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
| 决策迁移 | 2026-08-13 治理批次与行动项流水账已迁移至 [ADR-0001](../adr/0001-architecture-governance.md);本文档只保留发现快照 |

## 2. 总评

**判定:黄——健康系统,处于早期可逆腐朽阶段,代谢率仍高于腐朽率,但已出现两处"治理盲区已实际漏检"与一处"代码-推导自相矛盾",需在 v1.0.0 后的稳定期优先治理。**

- **做得好的**:17 个算法全部 NATIVE(0 个 api_only 假实现);registry 有防漂移测试锁死;PUSB benchmark 有 manifest sha256 锁定(真相分裂的最小化范例);文档-实现抽查 4 篇方法卡全部一致;双语 README 逐行同步(但源于近期人工治理而非结构性保障)。
- **要警惕的**:① 提取公共逻辑的机制健全(校验助手存在)但新代码不遵循——"机制存在"与"机制被执行"之间出现裂缝;② check_doc_links 声称覆盖实则排除最多引用密集的目录,漏检已实际发生(孤儿文档入库无人发现);③ 一处与自身推导注释矛盾的数学实现未被测试锁定。

> 批次治理记录(行动项、commit、闭环状态)已迁移至
> [ADR-0001](../adr/0001-architecture-governance.md);总评判定与
> 信号清单保留于此,作为下次复跑审计的对照基线。

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

已迁移至 [ADR-0001](../adr/0001-architecture-governance.md)(架构治理机制:
审计框架、单源助手、代谢率红线、复跑条件)。第一批 14 项与第二批 14 项均
已闭环,第三批 3 项已闭环(2026-08-13)。

## 6. 复跑指南

- **触发时机**:每发布一个 minor 版本后;或每引入 >5 个新文件时
- **复跑方法**:按本报告 §1 的四代理契约重新派发(代码/工程/文档/治理),重点复核本报告所有红项是否已治理、是否有新信号出现
- **代谢率检查**:治理腐朽的根本解是让"提取的机制被新代码遵循"——P1-6/7 与第二批单源化落地后,检查新代码是否用共享助手而非再次内联
- **单源助手清单**:以 `CONTRIBUTING.md` §5.1 为权威(消除清单第三份副本);
  复跑时核对新代码是否复用、check_test_quality 是否保持严格默认、
  slow 套件是否随 nightly 自动执行

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
