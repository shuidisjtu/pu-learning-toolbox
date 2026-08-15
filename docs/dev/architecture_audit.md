# 架构健康度审计报告

> 本文件为审计快照,保留「发现 → 治理 → 复核」对照基线;行动项流水账见 [ADR-0001](../adr/0001-architecture-governance.md)。
> 审计日:2026-08-09 | 快照更新:2026-08-16(治理复核)

## 1. 审计元信息

| 项 | 值 |
|---|---|
| 审计日 | 2026-08-09(仓库 HEAD c3501e9 + 审计 spec 5dab13a/6c97cb5) |
| 快照更新 | 2026-08-16:红项全部闭环复核;黄项 17 项中 11 项闭环,5 项部分修复,1 项原样 |
| 规模 | 171 个 .py;pu_toolbox ~15.8k 行;tests+benchmarks+scripts ~17.2k 行;docs 40 篇 ~9.8k 行 |
| 门禁基线 | 2026-08-09:738 passed;2026-08-15:907 passed,6 门禁全绿 |
| 决策迁移 | 治理批次与行动项流水账已迁移至 [ADR-0001](../adr/0001-architecture-governance.md);本文档只保留发现快照 |

## 2. 总评(2026-08-09 原判,作为对照基线)

**判定:黄——健康系统,处于早期可逆腐朽阶段,代谢率仍高于腐朽率,但已出现两处"治理盲区已实际漏检"与一处"代码-推导自相矛盾",需在 v1.0.0 后的稳定期优先治理。**

- **做得好的**:17 个算法全部 NATIVE(0 个 api_only 假实现);registry 有防漂移测试锁死;PUSB benchmark 有 manifest sha256 锁定(真相分裂的最小化范例);文档-实现抽查 4 篇方法卡全部一致;双语 README 逐行同步(但源于近期人工治理而非结构性保障)。
- **要警惕的**:① 提取公共逻辑的机制健全(校验助手存在)但新代码不遵循——"机制存在"与"机制被执行"之间出现裂缝;② check_doc_links 声称覆盖实则排除最多引用密集的目录,漏检已实际发生(孤儿文档入库无人发现);③ 一处与自身推导注释矛盾的数学实现未被测试锁定。

## 3. 治理复核(2026-08-16)

> 对 2026-08-09 审计全部红项与黄项的现状复核。证据来自 2026-08-15 全库 grep 核查,详见 `.superpowers/sdd/audit-yellow-items-check.md`(本地台账)。

### 3.1 红项 — 全部闭环 ✅

| 原发现 | 级别 | 闭环方式 |
|---|---|---|
| RBF 核公式 5 份独立实现 | 红(T1) | 单源化至 `utils/basis.py`(`build_rbf_basis`/`rbf_weights`),kldce 等改为调用 |
| check_doc_links 三空洞(research/project_management 整体豁免等) | 红(T3) | 豁免目录收缩为 `{"superpowers","figures"}`;research/ 由 check_math_rendering 单独覆盖 |
| check_math_rendering 静默假绿(相对 cwd 空扫) | 红(T3) | 锚定 `PROJECT_ROOT`(L21) |
| paper marker 空转(0 用例) | 黄→红(T3) | benchmarks 3 文件已用 `pytest.mark.paper` |
| KLDCE γ 分支代码-推导矛盾(差 +2) | 黄优先(S3) | 代码改为 `-1.0 - g` 单分支,与推导注释一致 |
| 决策日志停更(2026-08-06 后) | 黄(T3) | ADR 体系迁移闭环(ADR-0013/0014 等) |

### 3.2 黄项 — 治理闭环(11/17)✅

| 原发现 | 闭环方式 |
|---|---|
| PNULoss 死亡类 | losses/pnu.py 重构为模块级函数,被 estimators/risk/pnu.py 使用并有专属测试 |
| 校验助手 check_scalar_in_range 零调用 | 15 个调用点(PR #22 生效);check_positive 定义已删 |
| PenL1Estimator 别名零消费者 | 已删 |
| upu.py `_build_*` 私有别名虚构兼容 | 已重构为 _pack_theta/_unpack_theta |
| losses/pnu.py 无专属测试 | test_pnu_loss.py 已补,锁定双分支公式 |
| `_canonical_hash` 双份 | 单源至 serialization.py:32(PR #22) |
| list_algorithms docstring 与代码矛盾 | registry.py 明确 "Experimental entries are NOT filtered" |
| benchmark 测试模块级/逐函数 marker 双重声明 | 8 文件统一为模块级 pytestmark |
| DGPU 双重族归属 | method_selection.md 明确"族归属以 registry Fam.DEEP_PU 为准" |
| n_features/n_features_out 双键别名 | n_features_out 全库 0 处 |
| pipeline.py 单文件过大 | 输入、模型构造、CV/指标与报告组装拆入 5 个内部模块；主文件 1119→761 行，UI app 同步 624→404 行 |

### 3.3 剩余黄项 — 仍待治理(6 项)

| 项 | 现状 | 证据 | 建议 |
|---|---|---|---|
| S2 kldce/ldce 1−2ph 分母检查重复 | 推导已共享(solve_prior_from_positive_fraction),分母检查仍双份 | `risk/kldce.py:843-861` vs `risk/ldce.py:411-429` | 抽共享助手 |
| S4 sample_weight 七语义 | 剩 3 类:文档化忽略 6 家 / NotImplementedError(SelfPU) / 真实支持 6 家 | DistPU/LBE 静默忽略已文档化 | 统一为三档语义(支持/文档化忽略/NotImplementedError) |
| S4 .gitignore 逐目录例外 | 18 行→4 行(2 目录),模式仍在 | `.gitignore:59-77` | 统一忽略规则 |
| S4 check_test_quality 豁免名单 | 名单带理由 + review_exemptions() 提示,但仅 informational 不阻断 | `scripts/check_test_quality.py` | 明确退出机制 |
| T2 注册名 4 名字 | 类别名已删,registry 仍 4 名(注册名 + cpe/pen_l1/pe 别名) | `registry/builtin_methods.py:51` | 评估别名收敛或标注 deprecated |
| S1 set_global_seed / pu_validation_risk 测试专属 | 仍仅 conftest/测试使用 | `core/random.py:10`、`risk/upu.py` | 移至测试助手或标注测试专用 |

> 注:class_prior 范围校验 9 处内联已统一走 check_scalar_in_range(PR #22 覆盖),故不在剩余清单。

## 4. 复跑指南

- **触发时机**:每发布一个 minor 版本后;或每引入 >5 个新文件时
- **复跑方法**:按四代理契约重新派发(代码/工程/文档/治理),重点复核本报告 §3.3 剩余黄项是否治理、是否有新信号出现
- **代谢率检查**:治理腐朽的根本解是让"提取的机制被新代码遵循"——检查新代码是否用共享助手而非再次内联
- **单源助手清单**:以 `CONTRIBUTING.md` §5.1 为权威;复跑时核对新代码是否复用、check_test_quality 是否保持严格默认、slow 套件是否随 nightly 自动执行

## 5. 复核记录(2026-08-09 主上下文抽样,保留为历史证据)

| 发现 | 复核结果 |
|---|---|
| RBF 5 份实现(红) | ✅ 确认:5 处独立存在,公式数学一致但各自维护(2026-08-15 已单源化) |
| check_doc_links 三空洞(红) | ✅ 确认:47 行只匹配 .py;53-58 整体豁免 4 目录;61-65 白名单仅 3 文件(2026-08-15 已收窄) |
| check_math_rendering 假绿(黄→红) | ✅ 确认:84 行裸 glob 相对 cwd;B/D 双代理独立发现(2026-08-15 已锚定 PROJECT_ROOT) |
| KLDCE γ 分支矛盾 | ✅ 确认:推导注释 604-606 与执行代码 629-637 矛盾,差 +2(2026-08-15 已修) |
| paper marker 空转 | ✅ 确认:tests/ 0 个 `pytest.mark.paper`(2026-08-15 已落地) |
| CLAUDE.md:29 死链 | ✅ 确认:已迁移为 `docs/dev/project_structure.md` |
| project_structure.md 测试树漂移 | ✅ 确认:tests/unit/ 实际 12 目录,文档未收录 scripts/workflow_scripts(2026-08-15 已收录) |
| dev-workflow skill 数字过时 | ✅ 确认:SKILL.md:65 "705 passed(2026-08-08)"(2026-08-15 已改为动态收集) |
| nnpu.md §7.1 签名漂移 | ✅ 确认:卡签名缺 `class_prior`/`device`(2026-08-14 方法卡清洗已重构) |

> 本文档为纯审计记录,不包含任何代码改动。行动项与治理批次见
> [ADR-0001](../adr/0001-architecture-governance.md);本快照在每次复跑审计后更新。
