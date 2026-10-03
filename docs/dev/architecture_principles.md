# 架构维护原则

> 本文档回答"如何防止架构腐朽"：腐朽的信号是什么、根源是什么、用什么手段对抗。
> "当前架构为什么这样组织"是设计视角，见 [`architecture.md`](architecture.md)；
> 治理机制（审计触发、单源助手机制、代谢率红线）的决策记录见
> [ADR-0001](../adr/0001-architecture-governance.md)。本文档只余原则，不重复二者。

## 1. 架构的定义与价值

架构 = **被持续执行的决策** × **能被推理追溯的理由链**。

好的架构不是图纸，它回答：

- **什么不能做**——如层间不允许反向依赖（[architecture.md §2.1](architecture.md)）、
  分类器必须满足 fit/predict/decision_function 契约（[api.md](../user/reference/api.md)）；
- **在当前项目情况下，为什么不能做**——每条限制的取舍在 [ADR 体系](../adr/README.md)。

若理由链丢失，后人只能照抄"是什么 / 怎么做"，无法判断哪些可以演进。本项目的文档体系
为此分工：`architecture.md` 记为什么这样组织、`api.md` 记契约、方法卡记论文出处、ADR 记取舍。

## 2. 腐朽的根源

程序不只是代码的集合，还包括维护者对已有代码的构思：

- 为什么采用这种结构；
- 哪些模块与字段可改、哪些是核心依赖不可改；
- 哪些关键约束没有写进接口文档、却实际起着作用。

单靠文档无法根治：文档能记录结论，却记录不了判断力——即"为什么这样简化是对的"。
因此对抗腐朽的主要手段是**机械门禁 + 评审惯例**（§4），文档是它们的载体而不是替代品。

## 3. 腐朽信号与判定框架

判定框架（四信号三表现）由 [ADR-0001](../adr/0001-architecture-governance.md) 固化；
本节给出每类信号的**应手**——项目里已有的对抗机制。

### 3.1 先兆信号

| 信号 | 含义 | 应手 |
|---|---|---|
| 删除风险 | 删除任何东西的风险大于保留其成本，导致系统无法删除任何内容 | 变更以"消除"闭环：死代码即删（历史治理已删 PNULoss 死亡类、零消费者别名）、单源助手收敛 |
| 局部性丧失 | 改动的影响边界不可推理，一切改动默认全量回归 | 层间单向依赖（architecture.md §2.1）+ 注册表元数据驱动，改动面可沿依赖链推理 |
| 承重 bug | 某处行为错误、但下游依赖了该错误，bug 变成契约（甚至被写进文档） | 契约测试锁死行为；发现后按"修复 + 测试锁定"处理，不将就 |
| 疤痕组织 | 只增不减，用新分支 / 字段掩盖已废弃的僵尸 | 评审查同一概念第 3 次分裂；过时名走 `deprecated_aliases` 而非无限保留 |

### 3.2 表现特征

| 表现 | 应手 |
|---|---|
| 真相分裂 | 真相源分工：文件级职能 → [project_structure.md](project_structure.md)；API 契约 → [api.md](../user/reference/api.md)；公式/源码状态 → 方法卡；实现状态 → `registry/builtin_methods.py`；字段语义 → `core/tags.py` |
| 概念膨胀 | 命名治理：过宽注册名走 `deprecated_aliases` + `FutureWarning`；新名词先在契约中定义再使用 |
| 治理检验体系腐朽 | 永远通过 / 频繁误报的检查无价值：check_math_rendering 曾假绿（已锚定 PROJECT_ROOT）、check_test_quality 豁免计入非零退出码——门禁必须"能失败" |

## 4. 维护实践清单

1. **宪法级原则**：可维护性优先于短期开发速度；对外接口必须有文档与测试；新增代码
   必须复用单源助手（[CONTRIBUTING.md](../../CONTRIBUTING.md) §5.1），违反则触犯红/黄线。
2. **决策显性化**：有取舍即写 ADR，不事后批量补记；机械性结构用生成器
   （`generate_structure.py`）维护而非手绘。C4 类图工具不采用——本项目约定为
   文本依赖关系表（architecture.md §2.1）。
3. **低耦合高内聚**：一个模块一个职责、层间单向；新增模块着陆时以此为自检。
4. **持续重构，小步快跑**：重构前必须有覆盖测试；一次只解决一个问题。
5. **测试分层**：unit/math/property/contract/paper/gpu marker 金字塔
   （[project_structure.md §3](project_structure.md)），底层快、上层慢。
6. **机械门禁**：`check_doc_links`、`generate_structure --check`、`check_math_rendering`、
   `check_baseline_configs`、`check_skill_sync`、`check_format`——机械性一致性交给门禁，
   不靠人工核对。
7. **进化式架构**：能力以"字段 + 契约"渐进引入（双架构 4 能力字段、`_SYNC_FIELDS`
   镜像、provenance），避免一次性大重构。
8. **稳定模块标识**：长期无提交的模块视为稳定，不迁移不重构（YAGNI）。
   Python 源码分发下不做二进制物理隔离。

## 5. 审计历史

历次审计的完整发现与治理批次见 [ADR-0001](../adr/0001-architecture-governance.md)
（背景摘要）与 git 历史（各版快照正文）。

| 日期 | 范围 | 判定 |
|---|---|---|
| 2026-08-09 | 全库四代理审计（171 个 .py、40 篇文档） | 黄——健康系统、早期可逆腐朽；红项 6、黄项 17 |
| 2026-08-16 | 治理复核 | 红项全部闭环；黄项 17/17 闭环 |
| 2026-10-03 | 注释治理批次 A/B：门禁落地与三文件注释裁决 | 绿——接入时无遗留标记；三文件 25 条行尾注释中纯重复 1 条、同式重复 1 处，其余经裁决保留 |
| 2026-10-03 | 测试层批次 C：共享工厂提取 + Iris 冒烟回归 | 绿——工厂表收敛为单源；新增不依赖 torch 的拟合级冒烟层，覆盖全部 native 分类器；层内失败按「代码错误 / 输入不支持 / 依赖缺失」显式分档，无静默豁免 |
| 2026-10-03 | 审计批次 D：单源地图与依赖边界 | 绿——九个概念的真相源已建档（含四类重复判定与不可合并理由）；下层对实验层的依赖经静态与动态双路径确认为零，并由新增 ratchet 全层守住（此前守卫只覆盖单个模块） |
| 2026-10-03 | 迁移批次 E1：训练视图词表单源收敛 | 绿——角色名与 os/ts 视图收敛到 `core/training_views.py` 的 `Literal` 单一声明，运行时元组由 `get_args` 派生，两处报错文本逐字节不变；`ImageRole` 经证据判为同一概念并别名化。共享 token 但不同概念者——`SamplingAssumption`、`LEGAL_RUN_VIEWS`、`run_view` 祖传拼写，以及两处角色子集——明确保留并记明理由 |
| 2026-10-03 | 迁移批次 E2：JSON 标量序列化单源收敛 | 绿——八处调用表达式（含审计漏记的两处内联写法）收敛到 `utils/serialization.py` 的 `json_scalars`；六个摘要/缓存键值逐字节不变。**兼容性变更**：`prepare_survey_dataset` 的 `source_indices`/`test_indices` 现要求 JSON 标量，此前可哈希的 `tuple` 元素会被 `json.dumps` 静默写成数组而通过；三个生产调用点（均在 `scripts/prepare_survey_splits.py`）不传该参数，全仓库 12 处调用（3 生产 + 9 测试）亦无一传入，故无实际消费者受影响。**知情保留**两处同类转换：`experiment/strategies.py` 的 `label_view_sha256`（载荷是标签视图，`.astype(int)` 已保证整数）与 `diagnostics/uncertainty.py` 的报告载荷（输出侧）。行尾注释存量 144 条仍为 advisory，分区 strict 未启用 |
| 2026-10-03 | 迁移批次 E3a：JSON 摘要单源收敛 | 绿——JSON 摘要族 7 个实现体收敛到 `utils/serialization.py` 的两具名函数 `strict_canonical_hash`（严格 `allow_nan=False`，制品身份）与 `canonical_hash`（宽容，报告与清单载荷），二者只差 `allow_nan`、非有限输入上分道扬镳，不合并（同 E2 对 `json_safe`/`json_scalars` 的裁定）。**兼容性**：两个公开名 `survey_protocol.digest` 与 `survey_comparison.comparison_digest` 保留名字与签名、只改委托，`scripts/` 与既有测试零改动；删除三处私有实现体（`feature_adapter`/`image` 各自的 `_json_sha256`、`training_views` 的 `indices_sha256` 内联）与一个私有包装（`survey_comparison._survey_digest`，经项目所有者裁断）。**知情保留**第三种语义：`experiment/text.py:206` 的 `_json_sha256` 一字未动——多 `ensure_ascii=False`，含非 ASCII 码点的语料上摘要与上两者不同，且其产物 `cache_key` 直接是缓存文件名，改它会使已记录的 `texts_sha256` 与本地文本缓存失效。数组族（两份 `_array_sha256`，哈希 dtype/shape/bytes）与文件族（`text.py._file_sha256`，哈希文件字节）非 JSON 规范形式，留给批次 E3b。**证据**：15 组分割清单重算逐字节一致，1 个适配器缓存经严格与宽容两路复现同一值，1 次单元运行的 11 个摘要字段逐字节不变，PA/OA 控制点与治理前记录一致 |

复跑触发：每发布一个 minor 版本后，或新增 >5 个文件时（ADR-0001）。
