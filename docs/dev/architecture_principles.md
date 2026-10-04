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
| 2026-10-03 | 迁移批次 E3a：JSON 摘要单源收敛 | 绿——JSON 摘要族 7 个实现体：6 个收敛到 `utils/serialization.py` 的两具名函数 `strict_canonical_hash`（严格 `allow_nan=False`，制品身份）与 `canonical_hash`（宽容，报告与清单载荷），二者只差 `allow_nan`、非有限输入上分道扬镳，不合并（同 E2 对 `json_safe`/`json_scalars` 的裁定）。**兼容性**：两个公开名 `survey_protocol.digest` 与 `survey_comparison.comparison_digest` 保留名字与签名、只改委托，`scripts/` 与既有测试零改动；删除两处私有实现体（`feature_adapter`/`image` 各自的 `_json_sha256`）、把 `training_views` 的 `indices_sha256` 内联改调 `canonical_hash`，另删一个私有包装（`survey_comparison._survey_digest`，经项目所有者裁断）。**知情保留**第三种语义：`experiment/text.py:206` 的 `_json_sha256` 一字未动——多 `ensure_ascii=False`，含非 ASCII 码点的语料上摘要与上两者不同，且其产物 `cache_key` 直接是缓存文件名，改它会使已记录的 `texts_sha256` 与本地文本缓存失效。数组族（两份 `_array_sha256`，哈希 dtype/shape/bytes）与文件族（三处逐块哈希文件字节：`text.py._file_sha256`、`checkpoints.py._file_digest`、`split_archive.py.file_sha256`）非 JSON 规范形式，留给批次 E3b。**证据**：六处改道的替换式与被删函数体逐字同参（`allow_nan` 只在原处有处保留），故对任意输入等价；经验侧有 1 个适配器缓存经严格与宽容两路复现同一值、1 次单元运行的 11 个摘要字段逐字节不变、PA/OA 控制点与治理前记录一致；15 组分割清单重算一致锚住的是共享的宽容配方（那 15 处走 `datasets.py`，本批未改），属制品稳定性证据而非收敛证据 |
| 2026-10-04 | 迁移批次 E3b：数组与文件摘要单源收敛 | 绿——两组二进制内容摘要各收敛到一个具名函数：Arr-1 三处（`feature_adapter._array_sha256`、`image._array_sha256` 删除，`survey_protocol.array_digest` 委托）与 File-1 三处（`checkpoints._file_digest`、`text._file_sha256` 删除，`split_archive.file_sha256` 委托）→ `utils/serialization.py` 的 `array_hash` / `file_hash`；E3a 的 JSON 摘要族一字未动。**0-d 裁定**：三个数组实现**并非等价**——`survey_protocol.array_digest` 先从 `np.ascontiguousarray(value)` 的返回值读 shape，而该函数把 0-d 输入提升为 shape `(1,)`（`ndmin=1`），shape 串由 `[]` 变 `[1]`，与两处从原数组读 shape 的 `_array_sha256` 分叉；裁定为**取调用方原数组的 shape**（即 `array_hash` 的语义），理由是其背后已落盘制品更多。无任何冻结制品覆盖 0-d 输入，守卫只有 `tests/unit/utils/test_content_hashes.py` 的两条合成用例——helper 与**公开入口**各一条（入口那条是收口时补的：首轮只守 helper，把入口改回旧配方全套测试仍绿，评审以内存探针证实）。**兼容性**：两个公开名 `survey_protocol.array_digest` 与 `split_archive.file_sha256` 保留名字与签名、只改委托，`scripts/` 与既有测试零改动（`test_split_archive.py:198` 对模块全局名的 monkeypatch 仍生效）；删除四处私有体。**同批两处结构缺陷**：`feature_adapter._encoder_state_sha256` 去下划线改名 `encoder_state_sha256`（行为零改变，`survey_execution` 由函数内私有导入改模块级引用），`survey_protocol` 删一个只能重失败的回退分支（其内层重算与已被比较过的路径同一配方）。**证据**：15 组分割清单的 `manifest_sha256` 对照**已入库**索引（`produced_by_commit = 0d1335ce0dfba8afe5eedae315eba47c2e0acb5e`，治理前）逐字节一致，文本缓存 `cache_sha256` 对照本地缓存元数据一致，适配器 `feature_sha256` ×4 对照其 `adapter.json` 一致（20/20，`computed` 行改前改后 `VALUES IDENTICAL`）；两个公开名经循环断言确证转发（`file_sha256(path) == file_hash(path)`、`array_digest(x) == array_hash(x)`）；一次单元运行的 11 个摘要字段逐字节不变，PA accuracy `0.8686210640608035` / auc `0.9245534524126899` 与治理前记录一致。**覆盖边界（如实）**：六个被收敛实现体中有**两处调用点未被观测**——`checkpoints._file_digest` 的**调用点**未被冻结脚本或单元运行触及，`image._array_sha256` 的产物 `backbone_manifest.train_data_sha256`（记录值 `a3808025abf28e9faab49859742da60ef429fc4c8151e7382f3b8d75db3f188f`）在任何地方都**未与记录值比对**（冻结脚本核的是 `adapter.json` 的 `feature_sha256`，那是姊妹函数 `feature_adapter._array_sha256` 的产物；单元运行是 spambase/`native_2d`，从不调用 `fit_survey_image_preprocessing`；仅有的两处测试 `tests/unit/experiment/test_image.py:59`、`tests/unit/experiment/test_prepare_survey_splits.py:130` 只断言确定性与长度），「值未变」对这两处靠的都是**代码同一性**而非观测：`_file_digest` 的函数体与 `split_archive.file_sha256`、`text._file_sha256` 逐字相同，后两者在 15 份分割清单 + 1 份文本缓存共 16 个真实文件上被验证；`image._array_sha256` 与 `feature_adapter._array_sha256` 逐字相同，后者产物 `feature_sha256` ×4 在 4 份适配器上与 `adapter.json` 比对一致——均非被观测所得；`adapter.json` 与文本缓存元数据是 `data/` 下 gitignored 的本地缓存——三者中只有**已入库的 `split_artifacts_index.json`**（15/15）能在新克隆上复现。**登记为开口**：`benchmarks/` 的 13 处 `hashlib.sha256(`（横跨 10 个模块）与 `ui/app.py:46` 未收敛，前者产物是基准配置身份（收敛会改写已记录值，且不在本批两道 ratchet 的语料内），后者入参是 Streamlit 上传文件的内存字节块而非文件路径（与 `file_hash` 签名不同族），理由详见单源图 |
| 2026-10-04 | 迁移批次 E3b 收口：评审两项 P2 与一项 P3 | 绿——评审给「有条件通过，尚不能最终收口」，三项全部采纳。**P2-1 贡献者约束缺口**：`array_hash` / `file_hash` 已在单源图建档、被实验层多处复用，却不在 `CONTRIBUTING.md` §5.1 的「必须复用」表内——单源图是治理记录，替代不了面向贡献者的规则。补两行（位置、用途、选择边界），写明两者是**两个正交概念不是带开关的一个**（输入类型 / 编码 / 分块语义皆不同，不得合并）、既有公开名是委托包装、新增消费者直接复用不得内联复制，并把 `benchmarks/` 13 处登记为开口。**P2-2 公开入口无回归守卫**：0-d 裁定的守卫原本只覆盖 helper `array_hash`，而 `array_digest` 恰是三个旧实现里唯一取连续化副本的那个——评审以仅内存的回退探针证明「把入口改回旧配方，相关 41 条测试仍全绿」。补 `test_param_the_public_entry_point_carries_the_same_zero_d_ruling`：直接调公开名、钉住裁定摘要，并复现入口改前的配方（`4a95a9fb…d03`，与治理前记录的前后缀一致）作阴性对照；双向验证——改入口、或改它委托的 helper，测试均变红，其余输入下仍绿。**P3 注释解释有误**：测试注释称 `json.dumps(shape)` 与 `str(list(shape))` 不一致，实际两者都渲染为 `[3, 4]`；真正不能丢的是 JSON 的 `", "` 分隔符（会被 `replace(" ", "")` 一类改写抹掉）。摘要配方未变，只改解释。**顺带修一处既有漂移**：`docs/adr/0001-architecture-governance.md` 的决策 2 写「现 8 项」并逐个列举单源助手，而 §5.1 已有 12 行（E2 的 `json_scalars`、E3a 的 `strict_canonical_hash`、E3b 的 `array_hash`/`file_hash` 与 `stable_centroid_denominator` 均未计入）——该 ADR 本就声明「清单维护于 §5.1」，枚举属重复，已删枚举、只留指针与「不再复制」的说明。本地 tag `batch-e3b-closed` |

| 2026-10-04 | 迁移批次 E4：方法能力字段死写收敛 | 绿——批次 D 把「同一事实写两处」登记为**重复（可收敛）**，并给出「补一条断言」的收敛前提。**本批勘察推翻审计的两条前提**：① 那条断言**早已存在**——`tests/test_builtin_methods.py` 的事故守卫自 `4d5eebe`（PR #12，远早于审计）起就在注册前快照字面量、与类值比对，其 docstring 记着同一类事故（`upu` 的字面量写 `False`、类写 `True`，瞒了数月），故「只改字面量会被静默忽略」不成立；② 审计只点了 2 个字段，实际同步面是 **8 个条目字段**（`family`/`assumption`/`scenario`/`requires_class_prior`/`implementation_status`/`source_status`/`backend`/`maturity`）。**真正的缺口在守卫的反面**：那条守卫对「类未声明的字段」直接跳过，于是它守的恰是 182 处**死写**（写错也无害），跳过的是 10 处**活写**（字面量即权威）。10 处里 **7 处此前无任何守卫**（全部落在 `class_prior_estimation`），另 3 处已有守卫（cpe 的 `implementation_status` 由 `test_basic_implementation_status_distribution` 泛覆盖；`pusb`/`lbe` 的 `requires_class_prior` 由台账契约测试 `test_prior_semantics_consistent_with_registry_class_prior` 守）——准确口径是「cpe 的 8 处活写里 7 处无守卫」，评审以变异实验实证（同时改坏 7 个后全量 `-m "not slow"` 2592 passed / 0 failed；单独改坏 `implementation_status` 则变红）。**收敛动作**：删除 **182** 处死写（= 23 条目 × 8 字段 − 2），保留 **10** 处活写；事故守卫原地重写为**双向契约**（clause 1：类已声明的字段不得在字面量里重复出现；clause 2：类未声明的字段必须显式声明，并逐个钉住 `class_prior_estimation` 的 8 个值）——因删除后 `getattr` 仍返回 dataclass 默认值、属性访问分不出「省略」与「声明」，守卫改为**解析源码 AST**。**同批修的既有缺陷**：`experiment/method_ledger.json`（**已入库**）里 46 处 `registry:NNN` 行号引用**已全部腐化**——对照引入它的 `524c2b5`，当时多数指向正确（`upu`/`nnpu`/`dist_pu` 逐字段命中，`kldce` 引入时即 off-by-one），此后随文件增长漂移，**今日无一**仍指原字段，而没有任何门禁校验它们（同文件被校验的是备注头部枚举，不是括号里的锚）；本批删 182 行会让它们再错一次，故改为方法名式（`registry:upu`）并加防回潮测试 `test_basic_does_not_cite_source_line_numbers`。**证据**：另开 worktree 在 BASE 上跑**同一个**冻结脚本与 HEAD 对拍，24 法 metadata + `trainable_only` + `all` 共 26 行 **VALUES IDENTICAL**；源码 AST 普查 written `192→10`、dead `182→0`、live `10` 前后一致；三条阴性对照（死写回潮 / 活写缺失 / 活写改值）均实跑变红并还原；11 道门禁 exit 0；消费者侧（`tests/unit/{cli,advisor,ui,workflows}` + `tests/contract`）306 passed。**已知边界（如实登记，未解决）**：clause 1 无条件禁止**任何**条目（含未来的 `api_only`）写那 5 个仅类字段（4 个架构能力字段与 `label_semantics`），故 `api_only` 条目只能取 dataclass 默认值；今日 24 个条目全部已绑定、`api_only` 为 **0**，故该规则**空转**，clause 2 与 `api_only` 路径**均无真实样本覆盖**。**本批计划缺陷（实施方与评审各自发现）**：Task 3 断言初稿写成 `==`（两条款互为 XOR，须 `!=`）——照抄会在 182 处**正确**状态上变红、又对死写回潮静默通过；冻结脚本初稿用 `json.dumps(default=str)`，多元素 frozenset 的 repr 受 `PYTHONHASHSEED` 影响（同一 checkout 跑两次也可能 DIFFER），且其 `dead_total` **只按类声明算**、改前改后恒为 182，**不能**证明死写清零——已改为读源码 AST 自证。**命名约束（备案）**：新防回潮测试名刻意避开 `ledger`，因 `scripts/check_test_quality.py` 的类别关键词是**子串匹配**、`edge` 是 `ledger` 的子串，用含 `ledger` 的名字会把这文件「无边界用例」的诚实声明翻成「已覆盖」；实施方曾依门禁要求删掉该声明，被控制方否决、改为改名绕开（`test_method_ledger.py` 仍带着同一处门禁误报，属独立的小缺陷，本批未动），并把该约束写进测试文件注释 |

复跑触发：每发布一个 minor 版本后，或新增 >5 个文件时（ADR-0001）。
