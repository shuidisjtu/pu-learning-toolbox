# 注释治理

> 本文档回答「源码注释该写什么、不该写什么、门禁如何执行」。
> 为什么这样取舍见 [ADR-0019](../adr/0019-comment-quality-gate.md)；
> 更广的架构腐朽判定与维护实践见 [`architecture_principles.md`](architecture_principles.md)。

## 1. 判断标准

注释的价值不取决于数量，而取决于它是否表达了**代码本身无法表达**的信息。
代码优先通过命名、拆分、常量和低复杂度控制流做到自说明；能由代码直接表达的，
不用注释复述。

保留的注释分三类：

| 类别 | 说明 |
|---|---|
| 目的性 | 这段代码为何存在、服务于什么目标 |
| 概述性 | 为长函数或模块交代整体结构 |
| 不可由代码表达的信息 | 约束、取舍、来源、风险、数值稳定性、不变量 |

不以注释承载的内容按权威位置分流：

| 信息类型 | 权威位置 |
|---|---|
| 模块分层、依赖方向、数据流 | [`architecture.md`](architecture.md) |
| 文件与目录职责 | [`project_structure.md`](project_structure.md) |
| 公共 API 签名与语义 | [`api.md`](../user/reference/api.md) |
| 论文公式、来源与复现状态 | [research/method_cards/](../research/method_cards/) |
| 设计取舍与替代方案 | [docs/adr/](../adr/README.md) |

## 2. 机械门禁

| 门禁 | 检查 | 默认级别 |
|---|---|---|
| `scripts/check_comment_quality.py` | 遗留标记、裸 TBD、行尾注释 | error / error / advisory |
| `scripts/check_doc_links.py`（Rule 1） | 反引号路径引用是否存在 | error |

`check_comment_quality.py` 另有一条 `scan-error`：无法解析的 Python 文件（语法或词法
错误）会产生 error 级 finding，而不是静默跳过。

### 2.1 遗留标记

未完成工作统一写 `TBD`，不使用 `TODO`、`FIXME`、`XXX`、`HACK`。
判定为词边界匹配、大小写不敏感。

### 2.2 裸 TBD

`TBD` 之后必须出现 `:` 或 `(`（中间允许空白），即 `TBD: ...`、`TBD : ...`
或 `TBD(#123) ...`。

该规则只是「必须带可追踪上下文」的**语法代理**，证明不了上下文真的可追踪——
上下文是否充分仍是评审责任。`TBD` 大小写敏感：小写 `tbd` 视为普通散文，
不作为标记认领。

### 2.3 行尾注释

行尾注释默认不鼓励，仅以下工具链指令豁免：`noqa`、`pragma:`、`type:`、
`fmt:`、`ruff:`、`isort:`。其余行尾注释产生 advisory，不阻断。
按目录迁移时用 `--strict-inline` 局部提升为阻断。

### 2.4 引用存在性

以反引号包裹、以根目录白名单（`pu_toolbox`、`tests`、`scripts`、`examples`、
`docs`、`external`）之一开头、以 `.py` 或 `.md` 结尾的路径必须存在。白名单之外的
顶层目录（如 `benchmarks`、`.github`）不在检查范围内。

源文件按**整文件文本**扫描，因此普通字符串字面量里的反引号路径同样会被检查，
不只是注释与 docstring——这是该规则已知的暴露面（当前仓库 0 例触发）。

被门禁指出时按情况处理：修正路径，或去掉反引号——后者适用于该处并非对仓库
文件作出声明的情形。

两处刻意跳过：

- 含 `{`、`}`、`*`、`?` 的简写 token 不是单文件声明；
- `tests/` 不在扫描语料内——该目录中的失效路径是本门禁的负向测试夹具。

## 3. 已裁决的保留项

以下注释**刻意保留行尾形式**：它们表达的信息无法由命名或类型承载，且短注解贴行比
块前更易逐行对照。清理存量时不要把它们当冗余删除——删掉即丢失该信息。

| 位置 | 保留内容 | 为何不能由代码表达 |
|---|---|---|
| `pu_toolbox/core/config.py` | 两个 clip eps 的区间 | 区间端点与开闭；两者刻意一开一半开（`(0, 1)` 与 `(0, 1]`） |
| `pu_toolbox/core/config.py` | `MAX_PU_RATIO` 的 warn 语义 | 严重性：`validation.py` 只 `warnings.warn`，不阻断 |
| `pu_toolbox/core/config.py` | `MIN_POSITIVE_SAMPLES` 的 raise 语义 | 严重性：`validation.py` 直接 `raise`，与上一行形成对照 |
| `pu_toolbox/core/tags.py` | `SCAR`、`SAR` 的概率定义式 | 公式即定义；删除后该值在全仓库无处可寻 |
| `pu_toolbox/core/tags.py` | `TrainingCost` 四档的判据与数值界 | 判据（闭式 / 有界迭代 / 固定轮次非凸 / 未分类中性计分）与 `~1000 epochs` 界 |
| `pu_toolbox/core/tags.py` | `API_ONLY`、`NATIVE` 的语义 | 前者是「无训练逻辑」的实现状态契约，后者是来源与法务语义 |
| `pu_toolbox/losses/llsvm.py` | 13 处 shape 注解 | 签名用裸 `np.ndarray`，不携带形状；本库未引入 shape 类型标注 |

判定为**系统性风格问题、另议**（不在注释清理批次内处理）：

- `pu_toolbox/core/tags.py` 的 9 条 `# ── X ──` 分段标题与紧随其后的类名重复
  （9 条中 4 条逐字相同，其余仅大小写或空格有别），按 §1 标准应删；但该风格在本库
  源码中普遍使用、跨多个模块，只在单文件删除会造成文件间不一致，属全库风格决策。

## 4. 迁移策略

1. 先让门禁可失败，再清理存量；
2. 存量行尾注释按模块分批处理，每次一个模块，用 `--strict-inline` 锁定已迁移目录；
3. 删除注释前先判定类别：可由代码表达的直接删，其余改写为块前意图注释；
4. 任何注释改动不得改变代码行为、随机序列、序列化字段或公共 API；
5. 门禁误报必须有最小复现测试，修门禁而不加永久豁免。

## 5. 语言

- 源码注释使用英文；
- 面向用户与维护者的 Markdown 使用中文；
- 公式、协议名、类名、字段名与文档路径保持原始英文标识。
