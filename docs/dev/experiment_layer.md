# 实验层（experiment）设计文档

> 对应需求：PU 调研实验协议 [pu_survey_protocol.md](../research/pu_survey/pu_survey_protocol.md)
> §2.4（四份数据角色与 PA/OA 双协议）；执行状态见
> [survey_execution_plan.md](../research/pu_survey/survey_execution_plan.md) 任务分工表。

实验层是面向研究者/实验用户的公共 API 层，补齐 `PUPipeline` 缺失的「数据角色声明 + 显式模型选择协议」。
本文档只记本层**独有的设计机制**（D1–D5 各决策如何实现、为何这样设计、如何扩展使用）。
分层位置与数据流见 [architecture.md](architecture.md) §Experiment 与 §3.1；
公共 API 签名与参数契约见 [api.md](../user/reference/api.md) §实验层；
决策背景与备选方案见 [ADR-0018](../adr/0018-experiment-layer-public-api.md)；
模块文件清单见 [project_structure.md](project_structure.md) 目录树（`experiment/` 块）。

## 1. 关键设计机制

### D1 编排 vs 策略 —— 可注入策略接口的实现与使用

> 本节三个软件工程术语，先各说一句直觉含义：**可注入策略**＝把「生成数据 / 训练模型 / 选模」三道
> 会随实验变化的工序做成可插拔零件，runner 是固定流水线，换零件不改流水线
> （[Strategy 模式](https://refactoring.guru/design-patterns/strategy)）；**策略 ABC（抽象基类）**＝
> Python 里定义「接口契约」的方式——不写实现逻辑，只规定「要当这个零件，必须会哪个动作、声明哪个属性」
> （[abc 官方文档](https://docs.python.org/3/library/abc.html)）；**DIY 注入**＝照某个 ABC 写一个子类、
> 实现它的抽象方法，作为参数传给 runner（不继承 runner、不改 runner）。

**决策**：`ExperimentRunner` 固定编排骨架（[Template Method](https://refactoring.guru/design-patterns/template-method)，固定骨架、子步骤可替换）；数据生成、训练、选模为**可注入策略**，
各自一个策略 ABC（`Generator`/`Trainer`/`SelectionProtocol`）。不采用 [Bridge](https://refactoring.guru/design-patterns/bridge) 双层次（双轴继承，变化点尚少、过度设计）——变化点各自成轴，
研究者 DIY = 实现策略并注入，不继承 runner（ADR-0018 决策 3）。

**实现方式**：

- **骨架固定，注入点只有三处**。[`runner.py`](../../pu_toolbox/experiment/runner.py) 只暴露 `fit()`、
  不继承任何基类；`fit(model, train, pu_val, clean_val, test)` 按固定顺序编排——校验 → 生成 PU 视图 →
  候选训练 → PA/OA 离线选模 → 独立 test 评测 → 留痕，其中只有 `generate`、`fit`、`select` 三个动作是
  策略注入点。校验、执行顺序、候选重试策略、test 评测口径与 manifest 契约固定不可注入。
- **策略 ABC 是契约载体，不是运行时门禁**。[`protocols.py`](../../pu_toolbox/experiment/protocols.py)
  为三个策略各定义**一个抽象方法 + 一个声明属性**：

  | 策略 | 抽象方法 | 声明属性 |
  |---|---|---|
  | `Generator` | `generate(X, y_true, c, seed) -> (y_view, meta)` | `output_view = "pu"` |
  | `Trainer` | `fit(estimator, X, y, *, class_prior, val_pu, os_or_ts) -> RunTrajectory` | `trains_on_real_labels = False` |
  | `SelectionProtocol` | `select(trajectories, val_part, threshold_candidates, *, class_prior) -> SelectionArtifact` | —（仅 `name` 约定） |

  runner 用 **duck-typing**（[鸭子类型](https://docs.python.org/3/glossary.html#term-duck-typing)：只看对象行为、不强制继承）读取接口，全仓库对策略**没有一处** `isinstance(x, Generator/Trainer/
  SelectionProtocol)` 门禁；两处 `isinstance` 只做路由（`ProtocolOA` 才拿 `clean_val`、`ProtocolPA`
  才要求 π）。读取方式：`getattr` 读声明属性（取 fail-closed 默认值）、`inspect.signature` 探测
  `select` 是否接受 `class_prior`（绝不靠捕获 `TypeError`）、精确 `type()` 判断是否套逐 epoch
  checkpoint 捕获。ABC 的职责是**固化契约与文档**，使自定义策略无需继承特定类即可注入。
- **声明属性驱动一致性 fail-loud**。runner 在 `fit` 内校验：`output_view` 取值合法（拼写 `"Clean"`
  即 raise）、`output_view="clean"` 却声明 `mechanism="pn_oracle"` 不自洽即 raise、clean 视图与
  `trains_on_real_labels`/`class_prior` 的兼容性、`model.epoch_components` 与标签语义声明。默认值
  一律 fail-closed（如 `trains_on_real_labels` 默认 `False`），防止静默错配训练。

**为什么这样设计**：生成/训练/选模三个变化点各自成轴、互不交织，用「一个方法 + 一个声明属性」的
最小策略面即可 DIY；避免 Bridge 双层继承的过度设计（YAGNI，见 ADR-0018 备选方案）；runner 不继承、
策略不继承，两侧独立演化；零改动现有 `PUPipeline` 与分类器 `fit(X, y)` 契约。

**如何使用（DIY 扩展）**：实现一个策略 ABC 的抽象方法 + 声明属性，作为**实例**注入（非类；trainer
经 `config["trainer"]` 传入，runner 显式拒绝传类）。硬性契约：
- 生成器：`generate` 返回 `(y_view, meta)`，meta 建议含 `mechanism`（runner 读它做自洽校验），并声明 `output_view`。
- 训练器：`fit` 必须返回 `RunTrajectory`（checkpoint 覆盖逐 epoch × component 完整、指标有限、
  `decision_function` 在 pu_val 上返回同形有限分数）。
- 选模协议：`select` 返回 `SelectionArtifact`，其 `epoch` 必须等于所选 checkpoint 的 `epoch_position`。

**半注入边界**：是否套逐 epoch checkpoint 捕获由 `type(trainer) in (DeepFitTrainer, SupervisedTrainer)`
精确类型判断决定——自定义 trainer 不会自动获得 checkpoint 捕获，需自行处理。

### D2 零改动现有层

与 `PUPipeline`/分类器签名零改动；仅 nnPU `history_` 内部补记 `val_risk`（供深度轨迹读取），
SA 语义与早停逻辑不变。详见 ADR-0018 决策 2。

### D3 公共 API 与数据合约

**决策**：`ExperimentRunner.fit(model, train, pu_val, clean_val, test)`（协议 §2.4 第 7 条字面形态）
+ `DatasetBundle`/`DatasetPart` 数据合约。

**实现方式**：[`bundle.py`](../../pu_toolbox/experiment/bundle.py) 的 `DatasetPart` 是 `frozen`
dataclass（`view` 为 `Literal["pu","clean"]` 必填冻结字段），`DatasetBundle` 承载四路分区；
`validate_bundle` 三查——四份 indices 两两不重叠、四者 `view` 全为 `"clean"`、`test.for_selection
is False` 强制。

**为什么**：runner 不切分原始数据、只接受切好的四路数据——切分决定权与责任在协议/研究团队
（`scripts/prepare_survey_splits.py` 只执行、不擅自决定，见协议 §2.4 第 3/7 条）。

`model` 本身由调用方经 `registry.get_algorithm` 查表取类、实例化后注入，实验层不静态 import 算法
文件（解耦机制见 [architecture.md](architecture.md) §2.1 实验层注入链）。

### D4 视图语义 —— clean 入 / PU 运行时生成 / 防泄漏

**决策**：bundle 输入全部为 clean 视图（真实标签）；PU 视图由 `Generator.generate` 在**运行时**从
真实标签生成（不预先落盘）；PA 路径结构性接收不到真实标签。

**实现方式**分两段：

- **视图转换（clean → PU）**：`validate_bundle` 强制输入四份数据全为 clean 视图；对象 `frozen` 不
  就地改字段，`Generator.generate` 返回新标签数组 `y_pu`，由 runner **重建**新 `DatasetPart` 并置
  `view` 为生成器声明的 `output_view`。
- **防泄漏（PA 拿不到真实标签）**，三层叠加：
  1. **路由**：仅 `isinstance(proto, ProtocolOA)` 才拿 `clean_val`，其余（含 PA）拿 `pu_val_view`；
  2. **声明**：`output_view="clean"` 使 π 检查让位给 PA 的 view guard（报错权交给 guard）；
  3. **断言**：`ProtocolPA.select` 在 `view != "pu"` 时 raise（`strategies.py`），OA 对称要求 `"clean"`。

**边界（必须写明）**：上述「结构性接收不到真实标签」只对**内置 PA**成立。非 OA 的未知自定义协议在
clean 视图（PN oracle）运行里，`pu_val_view` 携带的是真实标签，只有 `ProtocolPA` 自己会拒绝——这不
构成对任意自定义协议的通用保护。oracle 用真实标签训练必须显式声明
（`CleanLabelGenerator.output_view="clean"` + `SupervisedTrainer.trains_on_real_labels=True`），
不能靠隐式复用 PU 通道。

### D5 轨迹语义

**决策**：`RunTrajectory.best_epoch` 为 **1-based position in `epochs`**（`epochs[best_epoch-1]` 为
最优记录）；`EpochRecord.epoch` 为来源标签（展示性，可能 0 基）。

**实现方式**（[`tracking.py`](../../pu_toolbox/experiment/tracking.py) 为纯数据类、无逻辑）：
- `best_epoch` 由 **Trainer** 写入（唯一写入点 `DeepFitTrainer.fit` 的 `argmin(val_risk)+1`），
  `FitTrainer`/`SupervisedTrainer` 不设该字段（保持 `None`）；runner 只校验、不写入。
- `SelectionArtifact.epoch` 同为 1-based position，由 `SelectionProtocol` 写入（有 checkpoint 时取自
  `checkpoint.epoch_position`，legacy 无 checkpoint 才回落 `trajectory.best_epoch`）；runner 校验
  `art.epoch == checkpoint.epoch_position`。

**为什么**：位置语义与 estimator 的 epoch 标签解耦（后者可能 0 基或任意索引）——轨迹内以位置索引、
跨层只认位置，避免来源标签歧义。

## 2. 边界与已知局限

- **正式资格**：PA 正式准则（proxy accuracy，R9）与逐 epoch checkpoint 独立恢复已实现；
  但 P2.0b/P2.0c 验收、合作者签署、完整 Self-PU OA meta-reweighting、CNN/full-batch oracle
  仍未完成，各单元保持 `formal_eligible=false`。阻断与验收见
  [p2_0a_review](../research/pu_survey/p2_0a_review.md)、
  [epoch_checkpoint_delivery](../research/pu_survey/epoch_checkpoint_delivery.md)。
- **PN oracle**：MLP 路径已接入（Phase 1）；CNN oracle 的 clean-val checkpoint 选择与
  backbone 对齐列为 Phase 2。见 [pn_oracle_integration](../research/pu_survey/pn_oracle_integration.md)。

## 3. 文档与代码的分工（实施载体约定）

类/模块 docstring 中 `Design notes` 仅记「为什么」（≤6 行）并链接本文件（§1 关键设计机制）；
长效规格文本只存在于本文件与协议（不重复记入代码注释）；被否决/被替代的决策（如 Bridge 方案）
不写入注释，决策记录归 `docs/` ADR 体系。
