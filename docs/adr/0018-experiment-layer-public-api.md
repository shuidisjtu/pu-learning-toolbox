# ADR-0018:实验层公共 API 与零改动契约

- 状态:已接受
- 触发复审:实验层需承载研究者自定义协议导致策略接口膨胀;或 `PUPipeline` 自身也需数据角色语义时

## 背景

PU 调研实验协议 §2.4 要求四份数据各带明确职责(`train` 只训练、`pu_val` 仅 PA 选模、
`clean_val` 仅 OA 对照、`test` 只评测),而既有 `PUPipeline` 面向非专家用户,是"单份数据 +
内部 PU 分层 CV + 平均分选模",没有数据角色声明与模型选择协议两个概念。"数据带角色"是
本次实验首次出现的用户需求,属正常架构演进而非早期设计错误——分类器层的 sklearn 式
`fit(X, y)` 契约是应当保留的底层设计。

## 决策

1. **新增独立 `ExperimentRunner` 层**(`pu_toolbox.experiment`)承载用户划分、PA/OA 双协议、
   模型选择记录及最终测试,作为工具箱面向实验/研究者用户的公共 API 层。
2. **零改动现有层**:不改 `PUPipeline` 的交叉验证语义,不修改分类器层的 `fit(X, y)` 契约;
   实验层与工具箱之间为单向依赖(实验层 → 现有层)。
3. **策略化接口**:`ExperimentRunner` 固定编排骨架(Template Method),数据生成/训练/选模/留痕
   各自为可注入策略 ABC(`Generator`/`Trainer`/`SelectionProtocol`);不采用 Bridge 双层次——
   变化点各自成轴,研究者 DIY = 实现策略并注入,不继承 runner。
4. **数据合约**:新增 `DatasetBundle`/`DatasetPart` 承载四路分区与视图语义(`view=="pu"` 使
   PA 路径结构性接收不到真实标签)。

## 备选方案

- **改造 `PUPipeline` 支持数据角色**:破坏面向非专家用户的一键分析语义,影响既有用户。否决。
- **Bridge 双层次**:变化点尚未充分多样化,双层继承过度设计(YAGNI)。否决。
- **分类器层加数据角色**:污染底层 sklearn 契约,违背单一职责。否决。

## 后果

- `pu_toolbox.experiment` 模块落地,P0 pilot 已实现四路合约、SCAR/SAR 生成、PA/OA 选模、
  PN oracle 与阈值选择(设计蒸馏见 `docs/dev/experiment_layer.md`)。
- `PUPipeline` 与分类器契约保持不变,既有测试不受影响。
- 研究者扩展 = 实现策略 ABC 注入,无需继承或修改 runner;策略接口先固化 SCAR 主实验,
  验证稳定后再扩展谱系(YAGNI)。
- 本决策的逐条记录见 `docs/dev/experiment_layer.md` §1 D1–D5。
