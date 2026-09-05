# PU 调研实验：技术实现与推进状态

> **本文件为实施方案与推进状态**：对应 [pu_survey_protocol.md](pu_survey_protocol.md)
> （方案要求纲要，本实验的协议真相源）。本文件聚焦方案各步骤的**技术实现思路与细节、
> 当前状态与进度审计**——如何用当前 Toolbox 落地、锁定哪些实现、当前差距与待办。
> 协议本身（实验怎么设计、什么算结论成立）以协议文件为准；两者不同步修改，本文件随
> 实现推进更新。

## 1. 工具箱现状差距与 `ExperimentRunner` 结论

### 1.1 差距清单（协议 §2.4 第 7-9 条）

| 要求 | 当前状态 | 依据与边界 |
|---|---|---|
| `fit(model, train, pu_val, clean_val, test)` 或等价接口 | 未实现 | 通用 `PUPipeline.fit_evaluate(X, y_pu, ...)` 只接收单个训练对象，并在内部执行 PU 分层交叉验证；没有用户传入的 `pu_val`、`clean_val`、`test` 参数。 |
| 用户给定数据、模型与参数后训练，并提供训练/数据生成 DIY 接口 | 部分实现 | 已有注册分类器、`classifier_params`、`prior_estimator`、`architecture` 以及 SCAR/SAR 生成器；但模型必须是已注册方法或继承 `BasePUClassifier` 的实例，不能直接接入任意 sklearn/PyTorch 模型。 |
| 读取四份用户数据并完成 PU 训练、PA/OA 选模与独立测试的官方示例脚本 | 未实现 | `examples/minimal/` 仅提供单功能示例；`benchmarks/` 使用内部固定协议，没有通用四份数据输入、双选模协议和独立测试汇总脚本。 |

个别算法的专用能力不改变上述结论。例如 `SelfPUClassifier.fit(..., validation_data=...)`
可接收 clean validation，但这不是全体算法共享的接口，也没有同时支持 PU validation、OA
对照与独立测试。现有 CLI/UI 同样主要面向单份训练数据加可选真实标签，不能表达本实验所需的
数据划分（§2.4）语义。

### 1.2 差距形成本质原因：工具箱的概念缺失

概念的缺失是接口构建方面的：

1. **数据"角色"声明**。当前工具箱把传入的数据视为"整个数据集"——训练与评估都在内部处理
   （PU 分层交叉验证），用户无法告诉和硬性规定工具箱"这份数据只能训练、那份只能测试"。而实验要求的
   §2.4 的数据划分恰恰相反：每份数据带着明确职责（`train` 只训练、`pu_val` 只供 PA 选模、
   `clean_val` 只供 OA 对照、`test` 只做最终评测），职责边界划分得十分清晰，但工具箱却无法提供如此细节的功能接口。
2. **模型选择协议**。当前工作流按交叉验证的平均分选模，并在全部数据上重新训练出最终模型；
   实验协议要求的是"完整训练轨迹 → PA/OA 各自离线选择 → 冻结的 checkpoint 独立测试"，
   后者需要轨迹记录与离线选择环节，现有流程只保留"平均分"这一个统计量，没有承载空间。

我要强调的是：严格来说这不能算为早期设计错误。当初关于`PUPipeline` 的设计主要是面向非专家用户（一键分析 + 推荐 + 报告），
"一份数据、内部划分"对他们是合理简化（虽然现在看来过于简化了），此前也从未有用户需要数据角色；分类器层的 sklearn 式
`fit(X, y)` 契约是应当保留的底层设计。本次实验是"数据带角色"这类用户需求首次出现，属于
正常的架构演进——补上缺口即可，不需要推翻现有部分。

### 1.3 结论与边界

**结论**：新增独立 `ExperimentRunner`（或 `PUExperiment`）层，承载用户划分、PA/OA 双协议、
模型选择记录及最终测试；**不改动 `PUPipeline` 的交叉验证语义，不修改分类器层的模型契约**。

**新增层边界确定**（什么进、什么不进）：

- **进**：四份数据（`DatasetBundle`）合约及校验（索引不重叠、PA 无真实标签、test 不参与选择）、训练 → PA/OA
  选模 → 独立测试的编排、全 epoch 轨迹记录与离线选择、选择制品与 manifest 留痕。
- **不进（复用或外置）**：数据切分（用户负责，协议 §2.4 第 7 条）；SCAR/SAR 生成（复用现有
  `preprocessing`）；类先验估计、指标计算（复用现有组件）；算法与 backbone 适配（见 §4、§5）。
- **对现有部分的影响**：零改动现有签名，实验层与工具箱之间为单向依赖（实验层 → 现有层）。
  对外新增一个公共接口 + 一个数据合约类型，属中等偏小的净新增。

**pilot 最小面**（遵循 YAGNI）：先实现四份数据合约、训练闭环、PA/OA 各自选模与独立测试、PN
oracle、阈值选择，通过 §7 验收清单后再扩展 SAR 细节与资源计量等，不一次铺满全协议。
（一次性全铺满风险过大，且影响范围难以评估，还是逐步来比较好。）

**实现侧自动验证要求**（协议 §2.4 第 2 条的落地）：自动化测试必须验证索引不重叠、PA 不读取
真实验证标签、测试集不进入训练或选择。

## 2. 数据管线实现细节

### 2.1 SCAR 标记（要求层级见协议 §2.3.2）

- 固定抽样数：$`n_{L}=\mathrm{round}(c\, n_{+})`$，从全部正例中**均匀无放回**抽取；不采用
  逐样本 Bernoulli 标注（保证扫描 $`c`$ 时实际标记率与名义 $`c`$ 对齐）。
- 记录 $`c_{\mathrm{realized}}=n_{L}/n_{+}`$；随生成 seed 与 manifest 保存。

### 2.2 SAR 压力测试（要求层级见协议 §2.3.2）

- 协议要求仅取 $`c\in\{0.1,0.5\}`$；须与 PU-Bench 的 SAR 设定保持一致。
- **锁定值（PU-Bench commit `2d95a19`，已对照源码核实）**：
  - 辅助 posterior 模型：sklearn `LogisticRegression(solver='lbfgs', max_iter=100,
    random_state=seed)`，在源 train 全部特征与**真实标签**上拟合，取 `P(y=1|x)` 作为 scores；
  - LBE-A：权重 `p ∝ scores^k`，`k = 10`（LBE 论文指定）；平滑 `p = 0.9·p + 0.1·uniform`；
  - LBE-B：权重 `p ∝ (1.5 + shrink_coef − scores)^k`，`shrink_coef = 1.0`（原实现 syn 函数取值）；
    负值截断为 0；全部为 0 时退化为均匀分布；无平滑；
  - 抽取方式：固定 `n_labeled`、**无放回**加权抽样；
  - 差异记录：PU-Bench 的 `n_labeled = int(n_pos · labeled_ratio)` 为向下取整，协议 §2.1 采用
    `round(c·n_+)`——实现时以协议口径为准并记录实际 `c`。
- 保持固定 $`n_{L}`$；每次保存请求与实际标记数、权重/score 版本和生成 seed 一并记录。

### 2.3 split 操作细节（要求层级见协议 §2.4 第 3-4 条）

- 保留 PU-Bench 的独立测试集；从原始训练源**分层**留出 10% 验证池，等分为 5% `pu_val` 和
  5% `clean_val`，其余 90% 为 `train`。
- **PU-Bench 独立测试集来源（commit `2d95a19`，已核实）**：
  - 自带官方测试集：MNIST、F-MNIST、CIFAR-10（`torchvision` `train=False` 测试集）、IMDB
    （自带 train/test 划分）、20News（自带划分）；
  - 无自带测试集的 Spambase、Connect-4、ADNI：`sklearn train_test_split(test_size=0.2,
    stratify=y, random_state=seed)` 从全量切出 20% 固定为 test。
- **验证池与 PU-Bench 做法的一致性**：PU-Bench 从源 train 内、在 PU 采样**之前**按真实标签
  分层切出单一路 val（策略名 `split_source_before_pu_sampling`，防止 case-control 重复抽到跨
  边界样本），并让 val 与 train 用同一 `labeled_ratio`/选择策略生成 PU 标签；config 各
  `param_sweep_*.yaml` 统一 `val_ratio: 0.01`。本实验沿用"PU 采样前切分、真实标签分层"的机制，
  在源 train 内切 10%（split 由实验侧实现，seed 驱动），`pu_val` 仅保留 PU 标签视图（剥离
  真实标签），`clean_val` 保留真实标签。
- `pu_val`/`clean_val` 不重叠；`clean_val` 与 `test` 保持自然类先验。
- 五个实验 seed 共同决定原始 split、SCAR/SAR 标记和训练随机性；同 seed 所有方法/PA/OA/
  PN oracle/所有 $`c`$ 共享同一底层 split；同 $`c`$ 共享相同 P/U 标记；扫 $`c`$ 仅重生成 $`S`$ 标签。
- 记录 seed、样本索引与 split manifest。
- 分层变量：真实（二元化）标签；实现工具：sklearn `train_test_split`。

## 3. backbone 实现细节（要求层级见协议 §2.5）

- 图像统一使用**随机初始化**的 ResNet-18 主体；灰度数据将首层卷积设为单通道，RGB 保持三通道，
  **不复制灰度为伪 RGB**。输入尺寸、首层设置、归一化和增强均须入 manifest。
- 表格与文本分别预注册数据集内共享的 MLP 规格；方法私有网络只能作为 `benchmark-adapted`
  路径报告。
- 所有可学习预处理统计量仅在该 seed 的 `train` 拟合并冻结，图像增强仅用于训练。
- 文本记录 `all-MiniLM-L6-v2` 的模型 revision、384 维输出及 embedding cache hash。
- 决策记录：原方案中的 ResNet-34 因当前 Toolbox 未集成且算力限制，不作为默认选择；已确认
  采用 PU-Bench 的数据集内统一训练与评估协议。

## 4. 双架构计划与本实验的前置工作项

### 4.1 决策记录

曾讨论：当前 Toolbox 正在推进"PU 双架构渐进式升级方案"，该计划是否是采用 PU-Bench 架构
（尤其在图像数据上比较传统 PU 方法与深度 PU 方法）的前置条件？**结论：双架构计划不是文本
SBERT 向量或表格 MLP 路径的前置条件；它是完成图像数据集公平比较的部分前置工作。** 本实验
暂采用 PU-Bench 的"同一数据集内统一表征、backbone、训练预算与调参预算"协议。

### 4.2 纳入实验计划的工作项

1. **统一图像 encoder/backbone 配置**：在 EncoderFactory/配置/报告中支持本实验选定的图像
   backbone、归一化和数据增强，并记录其版本与参数；当前已有的 CNN13、ResNet-18、ResNet-50
   不代表已复现 PU-Bench 的各数据集专用 CNN；
2. **完善算法能力台账与门禁**：逐算法声明 `native_mlp`、`native_cnn`、`tabular_only` 等
   能力，训练前拒绝不兼容的输入组合；
3. **完成必要的深度算法 encoder 适配**：针对本实验实际纳入图像榜单的方法，逐个评估并实现
   encoder 注入；不能仅改参数名或将完整模型伪装为 encoder；
4. **实现传统算法的 `cnn_feature_adapter`**（如主榜要求传统算法参与图像比较）：以固定或折内
   训练的 CNN encoder 提取二维特征，再训练传统 PU 算法。该路径必须单独调参、单独记录为
   `cnn_feature_adapter`；它不是传统算法原生 CNN 结果，也不得与端到端结果混合；
5. **加入跨路径公平性检查**：同一数据集主榜应校验数据划分、特征版本/backbone、训练预算、
   调参预算和随机种子协议一致；
6. **固定图像路径分组**：原生端到端 CNN 与 `cnn_feature_adapter` 二维特征路径分别成组，
   绝不混合排名。adapter、非原文训练接口或其他兼容性改写均须标为 `benchmark-adapted`。

### 4.3 独立的工作项

双架构计划完成后仍不足以完成整个实验：TS-OS 校准、PA/OA 模型选择、四份用户数据的
`ExperimentRunner` 接口及 SBERT 向量生成流程属于独立工作项，应与架构适配并行规划。

## 5. 算法接入与溯源现状

- 当前外部 Toolbox 源码审计显示：22 个目标方法（21 个 PU 方法 + 1 个 oracle）中仅 uPU、nnPU、
  KLDCE、Dist-PU、PUSB、LBE、Self-PU 七个已注册并绑定为可训练实现；其余方法及 PN oracle
  仍需接入。
- 完整主榜以 22 法全部通过相应门禁为发布条件；在此之前只可发布明确标为
  `pilot / partial benchmark` 的部分结果。
- 每个方法的**接入验收**依次包括：原论文和官方实现/commit 可追溯、固定小数据单元或冒烟测试、
  至少一个公开协议上的结果或行为对照。未完成前不得标为 `source-faithful`。

## 6. 复现与资源计量细则（要求层级见协议 §5.3-4）

- **中心超参数注册表**：候选池按论文/官方代码预注册并记录候选数，通过中心注册表统一管理；
  设计参考参考文献 1 对应代码库 `core/hparams_registry.py`。
- **环境分类**：只规定 GPU/CPU、显存等级和软件栈等大类，不将本机硬件写入方案；实际型号、
  驱动和资源限制写入 artifact。
- **成本口径**：单配置成本从模型初始化到最终 epoch，包含训练和每 epoch 验证，不含下载、
  SBERT 生成、split 和 PU 数据生成；共享预处理时间单列。总调参成本包含全部候选配置和五个
  seed；峰值显存取该全过程最大已分配 GPU memory。

## 7. pilot 前基础设施验收清单

pilot 前的基础设施验收至少覆盖：

- 四路 `DatasetBundle` 合约；
- SCAR/SAR 生成与 manifest；
- PA/OA 独立选择 artifact；
- PN oracle；
- 阈值选择；
- 资源/失败记录；
- 一个二维方法和一个 CNN 方法的端到端 smoke run。

## 8. 待确认与待办

截至 2026-09-05 无开放事项。此前两项待办已解决并回填：

- **test 测试集留出方式**：已对照 PU-Bench（commit `2d95a19`）核实，结论见 §2.3；
- **SAR 锁定 commit 值**：已锁定（commit `2d95a19`），参数回填见 §2.2。
