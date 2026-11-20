# 进度清单

> 总体策略：**framework-first**——先完成稳定框架与 API 契约，用 mock estimator 跑通链路，再逐个集成论文算法。当前全部注册方法均有 clean-room 核心实现；另有一个隔离的联合漂移 research 求解器。实现可运行与论文数值复现、研究结果审核分别判断；Survey 的当前能力与实验状态见 [Survey README](../research/pu_survey/README.md)。
> **Method Card 为可选文档**，新算法接入不要求必写。

## 阶段历史（已闭环）

Phase 0-9 已闭环（框架 → 核心风险估计 → 机制 → 推荐诊断 → SAR → 深度 → 分布漂移 → 联合漂移）；逐条明细与批次历史见 git log。

## 未完成项

- [ ] Phase 3 官方数据/历史环境全量运行（依赖外部官方数据与历史环境，非工具箱缺口）
- [ ] Phase 6 WConPU 官方视觉 + DGPU EDM paper-like 全量（依赖 CUDA/授权数据）
- [x] InfoMax 类别分组、batch size 与 KM 变体来源核对：原文仅说明 MNIST/Fashion-MNIST 各分成两组、表示学习按 4:1 mini-batch 交替更新，并采用 KM；未给类别 ID、数值 batch size 或 KM1/KM2。现配置 `[0,1,2,3,4]`、256、KM1 均保留为 provisional 工程选择，结果 `paper_claim=false`；详见[来源审计](../research/method_cards/InfoMax-PU.md)。
- ⚠️ v1 范围外：Phase 2 三经典包装器 + TIcE/AlphaMax 类先验估计

## 发布状态

### 2.0.0（目标版本，未发布）

- 旧 `run`、隐式 `auto` 选法和 `PUPipeline` 退出。显式选择 estimator 后训练、保存可信模型，
  再用独立 `review`；保留 `list-methods` 和四角色实验 API。已安装的 Skill 可运行
  `pu-toolbox skill install --force` 更新到当前安装包附带的版本。
- Elkan–Noto 的 `n_cv_folds` 改为 `internal_cv_folds`。倾向估计按观测 PU 标签分层做折外预测，
  只汇总留出的标记正例，移除旧折内 `sqrt` 补偿；新旧实现不承诺数值等价。
- 传统 PU v6 用原 commit/tag 和当时的配置重放；v7 用现行实现和 v7 配置写到独立目录，
  不回写 v6 结果。保留全部历史 1.x tag。

### 未发布（显式训练工作流退役）

- [x] 按 [ADR-0024](../adr/0024-estimator-and-experiment-workflows.md) 移除显式 Pipeline、单表候选评估、
  漂移训练比较及对应 CLI/UI、配置和报告生成；不设兼容层，不重写历史产物。
- [x] 保留 estimator、四角色实验、拆分与指标原语和独立 CLI，先验与验证边界见
  [迁移指南](../user/howto/pipeline.md)。
- [x] 定向验证 221 项通过、零失败或跳过；API、文档、技能、质量、依赖图与结构门禁通过。
  sdist/wheel 构建及隔离 wheel CLI 冒烟通过；未执行全量或远程 CI。

### 未发布（Elkan–Noto 标注倾向估计）

- [x] 按观测 PU 标签划分倾向估计折，移除旧版折内 `sqrt` 补偿；仍只对留出的已标注正例汇总预测。
- [x] 两类观测样本均需满足内部划分的折数要求；校准折数不超过训练数据中较小一类的样本数。

### 未发布（输入校验与评估折数命名）

- [x] 工作流、Elkan–Noto 的评估折数和样本权重校验，以及注册元数据校验，统一抛出
  `ValidationError`；同时保留 `PULearningError` 与 `ValueError` 捕获路径。
- [x] 外层评估用 `evaluation_cv`，Elkan–Noto 拟合内部用 `internal_cv_folds`；
  `shift-run` 分别设置评估和漂移折数。

### 未发布（注册表只登记可用实现）

- [x] 首次查询、发现或登记时加载内置算法，导入本身不初始化；用户不再需要手动预热。
- [x] 元数据与具体估计器一并登记，拒绝没有实现或抽象实现的条目。
  删除占位状态和重复的可训练过滤，保留来源、成熟度及能力声明。
- [x] 移除公开的手动初始化入口；脚本和安装验收通过查询自动初始化。

### 未发布（注册表命名与冲突检查）

- [x] 注册条目和别名统一为小写；查询仍不区分大小写，仅大小写不同的拼写不再登记为独立别名。
- [x] 一次校验全部名称后再写入；冲突不留注册残余。

### 未发布（模型选择指标方向）

- [x] `PUTuner` 与 `PUModelComparator` 按指标固定选优方向：风险、Brier 和 ECE 取最小值。
- [x] 移除 `higher_is_better` 构造参数；结果和序列化字段保留实际方向。
  `pu_negative_rate` 和别名仅供报告，不再用作 `scoring`。

### 未发布（对外文档英语化：贡献者入口）

- [x] [ADR-0022](../adr/0022-public-documentation-language.md) 取代 ADR-0008 的旧语言条款，
  对外英文单源、原路径维护；内部中文资料与 Skill 中文解释资源保留。
- [x] 贡献指南、PR 模板、文档总索引与五份外部开发者指南英语化；
  清理根中文 README 入口及失效引用。注册字段、可选依赖和打包说明按源码核对。
- [x] 依赖表生成器与现有测试仅同步英文展示；相关脚本测试 464 项通过，
  文档/API/Skill 等适用静态门禁及构建通过。
- [x] [PR #175](https://github.com/shuidisjtu/pu-learning-toolbox/pull/175) 已合并，
  远端与本地旧功能分支已删除。

本批交付贡献者入口；后续迁移状态见下节，公开导航按实际语言覆盖更新。

### 未发布（对外文档英语化：用户入口与概念）

- [x] 用户索引、快速入门与七篇概念页原地英语化，保留 OS/TS、SCAR/SAR、
  先验分布、PA/OA 及表征的独立维度与科学限制。
- [x] 快速入门补齐演示 CSV 的读取步骤；`pusb` 身份、推荐过滤及 LBE
  `sampling_policy` 说明按源码核对；修复受影响的五处标题锚点。
- [x] 快速入门/标记画像、采样先验、选模表征三组定向测试共 257 项通过；
  文档/API/Skill、格式、测试质量、数学渲染与生成一致性门禁通过。
- [x] 快速入门 CLI 链路与 6 段 Python 示例实际执行通过；11 个独立公式块
  与迁移前逐字一致。
- [x] [PR #176](https://github.com/shuidisjtu/pu-learning-toolbox/pull/176) 已合并，
  远端与本地旧功能分支已删除；本地 main 已快进同步至合并提交。

### 未发布（对外文档英语化：使用指南）

- [x] `docs/user/howto/` 全部 14 篇指南原地英语化，导航与受影响的入链锚点同步。
  API 参考与 Survey 复现资料按后续批次迁移。
- [x] 按源码校正 Pipeline 执行顺序、完整 PU 数据先验解析与逐折消费、动态 CNN 能力、
  OS/TS 画像审计和 Self-PU 独立验证/消融路径；不改变运行行为。
- [x] 人工复核 Skill 的研究实验与表征参考资料，用户指南路径和契约保持兼容；
  双客户端资源镜像及中文解释能力保留，Skill 无需因文档翻译重新安装。
- [x] 四组定向测试共 565 项通过；未安装 Streamlit，2 个 UI 集成测试模块在收集时跳过。
  文档/API/Skill、格式、测试质量、数学渲染、生成一致性门禁及构建通过。
- [x] 39 段 Python 文档片段、5 个诊断 CPU 示例及 8 项 CLI/Skill/CNN/Self-PU
  冒烟操作实际执行通过；另验证自定义生成器、训练器与选模策略的联合运行。
- [x] 全部 41 段 Python 示例的 AST 与迁移前一致，2 个独立公式块逐字一致；
  修复入链锚点，检查 9 处相关标题引用，无重复标题。
- [x] 标明既有 `10_self_pu.py` 的限制：OS 生成数据的总体先验直接传给默认 native-U
  风险模型；适配时应按训练分区换算。本批未改示例源码，冒烟通过不代表该先验假设正确。
- [x] [PR #177](https://github.com/shuidisjtu/pu-learning-toolbox/pull/177) 已合并，
  远端与本地旧功能分支已删除；本地 main 已同步至合并提交。
  完整多 seed benchmark、200 epoch CUDA 训练及浏览器 UI 交互未执行。

### 未发布（对外文档英语化：API 参考）

- [x] API 索引及 `api.md` 原地英语化，保留文件路径和完整章节覆盖；
  公共符号覆盖门禁核对 86 个导出/注册名，未借迁移拆分文件。
- [x] 按源码核对构造签名、默认值、直接训练的先验目标、PUSB 基线与固定阈值、
  复核零预算/逐行输出、Pipeline/CNN、独立支持集和 checkpoint 恢复边界。
  补齐 GradPU 三个执行策略参数及 profiler 的 `input_sampling`，修正复核调用中的非法裸 `*`。
- [x] Survey 当前版本解析、PA、正式资格与历史 manifest 解释分开陈述；
  修正核心异常继承范围与汇总参数默认值，不改变实现、配方或实验结果。
- [x] 六组定向测试累计 924 次通过，去重后 894 项，无失败或跳过。
  BBE/TEDn 拟合、PUSB/复核契约及四组画像/推荐/敏感性/诊断片段实际执行通过；
  敏感性片段按接口前提提供 prior 网格，未执行完整 GPU 或多 seed 研究。
- [x] 53 个代码/签名块完整保留，其中 3 块仅作上述契约修正；
  公式保留，38 处相关标题入链有效。导航、Skill 兼容性、文档/API、格式、数学渲染、
  测试质量、生成一致性门禁与构建通过，临时审计脚本及测试产物不提交。
- [x] [PR #178](https://github.com/shuidisjtu/pu-learning-toolbox/pull/178) 已合并，
  远端与本地旧功能分支已删除；本地 main 已同步至合并提交。

### 未发布（对外文档英语化：Survey 复现资料）

- [x] Survey 入口、实验协议、运行指南及数据来源页四份 Markdown 原地英语化；
  删除失效的等待外部 GPU 训练说明，不新增替代进度公告。
- [x] 保留协议/配方身份、历史 manifest、四角色与独立 clean support、PA/OA 及数据权利边界；
  明确 ADNI 槽位为替代数据，受试者隔离不可核验，历史 owner approval 不构成通用授权。
- [x] 按源码核对独立先验 API、PUSB 路径和固定标记数边界；区分公平比较要求与实际锁定预算。
  49 个数学片段逐字保留，8 个命令块及实际参数保留，修复四处标题入链。
- [x] 数据、协议、运行与报告四组定向测试累计 442 次通过，去重后 431 项，零失败或跳过。
  合成 CPU 验证完成配方检查、uPU 单元运行、批次 dry-run、汇总及两种导出路径；
  单 seed 产物保留 `formal_eligible=false`，不代表完整 GPU、多 seed 或论文数值复现。
- [x] 整体验收覆盖计划中的 35 份 `docs/` 目标、贡献入口与 PR 模板；更新公开语言导航，
  文档/API/Skill、格式、测试质量、数学渲染、生成一致性门禁与构建通过。
  JSON、图像、配方、运行实现、已保存结果和双客户端 Skill 资源未修改。
- [x] [PR #179](https://github.com/shuidisjtu/pu-learning-toolbox/pull/179) 已合并，
  远端与本地旧功能分支已删除；本地 main 已同步至合并提交。
  补齐 13 个最小示例的英文 API 引用注释，18 个文件无中文残留；执行代码保持不变。

英语化方案的五批 PR 均已合并，计划内全部用户、贡献者及 Survey 复现文档均为英文。
内部中文记录和 Skill 中文解释资源保留；迁移不改变公共 API、算法行为或已保存结果。

### 未发布（最小示例兼容性与采样语义修复）

- [x] 新分支 `fix/minimal-example-compatibility` 全量审查 18 个最小脚本；先用真实执行和
  专项回归复现 uPU 失效接口、nnPU/Self-PU 先验目标、PNU 零权重及复核模型分支问题。
- [x] uPU 改用公共 `pu_zero_one_risk`，明确 OS 诊断先验；uPU/nnPU/Self-PU 按训练
  标记比例换算 native-U 先验。Self-PU clean validation 仍为额外监督，不用于测试结论。
- [x] PNU 使用非零 PN/PU 混合权重；动态联合漂移示例独立生成 P 与总体混合 U，
  保持各域生成先验与实际风险池一致。漂移复核只在推荐加权时采用 weighted 模型。
- [x] 标明训练表准确率、PUSB/SAR 指标前提、平移仿真的协变量漂移限制与 oracle 审核角色；
  同步 Self-PU 指南/API、测试文件清单。算法实现、历史配置和保存结果未改动。
- [x] 18 个默认脚本及专项回归共 28 项通过；相关算法/先验/采样/漂移/新示例回归 222 项通过，
  零失败或跳过。无 PyTorch 时仅跳过依赖它的示例，CI 安装 torch 后执行完整覆盖。
- [x] 格式、注释、测试质量、文档/API、元数据、数学渲染、Skill 与生成一致性门禁及构建通过；
  本批仅验证合成 CPU 工程流程，不声称完成 GPU、多 seed 或论文数值复现。
- [ ] 本批 PR 审核与合并；验证产物保留在本地 `tmp/`，不提交。

### 未发布（AI workflow skill 更新，2026-11-13）

- [x] 以任务路由组织主入口和六份参考文件，覆盖分析、采样先验、调参比较、四角色 PA/OA、
  encoder/冻结特征、漂移、监控、复核及审计。skill 修订为 1.11.0，验证当前源码构建的同版本 wheel。
- [x] 两套客户端资源完整镜像；门禁覆盖资源树、内部引用和元数据。先复现参考资源漏检，
  修复后安装/同步专项 28 项通过。
- [x] 计划 CPU 回归 150 项、clean support/checkpoint 补充 77 项通过；15 个 Python 参考片段、
  日常与扩展 CLI、显式 CNN 小样本训练以及独立 OS/TS 演练完成。
- [x] wheel 在临时隔离环境安装，两个目标各含七份资源，外部目录 OS CLI 和 TS API 运行通过。
- [x] 使用指南、导航、CLI、架构与资源维护说明，以及根目录中英文 README 同步；
  文字直接陈述结果与适用条件。格式、测试质量、文档链接与生成目录门禁通过。

输入设计和先验目标通过现有 API 路由：CLI 使用 OS，独立 TS 使用显式 Python 参数，
Pipeline 记录总体解析与逐折训练消费。CLI 的显式采样参数作为后续工程项维护；recommend 路径按 [ADR 0023](../adr/0023-explicit-method-selection.md) 退役。

### 未发布（PU 研究能力整合）

- [x] P0 语义基线：固定四篇论文/两个参考库来源；核对采样、先验估计池与逐实现风险消费；
  实际执行 S1–S3 最小复现，形成 [ADR-0021](../adr/0021-pu-research-semantics.md)。
- [x] P1 公共知识：采样、先验、PA/OA、表征四篇概念页，SCAR/SAR 政策与来源对照，
  用户/开发导航；新增的 7 个 Python 文档片段实际执行通过。
- [x] P2-S1：同池定额入口 `make_fixed_count_labels`，旧名字兼容包装与 2.0 迁移计划；
  独立 TS 高斯生成器记录来源、总体参数和经验先验。
- [x] P2-S2：固定 Gaussian 旧场景，混合 U 教学路径使用新生成器；
  [示例 15](../../examples/minimal/15_pu_sampling.py)、[采样教程](../user/howto/pu_sampling.md)与 API 同步。
- [x] P2-S3：先验目标声明、ReCPE 复制流程、Pipeline 的解析与消费记录，连同 OS/TS 画像及指标回归。
  每折按实际标记比例换算训练参数；PUSBKernel 分别解析训练表与风险池先验；报告 schema 为 1.1。
- [x] P3：固定正例配额与均匀抽样单源；Pipeline/CVIR 共享先验换算；registry 消费目标声明。
  跨入口标签、全选后的随机流、元数据及层间依赖回归通过。
- [x] P4：示例 14–18 覆盖采样、SAR 政策、先验传递、独立 PA/OA、encoder 与完整 DIY；
  新增 encoder 教程，补齐可组合研究路径，CPU 实际运行和 31 项教学/快照回归通过。
- [x] P5：新运行记录采样、标记、风险池、先验输入/实际消费、监督与表征设置；
  报告按 seed 留存并输出摘要，历史结果按原身份读取。报告专项 109 项、最终定向回归 80 项通过，
  DIY 产物实际生成，API/协议/方法卡与适用静态门禁同步。

S1/S2 已按失败回归驱动完成实现；旧同池入口保留输出与随机轨迹。
文档按有证据支持的结论和适用条件直接陈述。专项与适用静态门禁结果见
[语义基线的验证记录](pu_research_semantics.md#7-本批验证记录2026-11-06)。

### 未发布（PU 数据入口重构）

- OS 已有数据与合成入口共用来源身份、标签预算和观测/审计/留痕契约。`make-demo-data`、
  示例 15 和两份工作流 Skill 已接入；旧同池 `case_control` 别名及分派已移除。
- 现行 OS 示例、测试夹具和传统/指定方法基准均迁到两个构造入口，删除旧 OS 标签、合成及分派函数。
  基准保留采样政策、补标、种子和顺序随机流；Survey 仍通过中立原语保持原有标签轨迹。
- 显式获取与离线读取按配置分离。MNIST/Fashion-MNIST 共用 IDX 格式实现，Survey 冻结 NPZ 走同一读取入口；
  不把类别映射和角色划分放进数据源配置。Survey 的 NPZ 产品、映射及整数 ID 用相同来源切片逐字段核对。
- 其他来源格式、独立高斯 TS、负类 U 教学构造及先验消费按后续工作包迁移；不改历史身份。

### 未发布（实验层收敛）

实验层保留公共四角色 API、独立 PA/OA 选模、当前 Survey 配方执行和结果报告生成能力。
训练产物及结果审核由研究团队维护；历史试点协议、队列、交接和阶段审批记录退出工具箱。
当前能力与实验状态见 [Survey README](../research/pu_survey/README.md)，
设计与扩展方式见[实验层设计](experiment_layer.md)，操作链见
[运行指南](../research/pu_survey/runbook.md)。

- **clean-support 覆盖扩展（2026-11-08）**：`survey-v3.2` revision
  `2026-11-08-clean-support-coverage` 将 PULNS/LaGAM/Self-PU full-meta 的 support 覆盖扩到
  IMDB、20News、ADNI（共 8 个数据集），文本从冻结 split 的 SBERT 特征 carve，ADNI 记录
  `group_policy="unverifiable"` 与 owner approval；配方注册表升至
  `survey-recipes-v1.3`。数据与监督预算按各次 manifest 解释，配方能力不自动代表数值复现。
- **uPU 拟合与 PA 先验分开**：内置 trainer 在 OS 原生 U 风险下用训练标记比例换算总体先验；
  TS-compatible 拟合与 PA 仍用总体先验。拒绝非法训练视图，以及对已声明独立 TS 的训练输入再做重构。
- **clean-support 执行门禁**：批量计划和续跑前检查五角色文件及原始样本 ID；单元运行在训练、
  选模和测试前校验支持集制品，并核对它与实际四角色输入逐行一致。样本行隔离不代表受试者隔离。

发布前检查：

- 公共 API 不读取 Survey 配置或项目历史文档；自定义策略仍可注入。
- 数据角色隔离、视图/先验换算、独立 support 预算与 PA/OA 选模约束仍生效。
- 结果读取拒绝损坏选择制品，并保留失败、缺 seed 和缺指标；均值/样本标准差及成本分账正确。
- 工作簿、诊断卡与结论可由现行入口生成；文献事实与实验观测分别记录。
- 运行适用的测试和质量门禁；不把未执行的 GPU、外部数据或结果审核报告为完成。

### 未发布（随下一版本发布）

- **PN oracle 接入**：`CleanLabelGenerator` + 视图/声明双向 fail-loud + `--oracle` 入口。见[实验层设计 §2](experiment_layer.md#2-当前能力与边界)。
- **双架构阶段 0-2**：Registry 4 能力字段、build_encoder 导出、nnpu encoder 试点、CNN 提示文案回归、契约路线 B 收口。见 [dual_architecture_plan §5](dual_architecture_plan.md)。
- **Survey 语义统一（issue #42）**：PUSB 拆为 `pusb`/`pusb_kernel`，先验门禁改由 registry 驱动，台账↔registry 一致性契约。采样视图见[实验协议 §2.3](../research/pu_survey/pu_survey_protocol.md#23-pu-data-generation)，先验语义见[实验协议 §3](../research/pu_survey/pu_survey_protocol.md#3-class-priors-and-related-parameters)。
- **Self-PU 声明收口（issue #38/#45）**：`input_ndims` 恢复 `{2,4}`，非原生 CNN 由 `native_architectures` 承载。
- **训练视图分层**：`core/training_views.py` 承载中立角色构造，实验层负责视图路由、先验语义与 manifest；estimator 不依赖 experiment。核心视图拥有并冻结角色位置与来源索引，借用的特征和标签只读。依赖方向见[架构 §2.1](architecture.md)。
- **SAR-OA 执行路径（issue #43）**：`--labeling-mechanism` 与 `--method` 正交，SAR 强制 OA-only。见[实验协议](../research/pu_survey/pu_survey_protocol.md)。

### 已发布版本

- 1.11.0（2026-08-29）：pu-workflow skill 扩展场景 + NaN/Inf 拒绝 + 最低版本要求升至 1.10.0
- 1.10.0（2026-08-29）：传统 PU 调优收尾（KLDCE b₀ 修复、契约 v2、六轮写回，ADR-0016 闭环）
- 1.9.0（2026-08-27）：七方法传统 PU benchmark + AP/balanced-accuracy/Brier/ECE 指标 + KLDCE 原生 SMO
- 1.8.0（2026-08-21）：联合漂移研究求解器 + shift-monitor/review CLI + UI 部署面板
- 1.7.0（2026-08-21）：配对漂移适配 + 窗口告警 + 不确定性/主动复核
- 1.6.0（2026-08-21）：分布漂移审计 + 协变量加权 + shift-audit CLI
- 1.5.1（2026-08-16）：CNN 序列化、PUTuner 坏参数隔离、UI 历史持久化
- 1.5.0（2026-08-15）：classifier_params + PUTuner + Streamlit UI

### 收尾统计

- **算法**：全部注册方法为 native 实现（当前集合见 registry / `pu-toolbox list-methods`）
- **质量门禁**：清单见 [README](../../README.md) 的 Quality gates 与 CI 的 quality job
- **v1 范围外**：Phase 2 三个经典包装器 + TIcE/AlphaMax 类先验估计
- **依赖外部**：Phase 3 官方历史环境、WConPU CUDA/授权数据、DGPU EDM/CelebA 全量运行

历史执行记录见 git log；关键决策见 [`docs/adr/`](../adr/)。
