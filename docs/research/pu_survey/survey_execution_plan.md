# PU 调研实验执行计划（pilot → 主榜）

> 定位：本文件是**执行路线与状态**，与协议的承接关系——
> [pu_survey_protocol.md](pu_survey_protocol.md) 是要求纲要，
> [implementation_plan.md](implementation_plan.md) 是现状差距与技术实现维度；
> 状态日期：2026-09-08。

## 1. 现状
- **实验层完备**：`pu_toolbox/experiment/` 34 个公共API，四路数据合约、PA/OA 独立选模、策略化接口、
  数据准备链（datasets/image/text/feature_adapter/training_views）、资源计量与失败语义、公平性门禁均已实现并入门禁覆盖。
- **22 目标方法**：7 个已实现可训练——uPU、nnPU、KLDCE、Dist-PU、PUSB、LBE、Self-PU（均有方法卡）；
  14 个**未出现**（无注册/无占位/无方法卡）：A 类 PAN、GEN-PU、PULNS、RP、CVIR、Holistic-PU、P3MIX，
  B 类 VPU、PULDA，C 类 PUET、Grad-PU、Robust-PU、Split-PU、LAGAM；第 22 个即 PN oracle（下述）；
  `api_only` 0 个。
- **PN oracle（2026-09-11 接入 MLP 路径）**：无注册项；经 `CleanLabelGenerator`（真实标签透传、
  `output_view="clean"`）+ `SupervisedTrainer` + 仅 OA 协议接入，脚本入口
  `run_survey_experiment.py --oracle`。CNN（图像）路径的 clean-val checkpoint 选择留待 Phase 2。
  详见 [pn_oracle_integration.md](pn_oracle_integration.md)。
- **能力声明现状**：代码级声明仅 `native_architectures`/`input_ndims`/`encoder_parameter`/`trains_encoder`
  四字段（有契约测试）——仅 nnPU 为双架构（mlp/cnn、{2,4}、encoder 注入），Self-PU 为 mlp/{2,4}，
  其余 5 个默认 tabular-only（{2}）——图像数据集上它们必须走 `cnn_feature_adapter`（benchmark-adapted）。
- **协议要求但还未实现/未声明的项**：
  1. 台账 6 字段（`native_sampling_assumption`/`run_view`/`calibration_applied`/`prior_semantics`/
     `adaptation_level`/模态与 backbone）在代码与方法卡中**均未声明**（仅协议文字 + 实验层 per-call 机制）
  2. ~~官方示例脚本（协议 §2.4 第 9 条）未实现~~ ✅ 2026-09-08 已实现（见 §2.2 的 1.3 先行小工作）
  3. 中心超参数注册表（实现计划 §6，参考 PU-Bench `core/hparams_registry.py`）未实现；
     当前候选池仅为 runner `config["candidates"]` 的运行态配置
  4. **标签语义无声明字段**（`fit` 的 `y` 是 PU / 监督 PN / PNU）：PU 的 `{1,0}` 与监督的 `{0,1}`
     数值同形，当前**仅靠约定区分、错配静默**。归 P3 接入前置，不阻塞 P2——
     见 [label_semantics_plan.md](../../dev/label_semantics_plan.md)

## 2. 推进路线

### 2.0 分工、依赖与验收（v2）

实施主体：**shuidisjtu**（数据、实验编排、结果留痕与文档）和 **HENG958**（算法接入、深度训练与 GPU 执行）。每项只有一名**主责**；协作者须在交付前完成复核。任务完成必须有测试、manifest、运行记录或 PR 链接等可复核证据，不能只以口头或代码存在判定完成。

| 编号 | 任务 | 前置 | 验收标准 | 状态 / 主责 |
|---|---|---|---|---|
| P1.1 | 环境与 GPU 验证 | — | `uv.lock` 可复现；目标环境完成 GPU smoke；版本、设备与验证记录可追溯 | ✅ 已完成 / shuidisjtu；HENG958 复核其执行环境 |
| P1.2 | 数据获取与版本审计 | — | 数据来源、版本、标签映射与许可记录入 manifest；ADNI 的准入状态明确 | 🚧 进行中 / shuidisjtu |
| P1.3a | 方法台账 | — | 7 个已实现方法的六字段有证据；每项经对应方法负责人复核 | ✅ 初版完成 / shuidisjtu；HENG958 复核 nnPU、Self-PU |
| P1.3b | 官方 Survey 脚本 | — | 四路输入、PA/OA、结果归档与 oracle 入口均有脚本级测试 | ✅ 已完成 / shuidisjtu |
| P1.4 | Pilot 数据产物 | P1.1、P1.2 | CIFAR-10、IMDB、Spambase 各 5 个 seed 的四路 split、预处理与 manifest 均通过合同验证 | ✅ 已完成 / shuidisjtu；HENG958 复核可执行性 |
| P2.0a | Pilot 共享规格与 oracle 对齐决策 | P1.3a、P1.3b | 共享 backbone/预算、训练路径分组与 PN oracle 对齐方式书面锁定 | 🚧 未完成 / HENG958；shuidisjtu 复核 |
| P2.0b | 标签语义门禁 | P1.3a、P1.3b | `label_semantics` 声明与 experiment/pipeline 检查点按计划完成，错误组合 fail-loud | 🚧 未完成 / shuidisjtu；HENG958 复核 |
| P2.1 | Pilot 跑批与运行制品 | P1.4、P2.0a、P2.0b | 每个计划单元产生完整 manifest、选择 artifact、资源/失败记录；oracle 按 `(dataset, seed)` 去重 | ⏳ 待办 / HENG958 |
| P2.2 | Pilot 聚合与审计 | P2.1 | 发布 `pilot / partial benchmark` 分层结果；检查路径隔离、复现字段和异常单元；不得生成跨数据集总排名 | ⏳ 待办 / shuidisjtu；HENG958 复核深度结果 |
| P3.1 | 缺失方法接入（经典/B 类） | P2.0a、P2.0b | 每方法完成实现、方法卡、台账、原文可追溯、冒烟与公开行为对照；使用已锁定的共享规格 | ⏳ 待办 / shuidisjtu：VPU、PULDA、PAN、RP、CVIR、PULNS |
| P3.2 | 缺失方法接入（深度/C 类） | P2.0a、P2.0b | 同 P3.1，另需 GPU smoke、设备/随机性与保存加载验证 | ⏳ 待办 / HENG958：PUET、Grad-PU、Robust-PU、Split-PU、LAGAM、GEN-PU、Holistic-PU、P3MIX |
| P3.3 | 深度 GPU 验证与调度 | P3.2 | GPU 预约、显存预算、失败/OOM 重试及结果路径均有记录；不与 P2.1 竞争同一窗口 | ⏳ 待办 / HENG958 |
| P4.1 | 中心超参数注册表 | 各方法候选参数已确定 | 候选池预注册、版本化；版本写入 artifact 并受 manifest 校验 | ⏳ 待办 / shuidisjtu |
| P4.2 | 主榜聚合与分析 | P3.1、P3.2、P3.3、P4.1 | 22 项全部通过门禁后，按四组结果和训练路径分层；结论区分文献事实、实验观测与推断 | ⏳ 待办 / shuidisjtu；HENG958 复核 C/A 深度结论 |

**依赖与升级规则**：P2.1 可在基础设施可用后做技术 smoke，但未完成 P2.0a 与 P2.0b 前不得将结果与 oracle 或跨方法结果混排；若为验证脚本而提前执行，产物必须标记为“需按 P2.0 规格重跑”。P2.1 与 P3.3 共享单卡时，HENG958 负责排定并记录 GPU 窗口；数据许可、共享规格、标签语义或资源不足造成阻塞时，主责须在计划的“开放问题与风险”中记录影响与下一步，并由两位实施主体共同决定升级、拆分或降级。Survey 分工在其范围内覆盖 ADR-0008 中较早的论文分配。


### P1 执行准备（pilot 前完成）

- **1.1 环境 ✅（2026-09-08）**：T600（驱动 596.52）上 pytest -m gpu 真实验证通过；
  `uv sync --extra research --extra dev`（torch 2.14.0+cu126 / torchvision 0.29.0+cu126 / lightning /
  sentence-transformers 5.7.0 / densratio / pytest / ruff）+ SBERT 模型落地
  （`all-MiniLM-L6-v2`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` 锁定、内容寻址缓存生效）；
  依赖由 `uv.lock` 固化（见 D3）。
- **1.2 数据获取**：种子子集先行 = `CIFAR-10 + Spambase + IMDB`（1 图 + 1 表 + 1 文）；各数据集获取渠道——MNIST/F-MNIST/CIFAR-10 自动下载，
  20News `sklearn.fetch_20newsgroups`，Spambase/Connect-4 UCI，IMDB HF/原始文本，ADNI 特殊（见 D1）。
- **1.3 先行小工作**（pilot 前，工作量小收益大）：
  1. **方法台账 JSON ✅（2026-09-08）**（`pu_toolbox/experiment/method_ledger.json`，程序真相源）：7 个已实现方法 × 6 字段，值按方法卡/论文
     填写，存疑处显式标注；实验脚本读取（TS-OS 判定/结果标注"原生适用 vs 假设违背鲁棒性"的依据）。
     【采用独立 JSON 方案；方法卡节、注册表字段扩展后期可做。经典 5 方法（uPU/KLDCE/Dist-PU/PUSB/LBE）
     已由 shuidisjtu 填写，nnPU/Self-PU 骨架+证据立好，待 HENG958 复核。】
  2. **官方示例脚本 ✅（2026-09-08）**（协议 §2.4 第 9 条交付物 + pilot 执行载体；
     `scripts/run_survey_experiment.py`）：
     读取四份数据 → c/candidates 配置 → `ExperimentRunner` → PA/OA → 结果归档；台账驱动先验必传判定；
     PN oracle 留待 P2 引入。
- **1.4 pilot 数据产物 ✅（2026-09-08）**：种子 3 集 × 5 seeds 四路切分完成
  （`data/splits/<dataset>/split_<seed>/` 四份 npz + split_manifest.json；
  `scripts/prepare_survey_splits.py` 生成，产物经 15 套合同断言 + 示例脚本端到端冒烟验证）。
  模态预处理：Spambase z-score(float32)、IMDB SBERT 384-d(revision 锁定)、
  CIFAR-10 uint8 NCHW + train-only 通道统计（统计均冻结入 manifest）。切分执行口径见 D2。

### P2 pilot 实验（结果一律标注 `pilot / partial benchmark`）

- 7 方法 × 3 数据集 × c∈{0.1,0.3,0.5} × 5 seed × PA/OA 双协议；PN oracle 对照
  （**跑批去重**：oracle 结果对 c 恒定，每 (dataset, seed) 跑 1 次共 15 次，
  聚合时广播到各 c 列并标注 `c_independent`；脚本用 `--oracle`）
- 模态-方法矩阵：表格/文本上 7 法均原生（文本＝SBERT 384 维特征 + MLP）；
  图像端到端仅 nnPU、Self-PU 原生（CNN）；其余 5 法图像走 `cnn_feature_adapter`
  （基准-适配、与原生路径**强制分组**，不混合排名）
- 候选池：pilot 阶段用论文默认参数 + 少量合手候选（中心注册表到 P4 引入）
- **依赖**：oracle 与各 PU 方法须在同一 backbone 规格下比较（协议 §2.5 第 4 条）——该规格
  属 P3 前置（见下）。pilot 若在其确定前启动，oracle 行须标注自身结构并单列，待规格落定后重跑
- 产出：pilot 榜单（SCAR-PA / SCAR-OA），交付前提=验证全链路（环境、数据、runner、manifest、资源计量）正常

### P3 算法接入（与 P2 并行推进；分工作如下：表 2.0）

- **前置：确定数据集内共享的 backbone 规格**（表格/文本 MLP：层数、宽度、激活、优化器、
  epoch 预算；图像 ResNet-18 已在协议 §2.5 锁定）。这是**方法接入的接口约定**——每个方法接入
  时都要按它实现，所以归属 P3，**不由 P4 的中心注册表承担**（注册表管的是超参候选池）。
  已实现的 7 个方法需回溯对齐：其表格路径默认各为 `nn.Linear(d, 1)`（深度类）或非网络
  实现（经典类），尚未共享同一规格
- **前置：标签语义声明**（`label_semantics`：`fit` 的 `y` 属 PU / 监督 PN / PNU）。同属接入时的
  接口约定——错配（如把 PU 估计器喂真实标签）当前静默，Phase 2 的深度 oracle 最易踩。
  方案、检查点与参考文献 2 的对照见 [label_semantics_plan.md](../../dev/label_semantics_plan.md)；
  P1 阶段（声明位 + 门禁，无行为变化）可与方法接入并行，P2 阶段须在 Phase 2 之前完成
- 14 个缺失方法的接入顺序：B 类（VPU、PULDA，风格接近已有 B 类）→ A 类（7 个，依赖论文及其源码复现）
  → C 类（5 个，深度/优化设计，需 GPU 验证）；具体主责以表 2.0 为唯一真相源。
- 每方法 = 实现 + 方法卡 + 台账登记（方法台账 JSON 同步更新）+ 门禁（原文可追溯、冒烟、
  公开结果对照，协议 §5）；**接入验收须确认使用共享 backbone 规格**，方法私有网络只能标
  `benchmark-adapted` 单列报告（协议 §2.5 第 4 条）；
  TS 原生方法在其训练循环内接入 `calibrate_ts_os_batch`

### P4 完整主榜

- 中心超参数注册表（参考 PU-Bench `core/hparams_registry.py`，注册表版本入 artifact）：
  管理各方法的**超参候选池**；backbone 规格由 P3 的共享规格确定，不在注册表内
- 22 方法全部通过四项门禁 → 四组榜单（SCAR-PA 主榜 / SCAR-OA / SAR-OA / PN oracle）→ §5.7 分析
- 发布条件（协议 §4 原文）：完整主榜以 22 个目标方法全部通过相应门禁为发布条件；
  在此之前只可发布明确标为 `pilot / partial benchmark` 的部分结果。

## 3. 决策记录（需讨论后确定）

| # | 决策 | 内容 | 日期 |
|---|---|---|---|
| D1 | **ADNI 数据获取** | **该数据集的获取似乎比较麻烦**，我会咨询一下学长。我目前的查证结论是（2026-09-08）：需通过 ADNI LONI 官网（adni.loni.usc.edu）在线申请——科研机构身份 + 接受数据使用协议（DUA）+ 研究用途描述，由 ADNI 数据共享与出版委员会（DPC）评审约 1-2 周，批准后经 LONI IDA 下载；限制：不得商用/重新分发、年度更新。决定后若申请通过，ADNI 加入后续实验矩阵；届时矩阵按"7+1"处理 | 2026-09-08 |
| D2 | 协议分工执行口径 | 数据准备协议 §2.4"工具箱不负责切分"字面与参考实现并存，产生了一个矛盾点——我会修改/补充协议的说法，并和学长说一声 | 2026-09-08 |
| D3 | 依赖锁与 CUDA 环境落地 | **uv.lock 已入库**（2026-09-08，chore(deps)）：替代 requirements.txt，PR 快层 CI 用 lock 确定性、nightly `--no-lock` 重新解析验证"最新可解析"（ADR-0012 修订）。**torch CUDA 配置**：pyproject `[tool.uv.index] pytorch-cu(cu126)` + `[tool.uv.sources]` 仅 `sys_platform=='win32'` 生效（CI Linux/macOS 保持 PyPI CPU 版；win 上 torch 2.14.0+cu126）；torchvision 并入 torch extra。多环境（T600/HENG958 主力机）由此保持一致 | 2026-09-08 |

## 4. 存在的开放问题与风险

1. **GPU 算力/显存**：shuidisjtu 本机 T600（4GB）不够强，所以主要进行轻量批与开发验证的工作，
   显存不足时（批大小/并行）需在实验记录中说明资源限制；
2. **数据集获取确认**：20News/IMDB/Connect-4/Spambase 的版本与标签编码需与协议锁定映射核对；
   数据版本或标签编码变化时必须提供映射转换审计记录（协议 §2.2 要求）
3. **结果可解释性**：pilot 结果必须区分"文献事实/本实验观测/推断"（协议 §5.7 要求，P2 起执行）

## 5. 留痕约定

- 每个实验产物：manifest 固化（协议 §5.6：代码 commit、依赖锁文件、Python/PyTorch/CUDA/GPU、
  数据与方法配置、注册表版本、seed、split/label manifest、选择 artifact、schema 版本）
- 结果按四组存储（SCAR-PA / SCAR-OA / SAR-OA / PN oracle），跨数据集只比较趋势，不生成总排名
- 本计划随执行更新：每完成一个阶段、每产生一个决策，更新 §1/§2/§3 相应条目
