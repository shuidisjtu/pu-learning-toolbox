# PU 调研实验执行计划（pilot → 主榜）

> 定位：本文件是**执行路线与状态**，与协议的承接关系——
> [pu_survey_protocol.md](pu_survey_protocol.md) 是要求纲要，
> [implementation_plan.md](implementation_plan.md) 是现状差距与技术实现维度；
> 状态日期：2026-09-08。

## 1. 现状
- **实验层完备**：`pu_toolbox/experiment/` 34 个公共API，四路数据合约、PA/OA 独立选模、策略化接口、
  数据准备链（datasets/image/text/feature_adapter/training_views）、资源计量与失败语义、公平性门禁均已实现并入门禁覆盖。
- **22 目标方法**：7 个已实现可训练——uPU、nnPU、KLDCE、Dist-PU、PUSB、LBE、Self-PU（均有方法卡）；
  15 个**未出现**（无注册/无占位/无方法卡）：A 类 PAN、GEN-PU、PULNS、RP、CVIR、Holistic-PU、P3MIX，
  B 类 VPU、PULDA，C 类 PUET、Grad-PU、Robust-PU、Split-PU、LAGAM，以及 PN oracle（无注册项，
  但实验层 `SupervisedTrainer` 作为基于真实标签的单点 fit 实现）；`api_only` 0 个。
- **能力声明现状**：代码级声明仅 `native_architectures`/`input_ndims`/`encoder_parameter`/`trains_encoder`
  四字段（有契约测试）——仅 nnPU 为双架构（mlp/cnn、{2,4}、encoder 注入），Self-PU 为 mlp/{2,4}，
  其余 5 个默认 tabular-only（{2}）——图像数据集上它们必须走 `cnn_feature_adapter`（benchmark-adapted）。
- **协议要求但还未实现/未声明的项**：
  1. 台账 6 字段（`native_sampling_assumption`/`run_view`/`calibration_applied`/`prior_semantics`/
     `adaptation_level`/模态与 backbone）在代码与方法卡中**均未声明**（仅协议文字 + 实验层 per-call 机制）
  2. 官方示例脚本（协议 §2.4 第 9 条）未实现（examples/ 无四份数据 + PA/OA 脚本）
  3. 中心超参数注册表（实现计划 §6，参考 PU-Bench `core/hparams_registry.py`）未实现；
     当前候选池仅为 runner `config["candidates"]` 的运行态配置

## 2. 推进路线

### 2.0 分工与并行性（v1）

实施主体假设：**shuidisjtu**（执行/数据/文档）+ **HENG958**（算法接入/深度方向）。并行分工如下：

| 任务 | 依赖 | 建议分工 | 并行性 |
|---|---|---|---|
| 1.1 环境 | — | shuidisjtu（开发/冒烟验证）；HENG958（P2/P3 ），基本的环境部署两个人都需要完成 | 与其它全部互不等待 |
| 1.2 数据获取+版本审计 | — | shuidisjtu | 与 1.3/P3 并行；仅 1.4 依赖 |
| 1.3① 台账 JSON | — | shuidisjtu 统稿+经典 5 方法；HENG958 填 nnPU/Self-PU | 可分两半并行填，最后合并 |
| 1.3② 示例脚本 | （骨架可先写） | shuidisjtu | 与 1.2/P3 并行 |
| 1.4 pilot 数据产物 | 1.1、1.2 | shuidisjtu（数据/流水线）；执行（HENG958） | 关键路径；3 集可分两批（CIFAR/Spambase 先行，IMDB SBERT 之后） |
| P2 pilot 跑批 | 1.3②、1.4 | **HENG958（执行）**；shuidisjtu（脚本/结果组织与分析） | GPU 独占窗口；期间两人在开发侧并行（P3 编码/台账/文档） |
| P3 方法接入 | — | shuidisjtu：B 类（VPU、PULDA）+ A 类经典（PAN、RP、CVIR、PULNS）；HENG958：C 类（PUET、Grad-PU、Robust-PU、Split-PU、LAGAM）+ A 类深度（GEN-PU、Holistic-PU、P3MIX） | 每方法独立"实现+方法卡+台账"循环，两人并行；`feature/<method>` 分支→PR，错峰合入避免 registry/api.md 冲突 |
| P3 GPU 验证（深度冒烟） | 与 P2 错开 | HENG958 | GPU 时间表轮换（单卡） |
| P4 中心注册表 | P3 各方法候选参数 | shuidisjtu | 以先接入方法先行验证，与 P3 收尾并行 |
| P4 榜单聚合/§5.7 分析 | 全部结果 | shuidisjtu（榜单+留痕）；HENG958（C/A 深度类别结论）、shuidisjtu（A/B 类别结论） | 分析按类别拆，shuidisjtu 最后合并 |

**唯一串行关键路径**：`1.1 → 1.2/1.3 → 1.4 → P2 跑批 → P4`；其余均可并行。


### P1 执行准备（pilot 前完成）

- **1.1 环境**：GPU 机器（驱动/CUDA 与 torch 匹配，`pytest -m gpu` 真实验证，先例 T600【这是我电脑上CUDA的型号】）+
  `uv sync --extra research`（torch/torchvision/lightning/sentence-transformers/densratio）+ 
  SBERT 模型落地（`all-MiniLM-L6-v2`，revision 显式锁定、内容寻址缓存）；依赖由 `uv.lock` 固化。
- **1.2 数据获取**：种子子集先行 = `CIFAR-10 + Spambase + IMDB`（1 图 + 1 表 + 1 文）；各数据集获取渠道——MNIST/F-MNIST/CIFAR-10 自动下载，
  20News `sklearn.fetch_20newsgroups`，Spambase/Connect-4 UCI，IMDB HF/原始文本，ADNI 特殊（见 D1）。
- **1.3 先行小工作**（pilot 前，工作量小收益大）：
  1. **方法台账 JSON**（`method_ledger.json`，程序真相源）：7 个已实现方法 × 6 字段，值按方法卡/论文
     填写，存疑处显式标注；实验脚本读取（TS-OS 判定/结果标注"原生适用 vs 假设违背鲁棒性"的依据）。
     【我的想法是：采用独立 JSON 方案；方法卡节、注册表字段扩展后期可做。】
  2. **我们需提供的示例脚本**（协议 §2.4 第 9 条交付物 + pilot 执行载体；落地于 `scripts/`，文件名随实现确定）：
     读取四份数据 → c/candidates 配置 → `ExperimentRunner` → PA/OA → 结果归档。
- **1.4 pilot 数据产物**：种子 3 集按模态流水线（文本 SBERT 编码+缓存；图像 train-only 通道统计；
  切分 produce split manifest）。切分执行口径见 D2。

### P2 pilot 实验（结果一律标注 `pilot / partial benchmark`）

- 7 方法 × 3 数据集 × c∈{0.1,0.3,0.5} × 5 seed × PA/OA 双协议；PN oracle 对照
- 模态-方法矩阵：表格/文本上 7 法均原生（文本＝SBERT 384 维特征 + MLP）；
  图像端到端仅 nnPU、Self-PU 原生（CNN）；其余 5 法图像走 `cnn_feature_adapter`
  （基准-适配、与原生路径**强制分组**，不混合排名）
- 候选池：pilot 阶段用论文默认参数 + 少量合手候选（中心注册表到 P4 引入）
- 产出：pilot 榜单（SCAR-PA / SCAR-OA），交付前提=验证全链路（环境、数据、runner、manifest、资源计量）正常

### P3 算法接入（与 P2 并行推进；分工作如下：表 2.0）

- 15 个缺失方法，我的建议顺序：B 类（VPU、PULDA，风格接近已有 B 类）→ A 类（7 个，依赖论文及其源码复现）
  → C 类（5 个，深度/优化设计，需 GPU 验证）：至于这里的任务分配，我还没有确定好，还需要评估
- 每方法 = 实现 + 方法卡 + 台账登记（方法台账 JSON 同步更新）+ 门禁（原文可追溯、冒烟、
  公开结果对照，协议 §5）；TS 原生方法在其训练循环内接入 `calibrate_ts_os_batch`

### P4 完整主榜

- 中心超参数注册表（参考 PU-Bench `core/hparams_registry.py`，注册表版本入 artifact）
- 22 方法全部通过四项门禁 → 四组榜单（SCAR-PA 主榜 / SCAR-OA / SAR-OA / PN oracle）→ §5.7 分析
- 发布条件（协议 §4 原文）：完整主榜以 22 个目标方法全部通过相应门禁为发布条件；
  在此之前只可发布明确标为 `pilot / partial benchmark` 的部分结果。

## 3. 决策记录（需讨论后确定）

| # | 决策 | 内容 | 日期 |
|---|---|---|---|
| D1 | **ADNI 数据获取** | **该数据集的获取似乎比较麻烦**，我会咨询一下学长。我目前的查证结论是（2026-09-08）：需通过 ADNI LONI 官网（adni.loni.usc.edu）在线申请——科研机构身份 + 接受数据使用协议（DUA）+ 研究用途描述，由 ADNI 数据共享与出版委员会（DPC）评审约 1-2 周，批准后经 LONI IDA 下载；限制：不得商用/重新分发、年度更新。决定后若申请通过，ADNI 加入后续实验矩阵；届时矩阵按"7+1"处理 | 2026-09-08 |
| D2 | 协议分工执行口径 | 数据准备协议 §2.4"工具箱不负责切分"字面与参考实现并存，产生了一个矛盾点——我会修改/补充协议的说法，并和学长说一声 | 2026-09-08 |

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
