# 不依赖合作者实验数据的独立推进（2026-10-05）

按[后续优先级](post_pilot_priority_plan.md)继续。这里只记录工程预集成与合成数据证据，
不代填方法负责人签署，不接受未收到的原始实验制品，也不修改历史冻结协议或对照矩阵。

## 已实施

| 内容 | 实际交付 | 明确边界 |
|---|---|---|
| PULDA 两阶段磁盘估算 | 新增 `checkpoint_epoch_count=warmup_epochs+pu_epochs`，runner 的无显式 epoch 预算分支可识别；显式协议预算优先 | 回收只减少最终留存，不减少跑批峰值；正式候选预算仍待确定 |
| PULDA / PUET 存储探针 | `scripts/profile_survey_candidate_storage.py` 输出严格 JSON；PULDA 两阶段权重恢复与两方法可信内存 pickle 恢复均逐预测比较 | 合成 48×3 特征、CPU、短预算；不是真实 CIFAR 存储上界或正式 GPU 资源审计 |
| PAN | 原论文式 (7) 的 D 最大化 / C 最小化、梯度隔离、稳定 logit 形式；独立 MLP 分类器和判别器 | 2-D 特征；原生 OS；拒绝未实现的 TS 风险替换；不是原论文图像数值复现 |
| Rank Pruning (`rp`) | 官方相关实现的 PU 特例：折内 OOF 噪声估计后求平均、排序剪枝、最终正例逆噪声权重 | 固定负标签噪声 `rho0=0`；不冒充一般双向标签噪声算法；原生 OS，未实现 TS |
| PULNS | 3 块状态、Bernoulli 动作、截断 log-odds 奖励、折扣回报及 REINFORCE；策略更新后重新选负例；reward probe 与最终 best classifier 分开 | 显式独立 clean support；支持身份交集检查，无身份时隔离仅由调用者保证；额外标签预算使普通 PU-only PA-ineligible |
| 同步 | 三方法 registry、公开导出、API、方法卡、方法台账与 capability/来源/采样契约 | 台账由 16 条增至 19 条；新增三方法负责人审核仍 pending；旧 8 方法准入草稿保持原范围 |

PULNS 不把 support 行送入 classifier 的梯度 minibatch，但 support 的真值会影响策略奖励和
内部 best-model 选择，因此**仍是训练监督**。不能复用实验 PA/OA selection 或 test 分区；
仅检查 train/support 身份不等于已证明与其它角色隔离。当前通用 runner 未提供其独立 support
路由，不把注册成功等同于可以直接运行冻结 Survey pilot。

## 来源与适配

- PAN：[AAAI 2021 原论文](https://ojs.aaai.org/index.php/AAAI/article/view/16953/16760)，
  式 (7)、Algorithm 1、§5.1 脚注 5；未核实作者官方代码，`source_status=not_found`
  只表示本次检索结果，不声称不存在。
- RP：[作者仓库](https://github.com/cgnorthcutt/rankpruning/tree/40ae17cc50fcfeec8d7a345a1c5426036d2bdb1a)，
  `rankpruning/rankpruning.py` 的 OOF、PU-only 噪声约束、排序剪枝与加权；采用独立 clone
  防折间污染，并显式设随机种子。完整差异见[RP 方法卡](../method_cards/RP.md)。
- PULNS：[AAAI 2021 原论文](https://ojs.aaai.org/index.php/AAAI/article/view/17064)，
  pp.8786–8788 的状态/奖励/算法及独立 P、U 构造；没有将原生 case-control 记成 OS。
  默认 OS 适配未做等价风险校准，显式 TS 请求拒绝；support 标签预算必须另立口径。
  论文推导组件不是作者源码重放，probe 拷贝与 Adam 重建等适配见[PULNS 方法卡](../method_cards/PULNS.md)。

三方法没有可安全直接替换的同形 U 风险项；台账均 `calibration_hooked=false`，
拒绝 TS 而非静默降级。不能据注册数量声称“校准全部完成”。

## 可重复检查

```bash
PYTHONPATH=. python scripts/profile_survey_candidate_storage.py
pytest -q tests/unit/estimators/test_pan.py tests/unit/estimators/test_rank_pruning.py \
  tests/unit/estimators/test_pulns.py tests/unit/experiment/test_multistage_disk_preflight.py \
  tests/unit/scripts/test_profile_survey_candidate_storage.py \
  tests/contract/test_capability_declarations.py tests/contract/test_classifier_baseline.py \
  tests/contract/test_ledger_registry_consistency.py tests/contract/test_p3_candidate_admission.py \
  -m 'not gpu'
```

存储探针 seed=0、input_dim=3 的实测：PULDA 2 个 epoch 权重各 2008 bytes，共 4016 bytes，
整体可信 pickle 3759 bytes；PUET 3 棵深度 3 的树 pickle 2259 bytes，无神经 epoch 轨迹。
序列化大小与 PyTorch/环境有关，测试不绑定此数值，更不外推正式候选上界。

首轮整体回归 **130 passed / 2 GPU deselected**（尚未计入后来增加的存储和空负例测试）。
Self-PU 既有无 clean-validation ablation 警告属预期披露，不是这三方法的完整数值验证。
首轮时新增 GPU 测试尚未执行；后续本机四方法 CUDA 证据见下文，不借用历史其它方法的 GPU 证据。

## 继续顺序

追加交付：GenPU 已完成独立可训练组件、台账、方法卡与公开 API；总体先验与 Du 的 TS-OS
训练池校准均接线。作者源码归档已锁定 sha256，论文与发布代码的明确分歧见
[GEN-PU 方法卡](../method_cards/GEN-PU.md)。不是未找到源码，也不是作者 demo 数值复现；
正式目标口径仍待负责人决定。GenPU 单项和通用 registry/API/台账回归 **115 passed / 1 GPU deselected**。
当前台账共 20 条（16 个既有 + PAN/RP/PULNS/GenPU）。

再追加：Holistic-PU 已补论文目标的可训练二维 MLP、台账、API 和[方法卡](../method_cards/Holistic-PU.md)，
单项与通用契约回归 **115 passed / 1 GPU deselected**。本地台账现为 21 条。
明确披露 signed 全时序评分 / variance 分割与作者简化邻差 / Jenks SSE 的差异，
固定预热并未完成 LZO。全部 contract 目录 **111 passed**。

路由修补：`os_or_ts` 参数存在并不一定代表 TS 风险替换已实现，PULNS 就是拒绝接口。
resolver 现在尊重台账显式 `calibration_hooked=false`：自动视图保持 OS，显式 TS 提前拒绝。
未声明此字段的旧方法行为保持不变；GenPU 真实 Du hook 自动走 TS，台账默认同步。
专用路由及历史路由/脚本/台账回归 **50 passed**。冻结 JSON 未修改。

P3MIX 已通过 ICLR 官方作者 slides 核查并补三个 batch 数学组件及
[方法卡](../method_cards/P3MIX.md)，但完整论文下载 403、作者源码未核实；
尚不注册 estimator、不新增“完成”的台账行，也不把基础 mixup 冒充 E/C 变体。

短时 CUDA 已独立验证：2026-10-05，物理 6 号 NVIDIA RTX A6000，驱动 550.54.14，
Python 3.12.2 / PyTorch 2.6.0+cu124 / CUDA 12.4。PAN、PULNS、GenPU、Holistic-PU
GPU 小样本/保存加载测试 **4 passed**（1.78s）；执行前后 6 号卡均为 1 MiB、0% 利用率，
不留常驻占卡进程。不是 Linux frozen-lock、多 seed 资源审计或正式数值跑批；RP 为 CPU 方法。

复核命令：

```bash
CUDA_VISIBLE_DEVICES=6 PU_REQUIRE_CUDA=1 pytest -q -m gpu \
  tests/unit/estimators/test_pan.py tests/unit/estimators/test_pulns.py \
  tests/unit/estimators/test_gen_pu.py tests/unit/estimators/test_holistic_pu.py
```

应先独立查卡是否空闲，不将本记录当未来 GPU 预约。

### 共享路径、runner 和存储补验

- 四角色 runner：PAN、RP、GenPU、Holistic-PU 正向运行及 manifest 检查通过；PULNS 未提供
  独立 support 时记录失败，selection/test 结果为空，不从其它角色偷取标签。共 **5 passed**。
- 共享冻结图像 encoder：已有 8 方法及新增 5 方法共 **13 passed / 5 GPU deselected**。
  PULNS 使用同一 encoder 编码独立第五角色，support 身份与四角色逐一不相交。
  这仍是合成图像的 2-D 接口验证，不是原生 CNN 训练或正式 representation 准入。
- 存储探针扩展至 7 方法及多个 seed，脚本回归 **14 passed**；21 份实测记录见
  [机器记录](data/candidate_storage_probe_20261005.json)。没有候选预算、显存峰值或正式选模声明。

| 方法 | seed 0/1/2 完整 pickle bytes | 更新步数 | 说明 |
|---|---|---|---|
| PULDA | 3759 / 3759 / 3759 | 4 / 4 / 4 | 两阶段共 2 个 epoch 权重、4016 bytes |
| PUET | 2259 / 1776 / 1862 | 不适用 | CPU 树结构随样本变化 |
| PAN | 4863 / 4863 / 4863 | 16 / 16 / 16 | 完整分类器与判别器可信序列化 |
| RP | 3184 / 3184 / 3184 | 不适用 | OOF 及最终 sklearn 拟合 |
| PULNS | 17906 / 17922 / 17994 | 24 / 25 / 28 | 独立 8 行 clean support；probe/实际更新均计数 |
| GenPU | 14213 / 14213 / 14213 | 36 / 36 / 36 | GAN 与 PN 阶段全部计数 |
| Holistic-PU | 4672 / 4672 / 4672 | 18 / 18 / 18 | 固定预热和伪标签阶段全部计数 |

探针样本为 48×3、CPU；耗时包含懒加载 import，首方法有冷启动成本，**不作速度排名**。
除 PULDA 外这次仅测完整可信 pickle，不把缺少 epoch 轨迹的记录说成 PA/OA checkpoint 验收。
序列化仅使用本次自己创建的对象，不加载下载源码归档中的 pickle。

### 后续独立工作

新增五方法准入交接草稿：[机器清单](data/p3_preintegration_extension_v1_draft.json)，
由 `scripts/prepare_p3_preintegration_extension.py` 只读生成，**6 passed**。
每方法直接引用台账来源/版本、原生假设、默认视图和适配差异，明确方法特有待决项；
PULNS 的独立标签预算及 PA-ineligible 不被省略，P3MIX 明确列为未注册组件。
预算/候选池/新协议为空、admitted=false、负责人决定为空；没有修改原八方法准入草稿或冻结矩阵。

测试质量补验：全仓库 `check_test_quality.py` 通过；测试名称与实际断言对齐，
补 CNN oracle、RP runner、共享 encoder 的真实 seed 复验，并为静态/跨模块测试登记
有对应其它测试文件的覆盖边界，删除已失效的台账 edge 豁免。
这些增量与审核脚本检查 **50 passed / 5 deselected**；P3MIX 候选身份/数学组件 **11 passed**。

最后一次扩大后回归：同上四目录 CPU 命令 **1511 passed / 24 deselected**（74.80s）；
另跑 builtin registry、公开导入、deep pipeline、CLI 参数及 info **70 passed**。
后者检出了高成本方法集合尚未包括四新增深度方法，已同步成本集合与多阶段预算说明；
保留 HIGH 元数据，未为让旧断言通过而降低成本类别。测试质量、全仓库格式、文档链接、
数学排版、80 项公开 API 和目录一致性检查均通过。
本批代码/文档保持本地未提交；没有提交或推送，也未将私人 Excel 或原始数据入库。

本批大范围回归：`pytest -q tests/unit/estimators tests/unit/experiment tests/unit/scripts
tests/contract -m 'not gpu'`，**1497 passed / 24 deselected**；未宣称其它测试目录全部覆盖。
文档链接、80 项公开 API、数学排版与全仓库 Ruff lint/format 门禁通过；既有 12 处
导入/类型写法与一处新增测试格式已机械修正，不改方法目标或旧结果。

冻结依赖预检：只读执行 `uv sync --frozen --dry-run --offline --no-cache --extra torch
--no-default-groups --no-install-project --python /opt/miniconda3/bin/python --no-python-downloads`，
退出 0、显示将装 40 包，包括 torch 2.13.0、torchvision 0.28.0、CUDA toolkit 13.0.3.0。
**dry-run 成功不是安装/运行成功**：无下载、无替换现有 `.venv`、无锁文件修改。
本机 550.54.14 驱动低于 NVIDIA 官方 CUDA 13.x minor compatibility 的 >=580，见
[官方兼容表](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html#cuda-driver)。
当前 cu124 smoke 与此锁文件环境不相同；未升级共享服务器驱动，也不擅自降级锁文件。
Linux frozen-lock GPU 验收仍待适配环境，历史审核/正式资格记录不改写。

P3MIX 再补训练候选池身份门禁：从观测训练 P 内选熵 top-k，拒绝无正例、重复/非法身份及
非法标签/预测；不会把高熵 U 当候选 P。仍未接完整 epoch/pool schedule，仍不注册 estimator。

1. P3MIX 完整论文/源码、pool 更新与 E/C 变体仍待核实；不注册假实现。
2. 新方法正式图像训练路径、完整 epoch 选模和资源上界仍待补；clean-support 方法必须走独立协议。
3. 已有方法来源行为与正式预算/选模草案继续补齐，再准备冻结依赖隔离 smoke。
4. 候选矩阵、PA 准则、support 标签预算及正式结果对照仍需负责人审阅；原始 645-run
   制品独立复算、OS/TS 配对差值不因这些工程交付而宣称已完成。

## 同日继续：原生图像路径与候选快照补齐

### PAN 原生 CNN

`PANClassifier(encoder=...)` 现支持四维图像；分类器/判别器分别 deepcopy 编码器，
独立训练编码器及头，不共享 BatchNorm 或优化器参数；不改调用方模板权重或 requires_grad。
四维输入没有 encoder 时拒绝，不静默 flatten。预测只切换分类器为 eval，恢复原模式；
空输入、完整空间 shape、encoder 输出与双网络 probe 均有校验。
公共 `PUPipeline(architecture="cnn", backbone="cnn13", classifier="pan")`、fold 隔离、
种子重跑、可信 pickle 和 epoch classifier 权重回放已验证。
能力声明、台账、API、方法卡与五方法交接草稿已同步；原作者架构/数值复现不作声明。

这轮顺带修复高成本方法提示：`refit=False` 只计 CV 折，不再多计全量重训。
PAN/CNN/台账/能力与 CV 针对性测试 **41 passed / 2 deselected**；
pipeline 与交接生成器测试 **25 passed**。

### Holistic-PU 与 GEN-PU 快照

Holistic-PU：warmup 与 pseudo_pn 每轮完成均调用 callback，epoch 连续零起始编号；
`history_["phase"]` 区分阶段，`checkpoint_epoch_count=warmup_epochs+max_epochs`。
捕获快照不改变种子/更新次数/伪标签目标；失败 callback 不吞异常、不标成功。
GEN-PU：只有合成 PN classifier 轮末有推断快照；GAN 阶段尚无可部署分类器，
不输出空壳快照。`checkpoint_epoch_count=classifier_epochs`、
`checkpoint_stage_="synthetic_pn"`；两阶段所有 optimizer steps 仍计训练成本。
两者 CPU 最后快照等于最终模型、早期快照不随最终模型修改而漂移；四角色 runner 接线通过。
专项 Holistic-PU/磁盘/runner **25 passed / 1 deselected**；
GEN-PU/探针/runner **32 passed / 1 deselected**。

阶段能否参与正式 PA/OA 选择、完整训练恢复和正式预算仍未完成；这些 hook 不代替预注册。
台账、方法卡、API 和交接机器草稿同步；PULNS clean-support 与 P3MIX 完整来源门禁不放宽。

### 新存储证据与回归

保留旧合成快照，新增 [v2 探针](data/candidate_storage_probe_20261005_v2.json)：
7 方法 × seeds 0/1/2，48×3 CPU，同样不作速度排名或正式资源上界。
每 seed：PAN 两轮分类器权重共 4016 bytes、完整 pickle 4931 bytes；
Holistic-PU 四轮权重共 8032 bytes、pickle 4782 bytes；
GEN-PU 一轮 PN 权重 2594 bytes、六网络 pickle 14248 bytes。
各快照及自己创建的可信 pickle 回放验证通过；不是 GAN/selector/optimizer/RNG 的恢复协议。
新图像路径真实磁盘/显存上界仍待测，不把二维微型模型的字节数推广为正式图像预算。

本轮补齐前大回归 **1520 passed / 25 deselected**；补齐后同四目录 CPU 回归
**1523 passed / 25 deselected**（74.78s）。三方法 GPU 训练→CPU 快照回放
**3 passed / 34 deselected**（1.74s），在当时空闲的物理 GPU 6 短时运行；不占用常驻卡。
全仓库 Ruff lint/format、测试质量、80 项 API、文档链接/数学排版、目录同步与 diff 检查通过。
最后补新增方法的回收接线：分别开启/关闭回收，核 PA/OA 所选 checkpoint 保留、
其余引用 reclaimed 与磁盘文件状态一致；RP 不捏造 epoch。该接线、两阶段磁盘预检、
PAN CNN pipeline 与折隔离 **20 passed**，不删除用户实验制品，仅测试各自临时目录。
本轮没有改冻结 protocol、comparison v1/v2/v3 或 uv.lock，没有 commit/push 或上传私人数据。

后续独立优先级：其它候选原生图像路径 → 已有适配来源行为/预算选模草案 →
P3MIX 完整来源可用后继续完整训练。负责人签署、原始 645-run 制品、配对实验数据仍独立待办。

## 再续：按 P3.1 优先补 PULDA 原生 CNN

PULDA 构造可注入 encoder，四维图像缺 encoder 拒绝；encoder deepcopy、独立解冻训练，
输出经有限二维特征契约校验，接原 MLP 头。稠密二维路径和两阶段目标继续保留。
图像/观测标签留 CPU，仅 P/U 与 MixUp batch 移到设备；伪标签刷新及推断分批 eval，
不更新 BatchNorm，预测恢复原模式；完整输入空间 shape 与空输入有门禁。
这是输入图像 MixUp 工程路径，非作者的完整 CIFAR 架构、增强/数据管线或论文数值复现。

新 CNN 单测检查 encoder 真更新、原模板冻结标志/权重不变、BN 仅训练 forward 更新，
两阶段 TS 总体风险角色计数、种子/clone/pickle、两阶段 checkpoint、batch 上界与空间形状。
公共 cnn13 pipeline 与 CV 折隔离亦已扩到 PULDA。
针对性 CPU 契约/台账/草稿/pipeline/折隔离 **57 passed / 2 deselected**。
方法卡、API、台账、原八方法准入草稿（范围仍八项、预算/null 与 admitted=false 保留）、
两方法规格草案和交接表均同步；没有改历史冻结协议/结果或提交私人数据。
先前 v1/v2 合成存储 receipt 保留为当时二维实现快照，不把其 bytes 推广到新 CNN；
正式 CNN 多 seed、frozen-lock 和真实资源上界仍未完成。

扩大回归：`tests/unit/estimators tests/unit/experiment tests/unit/scripts tests/contract`
的 CPU 集 **1533 passed / 26 deselected**（75.02s），新增 CNN GPU 集
**1 passed / 6 deselected**（2.29s）。GPU 测试还直接检查训练图像/标签 tensor 留在 CPU，
encoder 参数在 CUDA，最后权重可在 CPU 回放；没有启动正式跑批。
本地已提交 `1c76e04` 的 PULDA 与当前二维实现另做合成对照（30×3，seed=3，
warmup/PU 各 2 epoch，MLP4×1，batch P/U=4/8）：OS、TS 两次 logits 最大差均 0，
伪标签通过 1e-6 对照，各自更新次数均 12。此局部行为对照不等于作者源码数值复现。
格式/测试质量、80 API、链接、数学排版、目录及 diff 门禁通过；冻结协议/矩阵/uv.lock 无差异。

## 再续：Robust-PU / Split-PU 多阶段资源计数

核对真实 fit 分支后发现两个方法没有 max_epochs 或 checkpoint 总数声明，导致即便用户提供
bytes/component，runner 仍可能无法估算磁盘；也没有累计全部训练更新次数。
现在 Robust-PU 快照数量为 `pretrain_epochs+episodes`，inner_epochs 不产生额外快照；
Split-PU 上界为 `teacher_epochs+split_epochs+rounds*student_epochs`，早停只减少实际数量，
不能用于跑前低估峰值。两者 `optimizer_steps_` 覆盖全部 Adam/SGD 更新、每次 fit 重置。
训练目标、TS 的 warmup-only / teacher-only 范围、图像未实现及未正式准入声明均保留。

专项单测独立插桩优化器 step，对照累计数和 refit 重置；Robust 小例 22 更新、5 快照，
Split 小例 splitter 早停于 1 轮、4 快照，而上界 6。最后快照回放等于最终分类器，
捕获快照不改固定 seed 输出；snapshot TypeError 传播且不重复训练。
磁盘测试覆盖两方法上界/候选/重试乘数；显式冻结预算仍优先。
本批专项 **40 passed / 2 deselected**，不把这些合成数值当成论文行为/数值复现。
API 视图说明、方法卡、台账 checkpoint_accounting、八方法准入草稿同字段与目录说明同步；
`formal_budget_approved=false`、admitted=false/null 及旧冻结矩阵不改。

补齐后同四目录扩大 CPU 回归 **1541 passed / 26 deselected**（74.78s）；
台账/准入草稿契约另跑 **11 passed**。格式/测试质量、80 API、文档链接/数学、
目录同步与 diff 检查通过。本轮不涉及 CUDA 算子改变，未占卡或启动正式跑批。
未修改 protocol/comparison v1-v3/uv.lock，未提交或推送；正式存储 profile、来源数字对照、
CNN/增强完整路径及负责人审阅仍是待办，不因计数接口补齐而宣布 P3 完成。

## 再续：GradPU 主要来源及原实验协议核查

重新读取 AAAI 原文印刷页 7300：CNN13 明确移除 BatchNorm、灰度四层 MLP300，
batch250/三次运行/学习率渐降至0.1初值、灰度100/CIFAR200 epoch；实际实验从原训练集
划出 P 与500 validation 后剩余作为 U，而理论描述 U 为总体边缘。tanh 前梯度与 beta
线性增长也有同页证据。源事实与下一步接入约束已写入独立文档/机器清单，见
[GradPU 来源核查](gradpu_source_review_20261005.md)。
因此保留现有 BN 拒绝及理论 TS 声明，但明确 TS 校准不是原实验流程直接复现；
不静默删除默认 CNN13 的 BN，不把当前 MLP128/固定学习率说成论文配方。
台账增加来源核查指针与采样/骨干差异，八方法准入草稿同步指针，未动预算/null 或 admitted。
本次检索没有确认作者代码；同名点云上采样仓库不作为官方来源，not_found 保留。
无新增数值锚点/源码授权/签署；附录、validation 标签口径、无 BN CNN、正式规格仍待办。

现有 GradPU 行为/台账/准入契约 **27 passed / 1 deselected**；新增核查包状态断言后准入契约
另跑 **2 passed**，仅证明 pending/null/未准入状态，不自动证明人工原文读数。
格式、文档引用、目录与 diff 检查通过；冻结协议/矩阵/uv.lock 无改动。本轮未提交或推送。

## 再续：GradPU 无 BatchNorm 原生图像路径

`GradPUClassifier(encoder=...)` 接收无 BN 图像编码器，deepcopy/解冻训练，有限二维特征
接默认 MLP 或用户 raw-score 头；四维无 encoder 拒绝，BN 继续拒绝。
工具箱新增**显式** `cnn13_no_bn`（13卷积布局变体），默认 cnn13 保留原 BN/拓扑，
不静默替换也不声称作者完整网络。共享骨干列表让 config/pipeline/CLI/UI 一致接受新选项。
风险 U 的 TS 替换和插值均保留，插值 tensor 为原 NCHW 图像、梯度 create_graph=True。
数据仍留 CPU，优化批次上设备；预测分批 eval/模式恢复、完整空间 shape 与空行校验。

独立解析卷积例子验证输入梯度平方范数与卷积权重的二阶反传；真实拟合检查原图 tensor、
编码器更新和模板未污染、TS 风险池、种子/clone/pickle、epoch CPU 回放、公共 pipeline
实际骨干留痕及不同 fold 的权重隔离。专项 **42 passed / 2 deselected**；
单项 GPU **1 passed / 7 deselected**（2.07s），空闲物理 GPU 6 短时运行。
现有二维输入门禁已更新为缺 encoder 明确拒绝；预测错误仍保留 fitted 2-D/4-D 指示。

方法台账/能力、八方法准入草稿、来源复核 blocker 的正式规格措辞、方法卡、API 和
CLI/pipeline 骨干说明已同步。原生工程路径完成不代表来源/正式调参/图像预处理已复现，
签署/null/未准入状态及旧冻结矩阵不改。本轮没有提交或推送。

最终同四目录 CPU 扩大回归 **1548 passed / 27 deselected**（75.76s）；
配置/CLI 参数/模型配置/注册表另跑 **37 passed**。
早期回归只因旧预测错误文本断言失败，已保留 fitted 2-D/4-D 指示并重新完整回归通过，
没有删除形状门禁或降级梯度惩罚来通过测试。
全仓库格式/测试质量、80 API、文档链接/数学排版、目录与 diff 门禁通过。
GPU 6 测后回到 1MiB/0%，没有常驻占卡；冻结协议/比较矩阵/uv.lock 没有差异。

## 再续：Robust-PU 两阶段原生图像工程路径

Robust-PU 支持注入 encoder 的独立可训练副本，二维与四维输入接同一 raw-logit 头。
图像、标签、自步权重与 EMA 留在 CPU，优化批次才转设备；权重按原始行顺序计算，
原始 U 的后续伪负例池没有并入 P。TS 仍只作用于预训练风险，不扩大校准范围。
probe、权重扫描和预测在 eval 下执行，不污染 BatchNorm；模板权重和冻结标志不变。
形状、空预测、固定种子、clone/pickle、两阶段 epoch 快照和 CPU 回放均有测试。

专项检查的小例累计 5 次优化更新、7 次 BN 训练前向，符合两阶段真实路径。
GPU 6 的短时测试 **1 passed / 6 deselected**（2.06s），测后 1MiB/0%，没有常驻占卡。
完整四目录 CPU 回归 **1554 passed / 28 deselected**（80.82s）；台账/能力/准入契约
另跑 **23 passed**。原有全批预测与分批预测断言改为 atol=1e-7，以允许 float32 GEMM
产生的约 7.45e-9 舍入差，风险目标与梯度测试未放宽。

方法卡/API、能力声明、台账、八方法准入草稿、优先级与目录说明同步。
这只是原生图像工程能力，不是作者 CIFAR 增强/scheduler/完整预算的复现；
admitted=false、正式资源预算未批准、旧冻结矩阵不改，未提交或推送。

## 再续：Split-PU 分批资源路径与原始 U 边界

在接图像表征之前先移除全数据 GPU 常驻和整批教师扫描：数据保存在 CPU，
教师风险/splitter/student 优化只搬入当前批；冻结教师目标只计算一次、缓存于 CPU。
一致率与 margin 使用 eval 分批扫描，原始 U 的 easy/hard 分组不变，teacher-only TS 不变。
预测分批、模式恢复、空预测和 float32 溢出拒绝已补；只有一条 U 时明确拒绝，
因为两条非空 easy/hard 分支不可能同时成立，旧逻辑会生成空组并除零。

新增 CPU 测试插桩每次扫描数据位置/前向批大小，所有数据池在 CPU、批次不超过 batch_size。
专项 **26 passed / 1 deselected**；GPU 6 **1 passed / 14 deselected**（1.96s），
TS 实训后显式 restore(device="cpu") 回放通过，测后回到 1MiB/0%。
首次 GPU 断言未显式指定 CPU，因 restore 默认沿用训练设备而失败；修正为已有显式接口，
没有修改快照默认语义。快照/full-matrix 对分批 float32 比较采用 atol=1e-7，
不把数值舍入当作目标或采样变化。

以本地 HEAD 1c76e04 的可信模块为旧基线，小例 3 seeds × OS/TS 共 6 次对照：
最终 logit 最大差 0、teacher/split/student 历史一致、easy/hard 数量一致。
该证据仅针对合成 CPU 工程兼容性，不是公开论文或正式训练数值。
最后同四目录扩大 CPU 回归 **1557 passed / 28 deselected**（75.70s）。
API、方法卡、台账、原八方法准入草稿及优先级同步；能力仍二维 MLP，不声称 CNN。
格式/测试质量、80 API、链接/数学、目录检查通过；冻结协议/比较矩阵/uv.lock 未改，
未提交或推送。下一步仍需多层图像表征设计、官方增强/SimSiam 的明确适配边界和预算。

## 再续：Split-PU 真实 CNN 多层表征接入

已完成注入 encoder 的四维原图路径：教师、splitter、每轮 student 都独立 deepcopy
用户模板并解冻训练，原模板冻结标志/权重不变；教师训练后冻结，阶段/fold 无共享权重。
low MSE 使用实际 encoder 模块输出，high 余弦使用编码器最终二维向量，
不是将分类头的隐层冒充 CNN 低层。可指定 encoder_feature_layer，默认首池化层，
否则首 Conv/Linear；模块必须每次前向只执行一次、输出有限且保留 batch 轴。
临时 forward hook finally 清理，clone 保留梯度/原层输出，避免 inplace ReLU 覆写。
图像/目标仍留 CPU，分批训练和扫描；teacher-only TS 及原始 U 分组保持。

公共 cnn13 的 8×8 图像真实 Pipeline 测试发现单行 easy/hard 组在最终 1×1 BN 上失败，
不是测试假阳性。已加单行 BN 运行统计策略：保留 affine/CNN 梯度、恢复每层模式，
不复制/丢弃行；大组照常训练。异常前向也恢复 BN 模式，复用两次的特征模块明确拒绝
且 hook 无残留。该策略/低层选择/同模板阶段初始化均登记为工程适配。

共享 Pipeline 与 fold/CNN 测试 **28 passed / 1 deselected**（21.65s）；
最终补异常测试的 CNN 单测 **11 passed / 1 deselected**（1.75s）。
扩大四目录 CPU 回归 **1567 passed / 29 deselected**（77.48s），该回归执行于新增
异常清理测试之前，后者随后独立通过，不虚报为 1568 项整体回归。
最终 CNN 与旧 MLP 两条 GPU 路径 **2 passed / 24 deselected**（2.28s），
GPU 6 测后 1MiB/0%；均包含 TS 与显式 CPU 快照回放，未常驻占卡。

台账/能力、八方法准入草稿、API、方法卡、优先级与目录同步；格式/测试质量、
80 API、文档链接/数学、目录及 diff 门禁通过。
CNN 工程能力不代表官方图像增强、SimSiam 投影/predictor、作者完整 CNN 配方或
正式多 seed/资源预算已复现；admitted=false、候选池/协议/预算 null、负责人签署仍待办。
本轮公开源码重读请求未成功，未新增来源审阅/源码授权结论；沿用已有锁定来源和显式
适配边界，不以网络失败猜测作者细节。冻结协议/比较矩阵/uv.lock 未改，未提交或推送。

## 推送、远程治理合并与 CNN 存储实测

用户明确要求推送后，本地成果提交 d29da22；fetch 发现合作者 PR98 / 488d744，
合并保存为 19c68c2，普通 push 成功，原始 Excel 未上传。保留注释门禁、单源注册表、
中央测试工厂、稳定台账方法引用与架构文档。五新算法接中央 factory，类声明字段不在
registry 重复写；PULNS 无独立 clean support 的 Iris 用例显式不适用，专门 support 测试保留。
清理新门禁指出的过期豁免，不降低检查规则。静态 draft 引用与 canonical 台账同步，
null/未准入状态不变。合并专项 225 passed /23 skipped（9.21s）。扩展 CPU 回归
2518 passed /25 skipped /32 deselected，只因 UI 旧三方法名单断言失败；名单按当前八个
CNN 声明同步后 2 passed，未把这写成整套重新执行绿。合并前四目录为1568 passed。
全部格式/注释/文档/数学/目录/层间依赖/测试质量/元数据门禁通过。

推送后继续资源准备，新增脚本与[默认宽度 CNN 存储复核](candidate_cnn_storage_review_20261005.md)。
五方法 ×3 seeds CPU，所有阶段快照/最终分数/可信全对象恢复通过；另一次 PULDA CUDA
测量使用 GPU6 /A6000，峰值 allocated 308,205,056 /reserved333,447,168字节，测后1MiB/0%。
该数字仅本进程、小合成输入/阶段预算，不是正式 CIFAR 容量。29 项存储专项通过。
源码摘要在训练前后核验，CPU不伪报GPU峰值，请求CUDA失败不静默降级。
新记录不覆盖旧二维探针、不改旧冻结配置、不批准候选或正式预算。后续继续审阅所需
资源/选模规格、来源行为与剩余图像适配；正式结果制品和本人签署仍不能由工程测试替代。

最后重新执行全部 unit/contract/integration 与 builtin 测试（not gpu and not slow）：
**2535 passed /25 skipped /32 deselected**（185.36s）；这次包含修正的 UI 候选断言，
不是沿用上一次有失败的回归。跳过包括可选依赖与 clean-support 不适用 fixture，
GPU/slow 明确排除；PULDA CUDA 技术探针单独执行，不称 frozen-lock 验收。
新增来源在训练期间变化的拒绝测试随后另跑 **1 passed /15 deselected**（8.09s），
不虚增本次整套回归计数。台账只新增 technical_resource_evidence 指针、预算未批准且
protocol_binding=null；关联契约/方法台账/UI 专项 **31 passed**。

## P1-1：五方法来源、预算与选模工程准备

本次按 PULDA → PUET → Grad-PU → Robust-PU → Split-PU 补齐证据，见
[交付说明](p3_admission_evidence_20261005.md)和[机器包](data/p3_admission_evidence_20261005.json)。
四个作者仓库的 17 个文件均已按 commit/字节摘要核验；Grad-PU 保持未确认源码。
只读检查器核对当前构造默认值、checkpoint 上界、视图、证据引用和未准入状态，
可另行检查本地作者源码，但不下载或执行外部代码。

独立标量公式/梯度、PULDA 不等阶段/真实 step、PUET 边界与选择签名等新增测试，
连同检查器与既有准入契约专项 **45 passed**。四个来源实际检查返回 ok=true，
formal_admission=false。方法卡、台账、八方法草稿、索引和路线的准备状态同步。

特别核实 PULDA warmup cosine 在两实现中均按 pu_epochs；Split-PU teacher/student
返回末轮权重而非按 test-best 恢复，easy 组损失归一化差异另列。
本次不改生产训练逻辑，不新增公开数值锚点，不重新绑定旧结果，不清空正式阻断或代签。
私有 Excel 保持未跟踪，不进入本次提交。完整回归与静态门禁结果见本批交付说明。

## P1-2：公开结果库存与可比性准备

见[交付说明](p3_public_comparison_20261005.md)和
[机器草稿](data/p3_public_comparison_draft_20261005.json)。四篇原论文表格已渲染核列，
登记12条本人方法数值（不是其它方法的转载/邻列oracle），PULDA 全文受限明确保留未知。
五方法×三数据集15格中5格有读数但协议不符、7格仅在本批未观察到读数、3格来源不可读；
不是冻结 run 覆盖、更不是文献穷举；无新方法结果可作数值裁决。

只读检查器校核单位转换、axis身份、台账/准入草稿与冻结字节边界；实核四篇PDF和19个
作者源码登记文件，未执行外部代码。专项检查器/准入/P1-1回归 **64 passed**。
新增判读包括 Split-PU teacher50/代码20，Robust-PU均值百分数/std分数、非oracle PN，
GradPU邻列Sup.和deviation未知；所有值pending，阈值和numeric verdict空，不代签。
Skill 的 benchmark 证据纪律用于区分来源库存与真实制品验收；本次无结果目录，未运行训练。
台账、原八方法草稿、五方法卡、索引和执行路线已同步；完整验证见交付说明。
