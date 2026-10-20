# CVIR 来源差异与图像/分批工程复核（2026-10-15）

状态：P3.1/P3.2 独立技术补齐，HENG958 实施；shuidisjtu 方法/接口复核待办。
不代签 RV8，不是作者程序重放或正式算法准入，不改冻结 Pilot 的结果和协议。

2026-10-17 整改补记（#117）：本 Survey 的基础数据按协议 §2.3 始终按 OS 抽样，
CVIR 的运行路径当时明确为 **OS adaptation + 显式 α_U/来源**（`CVIRKnownPriorTrainer`），
并显式拒绝 `os_or_ts="ts"`。

**2026-10-19 更正**：原记“OS→TS 校准不属于本实验”表述有误。本实验不为 TS 方法重新
独立抽样 TS 数据；原生 TS 方法使用 **TS-compatible 校准**（训练视图逐 mini-batch 执行
$D_U \leftarrow D_U \cup D_P$）。CVIR 为原生 TS，应走校准视图；当前显式拒绝
`os_or_ts="ts"` 是实现缺口，待方法负责人接入后再复核。直接组件的 `input_sampling="ts"`
仍只声明调用方已提供原生 P/边缘 U，不用于本 Survey 的 OS 数据。冻结协议/矩阵与正式
准入状态不变。

## 2026-10-20 输入与推理状态补核

本轮承接 P3.2 现有 CNN 技术路径验收，不改变上述待接入的 TS-compatible 校准。
修复前回归确认公开评分把每个子模块的 `training` 强制改为 false；自定义 in-place
encoder 还可通过宽度 probe 改写调用方训练数据，以及通过评分改写验证/测试数组。

- resident / cpu_stream 推理均临时使用 eval，成功、前向异常或 eval 切换本身部分失败后
  恢复每个子模块原始模式；空输入先校验 fitted/shape，再直接返回，不触发网络。
- encoder probe 使用独立行副本；统一 logits 前向 clone 当前输入 batch，使训练/排序
  来源、调用方评分数组不被 in-place 操作改写。新增 provenance
  `forward_input_isolation="private_tensor_clone"`；会增加当前 batch 临时内存，旧测量
  不构成当前峰值或正式资源上界。未改 resident 全量评分/CPU-stream 分批策略。
- 学习率及公开负例 selector 的 α_U 须为有限实数，拒绝 bool、字符串、容器与非法区间；
  非 callable 回调在 RNG/网络构造前拒绝。复数输入不再悄悄舍弃虚部转为 float32。
- [回归文件](../../../../tests/unit/estimators/test_cvir_input_modes.py)覆盖逐层模式、
  部分 eval 失败、前向改写后失败、空评分、真实 BN/Dropout 卷积的 buffer 不变性、
  二维/四维 in-place 对照及提前失败。历史来源、默认数值配方、两阶段时序、风险权重、
  原始 U 排序/保留身份、总体 π 拒绝与 `calibration_hooked=false` 不变。

本轮不执行作者程序、真实数据训练或 GPU 跑批，不改冻结协议/结果/历史来源测量，
不代签方法负责人、正式配方或算法准入。

专项 CVIR/encoder/known-prior adapter/候选失败拟合回归 155 项通过、1 项无 CUDA 跳过，
0 失败/错误（本机 JUnit：`/tmp/cvir-input-modes-focused-20261020.xml`）。另对本项目
修复前提交 `d3f3ea7` 做 36 组合法 MLP 配置短预算对照：3 seed × warm0/1/2 ×
OS/native-TS 输入声明 × resident/CPU-stream。损失、权重、保留行身份、全部阶段更新
次数与分数逐位一致，最大分数差为 0；原生 TS 仅是兼容性测试参数，不证明合成数据
满足 case-control 分布，也不是 OS→TS 校准或作者程序重放。最终全仓结果另行追加。

收尾复测：旧 CNN 更新断言的模板现以独立 seed 初始化，避免全局随机模板造成零梯度，
未放宽断言。最终完整快速回归共 5,361 项：5,302 通过、59 条件跳过、0 失败/错误
（`/tmp/cvir-final-fast-fixed-20261020.xml`）。不含 slow/e2e，跳过不算 GPU/正式协议
验收。静态质量、源码/台账及配方草稿门禁在本机通过；远程多平台 CI 以推送后为准。

## 来源锁定与判读

[论文](https://proceedings.neurips.cc/paper_files/paper/2021/file/47b4f1bfdf6d298682e610ad74b37dca-Paper.pdf)
§3（印刷页4）定义 α 为 **U 内正类混合比例**；Algorithm 2 在印刷页 **6**，
更正本项目旧引用的 p5。论文采用按 α 加权的正例/保留负例风险，按训练误差收敛停止。

只读核查[作者仓库](https://github.com/acmi-lab/PU_learning/tree/5e5e350dc0588de95a36eb952e3cf5382e786aec)
`5e5e350dc0588de95a36eb952e3cf5382e786aec` 的干净 checkout；Apache-2.0，
没有执行、复制或导入其代码。文件 SHA256 和逐项判读见
[机器记录](../data/cvir_source_review_20261015.json)。
`source_status=official_exact` 按 core.tags 只表示作者源码可得，**不是实现忠实度或数值复现声明**。

| 项目 | 论文/作者代码位置及事实 | 本项目选择与残余差异 |
|---|---|---|
| 风险权重 | 论文 Algorithm 2 α 加权；`algorithm.py:233-236` 两类 CE 各占一半 | 保留论文加权 BCE；warm-start 保留既有未缩放两类损失之和。不是发布代码的 balanced CE |
| 负例数/平局 | `algorithm.py:274-276` 保留 `n_U-int(α*n_U)`，默认 argsort | 保留 `max(1,floor((1-α)*n_U))`、稳定排序。例 n_U=7/α=.3：项目4、作者5；不得称整数规则相同 |
| warm-start | `train_PU.py:116-123` 条件限定 TEDn，CVIR 分支不进 | 默认一轮是工具箱适配；可显式设0，不静默改旧默认 |
| loader/forward | `algorithm.py:205-228` zip 原始 P/U，逐 U 批 mask、空批跳过、P/U 合并 forward | 项目先筛全局原始 U，再分别打乱 P/保留 U，以较长池的批数和模起点配对，分别 forward；BN/随机增强不等价 |
| 优化器/预算 | `train_PU.py:26-43,107-114,197` CLI 默认 SGD .1 / momentum .9 / wd .0005、batch200、固定5000轮 | 默认 Adam .001、无 wd、batch128、warm1+CVIR20；发布 CIFAR 脚本实际2000轮，loader 还会覆盖批大小；均不是论文抽象收敛停机 |
| α 与真标签 | `helper.py:85-102` 显式构造 U 混合；`train_PU.py:199-202` CVIR 用给定α；`algorithm.py:263-280` 真标签仅作拒绝诊断 | 项目只接显式 α_U，不读取真标签诊断、也不接 BBE/TEDn；不把总体 π 或标记频率自动当 α_U |

作者数据构造对 U 正/负数量分别取整，非整数配置的实际有限样本比例可能略有偏差；
正式对照应核实际样本/索引，不能只引用 CLI α。当前未运行任何作者数据实验。
上述差异不修改原对照矩阵，也不新增文献数值锚点。

### CIFAR 配方的附加范围

`scripts/run_cifar.sh:6-11,26-31` 列出 seeds42/1432/8378、α=.5、ResNet/AllConv、
2000轮；脚本虽然传 warm-start100，CVIR 仍不进入 TEDn 专用 warm-start 分支。
`model_helper.py:44-46` 的 ResNet 入口为 ResNet18(num_classes=2)。
`data_helper/CIFAR.py:12-16,31-35` 中 binarized 正类为 {0,1,2,3,4}，
DogCat 正类为3、负类为5，不可凭数据集名字判断同任务。

`helper.py:319-337,355-385` 用固定 CIFAR mean/std（值见机器记录），不是 train-only
统计；crop/flip 用于 train loader，而排序复用该 loader，因此 eval 模式仍有数据随机增强。
本项目注入 encoder 接收现成数组，不自动加入这些增强。源验证 P/U 从原始 train=False
分区构造，不能自动等同本项目的独立四角色划分。
`β` 是 P 占 P+U 的比例，**不是 SCAR 标记频率 c**；源 CIFAR 的 U batch 按
`int(P_batch*(1-β)/β)` 设置。IMDb 的训练 batch 又固定为8（`helper.py:442-449`），
所以不能把 CLI 默认200当所有数据的实际批次。CVIR 默认还会额外计算验证先验估计诊断，
但不将估计值用于其 α；这些额外成本也没有在本项目固定α训练中重放。
未执行作者脚本、未取得与其同配方的训练结果。

## 工程交付与边界

- `encoder=None` 保留二维 MLP；四维 NCHW 必须注入 `torch.nn.Module`。
  独立 deepcopy、float32、联合训练，encoder 须输出有限 `(N, feature_dim)`；
  后接原 MLP64/单 logit 头，不自动 flatten 原始图像，不修改用户模板。
- `data_device_policy="auto"` 在无 encoder 时保留旧 resident 路径；
  有 encoder 时选 cpu_stream，可显式 cpu_stream，拒绝 encoder + resident。
  完整数据留 CPU，训练、全局 U 排序和预测每次送设备不超过 batch_size 行。
  全局排序仍须保存 O(n_U) CPU 分数/索引；**不是磁盘流式/常量主机内存**。
- 筛选始终只在原始 U 上，保留稳定规则；每轮先 eval 排序，后训练更新。
  两个模式的模型训练损失定义、抽样规则和默认预算不变。
- `optimizer_steps_` 累计 warm-start 和 CVIR 的全部更新；
  `epoch_provenance_` 逐轮记录 stage、stage_epoch、round_index、累计更新数。
  `checkpoint_epoch_count = warm_start_epochs + max_epochs`，两阶段均有推理快照；
  新上下文只表示成本/来源，不批准这些轮次的 PA/OA 候选资格。
- cpu_stream 快照携带推理批大小；旧 resident 的 replay 批大小仍256。
  快照可回放 encoder+头，但没有优化器训练恢复 API。
- fit/refit 异常清除 fitted 标记，不能把失败重训当成功；已完成 epoch 的 callback
  可预测并捕获快照。registry/UI 能力、API、台账与准入草稿同步。
- `class_prior` 始终拒绝；即使 U 经证明为总体边缘，也必须通过显式
  `unlabeled_positive_prior` 输入该已核比例。**没有新增 `os_or_ts` 风险校准接口**；
  直接组件的 `input_sampling="ts"` 只声明调用方提供原生 P/边缘 U，不是本 Survey 的 OS 校准。

## 验证与待办

新增 `test_cvir_source_encoder.py` 覆盖独立 BCE 读数/梯度、与发布 balanced CE 区别、
取整反例、模板/编码器更新、设备批次上限、每轮训练前全局原始 U 排序、
两个阶段/累计成本/快照回放、旧 resident/stream、clone/pickle、失败 refit、
维度拒绝、公共 CNN pipeline 及 `os_or_ts` 校准/总体先验拒绝。只使用合成 CPU 输入。

验证结果：新增 **17 项测试通过**；CVIR/台账/能力/准入/UI/共享特征定向组合
**63 passed / 6 skipped**。完整 Python3.11 CPU fast suite **3796 passed / 59 skipped /
0 failures / 0 errors**（3855项，267.414秒；slow/e2e 未执行）。本机临时 JUnit 为
`/tmp/pu-cvir-source-target-final-20261015.xml` 与 `/tmp/pu-cvir-source-full-20261015.xml`，
不是正式实验制品或远程 CI 结果。

旧默认兼容对照：与 `7a9679f` 隔离运行，NumPy seed31、31×5、9P/22U、α=.3、
MLP7、CVIR3轮、batch5/random_state5，warm-start0/1/2三种配置的 scores、
selected negatives、完整 history 和所有权重逐位相同。仅证明该样例，不外推跨平台。
另用未替换的真实共享 CNN13 做公共 pipeline CPU smoke：8×1×16×16、两折及最终训练，
warm0/CVIR1/batch2，输出有限，保持 CPU-stream；不代表作者网络或正式资源验收。

格式、测试质量、文档链接/API、元数据、Skills、数学、baseline、注释、目录/层依赖与
materialized draft 门禁通过；离线 sdist/wheel 构建通过，wheel 含新代码，不含上传的
私有 Excel/RAR/outgoing raw 包。未推送、发布或开始正式跑批。
冻结 protocol/comparison 规范摘要仍为
`287c2f45387f02714e5925b35dcb04f3f64e1e3faf8f740cab861fc5747dd84a` /
`340e18ce8fda933ccf3582e9d522aee7aac7aadfdf467737c7cb15a25dee1f4c`。

未完成：α_U 来源与有限样本转换/视图审批、正式共享网络/预处理/增强/选模与配方预算、
作者行为/数值同条件重放、批准范围下多 seed CPU/GPU 资源、HENG958 复核。
`admitted=false`，candidate protocol/budget/pool 均 null，旧冻结矩阵没有加入 CVIR。
