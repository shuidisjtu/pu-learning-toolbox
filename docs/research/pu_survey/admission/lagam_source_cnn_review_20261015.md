# LaGAM 来源与图像/分批工程复核（2026-10-15）

状态：P3.1/P3.2 独立技术补齐，HENG958 实施；shuidisjtu 方法/接口复核待办。
不是正式方法/预算批准，不改旧冻结协议、对照矩阵、结果或签署。

## 来源证据与两项重要边界

只读核对[作者仓库](https://github.com/llong-cs/LaGAM/tree/362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8)
的干净 commit `362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8`，没有复制/执行作者代码。
本次未在该 checkout 找到明确许可证；论文和补充材料直接获取 HTTP403，
不借搜索摘要宣称新一轮全文或数值审查。逐文件摘要和判读见
[机器记录](../data/lagam_source_review_20261015.json)。
`official_exact` 是源码可得性标签，不是本项目实现/数值忠实度认证。

1. **干净 support 是训练监督**：`train.py:350-364` 从 `valid_loader` 读真标签，
   对 meta loss 求训练标签扰动梯度；`utils/utils_data.py:96-112` 确认其真值来源。
   标准 PU-only PA 不符合资格。OA 也须独立的 support/selection 划分及标签预算，
   不能自动把本项目 clean_val 作为 support。
2. **采样定位有未决冲突**：`utils/utils_data.py:182-193` 从池中先移除验证，
   再移除已标记 P，发布 split 是 OS。既有 registry 为 CASE_CONTROL/台账 ts；
   本次明确登记冲突和待审项，**不自行改签 registry 定位或补 TS 风险替换**。
   当前运行默认仍为 os-compatible、calibration_applied=false，显式 ts 拒绝。

## 源码—工具箱逐项对照

| 项 | 作者锁定代码 | 本项目与判断 |
|---|---|---|
| 虚拟参数 | `train.py:338-348` 与 `models/meta_layers.py:60-71` 按 identifier 且 ndim>1 筛选，文档 classifier 范围只更新 weight、不更新 bias | 默认保留原 weight+bias；显式 `meta_update_bias=False` 新增 weight-only。仅范围对齐，不是完整源码重放 |
| 元标签与 EMA | `train.py:316-376,427-446` 基于当前软标签的两列扰动，epoch/epochs 的 rho；dataset 更新受 not no_verbose 条件控制 | 元更新仍基于原始硬观测标签的标量扰动；实际训练用持久 EMA 标签；rho 保留 endpoint 分母 epochs−1，不受控制台 verbosity 影响。已明确登记差异 |
| BCE/批次 | `utils/utils_loss.py:11-23` 概率截断1e-4、类别质量无保护，熵项可选；`train.py:305-306` 跳过全零标签批，训练 loader drop_last | 项目稳定 log-sigmoid、质量 clamp_min1，无熵项，保留纯 U 和尾批；不能称逐位等价 |
| 表征/聚类 | `models/resnet.py:91-134` 分类头与独立128维归一化 projection；`utils/utils_algo.py:142-204` Faiss GPU KMeans20iter/5redo | 注入 encoder + 原 MLP hidden 特征兼作对比/聚类，sklearn KMeans10init；不是源 projection/Faiss |
| 增强/对比 | `train.py:382-406` Beta(4,4)、弱强图像变换、classifier cutoff、可选 epoch>50 kNN | Beta(4,4) 保留，双视图仍为 Gaussian 特征/像素噪声，无 cutoff/kNN；不称作者图像增强重放 |
| 训练与测试 | `train.py:214-265` SGD，默认400轮含20warmup；调度见 `utils/utils_algo.py:54-67`；每轮 test best_acc 监控，save 写最终模型 | Adam常量学习率，默认20轮含2warmup；fit 不读取 test。源 best_acc 不是最终保存权重的 test-best 选择 |

在**相同稳定损失、目标标签和虚拟参数范围**下，标量约束扰动的方向是两列梯度差，
其负号选择可与两列 argmax 等价；新增独立双列与 NumPy 有限差分测试验证此窄命题。
这不消除上表 source soft labels、概率截断、BN 模式和虚拟 bias 等其它差异。

## 图像与内存工程契约

- 新 `encoder` 必须是可 deepcopy 的 torch Module，输出有限二维特征；
  四维 NCHW 必须提供，独立副本联合训练，再接原两层 MLP 及单 logit 头。
- `data_device_policy="auto"` 无 encoder 时保留旧二维 resident；有 encoder 时强制
  cpu_stream，可显式 cpu_stream，拒绝 encoder+resident。
  训练输入、support、观测/伪标签与组索引留 CPU；train/cluster/predict 每次设备输入
  ≤batch_size，support 每次 ≤support_batch_size。主机仍需完整输入和全量聚类嵌入，
  **不是磁盘流式或常量 CPU 内存**；对比矩阵仍是批内二次规模。
- 新 CNN 虚拟步用 no_grad + 临时 eval 特征，逐模块精确恢复模式；
  不改变真实 encoder 的 BN 缓冲、也不消耗 dropout 随机流。
  实际分类/MixUp/对比仍可更新 encoder。它不是源虚拟克隆 train-mode 特征行为。
- support 必须与训练同输入形状、显式真值两类。拒绝数组物理共享内存仅捕捉明显重叠，
  **不能证明跨副本/跨文件同样本或 support/selection/test 的索引独立**；正式协议仍须证据。
- `max_epochs` 已含 warmup，快照数不再另加 warmup。`optimizer_steps_` 记录真实更新，
  `virtual_steps_` 单列二阶虚拟步，`clustering_passes_` 单列全量聚类次数；
  `epoch_provenance_` 和新 checkpoint context 分 warmup/meta 记累计成本。
  推理快照包含 encoder+头，不包含训练续跑需要的优化器/伪标签/聚类状态。
- 支持集依旧不会由普通 pipeline、UI 或正式 trainer 自动提供。
  Registry 虽声明 CNN，UI 的 requires_clean_support 门禁仍使其不可用；
  标准 EpochCheckpointTrainer 缺 support 仍拒绝。测试中的显式绑定桥**仅为测试**，
  不代表生产 runner 扩展、额外监督批准或 PA 资格。
- fit/refit 普通异常（含后续 callback）撤销 fitted；原合法二维默认语义保留。

## 验证与后续

新增 encoder/stream 与虚拟范围两个测试模块，覆盖独立梯度、源参数范围、
BN/dropout/RNG 与异常模式恢复、独立模板训练、设备批次、CPU 标签/聚类、
阶段成本/推理快照、确定性/pickle、support 与 UI/标准 trainer 拒绝。
只用合成 CPU 输入，不宣称新图像路径 CUDA、吞吐或正式资源验收。

验证结果：新增 **27 项测试通过**（encoder/stream18、独立虚拟范围数学9）；
LaGAM/台账/能力/准入/UI/共享特征组合 **80 passed / 6 skipped**。
完整 Python3.11 CPU fast suite **3823 passed / 59 skipped / 0 failures / 0 errors**，
3882项、260.740秒；slow/e2e 未执行。JUnit 本机临时记录为
`/tmp/pu-lagam-cnn-target-final-20261015.xml`、`/tmp/pu-lagam-cnn-full-20261015.xml`，
不是正式实验结果树或远程 CI 结果。

默认兼容对照旧 `2abece9`：NumPy seed31、23×5、8P/15U、独立10行 support、
MLP7、epochs3/batch5/support_batch6/clusters2/random_state5，warmup0/1 两种配置的
scores、pseudo_labels、完整 history 和所有权重逐位一致；仅证明该有限样例。
实际共享 CNN13（未替换）CPU smoke 使用12×1×16×16合成训练图像、独立6行 support，
warmup1/total2/train batch4/support batch2，真实更新/虚拟步/聚类次数为6/3/1，
分数有限、数据/伪标签驻 CPU；不等于正式图像规格或多 seed 资源验收。

格式、测试质量、文档链接/API、元数据、Skills 同步、数学、baseline、注释、目录/层依赖、
materialized draft 门禁与离线 sdist/wheel 构建通过。wheel 包含新代码，源包不含用户
RAR、私有 Excel 或 outgoing raw 包；未推送/发布/正式跑批。
冻结 protocol/comparison 摘要仍为
`287c2f45387f02714e5925b35dcb04f3f64e1e3faf8f740cab861fc5747dd84a` /
`340e18ce8fda933ccf3582e9d522aee7aac7aadfdf467737c7cb15a25dee1f4c`。

下一步需 HENG958 对 source OS 与 registry TS 定位、虚拟参数范围/目标差异、
独立 support 标签预算和 CNN 模式作逐项复核；仍需授权/全文、正式共享规格/预算、
公开数值与正式多 seed 制品。候选仍 admitted=false，protocol/budget/pool 均 null。
