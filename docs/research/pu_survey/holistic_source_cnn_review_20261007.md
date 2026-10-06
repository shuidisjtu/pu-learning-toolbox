# Holistic-PU：停止/阶段来源复核与端到端 CNN

状态：**独立工程准备，pending_method_owner_review**。承接
[优先级](post_pilot_priority_plan.md) 的缺失算法/原生图像路径，不改冻结 pilot、
历史结果或本人签署。台账和五方法扩展草稿同步，`admitted=false`。

## 1. 本轮核查来源

- [正式论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/d5c0f9585592bad5251133813893a6c0-Paper-Conference.pdf)，§2.2、§2.3。
- 由[会议摘要页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d5c0f9585592bad5251133813893a6c0-Abstract-Conference.html)
  的 Supplemental 链接下载 ZIP，内部 `Supplementary Material-Beyond Myopia/attachment.pdf`。
  **内部 PDF 是带行号的匿名投稿稿，不能冒充正式主文版本**；定位以它自己的 PDF 页码为准。
  ZIP SHA-256：`f18e3648ace550d04bec5876892aed88c0db5d40921152680f7afa9a18ff5e56`；
  attachment PDF：`0e622fcad83edb94b169e48778a73205cb0ec71ad2dfe55a443f086cd4657fd6`。
- [作者仓库锁定版本](https://github.com/wxr99/HolisticPU/tree/4d4ce7d6ba29722995d308293374c0f475d988d3)，
  只读审查，不复制/运行作者代码，不加载其 pickle 或下载数据。
- [LZO 原论文 v2](https://arxiv.org/pdf/2012.13309v2)，Algorithm 1/2。
  LZO 是基于类内混合构造验证集的候选选择，不是一个凭空指定的 patience 规则。

锁定代码文件摘要（用于来源定位，不是工具箱训练制品）：

| 文件 | SHA-256 |
|---|---|
| `main.py` | `c5730aaf5c51b366fc3597f1ae8597eae574d6a0f43d7b0c6cccfdab387623bb` |
| `statistic.py` | `b2ab1e565fd13ac28d57f215cbf9f2844972aebc778518aa2fa5961cc472cee0` |
| `train.py` | `a686d35e4253895e20ccae4422213ede7d3de1fd9b6db074fedde38da95e41aa` |

## 2. 可以证伪的判定

| 核查项 | 结论及定位 | 对当前实现的影响 |
|---|---|---|
| 论文是否要求 LZO 选预热终点 | 确认：正式主文 §2.3；附带 PDF 页23 §F 说明用原始已标记训练样本的 mixup 验证集选停止迭代 | 固定 warmup 不能称论文完整路径 |
| 锁定代码是否执行 LZO 停止 | 未发现该执行闭环：`main.py:284` 固定算 warming_epochs，`train.py:213` 固定循环，`train.py:312` 只记录 U；`train.py:168` validation 调用被注释 | 代码默认固定预算与论文不同，不能把未调用 helper 当已实现选模 |
| validation helper 的数据含义 | `statistic.py:97-109` 同批 mixup 后仍用原 targets；被注释的调用传 labeled_trainloader；`main.py:103` alpha 默认0.5 | 不能推导成需要真负验证集，也不能据未调用函数认定最终实验采用何种准则 |
| 后段是否应继续预热模型 | 推翻此前完整训练的隐含等价：附带 PDF 页22 Algorithm 2 第14步重新初始化；`main.py:329-335` 删除预热模型并新建后段模型 | 当前继续同模型/Adam 是适配；台账/方法卡补显式差异，本轮不悄悄改默认训练语义 |
| 后段最优日志是否等于 PA/OA 验收 | 否：`train.py:169,179-180` 按 test_acc 更新 best_acc 并记录；不能仅据这些变量推断最终 checkpoint 选择/论文表格来源 | 不将该 test 指标接入工具箱选模，不增加真值泄漏 |

LZO 构造规模、固定还是每轮重抽、终点候选范围、平局处理、选中终点后状态恢复、
后段初始化/预算和独立 PA/OA 资格仍需清晰规格。正例混合的标签本身不需要额外真负，
但“可以构造”不等于该口径已正式准入。算法来源复核与负责人批准分开。

## 3. 已完成的 CNN 工程路径

`HolisticPUClassifier(encoder=...)` 支持 4-D NCHW；输出必须为 batch×feature_dim，
head 为 Linear→ReLU→Linear。一份 deepcopy 编码器随 balanced warmup 和 pseudo-PN
一起优化，不是 frozen-feature 包装。原始模板的参数、梯度标志和 BN buffers 保留。

保留原始 U 行身份与论文 pairwise/variance 目标；数据/标签留 CPU，优化、轨迹与预测分批。
轨迹/预测用 eval，预测返回后或出错时恢复所有层的模式。singleton 尾批采用 BN 运行统计，
保留样本和 affine 梯度；未实现作者增强、EMA、CNN7 或原优化器调度。

默认二维 MLP 与 fixed warmup 仍保留；无外部先验、无 clean support、显式 TS 拒绝，
校准字段仍 false。`warmup/pseudo_pn` 阶段元数据、累计更新和两阶段存储计数沿用。
快照仅推断回放，不是 optimizer/RNG/轨迹断点续训，也不决定正式阶段选模资格。

## 4. 验证与下一项

新增 `test_holistic_pu_cnn.py`：两阶段真实参数更新、模板/BN 隔离、singleton、
种子/clone/refit/pickle、shape/output 拒绝、分批 eval 及阶段快照回放。
公共 pipeline 增加 Holistic-PU CNN 两折执行和独立模板测试；能力声明/台账/草稿契约同步。
本轮仅 CPU 合成工程测试；CUDA 测试存在但 CPU 环境下跳过，不补写 GPU 实测或数值复现。

隔离环境为 Python 3.11.12、torch 2.13.0+cpu，CUDA unavailable；不是正式 frozen-lock GPU 验收。
定向回归 **104 passed / 2 skipped**，另两份能力/台账契约 **22 passed**；11 项静态门禁通过。
与父提交 `356ebef` 的二维分支另做种子 0/1/7 缩短合成核对：输出分数、完整概率轨迹、
伪标签逐 bit 相等，累计更新均17；不是公开数据数值复现。P1-1/P1-2 和五方法选模/预算
绑定检查继续通过，全部准入/数值 verdict 仍未开启。

最终快速全量回归（`not slow and not e2e`）：**3110 passed / 56 skipped / 36 deselected**，
161 warnings，197.84s。首次全量的旧 UI 固定候选断言已修正，最终全量重跑 exit0；
新增能力不代表远端 CI 或其它 Python/CUDA 矩阵已经运行通过。

下一独立项：明确并实现来源支持的后段重新初始化变体和 LZO 选终点接口，
同时为 GEN-PU/PULNS 补原生训练路径或来源行为证据。P3MIX 的 OpenReview 官方 API
本轮仍返回 HTTP403，保留 batch 组件未注册状态；这不阻断其它方法自主推进。
正式候选、图像配方、预算/存储、多 seed GPU、公开数值及负责人签署仍未完成。
