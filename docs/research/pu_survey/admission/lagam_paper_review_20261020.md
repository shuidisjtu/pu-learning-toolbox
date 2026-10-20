# LaGAM 全文与发布实现交叉复核（2026-10-20）

状态：`pending_method_owner_review`，无负责人签署，不代表数值复现、协议批准或正式准入。
本项推进 P3.1 来源/行为证据，不重开 P2.2 review，不采纳根目录待审协议草案。

## 来源与历史记录

本轮成功读取 CVF 的[正文 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Long_Positive-Unlabeled_Learning_by_Latent_Group-Aware_Meta_Disambiguation_CVPR_2024_paper.pdf)
及[补充材料](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Long_Positive-Unlabeled_Learning_by_CVPR_2024_supplemental.pdf)，
分别为 10 页与 7 页。正文 SHA-256 为 `31c76526cfa5797d836e2b0108e385334902d8bdc77043658efaea4f887f1420`，
补充材料为 `989d49b00699e6d1796c5498a2ff5e9a3a3aa96bea29183c65231e3cc2ea6378`。
通过 HTTPS 下载字节并在内存解析；没有提交 PDF 或复制/执行上游代码。
本记录补齐[10-15 来源核查](lagam_source_cnn_review_20261015.md)的全文获取缺口，
不覆盖当时 HTTP403 的历史事实，也不把“获取成功”当作来源许可已解决。

作者代码仍锁定
[`llong-cs/LaGAM@362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8`](https://github.com/llong-cs/LaGAM/tree/362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8)。
本轮重读 `train.py`、`utils/utils_loss.py`、`utils/utils_data.py`、`README.md`，
摘要与旧记录一致。页码、物理代码行号、摘要、逐项结论与阻断见
[机器记录](../data/lagam_paper_review_20261020.json)。正文 PDF 页码加 23137 即印刷页码。

## 公式—源码—当前实现

| 核对项 | 原始来源 | 工具箱及结论 |
|---|---|---|
| 分类损失 | 正文 p.3/印刷23140 式 (1) 及 L_cls 定义为逐样本 BCE 均值；源码 `utils/utils_loss.py:11-23` 为两列标签质量归一化并裁剪概率 | 当前 `_balanced_soft_bce` 是正负软质量各自归一化，分母下限为 1，使用稳定 log-sigmoid；不是正文普通均值，也不是所有边界下的源码损失 |
| 层次对比 | 正文 p.3–4 式 (2)–(5) 为 ID 加 LGA；LGA 正对是“同簇且同二分类预测”并上局部邻居 | 当前仅 ID 加同簇项，无 binary cutoff/kNN；保留已有明确适配，不能称完整 LGA。归一化 hidden 特征并非作者独立 128 维 projection |
| 元更新 | 正文 p.4–5 式 (6)–(11) 在当前伪标签上求支持集元梯度；补充 p.3 式 (17) 解释标签扰动 | 当前元步仍扰动原始 hard-observed 标签，实际训练才用 EMA 软标签；不是论文的同一标签轨迹。默认虚拟 weight+bias，显式 `meta_update_bias=False` 只对齐作者 weight-only 范围 |
| EMA/调度 | 正文 p.5 为 20 轮 warmup、EMA 0.95→0.8；补充 p.1 为 400 总轮、SGD、batch64、lr0.001 | 当前 20 总轮含 2 warmup、Adam/常量 lr；rho 分母 epochs−1 与作者 epochs 不同。`not no_verbose` 对作者标签持久化的门控也不是本项目语义 |
| 聚类、增强 | 正文 p.3 逐轮 k-means；补充 p.1 单独 projection、100 簇、50轮后10邻居 | 当前 sklearn 聚类、Gaussian 双视图，无源图像变换/Faiss/projection/cutoff/邻居；5 簇缺省不能泛称论文设置，见下节 |

标量约束扰动与两列扰动只在**同损失、同目标、同虚拟参数范围**下有方向关系；
已有有限差分测试只证明该窄命题，不消除正文普通 BCE、源码裁剪、原始/软标签或 BN 的差别。
本轮不为“贴合论文”静默替换默认目标，也不新增一个未经批准的正式 recipe。

## 新登记的三个可证伪分歧

1. **零梯度投影存在论文内部边界冲突。**正文 p.5 式 (10) 写 δ≥0 时为 1；
   补充 p.1 Algorithm 1 第 11 行却写 `I(δ>0)`。作者 `train.py:366-369`
   对两列负梯度取 argmax，列相等时选第 0 类，随后将观察 P 固定为 1。
   当前 scalar `gradient<0` 在梯度为零时也将 U 判为 0，保留该行为；
   不可把正文“含等号”登记为已逐点对齐。独立零梯度例子区分两种规则。
2. **对比损失系数 β 不能直接从补充材料抄入 API。**正文 p.5 式 (12) 明确写
   分类损失加 β 倍对比损失；补充 p.6 Table 15/C.4 又称 β 默认 0，
   同时用其说明足够强的对比表征。锁定 `train.py:147-169,397-408` 默认
   `using_cont=1`，直接加 `loss_cont`，未声明 β 参数；不能据此倒推出 β 是
   指数、补数或另一个损失权重。当前 `contrastive_weight=1` 直接乘对比项，
   设 0 确实关闭该项，不解释为补充材料的 β=0。**待负责人裁决配方。**
3. **簇数具有论文/README 配方差异。**正文 p.8 Table 6 标注默认 100 簇，
   补充 p.1 A.3 称所有数据集均为 100 簇；
   `train.py:151` 缺省也是 100，但 `README.md:9,15` 的 CIFAR-10 命令
   显式设 5，CIFAR-100/STL-10 命令设 100。工具箱缺省 5 仅与前一命令参数
   相同，不是全数据集的论文默认。无对应 checkpoint/日志，不能判定论文表格
   实际用了哪档；保留两个来源，不从结果好坏反推。

推翻以上记录需给出对应版本 PDF 或该锁定 commit 的原文/代码位置；
解决冲突还需方法负责人声明采用哪一口径，不能由工程测试替人签署。

## Clean support 的实际监督与采样

正文 p.4 式 (6)–(9)、补充 p.1 Table 7 明确 support 干净且与训练不交；
正文 p.5 说从 validation 取 500。发布代码实际上在
`utils/utils_data.py:157-200` 按**原始多分类类别**各取 `floor(M/K)`，
先从原训练池移除 support，再取观察 P，再将剩余设 U。`train.py:350-364`
使用 support 真标签求元梯度。当前 preparer 的无原始子类配额均匀抽样不是该采样重放。

以完整均衡 CIFAR-10、README 的正类 `{0,1,8,9}` 和 M=500/nP=1000 为条件，
可推导 support 每原类别 50，即 **200 正/300 负**，不是二分类各 250；
每正类取 250 个观察 P；剩余 PU 训练为 49,500，其中 P=1,000、U=48,500。
这是源码条件推导，不是新跑批或已经物化的本项目 split。
原始类别不均衡/稀少时切片或整数除法还可能使实际数量少于请求，不能无条件套用公式。

补充 p.3–4 B.3 将 support 称为 validation 的特殊用途，并主张可直接 test 而不额外选模。
本记录尊重该来源陈述，但其真标签经“元梯度→伪标签→实际训练”仍构成监督通道；
不因虚拟步不保存参数而将额外标签预算记为零。若本项目另做 OA 选模，selection 仍须
与 support/train/test 分离。补充 p.6–7 Table 16/C.4 的小支持集结果还部分依赖原子类
知识，不能用来证明当前均匀抽样必然等效或“越少越好”。

现有 `LaGAMCleanSupportTrainer` 仍仅 exploratory SCAR OS/terminal OA；
PU-only PA 拒绝、显式 TS 拒绝、无校准 hook。新的 PA+clean-support 口径、真实支持集
来源/组身份、唯一标签预算和统一骨干必须另行批准；本轮不把待审草案当成已生效协议。
补充 p.4 Table 9 的 Alzheimer 正类是 Moderate Demented，也不能与本项目
NonDemented-positive 替代数据直接对照或称为官方 ADNI。

## 技术验收与剩余工作

[数值行为测试](../../../../tests/unit/estimators/test_lagam_paper_behavior.py)使用独立 NumPy
损失/对比参考式和精确零梯度例子，验证当前平衡 BCE、分母保护、ID+同簇项及论文边界差异；
另记录两轮元步实际接收 hard-observed 标签而非 EMA 目标。
[来源登记契约](../../../../tests/contract/test_lagam_paper_review.py)核对新旧来源关联、
PDF/代码身份、三项分歧、采样算术和未准入/未校准边界。均不联网或运行作者代码。

本轮新增 15 个参数展开用例全部通过；包含既有 LaGAM、台账/registry 和来源契约的
专项为 **84 passed / 1 skipped / 0 failures / 0 errors**（85 项，2.621 秒），
本机 JUnit：`/tmp/lagam-paper-followup-final-20261020.xml`。
整仓 `pytest tests/ -m 'not slow and not e2e'` 为 **5,317 passed / 59 skipped /
0 failures / 0 errors**（5,376 项，328.029 秒），
本机 JUnit：`/tmp/lagam-paper-final-fast-20261020.xml`。
静态格式、文档/API、数学渲染、测试质量、目录/层依赖、元数据、注释、skills 同步、
baseline、候选 fit 身份及 materialized draft registry 门禁均通过。
draft registry 的通过不是正式 registry 锁定；CUDA 条件跳过与未运行的 slow/e2e
也不算通过。既有代价/solver/NVML 等告警仍在，不宣称远程 CI 已通过或零告警。

已补全文获取与交叉表，不等于完成来源许可、差异裁决、数值复现或 RV8。
下一步首先由方法负责人确认投影边界、β/簇数配方、完整/消融实现定位与 support 预算；
之后在获批规格下补公开数据、多 seed/GPU 和资源证据。本轮不改冻结协议、比较矩阵、
旧 645-run 结果、签署或正式准入，不提交用户 PDF/Excel/数据包/协议草案。
