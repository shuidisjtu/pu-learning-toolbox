# P4.2b：三个无统一基准方法的定性交代（2026-10-25）

`genpu`、`pulns`、`split_pu` 在 v3 可运行，但不在任何统一基准里，各自论文的数据集、
条件轴与指标也和本项目不同。按 D37（2026-10-25 用户裁定）**只做定性交代**：不为凑锚点
改数据集、改指标或改阈值，也不把论文数字硬套到我们的单元上。

**产物**：`survey_comparison_v6.json`（`survey-comparison-v6`）把这三条 profile 的 71 个
v3 结果单元登记进矩阵 —— 70 条 `no_direct_anchor` + 1 条 `non_runnable` 标记行；
`survey_comparison_profile_map_v3.json`（`survey-profile-map-v3`）把三条 profile 由
`not_in_matrix` 转 `mapped`。锚点仍 64 条、判定规则不变。

## 1. GEN-PU

- **论文**：Hou 等，IJCAI 2018，DOI 10.24963/ijcai.2018/312。Table 1（PDF 第 4 页、印刷页 2259）
  报的是 **MNIST / USPS**，生成器为稠密网络（100 维噪声 → 两个 256 维隐藏层 → 784/256 维 tanh 像素）；
  作者 demo 同样是稠密网络。
- **我们的单元**：spambase、imdb（`native_2d`，共享 MLP）与 cifar10（`native_cnn`，ResNet-18），
  SCAR c 网格 + PA/OA 选模。配方已冻结（#126，2026-10-24），正式预算 100 GAN + 100 PN，
  首跑 2.666 h（见[正式预算收据](genpu_formal_run_20261025.md)）。
- **为何没有直接锚点**：数据集不同（MNIST/USPS vs spambase/imdb/cifar10）；架构不同
  （论文稠密生成/判别 vs 我们共享 MLP/CNN 路径）；选模不同（论文报发布模型，我们按 c 网格选）。
  PU-Bench 的 `cgenpu` 是该方法的**变体**而非本行（D37），不能当直接锚点。
- **归类**：70 个单元全部 `no_direct_anchor`。

## 2. PULNS

- **论文**：Luo 等，AAAI 2021，DOI 10.1609/aaai.v35i10.17064。Table 3（印刷页 8788）报
  **test error rate** 的均值±标准差，轴是 U 内正类比例 γ（七个 benchmark × 三个 γ）；
  p.8787 明确需要**真实正/负验证集**作为终端奖励的**训练信号**（不只是 OA 选模）。
  CIFAR-10 正类 `{0,1,8,9}`，其余方法用 100 隐层 MLP。
- **我们的单元**：只有 spambase `native_2d` 可运行（cifar10 `native_cnn` 在 v3 标 non-runnable），
  用从 train 中拆分并单独收据化的独立 clean support，走 SCAR c 网格与 accuracy。
- **为何没有直接锚点**：轴不同（论文 γ 固定 U 内正类比例，我们扫标签频率 c；两者不能互算）；
  指标不同（error rate vs accuracy）；support 标签来源不同（论文自己的划分 vs 我们从 train 拆出）；
  且该行消耗额外正负标签（`extra_clean_support` 比较组），本就不能与 PU-only 行同榜。
- **归类**：10 个可运行单元 `no_direct_anchor`；cifar10 `native_cnn` 的 1 个标记单元 `non_runnable`。

## 3. Split-PU

- **论文**：Xu 等，ACM MM 2022，arXiv:2211.16756。Table 1（PDF 第 6 页）在 CIFAR-10 上按
  **固定 n_P（500/1000/3000）** 报 accuracy 89.18±0.12 / 90.51±0.10 / 92.51±0.10（5 次重复），
  官方代码用 CNN13 + 真实弱/强增强 + SimSiam 一致性，且训练中反复读取测试标签报告准确率；
  论文 50 轮、锁定代码 20 轮、本工具箱 teacher 10 轮是三个不同配方（已登记，不选择性抹掉）。
- **我们的单元**：spambase、imdb（`native_2d`）与 cifar10（`cnn_feature_adapter`），
  SCAR c 网格 + 外部 PA/OA 选模，Gaussian 弱/强扰动替代官方增强。
- **为何没有直接锚点**：轴不同（n_P vs c）；增强/一致性配方不同（官方增强 + SimSiam predictor
  vs 我们的 Gaussian 扰动 + 余弦一致性）；选模口径不同（论文代码用测试标签监控，我们从不让
  `fit` 看到测试标签、选模由外部 runner 分开执行）；非 CIFAR 行论文根本没有对应数据集。
  v4 里 Wang 等的 cifar10 **行级**映射是其 PA/PAUC 行，不覆盖我们的 split_pu 单元。
- **归类**：30 个单元全部 `no_direct_anchor`。

## 4. 为什么不是 `background_only`

本矩阵里 `background_only` 的含义是「**存在已发表数字，但不得据此告警**」（v6 中 3 条 PN
oracle 单元，各带一条锚点）。这三条方法一个可用的论文数字都没有挂到我们的单元上，套
`background_only` 会暗示存在被刻意压低的锚点；`no_direct_anchor` 才是「单元格已登记、
没有直接锚点」的实话。

## 5. 影响与边界

- 覆盖率门禁：359 个单元全部有映射，`outside_matrix` 由 27 行降到 19 行；锚点仍 64 条，
  无孤立、无重复引用；`decision_rules` 与 v5 完全一致。
- **已跑的 v3 结果不被重新裁决**：历史 manifest 仍记运行当时的状态；新运行才会拿到 v6 与
  map v3 的比较块。
- 这些单元没有数值判决：聚合报告会把它们记为 `no_direct_anchor` 并给出理由，**永不进入数值排名**，
  即「摘要中只作背景展示」。
- v5 与 map v2 仍在盘上，可显式按路径加载；生效路径已切到 v6 / map v3。

## 6. 什么情况下可以重新裁定

只有当下列之一发生时，才以**新版本**（而不是原地改 v6）重开：统一基准真的发布了这三条方法在
我们数据集、c 网格与选模口径下的数字；或我们另行跑通论文原始设定（n_P/γ 轴 + 官方增强 +
论文轮数）并把它登记为**单独的行**。两条都不改变本记录对现有单元的裁定。

## 7. 验证

- `check_survey_profile_map.py --require-present`：`survey-profile-map-v3`，16 mapped / 7 not_in_matrix；
- `validate_comparison_coverage(v3, v6, map v3)`：359 单元 / 359 映射 / 7 行级 / 64 锚点 / 19 outside；
- 新增 `tests/unit/experiment/test_survey_comparison_v6.py` 钉住「只加 71 个单元、不加锚点、
  不改规则、只升三条 profile」；既有 v5 测试仍按 v5 自身内容通过（历史制品不改写）。
