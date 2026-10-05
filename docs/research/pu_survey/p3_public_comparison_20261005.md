# P1-2：五方法公开结果对照准备（2026-10-05）

承接 [P1-1](p3_admission_evidence_20261005.md)，对应 P3.1/P3.2 的公开来源与可比性准备，
**不是执行计划 P1.2、P2.2 正式验收或新的冻结对照矩阵**。
本批可自主完成的来源库存、协议差异、机器检查和交接已完成；数值复现与负责人审阅未完成。

机器事实见[公开对照草稿](data/p3_public_comparison_draft_20261005.json)。
所有读数保持 `pending_review`，所有单元 `numeric_eligible=false`；没有新方法实验结果、
数值 verdict、签名或判定阈值。旧执行矩阵、对照 v1/v2/v3、P1-1 证据与 uv.lock 字节摘要绑定，
不修改原矩阵或将新方法塞进历史 645 runs。

## 1. 原始来源与可证伪读数

只登记原论文中**本方法自身**的结果，不混入邻列 PN、其它论文的 nnPU 引用或第三方复现。
本表是少量待审读数库存，不是论文表格全文、穷尽文献清单或已批准的 numeric anchors。
四篇 PDF 均下载到临时目录、记录 SHA256 并渲染原表核列；论文未随代码入库。

| 方法 / 位置 | 条件轴 | 原表读数 | 离散度及重复数 |
|---|---|---|---|
| PUET，Table 2，PDF8 / 印刷8，nonnegative/quadratic | CIFAR-10，n_P=1000 | accuracy 79.74 (0.37) | SD，均为百分点，5 次拟合 |
| GradPU，Table 1，PDF6 / 印刷7301，GradPU 列 | CIFAR-10，n_P=1000 / 3000 | accuracy 90.1±0.2 / 91.9±0.1 | 3 runs；正文仅说 deviation，暂不指定 SD/SEM |
| Split-PU，Table 1，PDF6，Ours 行 | CIFAR-10，n_P=500 / 1000 / 3000 | accuracy 89.18±0.12 / 90.51±0.10 / 92.51±0.10 | SD，百分点，5 repetitions |
| Robust-PU，Table 3，PDF7，ours 行 | CIFAR-10，π_U=0.2 / 0.4 / 0.6 | error 7.58 (.004) / 10.26 (.006) / 10.73 (.005) | 10 trials；均值为百分数，代码 std 为分数 |
| Robust-PU，同表 spambase 列 | spambase，π_U=0.2 / 0.4 / 0.6 | error 6.40 (.006) / 9.79 (.014) / 11.25 (.009) | 同上；不把 π_U 当 c |
| PULDA | 全文尚未取得 | 不登记数值 | 未知，不意味着原论文没有数值 |

主要出处：

- [PUET 原论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/98257285340854262185500e59bc0f28-Paper-Conference.pdf)，§5/5.1、Table 1/2。
- [GradPU 原论文](https://ojs.aaai.org/index.php/AAAI/article/download/25889/25661)，印刷7300–7301。
- [Split-PU arXiv v1](https://arxiv.org/pdf/2211.16756v1)，§4 和 Table 1。
- [Robust-PU arXiv v1](https://arxiv.org/pdf/2308.00279v1)，§4 和 Table 3。
- [PULDA 出版入口](https://ieeexplore.ieee.org/document/10264106)及
  [作者列出的全文入口](https://qianqianxu010.github.io/publications/)，本次均受 JavaScript/访问校验限制；
  只沿用锁定作者代码，不拿 Dist-PU 的 91.88±0.52 当 PULDA 的结果。

逐条推翻条件：读数、行列、单位、重复数或所列协议与原表/正文不同；提供原页或
仓库+commit+文件行号即可提出更正。GradPU 的 90.5 / 92.9 属相邻 Sup. 列，不是 GradPU；
PUET 的 79.86 属 logistic 列，不是本适配的 quadratic；Split-PU 表内 nnPU 无星号行
不能因所在论文而被归为 Split-PU 作者自行产出的结果。本批不登记这些旁列锚点。

## 2. 协议差异：不能用同名数据集代替同一实验

目标参照仅为历史 pilot 的协议上下文，**不是替新方法选定 recipe**：SCAR c=0.1/0.3/0.5，
统一角色/PA-OA 选模、train-only 统计、ResNet18 图像参照和特征路径，5 个 split seeds。
新方法的候选协议、表征和预算仍待批准；MLP/注入 CNN 默认事实另见 P1-1。

| 方法 | 来源事实与参照差异 | 必须补齐才能谈数值一致 |
|---|---|---|
| PUET | 固定 P1000、整个训练集作 U；CIFAR 输入3072维、100棵树；五次拟合不等于五个 split；PA/OA 未建立 | 原像素/冻结特征需分口径；同采样、split、CPU 树预算；接受 P1-1 的四项树适配差异 |
| GradPU | P1000/3000、validation500、剩余 U；CNN13 无 BN、batch250、衰减 LR；3 runs | 不能把 TS 风险并集当原数据流程；确认验证标签、选模与 deviation 定义，补源码/完整配方 |
| Robust-PU | 扫 U 内 π，不扫 c；CIFAR P2000/U4000、spambase P400/U800；10 trials；代码两阶段 clean-val best | π_U 与总体 π 分开；干净验证预算和阶段初始化与 PA 不等价；图像、预算和指标转换单列 |
| Split-PU | P500/1000/3000 后剩余 U；CNN13、真实弱强增强/SimSiam、5次重复；code 返回末轮权重 | 同数据与 stage recipe；Gaussian/余弦和 easy 组归一化不是源配方；外置 PA/OA 未由 test 监控建立 |
| PULDA | 代码固定 P1000、总体 U，ImageNet 统计、自定义 CNN、test 最高准确率报告 | 先取得原表及论文实验协议；不可推断论文数值等同锁定代码，更不可与 Dist-PU 混用 |

以上含未知项，未核 normalization、阈值网格或 split 身份时就写未知，不将未知记作一致。
所有方法 CIFAR 正类 `{0,1,8,9}` 有来源支持，但**这一项一致不推出其它项一致**。
数字 c、π_U 和 n_P 也不能互相填充：即使给出 c=n_P/N_+ 的反算值，仍须核总体/验证划分和采样。

### 新补的三项重要边界

1. **Split-PU 论文/代码周期不同**：论文 PDF6 写基模型50轮；锁定
   `82fb9730597a65a12156736419d4588675bc24d5` 的
   [main.py:13-22](https://github.com/loadder/SplitPU_MM2022/blob/82fb9730597a65a12156736419d4588675bc24d5/main.py#L13)
   是20轮。本工具箱默认 teacher10 又是第三个配方。没有运行日志证明哪一个生成表格，
   两边都登记，不选择性消掉矛盾；预算一致仍为待办。
2. **Robust-PU 均值和误差棒混合单位**：锁定
   `34d950f2c6e56510855a922acb5f84b6459773ef` 的
   [main.py:511-515](https://github.com/woriazzc/Robust-PU/blob/34d950f2c6e56510855a922acb5f84b6459773ef/main.py#L511)
   显式打印 mean×100、std(fraction)，后者使用 NumPy 默认 ddof=0。
   因此 error7.58转换为 accuracy92.42，(.004)转换为0.4个百分点，不是0.004个百分点；
   error→accuracy 不改变 spread。本次代码支持这一单位判读，**不证明作者表格由该次代码运行产生**，
   归属和单位判断仍 pending；不据此计算判定阈值或把 SD 再误读成 SEM。
3. **“无先验”“PN”不能跨变体照搬**：Robust-PU 论文称其方法无需先验，
   本工具箱组合的 nnPU warmup 明确需要 prior；不能把整个组合称为 prior-free。
   该文 PN 将 U 当负类，并非本项目真值全监督 PN oracle。不能由列名 PN 合并两类 baseline。

## 3. 机器可核的边界与覆盖

`scripts/check_p3_public_comparison.py` 只读检查：12条读数 / 5方法×3数据集15格，
来源类型、axis 身份、单位方向、台账指针、未准入及冻结文件摘要。当前覆盖为：

- 5 格有已核原表读数，但协议不匹配；
- 7 格在本批审阅范围中未登记读数；**不宣称全世界没有对应结果**；
- 3 格 PULDA 全文来源暂不可读。

均不能数值裁决。覆盖不是按 seed/c/PA-OA 展开、不是正式结果单元或幽灵 run；
新方法尚不在冻结 pilot，不存在可与这些数字配对的新方法正式结果。
读数归一化输出仅统一显示单位，没有 Delta、显著性、通过/失败或排名。
manual 页码/读数可能错误，静态校验不能证明它们真实；负责人须按推翻条件复核。

```bash
PYTHONPATH=. python scripts/check_p3_public_comparison.py
pytest -q tests/unit/scripts/test_check_p3_public_comparison.py \
  tests/contract/test_p3_candidate_admission.py
```

如已合法取得相同版本 PDF，可加 `--paper-dir /path/to/papers`，文件名
`puet.pdf`、`gradpu.pdf`、`robust_pu.pdf`、`split_pu.pdf`；检查真实摘要，不解析/运行文档内容。
作者 checkout 可加 `--source-dir method=/path/to/source`，承接 P1-1 的 commit 与摘要，
另核 PULDA cifar10 和 Split-PU dataset 文件。本机已核四篇PDF及四仓库共19个登记文件，
P1-1 的17文件记录保留。checker 不下载、不训练、不接收私有 Excel 或实验权重。

## 4. 下一步与签署顺序

1. 负责人独立核这12条原表读数/单位、两个 discrepancy 和各项协议未知；
   不确认的条目维持 pending，不随整体签署接受。
2. 选择“源配方复现”还是“共享协议下的 benchmark-adapted 比较”；
   后者不能通过对齐几个参数就宣称原论文准确率复现。
3. 登记新版本的采样/表征/标签预算、方法变体、stage/阈值选择、重复口径和存储规格；
   在新实验前预注册 numeric 资格、阈值与误差棒换算；本批不事后调参/填 tolerance。
4. 在获批资源下生成真实结果/manifest/selected checkpoint，再独立复算并作对照裁决。

P1-2 自主准备闭环不消除 P3 负责人决策、P3.3 正式资源验证或 pilot 原始制品待交付。

## 5. 本批验证

- 新检查器、原八方法准入契约及 P1-1 检查器专项 **64 passed**。
- 全量 CI 快层选择式 `pytest tests/ -m 'not slow and not e2e' -q --tb=short`：
  **3039 passed /55 skipped /36 deselected**（243.94s）。55 skips保留可选依赖/无CUDA等原因，
  36项slow/e2e未执行；不是全部测试、正式训练或GPU多seed验收。
- 隔离Python3.11.12/PyTorch2.13.0+cpu，其余数值依赖按锁定版本；CPU Torch变体
  不是Linux uv.lock 的CUDA制品。禁止GPU使用，不称 frozen-lock GPU复验。
- 四篇PDF字节及四个来源共19文件摘要实核成功；原文表格另经渲染查看列名。
  禁用torch导入时离线检查成功，不要求深度学习可选依赖；不执行作者源码或权重。
- 11项静态门禁通过：格式、测试质量、文档链接、API、元数据、数学、Skill同步、
  baseline配置、注释、目录及层间依赖。`git diff --check`通过。

本批没有推送或新远程CI结论；上述只报告本地证据，不沿用上批CI绿色作为本批证明。
