# P1-1：已有五方法准入证据补全（2026-10-05）

对应执行计划 **P3.1 / P3.2 的准备切片**，不是新的实验编号。
顺序为 PULDA → PUET → Grad-PU → Robust-PU → Split-PU。
本批独立工程交付完成；**方法负责人审阅、正式预算/候选与数值准入仍待办**。
不修改旧冻结协议、对照矩阵、645-run 表格、历史 manifest 或存储探针读数。

## 1. 可检查的交付

- [机器证据包](data/p3_admission_evidence_20261005.json)：五方法完整构造默认值、
  校准范围、预算单位、快照计数、选模/存储边界、来源位置与适配差异。
- `scripts/check_p3_admission_evidence.py`：离线核默认值与真实 property、台账、冻结摘要、
  测试指针及角色隔离声明；可选核本地作者源码 commit 和逐文件 SHA256。
  不下载、不 import/执行作者源码，不训练、不写文件；不需要安装 torch 即可做静态核验。
- `test_p3_source_behavior.py`：独立标量公式对照，补 PULDA 三批 EMA/旧图断开/有限差分梯度、
  PUET 节点风险边界、GradPU raw-gradient 二阶反传、SPL 三规则边界、Bernoulli JS/teacher detach。
- `test_p3_budget_selection_evidence.py`：PULDA 不等阶段/尾批/真实 Adam 更新/重新 fit 计数，
  PUET tie/零增益与 CPU 非 epoch 预算；五方法 fit 均没有验证/test 真值或任意 kwargs 入口。
- 五方法方法卡、方法台账和原八方法准入草稿已互相链接；机器检查的变异测试锁住 pending/null。

静态检查证明的是**登记与当前代码一致**，不证明人工原文读数正确；
公式测试不是作者仓库训练重放，更不是论文表格数值复现。
PA/OA 角色声明由既有四角色 runner 测试支撑，签名检查本身不能证明任意调用方无泄漏。

## 2. 新补的来源与差异

| 方法 | 本批确认/补登记 | 判读边界 |
|---|---|---|
| PULDA | 锁定 `7b3dcad`；三批 EMA 状态与梯度；作者 U 尾批丢弃、循环 P、测试最高准确率、ImageNet 统计和缺失模块；两实现 warmup cosine 均随 pu_epochs | 我方保留尾批、P 有放回抽样；TS 风险并集不等于完整上游数据/RNG/MixUp 重放，不把 warmup 日程自称独立周期 |
| PUET | 新锁定 `5cb15d7e021a6e24d4cd300278500c38729e2775`；逐文件摘要；叶节点随机 tie、裁边阈值区间、零/负增益划分、父节点 min_samples_leaf | 我方负类 tie、完整区间、严格正增益、子节点最小样本数是明确适配；仅 nnPU/quadratic |
| Grad-PU | 沿用原文来源核查；独立验证 raw-score 梯度惩罚和二阶反传，默认参数与预算入包 | 未确认作者源码；不把固定 LR/MLP128/batch256 或 TS 流程当原论文配方 |
| Robust-PU | 锁定 `34d950f`；SPL 边界、默认 400/100 对比我方 10/20、预训练/episode 的 clean-validation best 恢复与 patience | 我方内部固定预算和最后预训练模型；PA/OA 外置，官方 restart/scheduler 全分支未覆盖 |
| Split-PU | 锁定 `82fb973`；JS 标量对照；easy 损失归一化、分阶段 LR/预算、增强/SimSiam 与测试监控 | 官方 teacher/student 返回末轮 state_dict；unused best_acc 不证明 test-best 选模；我方两组均值损失不是上游拼接行均值 |

逐项 commit + 文件行号和 SHA256 在机器包中。关键位置：

- [PULDA train.py](https://github.com/jiangyangby/PULDA/blob/7b3dcad95bd7caa0a9477af37a05764fbe6e27bc/train.py#L199)
  与 [PUSampler.py](https://github.com/jiangyangby/PULDA/blob/7b3dcad95bd7caa0a9477af37a05764fbe6e27bc/dataTools/PUSampler.py#L15)。
- [PUET tree.py](https://github.com/jonathanwilton/PUExtraTrees/blob/5cb15d7e021a6e24d4cd300278500c38729e2775/tree.py#L274)。
- [GradPU 原文](https://ojs.aaai.org/index.php/AAAI/article/download/25889/25661)，印刷页 7299–7300，式 (5)–(7)/Algorithm 1。
- [Robust-PU best-validation 恢复](https://github.com/woriazzc/Robust-PU/blob/34d950f2c6e56510855a922acb5f84b6459773ef/main.py#L396)。
- [Split-PU easy 归一化与最后权重返回](https://github.com/loadder/SplitPU_MM2022/blob/82fb9730597a65a12156736419d4588675bc24d5/splitpu_utils.py#L162)。

本批只读取作者仓库；未复制其实现进包，未运行外部源码或未知权重。
Robust-PU/Split-PU 的未决许可证、GradPU 的未确认源码不被这些测试解除。

## 3. 预算、存储和选模交接

| 方法 | 预算单位 | 推断快照上界 | 额外成本/资格注意 |
|---|---|---|---|
| PULDA | warmup + PU 两阶段 epoch | warmup_epochs + pu_epochs | U 尾批仍更新；两阶段 Adam；不能只计 PU 轮次 |
| PUET | 树数、深度、特征/阈值候选 | 不适用神经 epoch | 森林持久化/CPU 内存另计；GPU 条款调整需负责人决定 |
| GradPU | epoch 与真实 optimizer_steps_ | max_epochs | 输入梯度与二阶图影响成本；无 BN 图像结构另登记 |
| Robust-PU | pretrain + episode × inner_epochs | pretrain_epochs + episodes | inner_epochs 增训练更新，不增加逐 episode 快照 |
| Split-PU | teacher + splitter + rounds × student | teacher_epochs + split_epochs + rounds × student_epochs | splitter 早停只降低实际值；跑前不能用早停低估峰值 |

这张表和构造默认值均为**当前代码事实，不是批准的统一候选预算**。
神经快照只恢复分类器推断；不含优化器/RNG/EMA/伪标签/全部辅助网络的续训状态。
已选权重回收不降低训练中峰值；真实结构的 bytes/component 和保守容量上界仍待批准后测量。
旧存储 JSON 对应修复前源码摘要，原样留档，不冒充本批新版源码重测结果。

四角色边界固定：train 训练；pu_val 供 PA；clean_val 供 OA；test 只做最终评价。
每方法哪些阶段可进入正式候选、阈值池/tie-break/候选数和资源预算须预注册；
工程上可回放 teacher/warmup/episode **不自动允许它们进正式 PA/OA 候选池**。

## 4. 复核命令和剩余边界

```bash
PYTHONPATH=. python scripts/check_p3_admission_evidence.py
pytest -q tests/unit/estimators/test_p3_source_behavior.py \
  tests/unit/estimators/test_p3_budget_selection_evidence.py \
  tests/unit/scripts/test_check_p3_admission_evidence.py \
  tests/contract/test_p3_candidate_admission.py
```

取得相同 commit 的作者 checkout 后，可重复添加
`--source-dir pulda=/path/to/PULDA`、`--source-dir puet=/path/to/PUExtraTrees`、
`--source-dir robust_pu=/path/to/Robust-PU`、`--source-dir split_pu=/path/to/SplitPU_MM2022`。
校验范围为登记文件的真实字节；不是检查所有上游文件、更不是成功执行上游项目。
本机已对四 checkout 的 **17 个文件**核 SHA256；GradPU 无作者 checkout，不伪造 commit。

下一步由负责人确认方法特定选模/预算/表征及适配接受条件；之后才可冻结新版本和正式验证。
不代签，不清除 common/specific blockers，不登记新 numeric 锚点，不占用完整主榜资源。

## 5. 本批验收记录

- 新增来源行为/预算/检查器与既有准入契约专项：**45 passed**。
  台账文案全部同步后，加方法台账一致性复验：**55 passed**。
- 与 CI 相同选择式的全量快层：**2999 passed / 55 skipped / 36 deselected**，238.25s。
  命令为 `pytest tests/ -m 'not slow and not e2e' -q --tb=short`；本次禁用 GPU，
  skips 保留可选依赖/无 CUDA 等明确原因，slow/e2e 未执行，不称全测试或 GPU 验收。
- 环境：隔离 Python 3.11.12、PyTorch 2.13.0+cpu；数值依赖按锁定版本安装，
  Torch 使用官方 CPU 变体而非 Linux uv.lock 的 CUDA 制品。不是 frozen-lock GPU 复验。
- 11 项静态门禁通过：格式、测试质量、文档链接、API、项目元数据、数学渲染、
  Skill 同步、baseline 配置、注释、目录生成及层间依赖；`git diff --check` 通过。
- 四 checkout 的 17 文件核验通过；另以禁用 torch 导入方式核验离线检查器成功，
  证明静态检查不要求安装该可选依赖。不证明未知上游文件、人工读数或训练数值。

以上均为本地验收；本次尚无新提交的远程 CI 结果，不能借用此前绿色 CI 作为本批验收。
