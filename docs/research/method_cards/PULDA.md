# Method Card: PULDA

> 参数签名与行为契约以 [API 参考](../../user/reference/api.md) 为准。本卡区分论文方法、作者发布代码与工具箱 MLP/注入 CNN 适配；当前接入不表示 Survey P3.1 已验收。

## 论文与来源

| 字段 | 内容 |
|---|---|
| Paper | Positive-Unlabeled Learning with Label Distribution Alignment |
| Authors | Yangbangyan Jiang, Qianqian Xu, Yunrui Zhao, Zhiyong Yang, Peisong Wen, Xiaochun Cao, Qingming Huang |
| Venue | IEEE TPAMI 45(12), 15345–15363, 2023 |
| 作者代码 | [jiangyangby/PULDA](https://github.com/jiangyangby/PULDA)，锁定 `7b3dcad95bd7caa0a9477af37a05764fbe6e27bc`，MIT |
| 来源状态 | `official_related`：损失与两阶段流程对照发布代码；网络、输入模态与工程接口为工具箱适配 |

PULDA 是 Dist-PU 的扩展，需要类先验 $\pi$。发布仓库只提供 CIFAR-10 入口和自定义 CNN；锁定提交的 `losses/factory.py` 还导入了仓库中不存在的 `entropyMinimization` 模块。因此本实现以现存 `distributionLoss.py`、`twoWaySigmoidLoss.py`、`customized/mixup.py` 和 `train.py` 为可审计依据，不把“能够导入工具箱组件”等同于原仓库端到端复现。

2026-10-18 论文全文已取得并核对（TPAMI 2023, 19 页，
SHA-256 `a2cffd8e0b3275aebf8ab89a29bea11a07b598f252f228df479c8db0e16c4070`）。
论文区分 PULDA（oracle class prior，目标 `R_lab(f)`）与 PULDA\*（BBE 估计先验，
目标 `R_lab(f,h)`）；本实现对应 PULDA。协议 §3 第 2 条允许主实验向需要总体 prior
的算法传入真实 `π_population` 并标注 known-prior（oracle-prior），因此固定先验变体
符合本 Survey 主实验；PULDA\* 的 BBE 流程不在本轮实现范围。

## 核心目标

令 $p_i=\sigma(f_\theta(x_i))$，$d_T(a,b)=\mathrm{softplus}_T(a-b)+\mathrm{softplus}_T(b-a)$。发布代码的标签分布项为

```math
L_{LDA}=2\pi\left(1-\mathbb E_P[p]\right)
+d_T\left(\mathbb E_U[p],\pi\right).
```

为避免仅匹配一阶分布产生退化解，双向 sigmoid margin 定义

```math
C^+(z)=\sigma(z)\sigma(-(z-m)),\qquad
C^-(z)=\sigma(z+m)\sigma(-z),
```

```math
L_{2way}=\pi\mathbb E_P[C^+(f)]
+d_1\left(\mathbb E_U[C^-(f)],\pi\mathbb E_P[C^-(f)]\right).
```

预热阶段优化 $L_{LDA}+L_{2way}$。未标记预测均值和两个负向 margin 矩使用发布代码一致的指数移动平均；第一次直接用当前批次，后续把距离项除以 $1-\alpha$。第二阶段从预热模型初始化所有伪标签，每批强制 P 目标为 1，以 $\lambda\sim\mathrm{Beta}(a,a)$ 做 MixUp，并优化

```math
L_{LDA}+L_{2way}+w_{mix}L_{BCE}^{mix}.
```

每次更新后用该批次更新前的模型分数刷新其持久伪标签，对齐作者训练顺序。

论文 Eq. (31) 的 MixUp 目标是当前模型对原始样本的预测 `σ(f(x1))`、`σ(f(x2))`；
作者发布代码实际使用持久伪标签并强制 P 目标为 1，工具箱跟随作者代码。该
paper/source 差异保留登记；按 2026-10-18 用户决定，尊重队友实现，等训练结果出现
明显错误时再处理。

## 当前实现与边界

- 注册名 `pulda`，别名 `label_distribution_alignment`；支持稠密二维数值特征、注入 encoder 的四维图像、CPU 和显式 CUDA。默认两层 MLP64 是工具箱技术适配，不是作者 CIFAR CNN。
- 默认值对齐发布脚本的关键训练量：预热/PU 各 60 epoch、P/U 批次 16/128、温度 3.5、EMA 0.85/0.5、margin 0.6、MixUp 权重 4.2、Beta 参数 11；优化器为两阶段 Adam + cosine schedule。
- `decision_function` 返回原始 logit，`predict` 在 0 阈值分类，`predict_proba` 仅为 sigmoid 分数，未经过独立概率校准。非空 `sample_weight` 明确报错。
- 已有公式 golden、两阶段轨迹、确定性、类先验覆盖、注册/pipeline 与逐 epoch 权重恢复测试；2026-09-19 在 RTX A6000（限定 0 号卡）完成 CUDA smoke，2026-09-28 又通过全局 PyTorch 2.6.0+cu124 环境下的原生与 TS 特征路径单次 CUDA smoke。这些技术 smoke 不替代 frozen-lock 或正式多 seed GPU/资源验收。
- 当前为 **P3.1 技术预集成**。方法台账已登记；`fit(os_or_ts="ts")` 的 LDA 与 two-way margin 的 U 期望角色逐批取 `U∪P`，EMA 同步该角色。正例项、总体先验、MixUp 物理池和持久伪标签索引不变；默认视图为 `ts-compatible`，逐 run 实际值以 manifest 为准。方法身份/适配已于 2026-10-18 复核；冻结 Survey 执行矩阵尚未登记，共享 backbone/正式 CIFAR 规格、公开数值对照、多 seed 资源记录与正式准入仍未完成，不得进入正式榜。

## 原生 CNN 工程路径（2026-10-05）

2026-10-18 修复[逐子模块推理模式恢复](../pu_survey/admission/pulda_gradpu_inference_modes_20261018.md)：
预测及异常退出不再覆盖调用者单独设置的Dropout/BN模式；不改变训练BN策略或默认数值。

2026-10-18 根据 #119 补[共享图像接线](../pu_survey/admission/pulda_gradpu_image_review_20261018.md)：
可选 `train_transform` 在两个阶段逐训练 batch 重新采样，先增强、再 MixUp、最后
encoder 内归一化；推理和伪标签扫描不增强。新增独立技术构造入口使用同 seed
ResNet-18，并验证训练 BN 与模板隔离；默认仍不启用增强。统一增强/BN/表格 MLP、
正式 recipe/runner 与120快照资源实测仍未验收，不构成正式准入。

`encoder=nn.Module` 输出须为有限二维特征；其 deepcopy 被解冻训练，调用方原模板和不同 CV 折
互不影响，后接原 MLP 头。四维图像缺 encoder 时拒绝，不静默 flatten。probe 在 eval 下执行。
训练图像/观测标签保留 CPU，仅 P/U 或 MixUp batch 搬入设备；预热后伪标签以 eval 分批计算，
不让 BatchNorm 读取整份训练集或更新统计。预测同样分批，恢复原模式并校验完整空间 shape。
默认二维 MLP 的目标与阶段预算不变；图像输入上的 MixUp 是输入插值，不冒充作者完整增强协议。
两阶段 epoch 权重只支持分类器推断回放，不包含优化器/RNG/EMA/伪标签的续训状态。
单元形状/BN/模板/种子/pickle/快照测试为 `test_pulda_cnn.py`；流水线与折隔离另有集成测试。
这不决定正式 backbone、候选预算或方法负责人接受，也不改历史冻结结果。

## 准入证据补全（2026-10-05）

2026-10-07 补[阶段/预算规格](../pu_survey/p3_stage_budget_review_20261007.md)：
warmup/pu_mixup 快照保留阶段内位置与累计更新数；当前技术 selector 仍遍历两阶段。
哪些阶段可获正式 PA/OA 资格仍 null，训练目标、校准范围和默认预算未改。

P1-2 [公开对照准备](../pu_survey/p3_public_comparison_20261005.md)现将 PULDA 状态更正为
`full_text_reviewed_anchor_pending`。全文曾于 2026-10-18 核对并记录上述摘要，但该 PDF
不在当前工作区；2026-10-19 从[第一作者主页](https://jiangyangby.github.io/publications/)
重查，其 PDF 入口指向[IEEE 原文页](https://ieeexplore.ieee.org/document/10264106)，本轮浏览器
只能取得 JS/机器人验证页，无法重读表格。故目前不登记数值锚点；不以 Dist-PU 数字替代，
也不将“本地未能重新取得”解释为原论文没有结果。要抽取锚点需重新提供与上述 SHA-256 匹配的 PDF。

见[五方法证据包](../pu_survey/p3_admission_evidence_20261005.md)。锁定作者
`dataTools/PUSampler.py:15-41` 丢弃 U 尾批并循环抽 P；本实现保留尾批、P 有放回抽样，
同 epoch 不保证相同更新数。作者 `train.py:120` warmup cosine 用 pu_epochs，
本实现亦用 max(1, pu_epochs)，不是独立的 warmup 长度；不等阶段测试锁住此耦合事实。
`train.py:199-235` 跟踪 test 最高 accuracy，不等同于我方外部 PA/OA 的选模和末轮推断。
`dataTools/factory.py:14` 的 ImageNet 统计亦非我方 train-only 统计。
上游 U 的擦标签 P 副本构造不使 TS 风险角色并集等于完整 MixUp/RNG 流程重放。
新增三批 EMA 梯度对照与不等阶段/尾批更新计数测试；正式阶段准入、候选和预算仍待审。

2026-10-18 复核：按用户决定，尊重当前“风险损失逐批 `U∪P`、MixUp/伪标签保留原始 U”
的协议适配，不声称作者数据池/MixUp/RNG 完整重放。PULDA 方法身份 =
known-prior/oracle-prior PULDA；PULDA\* 的 BBE 先验估计不在本轮。
stage/budget、候选/资源和正式准入仍单独讨论。

**shuidisjtu 签署（授权代录，2026-10-18）**：仅确认上述方法身份与适配；
不包含 PULDA\*、warmup/pu_mixup 阶段资格、60+60 预算、候选/资源规格或正式 Survey 准入。

2026-10-18 阶段/预算/阈值决定（与上述签署分开记录）：PA/OA 仅允许
`pu_mixup` 阶段；`warmup` 只作诊断。正式预算采用原生 60+60、P/U
16/128、两阶段 cosine `T_max=pu_epochs=60`；120 个快照、不早停，
`pu_mixup` 内 60 个 epoch 全部可选。候选池为单一默认配置，
`class_prior=π_population` 固定，不扫描先验。分类阈值固定为 logit 0
（当前实现默认，等价于 sigmoid 0.5），阈值候选集为 `{0}`，PA/OA
只选 checkpoint、不调阈值。该决定是作者默认预算口径，
但整体方法/表征仍是 benchmark-adapted；共享表示、存储/资源、
公开数值和正式 Survey 准入仍未批准；机器草案的
`formal_decisions` 仍为 `null`。

2026-10-18 图像表征决定：按协议 §2.5 使用 ResNet-18；PULDA 采用
**端到端 ResNet-18**（注入 `encoder` 可训练），不走 feature-adapter，
结果标为 `end-to-end CNN` / `benchmark-adapted`，不声称作者 CNN 或论文
数值复现。共享初始化、BN、每 epoch 增强、输入/归一化及表格 MLP 共享规格
待 [#119](https://github.com/shuidisjtu/pu-learning-toolbox/issues/119)
与队友确认；本决定不改变阶段/预算/阈值，也不构成正式准入。
