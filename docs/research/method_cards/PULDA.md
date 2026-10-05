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

## 当前实现与边界

- 注册名 `pulda`，别名 `label_distribution_alignment`；支持稠密二维数值特征、注入 encoder 的四维图像、CPU 和显式 CUDA。默认两层 MLP64 是工具箱技术适配，不是作者 CIFAR CNN。
- 默认值对齐发布脚本的关键训练量：预热/PU 各 60 epoch、P/U 批次 16/128、温度 3.5、EMA 0.85/0.5、margin 0.6、MixUp 权重 4.2、Beta 参数 11；优化器为两阶段 Adam + cosine schedule。
- `decision_function` 返回原始 logit，`predict` 在 0 阈值分类，`predict_proba` 仅为 sigmoid 分数，未经过独立概率校准。非空 `sample_weight` 明确报错。
- 已有公式 golden、两阶段轨迹、确定性、类先验覆盖、注册/pipeline 与逐 epoch 权重恢复测试；2026-09-19 在 RTX A6000（限定 0 号卡）完成 CUDA smoke，2026-09-28 又通过全局 PyTorch 2.6.0+cu124 环境下的原生与 TS 特征路径单次 CUDA smoke。这些技术 smoke 不替代 frozen-lock 或正式多 seed GPU/资源验收。
- 当前为 **P3.1 技术预集成**。方法台账已登记；`fit(os_or_ts="ts")` 的 LDA 与 two-way margin 的 U 期望角色逐批取 `U∪P`，EMA 同步该角色。正例项、总体先验、MixUp 物理池和持久伪标签索引不变；默认视图为 `ts-compatible`，逐 run 实际值以 manifest 为准。冻结 Survey 执行矩阵尚未登记，且共享 backbone/正式 CIFAR 规格、公开数值对照、多 seed 资源记录与合作者复核仍未完成，不得进入正式榜。

## 原生 CNN 工程路径（2026-10-05）

`encoder=nn.Module` 输出须为有限二维特征；其 deepcopy 被解冻训练，调用方原模板和不同 CV 折
互不影响，后接原 MLP 头。四维图像缺 encoder 时拒绝，不静默 flatten。probe 在 eval 下执行。
训练图像/观测标签保留 CPU，仅 P/U 或 MixUp batch 搬入设备；预热后伪标签以 eval 分批计算，
不让 BatchNorm 读取整份训练集或更新统计。预测同样分批，恢复原模式并校验完整空间 shape。
默认二维 MLP 的目标与阶段预算不变；图像输入上的 MixUp 是输入插值，不冒充作者完整增强协议。
两阶段 epoch 权重只支持分类器推断回放，不包含优化器/RNG/EMA/伪标签的续训状态。
单元形状/BN/模板/种子/pickle/快照测试为 `test_pulda_cnn.py`；流水线与折隔离另有集成测试。
这不决定正式 backbone、候选预算或方法负责人接受，也不改历史冻结结果。

## 准入证据补全（2026-10-05）

见[五方法证据包](../pu_survey/p3_admission_evidence_20261005.md)。锁定作者
`dataTools/PUSampler.py:15-41` 丢弃 U 尾批并循环抽 P；本实现保留尾批、P 有放回抽样，
同 epoch 不保证相同更新数。作者 `train.py:120` warmup cosine 用 pu_epochs，
本实现亦用 max(1, pu_epochs)，不是独立的 warmup 长度；不等阶段测试锁住此耦合事实。
`train.py:199-235` 跟踪 test 最高 accuracy，不等同于我方外部 PA/OA 的选模和末轮推断。
`dataTools/factory.py:14` 的 ImageNet 统计亦非我方 train-only 统计。
上游 U 的擦标签 P 副本构造不使 TS 风险角色并集等于完整 MixUp/RNG 流程重放。
新增三批 EMA 梯度对照与不等阶段/尾批更新计数测试；正式阶段准入、候选和预算仍待审。
