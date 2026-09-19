# Method Card: PULDA

> 参数签名与行为契约以 [API 参考](../../user/reference/api.md) 为准。本卡区分论文方法、作者发布代码和工具箱二维特征适配；当前接入不表示 Survey P3.1 已验收。

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

- 注册名 `pulda`，别名 `label_distribution_alignment`；支持稠密二维数值特征、CPU 和显式 CUDA。默认两层 MLP64 是工具箱技术适配，不是作者 CIFAR CNN。
- 默认值对齐发布脚本的关键训练量：预热/PU 各 60 epoch、P/U 批次 16/128、温度 3.5、EMA 0.85/0.5、margin 0.6、MixUp 权重 4.2、Beta 参数 11；优化器为两阶段 Adam + cosine schedule。
- `decision_function` 返回原始 logit，`predict` 在 0 阈值分类，`predict_proba` 仅为 sigmoid 分数，未经过独立概率校准。非空 `sample_weight` 明确报错。
- 已有公式 golden、两阶段轨迹、确定性、类先验覆盖、注册/pipeline 与逐 epoch 权重恢复测试；2026-09-19 在 RTX A6000（限定 0 号卡）完成 CUDA smoke。该单次技术 smoke 不替代正式多 seed GPU/资源验收。
- 当前为 **P3.1 技术预集成**。尚未登记冻结 Survey 台账/执行矩阵，也未完成共享 backbone、CIFAR 图像路径、公开数值对照、多 seed 资源记录与合作者复核，不得进入正式榜。
