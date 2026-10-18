# Method Card: PAN

## 来源与范围

Hu et al., AAAI 2021：[论文与 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/16953)，
DOI 10.1609/aaai.v35i9.16953。2026-10-18 复核定位到作者相关源码
[`morning-dews/PAN`](https://github.com/morning-dews/PAN) @
`5be703639e4c1eb2c63d869ba150d84ebf0c116e`；README 写“Code for paper
Predictive Adversarial Learning from Positive and Unlabeled Data”，论文正文未直接给出
该链接，仓库无 LICENSE，`source_status=official_related`。本组件按论文式 (7)、
Algorithm 1 独立实现，不复制源码；作者源码与论文的目标/采样差异见
[源码差异](#源码差异2026-10-18)。论文/代码许可不能互相替代，registry 源许可暂为
`needs_review`。

## 目标与训练

令 D、C 分别为判别器和预测器；P、U 期望分别取均值：

```math
V(D,C)=E_P\log D(x)+E_U\log(1-D(x))+
\lambda E_U[(\log(1-C(x))-\log C(x))(2D(x)-1)].
```

每步先最大化 V 更新 D，再从 U 重新采样，最小化最后一项更新 C；非活动网络停止梯度。
使用 logits 恒等式 `log(1−sigmoid(c))−log(sigmoid(c))=−c` 避免极端概率溢出。
不生成新样本、不要求先验，预测来自 C，不误用 D。Adam 与论文一致；默认二维 MLP128、epoch/λ
为工具箱工程配方，不是已批准公平预算或论文图像架构。

## 抽样与校准

论文印刷页 7810（PDF 第 5 页）脚注 5 明确 **single-training-set**；P 随机标记，U 为剩余样本。
因此 `native_sampling_assumption=os`、`run_view=os-compatible`、`calibration_applied=false`。
**2026-10-18 复核发现作者源码 `cfg/*/uPU` 用 `overlap=True`：U 为完整训练池（含被标注
正例），P 为标注正例副本，属 case-control/overlap，不是论文正文的剩余 U；PU-Bench fixture
也标 `scenarios: [case-control]`。** 本轮决定正式 Survey 跟随论文的
single-training-set/OS；源码差异见[源码差异](#源码差异2026-10-18)。
显式 ts 报错，不替换为 U∪P、不把有标签正例伪装为负例。SAR 只能视为假设违背压力测试。

## 接口与证据

`PANClassifier`，注册 `pan` / `predictive_adversarial_pu`；参数见 [API](../../user/reference/api.md)。
二维稠密输入或 4-D NCHW、可注入 C/D torch 网络、CPU/CUDA、clone、pickle、固定 seed、epoch callback。
snapshot 仅保存推理所需 C，非训练续跑；D 和优化器预算要计入资源，不能因只保存 C 隐去训练成本。
非空外部 sample_weight 报错；sigmoid 为模型分数，不承诺统计校准。

式 (7) probability/logit golden、minmax 梯度隔离、极端数值、训练/权重恢复及 OS-only 门禁见
`tests/unit/estimators/test_pan.py`。合成 smoke 不是公开数值复现，不进 `survey-v1.2` 冻结矩阵。
来源/差异证据：作者 `main.py:351-423`、`helper/pu_learning_dataset.py` 的 `overlap=True`、
PU-Bench `train/pan/core.py`、PUBench `core/algorithms.py:1694`；见
[Issue #123](https://github.com/shuidisjtu/pu-learning-toolbox/issues/123)。

## 源码差异（2026-10-18）

作者源码 `main.py:351-423` 的 PAN 分支与论文 Eq.(7) 不一致：

- 论文 U 项是 `E_U log(1-D(x_u))`；源码是
  `sum(max(log(1-D(u_h))-log(1-mean(D(u_h))), 0))`，只取前 `n_P` 个 U。
- 源码用 `eps=1e-2` 平滑，并对 P/U_h/U 分别求和；论文 Eq.(7) 是期望。
- 源码 C 更新复用同一 batch；论文 Algorithm 1 写重新采样 U；本实现跟随论文重新采样。
- 源码 `main.py` 默认 `n_epochs=200`、Adam `betas=(0.5,0.999)`、`weight_decay=1e-3`；
  本实现默认 `max_epochs=100`、Adam 默认 betas、无 weight decay。
- 源码 `PuDataIterator` 按数据集 P/U 比例组 batch；论文写 1:1；本实现使用 1:1。
- 源码配置 `overlap=True`：U 为完整训练池、P 为标注正例副本（case-control/overlap）；
  论文/本实现为 single-training-set、U 为剩余正负样本。

PU-Bench `train/pan/core.py` 与 PUBench `core/algorithms.py:1694` 的 PAN 都跟随源码的
max-margin/sums/eps 形式；本实现跟随论文 Eq.(7)。2026-10-18 决定（shuidisjtu）：
**保留当前 paper-faithful 实现**，正式 Survey 采样与论文一致；上述差异只登记，
不改 `pan.py`，不声称 source-faithful。详见
[Issue #123](https://github.com/shuidisjtu/pu-learning-toolbox/issues/123)。

## 端到端 CNN 工程路径（2026-10-05 追加）

`encoder` 是共享**初始化模板**而不是拟合后的共享网络：C、D 分别 deep-copy，分别拥有
权重、梯度和 BN 统计；两副本解冻后各自训练。源模板即使冻结也不被改动。
4-D 无 encoder 直接拒绝，不展平冒充 MLP；encoder 输出须为有限 2-D 特征。
有 encoder 时 `model` / `discriminator` 是独立 heads，默认各一层 linear；无 encoder 时
保持原二维 MLP 配方。C 的 epoch snapshot 包含它的编码器与 head，不包含 D 或 optimizer。
整个可信 pickle 保留 C/D；不加载外部 pickle。输入批次按需传 GPU，不把整份图像搬入显存。

probe 与非活动网络使用 eval/no-grad，因此 C/D 的 BN 只由各自更新的 batch 改变；
测试分别计数并确认权重确实变化，而不是仅声明 `trains_encoder=true`。
`tests/unit/estimators/test_pan_cnn.py` 验证 clone、固定 seed、输入/encoder 门禁、
保存加载、权重恢复与 GPU；`tests/integration/test_pan_pipeline_cnn.py` 验证公开 pipeline
的 native_cnn provenance，`test_cv_fold_isolation.py` 验证两个对抗 encoder 跨折均不泄漏。
这是可训练 CNN 接口，不宣称论文完整结构、预处理/增强、训练预算或公开表格重现。
