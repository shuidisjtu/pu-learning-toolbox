# Method Card: PAN

## 来源与范围

Hu et al., AAAI 2021：[论文与 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/16953)，
DOI 10.1609/aaai.v35i9.16953。2026-10-05 未定位到可确认的作者代码，`not_found`
仅表示本次检索状态。本组件按论文式 (7)、Algorithm 1 独立实现，不复制第三方 benchmark 代码。
论文/代码许可不能互相替代，registry 源许可暂为 `needs_review`。

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
显式 ts 报错，不替换为 U∪P、不把有标签正例伪装为负例。SAR 只能视为假设违背压力测试。

## 接口与证据

`PANClassifier`，注册 `pan` / `predictive_adversarial_pu`；参数见 [API](../../user/reference/api.md)。
二维稠密输入或 4-D NCHW、可注入 C/D torch 网络、CPU/CUDA、clone、pickle、固定 seed、epoch callback。
snapshot 仅保存推理所需 C，非训练续跑；D 和优化器预算要计入资源，不能因只保存 C 隐去训练成本。
非空外部 sample_weight 报错；sigmoid 为模型分数，不承诺统计校准。

式 (7) probability/logit golden、minmax 梯度隔离、极端数值、训练/权重恢复及 OS-only 门禁见
`tests/unit/estimators/test_pan.py`。合成 smoke 不是公开数值复现，不进 `survey-v1.2` 冻结矩阵。

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
