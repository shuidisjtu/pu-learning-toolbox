# Self-PU 使用指南

`SelfPUClassifier` 是 sklearn 风格的 PyTorch PU 分类器，实现 Self-PU 的三个核心阶段：
动态 self-paced trusted set、clean-validation meta reweighting，以及双 student/EMA teacher
蒸馏。当前实现是可测试的 clean-room 核心，不等同于 MNIST、CIFAR-10 或 ADNI 的完整
论文复现。

## 1. 数据协议

训练集只包含 P/U 标签：

- `y_pu == 1`：可靠的 labeled positive；
- `y_pu == 0`：来自边缘分布的 unlabeled；
- `class_prior`：已知或仅由训练 fold 估计的 $`P(Y=1)`$。

完整 self-calibration 还需要独立的 `(X_val, y_val)`。这里的 `y_val` 是真实二分类标签，
可编码为 `{0, 1}` 或 `{-1, 1}`，并且必须同时包含两类。验证集用于 meta-gradient 和最终
teacher 选择，不能使用测试集替代。

## 2. 完整训练

```python
from pu_toolbox.estimators.deep import SelfPUClassifier

classifier = SelfPUClassifier(
    class_prior=0.3,
    warmup_epochs=10,
    self_paced_start=10,
    self_paced_end=50,
    distill_start=50,
    max_epochs=200,
    max_trust_ratio=0.25,
    pace_1=0.20,
    pace_2=0.30,
    require_validation=True,
    random_state=42,
    device="cuda",
)

classifier.fit(
    X_train,
    y_pu_train,
    validation_data=(X_clean_val, y_clean_val),
)
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
scores = classifier.decision_function(X_test)
```

默认 backbone 是适用于表格或展平输入的两层 MLP。图像输入可直接使用 NCHW 数组并传入
输出形状为 `(n_samples,)` 或 `(n_samples, 1)` 的自定义 PyTorch `backbone`。

## 3. 无 clean validation 的消融模式

未提供 `validation_data` 且 `require_validation=False` 时，模型会发出 `UserWarning`，并：

- 继续运行动态 trusted set；
- 不启用 validation meta reweighting；
- 继续运行 student/teacher distillation；
- 用训练集 nnPU risk 选择最终 teacher；
- 设置 `calibration_mode_="ablation"`。

这条路径不能报告为完整 Self-PU。正式实验建议始终设置 `require_validation=True`，让缺失
clean validation 直接失败。

## 4. 训练阶段和状态

两个 student 使用不同的最终 pace，实际 trusted ratio 为
`min(pace_i, max_trust_ratio)`。每次更新都重新排序完整 U pool，所以旧样本可退出，且正负
方向始终成对选择。保存的 target 是模型概率而不是硬 0/1。

拟合后可审计：

| 属性 | 内容 |
|---|---|
| `trusted_indices_` | 两个 student 的全局样本索引、soft labels 和选择方向 |
| `trusted_history_` | 每 epoch 的目标、实际规模、进入和退出数 |
| `reweight_history_` | meta 是否启用、fallback、CE 支持率和权重质量 |
| `distillation_history_` | hard-sample 比例及 student/teacher MSE |
| `training_history_` | nnPU、trusted CE、calibrated CE、一致性和总损失 |
| `best_teacher_index_` | 最终选择的 teacher 1 或 2 |
| `teacher_selection_basis_` | clean validation accuracy 或消融 nnPU risk |

`reweight_gamma` 限制有 CE 权重的样本比例；若某列 meta influence 全部非正，会使用稳定
均匀回退并记录 `ce_fallback` 或 `pu_fallback`。`sample_weight` 当前会明确抛出
`NotImplementedError`，因为静默地只加权部分损失会破坏算法语义。

## 5. Checkpoint 与复现边界

```python
checkpoint = classifier.get_training_checkpoint()
```

checkpoint 包含两个 student、两个 teacher、optimizer/scheduler、trusted set、四类历史、
类先验和最终 teacher 选择。返回的是独立副本，可交给 `torch.save`；当前接口不承诺跨不同
backbone 自动恢复实例，恢复时应使用相同构造配置和模型结构。

当前实现已经覆盖算法核心和统一 estimator contract，但以下仍属于独立 paper-like 工作：

- 官方 MNIST/CIFAR-10/ADNI 数据划分与预处理；
- 论文对应的 6-layer/13-layer/multi-branch backbone；
- 五随机种子、完整 200 epochs、消融和论文表格对齐；
- 与官方 commit、历史依赖环境的数值比较。

最小可运行示例：

```bash
python examples/minimal/10_self_pu.py
```
