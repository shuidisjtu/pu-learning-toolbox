# PULDA / PUET 准入规格草案（2026-10-04）

状态：**draft / pending_method_owner_review**。承接 [后续优先级](post_pilot_priority_plan.md) 的第 2 项。
只记录代码事实、技术接口证据及待决问题；不建立正式候选池，不扩展冻结矩阵，不宣称论文数值复现。

## 1. 预算与表征必须分别处理

| 项 | PULDA | PUET |
|---|---|---|
| 当前输入 | 稠密二维 MLP64/depth=2；或四维图像+注入可训练 encoder/MLP 头 | 稠密二维；CPU/NumPy 树 |
| 原生图像 | 已有独立 CNN 编码器工程路径；非作者 CNN/完整预处理复现 | 尚无；图像只可另立 feature-adapter 口径 |
| 当前预算参数事实 | warmup 60 + PU 60 epoch，P/U batch 16/128 | n_estimators=100，max_depth=None，min_samples_leaf=1，max_features=sqrt，max_candidates=1 |
| 训练形态 | 两阶段 Adam/cosine；每步/epoch 成本受 U 池大小影响 | 单次森林 fit；不能把 100 棵树称为 100 epoch |
| 默认 bootstrap | 不适用树采样 | False；True 是不同训练采样选择，不能事后切换 |
| 选择制品 | 有 epoch_callback，可捕获每 epoch 权重进行独立 PA/OA 选择 | 无 epoch callback，使用单模型 trajectory；PA/OA 可各选阈值，不能假造 epoch 权重 |
| 存储 | 需要新预算/结构的实测序列化 profile；不可无证借用 mlp128 配额 | 应登记森林模型持久化，不能借神经 epoch checkpoint 配额 |
| 资源 | CPU 技术测试可执行；GPU、多 seed、frozen-lock 尚待正式验证 | CPU 计时/内存；GPU smoke 不适用，任务条款修改由负责人确认 |

依据：[PULDA 方法卡](../method_cards/PULDA.md)、[PUET 方法卡](../method_cards/PUET.md)，
以及 `pu_toolbox/estimators/risk/pulda.py` / `pu_toolbox/estimators/risk/puet.py` 的构造和 fit 签名。
上述默认参数是**当前组件事实**，不是已批准的 Survey 候选值或统一公平预算。

## 2. 标签、先验、视图与选模

- 两方法当前都使用总体 class prior；运行时从批准的数据生成 metadata 取得，不根据 test 标签估计。
- 裸 estimator fit 默认 OS，而台账默认运行视图 TS；正式 runner 必须显式正确路由，不能以裸 fit 默认值替代台账。
- PULDA：TS 改 LDA / two-way-margin 中的 U 风险输入和对应 EMA，原始 MixUp/伪标签池不变。
- PUET：TS 将 P 同时作为 U 风险角色，U 质量分母采用 n_U+n_P；仍保留原 P 角色。
- 下述 smoke 使用现有 ProtocolPA / ProtocolOA 检查接口：前者仅 PU validation，后者 clean validation。
  这不代替 PA 准则方法学复核，也不决定未来正式协议必须为所有方法采用同一选模方式。
- 禁止 clean_val / test 用于候选生成、梯度训练或先验选取；角色互斥先由 DatasetBundle 门禁核验。

## 3. 已执行的独立工程验证

新增 `tests/unit/experiment/test_p3_pulda_puet_runner_smoke.py`：两方法均完成互斥四角色数据、
SCAR 标记、显式 TS、独立 PA/OA 选择、测试 accuracy/AUC 及 manifest 留痕。
**2 passed**；manifest 不是 versioned_pilot，也不携带 recipe registry binding。

测试只用合成小样本、PULDA 1+1 epoch、PUET 3 棵浅树；不将 accuracy 值当作论文复现。
历史记录（2026-10-04）：PULDA 出现一条明确的磁盘门禁警告：未声明新正式 budget/storage profile，无法估算 checkpoint 磁盘，
技术 smoke 跳过跑前估算。这是**正式准入仍缺存储 profile 的证据**，不是已通过磁盘容量验收。

2026-10-05 更新：PULDA `checkpoint_epoch_count` 已让 runner 在无显式 epoch budget 时按两阶段
总轮数估算；冻结预算仍优先，候选/重试乘数与回收后的峰值语义已由测试锁定。
合成存储探针和逐预测恢复证据见[独立推进记录](independent_progress_20261005.md)。
此修补不填写新正式 budget/storage profile，不据小 MLP 的 bytes 推算 CIFAR 上界。

后续 CNN 技术实测见[存储复核](candidate_cnn_storage_review_20261005.md)：
PULDA 默认宽度 CNN/小 head 的每份推断快照 40,792,242 字节，三种子一致、恢复通过；
另有一组合成 CUDA allocator 峰值。它不是批准结构/正式图像预算的上界，正式配额仍待决。

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest -q \
  tests/unit/experiment/test_p3_pulda_puet_runner_smoke.py
```

## 4. 正式接入前的顺序

1. 负责人确认训练阶段预算、候选超参数与 PA/OA 适用性；分别确定树预算与两阶段神经预算，不能伪作同预算族。
2. 以新版本协议登记参数/表征/先验/选模/比较组；新矩阵不能仅向旧矩阵添方法名。
3. 测量真实模型序列化体积与峰值内存，补预算/存储 profile；接入 assembler/runner 参数锁与存储门禁。
4. 预注册来源行为/公开数值可比条件，完成多 seed 资源留痕，最后方法负责人复核。

当前不得把任一步的待决参数填成已签署/locked；本页可供审核，但不代表批准本身。
