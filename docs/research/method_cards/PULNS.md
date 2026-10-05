# Method Card: PULNS

Luo et al., AAAI 2021：[原论文](https://ojs.aaai.org/index.php/AAAI/article/view/17064)，
DOI 10.1609/aaai.v35i10.17064。实现依据为印刷页 8786–8788 的状态、奖励、折扣 REINFORCE、
Algorithm 1。2026-10-05 未定位到可确认的作者代码，登记 `not_found`，不把 PULSNAR 等近似名称当来源。

## 原文行为与当前实现

selector 状态拼接当前样本隐层特征、已选负例均值、P 均值；无已选负例时使用 U 均值。
Bernoulli 动作 1 表示选为负例，selector 为 64-32-1 MLP。
初始 classifier 以 P=1/U=0 预训练；每 episode 顺序采样 U、训练临时奖励 probe、计算干净 support
accuracy 减去历史最高 probe accuracy 的终端奖励，更新 selector 后重新选 N，再训练实际 classifier。
保留实际 classifier 中 support accuracy 最好的模型；奖励 baseline 与返回模型的 best score 分开。

```math
r_i=\mathrm{clip}((1-2a_i)\mathrm{logit}p(x_i),-1,1),\qquad
v_i=\sum_{t=i}^{|U|}\beta^{t-i}r_t+\alpha r_{terminal}.
```

policy loss 为 `−sum(v_i log π(a_i|s_i))`；终端奖励不随剩余步数折扣。
三类质心/特征和奖励停止梯度，仅 policy log-prob 求 selector 梯度；reward probe 和实际训练成本都计入。
空负例集合不强制选一行，记录跳过更新次数。默认 MLP32、Adam 重置与短训练配方是工程适配，
不声称作者完整架构、优化器状态管理和公开数值复现。

## 标签预算与校准

论文 p.8787 明确需要真实正/负 validation 来计算终端奖励；这是**训练信号**，不是仅 OA 选模。
组件要求显式独立 `support_data=(X_support,y_clean)`；注册 `requires_clean_support=True`，
自动 PU-only 推荐排除它。缺失支持集直接报错。不能用现有 clean_val/test 偷代 support，也不能
生成“PA 可用”结论。可传 `train_indices`/`support_indices` 验证二者互斥，缺失 IDs 时明确为
`unverified_caller_responsibility`；对其他角色的隔离仍需新实验协议。

论文 P/U 分别采样、U 的正类比例 γ 固定，登记原生 TS；该 RL 选择目标没有可安全同形替换的
未标记风险项，因此 `calibration_applied=false`，显式 ts 报错，默认 OS 只是未校准工程输入。
论文中的“calibrated negative sample set”不是本项目 TS-OS 风险校准，不因词语相同登记为已校准。

## 接口和证据

公开 `PULNSClassifier`，注册 `pulns` / `negative_selector_pu`，稠密二维/CPU/CUDA；
真实支持集、clone、pickle 和模型 sigmoid 分数，非空 sample_weight 报错。不提供逐 epoch
callback，也不把完整 episode 中的多次训练伪装成一次 epoch。
公式 golden、支持标签/身份门禁、分类器 batch 不含 support、probe baseline 与返回模型隔离、
固定种子和保存加载见 `tests/unit/estimators/test_pulns.py`。
正式 Survey 尚未准入，需要独立 clean-support 标签预算、协议/保存与来源复核；不能直接跑旧 pilot。
