# 神经 LBE 独立工程接入（2026-10-14）

补齐线性 LBE 尚不具备的双网络 Adam generalized-EM 工程路径，以新名称
`lbe_neural` / `NeuralLBEClassifier` 登记。原 `lbe`、冻结协议/比较矩阵与历史 645-run 数值均不变。
这是独立技术预集成，不是作者包执行、论文表格复现或人工准入签署。

## 来源与采用范围

沿用[10-12 源码复核](../data/lbe_source_review_20261012.json)，本轮重新核验用户包的
`LBE_MLP/train.py` 与 `em.py` 字节 SHA256，均与原复核记录一致。
用户包 `86034d5b…` 与旧官方锁 `79cc2c36…` 不等价；采用前者的 MLP 时序，不覆盖旧官方来源锁。

- `em.py:3-16`：latent class 联合概率归一化；P 后验固定为 1，U 按本卡公式更新。
- `train.py:27-45`：class-balanced observed-label classifier 预训练，随后用
  `s+(1-s)*r` 的固定权重训练 propensity。每个模型 20 steps。
- `train.py:62-81`：20 个 E rounds，各有 **40** 个 M updates（不是未使用的 `M_STEPS=20`），
  每个 update 重新无放回取 10% 训练行；q detach 后固定，propensity BCE 必须乘 q。
- 源 demo 的读数据、全池统计、训练真值阈值和 repeated test diagnostics 不迁入 fit。
  repeated test 调用在该源程序中是诊断，不据此指控它按 test 选模。

独立源码仅实现数学公式/已核流程，不复制、执行或再分发上游代码。
官方变体归属和上游 license 仍待核；registry 使用 `official_related` / `needs_review`，不是 exact replay。

## 工程能力与差异

默认双 MLP100/tanh；支持各自 deepcopy 的可训练 encoder+head，两个网络不共享参数或模板所有权。
CPU 保存原始数据、观察标签和 E 后验，设备前向按 batch_size 分块；每次源 optimizer update
累积各块按行数加权的梯度后只执行一次 Adam step。

| 内容 | 本实现 | 不应据此声称 |
|---|---|---|
| E/M 数值 | log-space posterior + stable BCE-with-logits，q 停止梯度 | 与源码 epsilon 概率运算逐位相等 |
| 小样本抽样 | `max(1,floor(N*fraction))` | 源程序 0 行采样也能运行 |
| CNN | 独立注入两个可训练 encoder，分批预测 | 原作者 CNN/增强或 BN/dropout 分批语义相同 |
| 预算 | 默认预训练 40 + EM 800 = 840 次 Adam；20 EM callbacks | 840 个 checkpoint 或正式资源上界 |
| checkpoint | classifier 推理快照和 EM/累计 optimizer provenance | optimizer/propensity 完整续训状态 |
| 持久化 | standalone pickle 保存两网络，保留概率输出 | untrusted pickle 可安全加载 |

`checkpoint_epoch_count` 显式对应 EM 轮数，已有 runner 预检可消费该值；真实网络序列化/GPU峰值仍待批准配置测量。
不传 clean_val/test，不在 fit 里选择阈值或 best-test；训练似然与 Q-loss 分开记录，均为诊断，不能据此保证单调或全局收敛。
原生 OS joint likelihood 不适用 TS 的 U∪P 替换：显式 TS 和 sample_weight 均拒收，未假报已校准。

## 接入与复核边界

- 根包、registry（别名 `neural_lbe`）、方法台账、能力声明和测试 factory 同步。
- 通用 CLI/PUPipeline 可通过注册名访问；正式 Survey matrix 未追加该名称，未生成 locked recipe 或放行主榜。
- 代码由 shuidisjtu 侧独立工程推进，台账明确 HENG958 的 `pending_collaborator_review`；没有代签。
- 原 12 项新增方法的 RV8 待审清单不被此变体隐式改变；此变体单独待接口/方法学/预算复核。
- 待完成：来源版本/许可确认、正式神经/CNN 输入规格与预算/选模批准、资源测量、真实数据/GPU及独立制品验收。

## 验证

独立 joint normalization/解析梯度、极端 logits、固定 E/M 次数、预训练成本、分块梯度等价、
双 CNN 副本训练、clone/pickle/预测拒收、RNG/先验不变性、台账/registry、checkpoint 恢复和四角色 runner 均有回归。
test 的真值和特征被替换后，训练与 PA/OA 选择模型不变；这只是合成链路隔离证据。
最终验证（Python 3.11、CPU）：新增三文件 **47 项定向回归通过**；完整 fast suite
`pytest tests/ -m 'not slow and not e2e'` 为 **3749 passed / 59 skipped / 0 failures / 0 errors**。
registry 类声明单源门禁、UI CNN 能力清单与共享 factory 已同步，未放宽现有检查。
格式、测试质量、文档链接/API 覆盖、元数据、Skills 同步、MathJax、baseline 配置、注释质量、
目录结构/层依赖和 materialized draft 检查通过；源码包/wheel 离线构建成功。
包内包含新增实现，不包含上传 RAR 或私人 Excel；未发布或上传包。
不据此声明跨平台 CI、slow/e2e、正式 GPU/真实数据/作者源码重放或人工验收。
