# Robust-PU / Split-PU / Holistic-PU 阶段预算（2026-10-24）

承接 [GEN-PU/PULDA 阶段预算](staged_budget_20261024.md)，新增
`survey-staged-budget-2`，把三方法的原生二维路径接入同一 draft CLI、pilot、
容量预检与训练后成本收据。**不是正式 v3、P4.1-B 批准或论文数值复现。**
schema 1 继续仅对应原两方法，旧 v1/v2、算法默认、冻结台账/比较矩阵与历史结果不改。

## 1. 实际接通的区别

| 方法 | 显式锁定的预算/停止参数 | 快照与已付成本 |
|---|---|---|
| Robust-PU | pretrain / episodes / inner_epochs、batch、宽度、两段学习率 | 快照数为 pretrain+episodes；episode 内多轮均计优化器更新，不额外造快照。TS pretrain 仍按原 P/U 分批，只有 nnPU 风险角色校准；零 pretrain 仅允许 OS |
| Split-PU | teacher / splitter / student 轮数、rounds、batch、宽度、学习率、agreement threshold | 容量按 teacher+split_cap+rounds×student 上界预留；实际 splitter 在首个达到阈值的轮结束，或跑满 cap；核对全部 teacher/splitter/逐 round student，不能把早停后的完整轨迹误报缺失 |
| Holistic-PU | warmup / PN 轮数、batch、宽度、学习率、warmup_selection、continue/reinitialize、LZO alpha/size | 全 warmup + PN 均保存；LZO 全视野评估后回选状态，但所有已执行 warmup 更新/前向仍计费，不因回选或重置优化器“退款”。保持仅 OS |

三方法均要求命名 budget 的参数与 profile 锁、unit 指针一致；架构使用自己的 native
MLP 身份，不冒充共享 MLP。schema 2 不接受 schema 1 的方法或反向混用。
新增的默认架构守卫也用于 schema 1：禁止注入 model/encoder/transform 后仍使用默认
native_2d 身份和预算；不收紧原生估计器的公共 API，独立图像入口保持。

训练后、PA/OA 选择前验证全部实际快照、stage/epoch/round、逐阶段及累计更新、
文件体积上界。Split-PU 另外核对停止历史、easy/hard 对原 U 的非空划分和 student
round 覆盖；Holistic-PU 核对 LZO 候选、最早 argmin、train-P-only mixup 来源、
评估行/批次数、被回滚但仍已付的更新，以及 PN 是否重置优化器。
成功候选写 `survey-staged-budget-receipt-2`：实际快照数与声明上界分列，
预算摘要、阶段成本、实际字节与 `formal_budget_approved=false` 均保留。

同时修复汇总的预算投影：嵌套阶段预算不再回退成 `epochs=1/batch=1`。
旧 fairness gate 读取实际快照上界和 batch；PULDA 投影到驱动训练循环的 U batch，
P/U 是一组设置，不登记成两个搜索候选。详细诊断比较另核对 schema、算法及全部
显式参数，因此相同快照数但不同 GAN/inner_epochs/P batch/停止规则也不能混合。
不同磁盘字节上界不等于不同训练配方；仍由各自容量守卫核验。旧非阶段 budget 比较不变，
新扩展即使诊断比较通过仍不可正式聚合，未产生跨方法预算公平性的额外批准。

## 2. 来源、选择与验证边界

依据本仓库已接入的原生实现，而不是重新解释论文：

- `estimators/deep/robust_pu.py` 的 `checkpoint_epoch_count` 与两段训练循环：
  pretrain 每轮 `max(ceil(P/B),ceil(U/B))` 次更新；episode 每次
  `inner_epochs×ceil(N/B)` 次更新。
- `estimators/deep/split_pu.py` 的 teacher、agreement break、`_three_group_batches`：
  splitter 每轮 `ceil(N/B)` 次；student 每轮 `max(ceil(P/B),ceil(easy/B),ceil(hard/B))`
  次，按原生 round/epoch 标记重建，而不把全部轮次摊成同一个 stage epoch。
- `estimators/deep/holistic_pu.py` 的 full-horizon LZO、状态回选与 `_record_epoch`：
  warmup 每轮 `ceil(U/B)` 次，PN 每轮 `ceil(N/B)` 次；LZO 的 forward work 单列，
  不混算成 optimizer step。`N=P+U`，各角色以 core training view 为准。

预算层不改变上述目标、采样、停止/回选或初始化。选择范围沿用原生可部署快照全集，
尚未实现任意 stage 子集裁剪。Split-PU #122 的 JSD/TS/台账裁决仍保留，
没有通过预算工程测试代替审批或关闭 issue；不重新请求复核已由证据覆盖的接线选择。

专项为合成四角色 CPU 工程验证：实际 CLI 的 Robust/Split OS/TS 与 Holistic OS、
两 seed 原生 fit/记录 fit 的权重和分数逐位一致；Holistic fixed/LZO 与
continue/reinitialize 均覆盖。固定 fixture/seed 同时触发 Split-PU 提前结束及跑满上界，
Holistic 回选第2轮时仍付完整4轮 warmup。它们不是正式数据或论文效果对比。
实际 pilot 子进程也已运行，提前停止的 Split-PU 轨迹能完成 PA/OA、第二次直接续跑。
另外通过两次独立 CLI 子进程验证 Holistic LZO 的 continue/reinitialize：
各6份快照、14次已付优化器更新、12行 LZO 前向，并输出 PA/OA；仍为合成 CPU 验证。

新增53项；最终专项/回归86项通过、0跳过，
`/tmp/survey-episodic-budget-final-release-target-20261024.xml`。
前序91项身份/执行回归也通过，范围有重叠，不能将91和86相加当作新增项数。
中间快层在补汇总修复期间触发一次来源摘要不一致；冻结后来源专项已通过，
最终完整快层 **5,723 passed / 59 skipped / 0 failures/errors**，
共5,782项、370.979秒，收据
`/tmp/survey-episodic-budget-final-full-fast-20261024.xml`；
不使用中间计数作为交付通过证据。既有环境/成本/收敛告警保留，条件跳过不是 GPU 通过。
格式（含精确暂存快照的完整门禁）、测试/注释质量、文档/API、元数据/数学渲染、
skills、目录/层依赖、baseline、候选来源契约、旧 draft registry 和当前 proposal 检查通过。
离线 wheel/sdist 构建和内容核对、仓库外安装后的实际 Robust-PU 训练/阶段收据也通过；
wheel 源码与最终源码一致，不含根上传或数据/模型 payload。
用户原有 benchmark 修改及上传不编辑、不提交；未声称远程多平台 CI 已通过。
当前源码 planning JSON 只刷新来源绑定，
22算法/23 profile/184阻断请求、null 正式预算和人工签署不变。

## 3. 仍未完成

- 其他方法族与正式五角色 support 的版本化消费。
- native CNN / feature adapter 的相应架构、预算和资源绑定；当前三方法此草案仅 native_2d。
- 正式 v3 的候选/seed/预算/阶段资格及新版 registry/comparison。
- 相应真实数据资源验收与数值比较；本批无 GPU 实验、无 PN-CNN 5 seed，未代签或放行。
