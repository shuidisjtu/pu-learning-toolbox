# Self-PU 完整 clean-meta 的显式 TS 接线（2026-10-23）

已新增 `SelfPUTSCleanSupportCheckpointTrainer`，复用现有五角色 support 制品与
`run_clean_support_experiment.py`，显式指定 `--training-view ts` 后可完成单候选
SCAR TS-compatible、真实 clean-meta、两 teacher 流的 `PA+clean-support`/OA
独立选择与回放。**仍是独立 CPU/MLP 工程链，不是正式 v3 单元或预算批准。**
默认 OS/terminal、原 OS checkpoint adapter、LaGAM/PULNS 与冻结 Pilot no-meta 消融不变；
未改 Self-PU 估计器目标、默认参数、冻结 ledger/协议/比较矩阵、旧结果或人工签署。

## 接线与校准边界

之前缺的不是 estimator 的 TS 实现，而是 support adapter 与 CLI 固定传 OS。
新路径显式传 `os_or_ts="ts"`，不能只在 manifest 写一个校准标记；旧适配器仍拒绝 TS。
Self-PU TS 只接受 `meta_checkpoints`，LaGAM/PULNS 或终点模式不被偷偷切到 TS。
新 plan 身份为 `clean-support-cpu-ts-checkpoint-plan-v1`，视图纳入摘要并在执行前重建校验；
OS plan schema 保留，缺省仍是 OS。

- 训练只收到 train 的观测 PU 标签、总体先验、独立 support；不转发 pu_val/clean_val/test。
  support 必须已从原 train 移除并隔离，正负标签均计训练期标签预算。
- 强制 `require_validation=True`、`distillation_scope=paper_population`，且实际执行 meta 更新；
  缺失 support、仅 warmup、降级消融、角色/组冲突、SAR、普通 PA 或正式协议/registry 都拒绝。
- 原 TS 负边缘项按“未信任 native U 的加权负损失”与 P 的行数质量混合。
  非均匀 meta 权重下**不是普通并集均值**，不声称已证明总体混合恢复；不为 P 学 meta 权重。
  trusted/pace/伪标签/meta 的候选人口仍是 native U，不将 P 或 support 加入其身份池。
  改变权重轨迹后实际 trusted 成员可改变，“人口不变”不是“选择结果必相同”。
- student consistency 仍覆盖 P+未信任 U、teacher consistency 覆盖 P+全部 U；
  TS 不扩张这些定义域。已标记 P 不会被 trusted manager 赋伪标签。
- 只记录已完成 meta-active epoch 的两份 EMA teacher，warmup 不参与本工程方案选模。
  warmup 的更新仍计成本；`optimizer_steps` 来自 optimizer.step 后写入的 student 记录，
  不把虚拟 meta 探针、EMA 或快照数量当作 optimizer step。
- PA 必须明确叫 `PA+clean-support`，不是纯 PU PA；外部两轨独立选择 teacher/epoch/阈值，
  不能用内部 support 选出的单个最终 teacher 替代。test 只用于选后评价和回放。
- 共享 checkpoint bridge 的训练后字节/收据拒收也会撤销 fitted 状态、移除成功 provenance，
  保留文件；异常/取消不改跑裸 fit。该失败守卫同时覆盖原 OS 三方法，不改变正常训练。

来源与差异沿用[锁定代码行为说明](../admission/selfpu_source_behavior_review_20261019.md)、
[一致性定义域说明](../admission/selfpu_consistency_scope_review_20261019.md)与
[RV6 技术核验](../reviews/rv6_ts_calibration_technical_review_20261019.md)。
原作者程序的 test/meta/选模数据流不照搬，本适配器只用独立 clean support；
TS 是项目已有质量守恒校准扩展，不描述成原论文/作者程序完全重放。
本轮只接线与验证，非新增数学目标，不要求合作者重审已验证的工程细节；
正式 full-meta TS 口径、阶段/预算/单元仍走版本化协议，不代签 RV5/RV6 或 v2 review items。

## 真实 Spambase 同条件短运行

只读复用 `data/clean-support-20261022-yg0avvvx/spambase/split_0`，未重建/改写输入。
新建 OS/TS 两结果目录 `data/selfpu-support-paired-20261023-{os,ts}-v4`；
此前同批初测目录保留，不覆盖历史结果。
公开机器摘要：[selfpu_ts_support_validation_20261023.json](../data/selfpu_ts_support_validation_20261023.json)。

| 配对条件 | 两次实际值 |
|---|---|
| split / seed / SCAR c | 同制品 / 0 / 0.5 |
| 剩余 train / pu_val / clean_val / test | 3212 / 184 / 184 / 921 |
| support / 监督预算 | 100（46正、54负）；观测 P630 + support100 = 730标签 |
| 总体先验 | 同制品的0.39404477287546186，不从 test 估计 |
| 预算 / 模型 | MLP hidden4，batch64，3 epoch；warmup0、self-paced 0→1、distill2 |
| meta / teacher 快照 | 6条 active meta 记录；3轮×2 teacher = 6份快照 |
| 快照 payload 峰值 / 保留 / 回收 | 17,616字节 / 2,936字节（1份）/ 5份 |
| 告警 / 重试 / 回放 | 0 / 0 / 两轨均成功 |

输入收据、seed、参数、先验及 train/pu_val PU 标记摘要已逐项配对，只有训练视图切换。
原生实现每 epoch 抽一对 P/U batch、各 student 更新一次，**不是对3212行完整遍历**；
3 epoch 实际6次 student 更新。没有为探针改写原生训练循环，也未把这种短预算当正式预算。
单组件声明65,536字节，容量预检按3轮×2组件×1候选×2尝试=786,432字节；
不以选后1份抵扣峰值，不把 JSON/pickle、optimizer、RAM 纳入 weights-only payload。
两轨可选同一文件，阈值仍各自独立。

该短运行的 test 指标仅用于回放自检，不能用于判断 TS 比 OS 更好，或评价校准是否提高精度。
数据是完整已准备 Spambase 角色，但训练仅单 seed/3轮/小 MLP，不是正式预算、多 seed 或论文数值复现。
加载器核对产品摘要/IDs/预算，不在本次运行重读 raw；
`support_source_verified=false`、`raw_label_mapping_verified=false` 保留，不能拿已有 raw 审计文档
偷偷把运行中的来源断言升级为本次重新核验。既有 raw 逐行材料见[数据交付](clean_support_materialization_20261022.md)。
基线 `6de2daf` 加本轮 dirty 源码，以实际逐文件 SHA 绑定，不倒填之后的提交/环境锁。

## 复用命令

保存一个显式短测参数 JSON（下例非正式配方）：

```json
{"device":"cpu","hidden_dim":4,"batch_size":64,"max_epochs":3,"warmup_epochs":0,"self_paced_start":0,"self_paced_end":1,"distill_start":2}
```

```bash
PYTHONPATH=. python scripts/run_clean_support_experiment.py \
  --artifact-dir data/clean-support-20261022-yg0avvvx/spambase/split_0 \
  --method self_pu --c 0.5 --parameters /path/to/parameters.json \
  --training-view ts --selection-scope meta_checkpoints \
  --checkpoint-bytes-per-component 65536
```

默认只规划；执行另加 `--execute --out-dir <不存在的新目录>`。
将视图改 OS 可做同条件配对，不能在计划生成后直接改字段绕过摘要。
输出包含实际资源、完整角色/预算/视图、PA+clean-support/OA 选择与可信模型回放。
数据/模型不入 Git，只登记摘要；禁止加载未知来源 pickle。

## 验证与后续

测试验证新 TS 参数确实抵达原生 fit，负风险实际启用角色混合、known P 不进入 trusted population；
同配置、两 seed 的原生 full-meta TS 与桥接训练逐位比较四网络权重/history/评分一致。
选模/test 角色扰动不改训练权重，warmup 排除但成本保留，超字节拒收撤销 fitted、保留证据。
CLI 验证显式 dry-run、OS 默认、TS 不错接其他方法、计划视图篡改和真实双模型回放。
公开收据契约只核元数据，不等于 CI 能访问本机输入或独立复现实验。

最终完整快层：**5,581 passed / 59 skipped / 0 failures/errors**，共5,640项，
370.947秒；本机 `/tmp/selfpu-ts-support-final-full-fast-20261023.xml`。
既有 NVML/计算成本/收敛等告警保留，不能把全仓回归称作零告警。
新增26项与既有 checkpoint CLI 合计35项专项通过、0跳过；本机
`/tmp/selfpu-ts-support-release-target-20261023.xml`。此前原生视图/结构与新路径专项54项亦通过。
本批未做 GPU 实验，普通环境 CUDA skip 不算 GPU 通过。

当前源码、五角色输入和输出/选中权重摘要已与实际文件核对；回收文件确实不存在。
Self-PU 原生 estimator 与基线 `6de2daf` 字节相同；改动只在适配器、命令与文档/测试。
格式（不含原有未提交 benchmark 修改）、文档/API、元数据/数学渲染、skills、测试/注释质量、
目录/分层依赖、baseline、候选 fit 来源契约和既有 draft registry 检查通过。
原 `benchmarks/deep_pu/official_data.py` 修改与上传保留、不提交，不代替用户格式化。
离线 wheel/sdist 构建和包内容检查通过；独立临时 target 安装后在仓库外实跑
TS full-meta 双轨选择与可信模型回放通过，复用本机依赖，不是新建 frozen-lock 环境。
不据此声称远程多平台 CI 已通过。

下一步：protocol v3 的独立 full-meta profile/unit、support 正式消费、预算与阶段资格冻结，
再按新规格做正式数值/资源验收。当前工程 TS 入口不替代这些要求，不修改人的签署。
PN-CNN 5 seed 按用户指令继续暂停。
