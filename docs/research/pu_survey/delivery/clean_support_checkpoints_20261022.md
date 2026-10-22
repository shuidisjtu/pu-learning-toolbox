# Clean-support 独立 checkpoint 双轨接线（2026-10-22）

> 2026-10-23 补记：下文保留原 OS 交付截点；另新增[Self-PU 显式 TS full-meta 入口](selfpu_ts_support_20261023.md)。
> 原 OS adapter/default 不变，LaGAM/PULNS 不随之启用 TS；正式 v3/预算仍待。

## 本批交付

按最新执行计划 §1.5 推进 P3.2；不重开 P3.1/RV8，不启动正式数据跑批。
新增 `SelfPUCleanSupportCheckpointTrainer`、`LaGAMCleanSupportCheckpointTrainer`
（`pu_toolbox.experiment.support_checkpoints`），显式继承现有全角色/support 审计。
旧 terminal OA adapter 与 CLI 默认行为不变；Self-PU 估计器默认消融和旧 Pilot 不变。

- 只支持 exploratory SCAR OS，拒绝 versioned Survey/SAR/TS 和普通 `ProtocolPA`；
  PA 必须显式使用 `ProtocolPACleanSupport`，输出键为 `PA+clean-support`，绝不写成纯 PA。
- 实际训练只得到 train 的观测 PU 标签、显式 support 及适用的总体先验；bridge 不向 fit
  转发外部 `pu_val`/`clean_val`/test。没有逐候选重复训练、optimizer 重置或支持集暗取。
- Self-PU 强制 `require_validation=True`、`distillation_scope=paper_population`，且验证
  真实 meta 更新发生；同时记录两个 EMA teacher 的所有完成 meta-active epoch。
  外部 PA/OA 各自从两个 teacher 流选择，不能复用最终 support 选出的 teacher 冒充双轨选择。
- LaGAM 只记录完成 warmup 之后的 meta epoch。warmup 不属于本探索方案的候选池；
  这是显式工程选择，不是已批准正式阶段资格。PULNS 的 episode checkpoint 暂未接通，
  这是第一批截点；同日第二批已补齐，见末节。旧终点 OA 入口始终保留。
- 每个外部协议独立选择 candidate、checkpoint/component、阈值及验证仿射；
  原 PA proxy criterion 不变，PA 只见 PU validation；OA 只见 clean validation。
- 复用 weights-only writer，记录逐快照身份、实际字节与全部选择信息；选择完成后，
  可按 D25 回收未选中 checkpoint，保留选中权重和被回收项原始路径/SHA。
  同时保存两份选中预测模型，仅允许加载本次自己生成的可信 pickle。

## 命令入口

复用 `scripts/run_clean_support_experiment.py`，仍默认 dry-run/terminal OA。
以下仅为**合成 CPU 工程示例**，不是正式配方、支持集预算或共享图像规格：

```json
{"device":"cpu","hidden_dim":4,"batch_size":32,"max_epochs":3,"warmup_epochs":0,"num_clusters":2}
```

将上面的 LaGAM 参数保存到自己的 JSON 后，在仓库根运行：

```bash
PYTHONPATH=. python scripts/run_clean_support_experiment.py \
  --artifact-dir path/to/independent-support-product \
  --method lagam --c 0.5 --parameters path/to/parameters.json \
  --selection-scope meta_checkpoints --checkpoint-bytes-per-component 65536

# 显式执行一次，输出目录不得存在：
PYTHONPATH=. python scripts/run_clean_support_experiment.py \
  --artifact-dir path/to/independent-support-product \
  --method lagam --c 0.5 --parameters path/to/parameters.json \
  --selection-scope meta_checkpoints --checkpoint-bytes-per-component 65536 \
  --out-dir path/to/new-results --execute
```

Self-PU 同时须提供 `self_paced_start`、`self_paced_end`、`distill_start`，且预算必须进入
meta-active epoch。CLI 仅接受一个显式 CPU/MLP 候选；数组 API 的 runner 可显式传候选池，
各候选都必须满足 meta 预算和完整分支约束。候选搜索的正式预算没有在此获批。

新 plan schema 为 `clean-support-cpu-checkpoint-plan-v1`，绑定产品收据、PU 标记、构造
参数、所用源码/锁文件、选择协议、snapshot 上界和容量估算。`plan_payload_sha256`
防止误改计划后执行；执行前还重新构建并对账，不是人为批准的电子签名。
输出包含 plan/environment/manifest/summary、`selected_PA+clean-support.pkl`、
`selected_OA.pkl` 及选中的 weights-only checkpoint。两份模型都复核原始分数、固定
验证仿射/阈值与 test accuracy 回放；绝不使用 test 重新选择。

## 容量与边界

容量参数是调用方明确提供的**单组件序列化字节上界**。上例 65,536 只用于小 MLP，
不是 CNN 或正式资源上界。runner 在同一实际 writer 根目录预检，按最大候选 epochs ×
组件数 ×候选数 × 2 次尝试保守估算；warmup 过滤和选后回收不被用来隐瞒训练期峰值。
若实际单文件大于声明值，本尝试拒收并保留证据；普通探索性磁盘不足告警按原 runner
规则处理，CLI 对告警/重试均不接受为成功工程证据。

`extra_supervision` 与 candidate provenance 保留全部角色/组隔离、支持标签数、实际 P 数、
总训练标签预算、未核验来源和额外监督分组。`pa_eligible=false` 指纯 PU PA；新的
`pa_clean_support_eligible=true` 只表示本探索接口可执行额外监督 PA。
`formal_admission=false`、`budget_approval=not_granted_by_adapter` 保持不变。

机器协议 v2 的 `v2_p4_clean_support_policy`、`v2_p10_self_pu_scope` 仍待复核；
本入口保留这些 pending 项，不代签、不添加/修改 v2 执行单元，不修改历史 645-run、
比较矩阵或原制品。正式接入仍需 protocol v3 的完整分支单元、数据/预算和后续验收。
Self-PU native-TS 校准的正式 full-meta 接线也不能用本 OS-only 路径替代。

## 验证范围与后续

真实合成 CPU 拟合覆盖 Self-PU/LaGAM、候选池、两 teacher 流、meta-only 快照、PA/OA
独立选择、test/其他选模角色扰动、pickle 回放、磁盘估计和 D25 回收；守卫覆盖角色交叉、
普通 PA/重复协议、失效 support、meta 预算、消融降级与缺失容量参数。
CLI 覆盖 dry-run、实际制品加载/训练/两模型回放、计划改动拒绝；原终点 OA 测试继续执行。
没有读取用户私有 Excel、压缩包或启动真实数据/GPU 跑批。

第一批后的剩余项：PULNS episode 接线（随后第二批已补齐）、真实 support 制品、正式 profiles/units、Self-PU 全量正式
协议路径、图像/CNN 规格与真实 smoke；CVIR 校准及其他方法 recipe 仍按执行计划推进。
本工程交付不等于 P3.2 整体完成。测试结果在本批验收后另附，不外推远程 CI 已通过。

第一批最终验收：专项 **145 passed / 0 skipped / 0 failures / 0 errors**（5.999秒），
本机 XML `/tmp/support-checkpoint-target-final-fixed-20261022.xml`。
完整快测 `pytest tests/ -m 'not slow and not e2e'` 为 **5,381 passed / 59 skipped /
0 failures / 0 errors**（5,440项、321.634秒），
XML `/tmp/support-checkpoint-full-fast-fixed-20261022.xml`。
格式、文档/API、目录/层依赖、测试质量、数学渲染、元数据、skills/注释、baseline、
候选 fit 来源绑定和 materialized draft registry 门禁通过。CUDA 条件跳过及未执行的
slow/e2e 不计通过；既有 solver/代价/消融/NVML 告警保留，不声称 GPU 或远程 CI 验收。

## 同日第二批：PULNS completed-episode 接线

新增 `PULNSClassifier.fit(..., epoch_callback=None)`，默认 None 不改变原轨迹。
每个 episode 的 classifier 更新、support 评估和 history 完成后调用一次，标签从0开始；
不额外回调预训练或内部 classifier epoch，不保存 reward probe / selector。
回调发生在最终恢复 support-best 模型之前，因此各快照真实代表该 episode 的当前
classifier，不会把同一内部 best 重复包装成各 episode。正常终点仍保持原 support-best。
回调异常/取消由原 failed-fit guard 撤销公开拟合状态；不重试裸 fit 或悄悄省略回调。

`PULNSCleanSupportCheckpointTrainer` 接入同一角色审计、容量预检、weights-only writer、
`PA+clean-support`/OA 独立选择与选后回收。API provenance 用
`selection_scope=completed_episodes`、`captured_episodes`，history/stage cost 保持实际
所有 pretrain/reward-probe/policy/classifier 更新，不将 snapshot 数当成训练步数。
CLI 同样显式用 `--selection-scope meta_checkpoints`（这个模式名也覆盖 RL episodes），
支持 `--method pulns`，但参数须换成 PULNS 的显式预算，例如：

```json
{"device":"cpu","hidden_dim":4,"batch_size":32,"pretrain_epochs":1,"episodes":3,"classifier_epochs":1}
```

上述仍只是合成 CPU 小模型示例，不批准正式100/500支持集、训练预算或GPU规格。
纯 PU PA 不适用、TS 校准不适用、原终点 OA、v2 pending 和 formal_admission=false 均保留。
本轮同步当前候选 fit 契约的 PULNS 源码 SHA/新增 callback 签名，不重写历史资源测量。
新增 callback 默认兼容、阶段/时序、单 episode、早失败、异常/取消与隐式 support 拒绝回归；
共享双轨和 CLI 测试扩展到三种方法。正式数据、完整协议单元和 P3.2 其他事项仍未完成。

第二批最终验收：专项 **226 passed / 2 CUDA-only skipped / 0 failures / 0 errors**
（228项、17.334秒），XML `/tmp/pulns-support-final-target-20261022.xml`。
完整快测 **5,408 passed / 59 skipped / 0 failures / 0 errors**
（5,467项、321.600秒），XML `/tmp/pulns-support-full-fast-20261022.xml`。
对本项目修改前 `5c7d331` 的 PULNS 另做3个 seed × 1/2 episodes 共6组对照，
默认分数、history、selected-negative IDs、classifier/selector 全部权重及阶段更新数逐位一致。
仅证明这些合成配置的默认兼容，不是作者代码或论文数值复现。

另以真实独立 Python CLI 子进程执行三种方法：从同一400行合成输入建立5角色产品，
各执行一次双轨 CPU 训练，均保存两份选中模型并验证分数、阈值与结果回放；
均为0捕获warning、0重试。制品仅在本机 `/tmp/pu-support-cli-20261022-v3vsmjdc`，
不提交数组、pickle 或权重；临时目录不保证长期保留。环境为当时的 dirty checkout，
各计划绑定实际源码摘要，不冒充 frozen-lock、正式 Spambase 或 GPU 实验。

两批实现分别提交推送；本地静态、候选来源契约、baseline、draft registry、P3准入证据/
公开对照检查通过，后两者是既有记录的机器检查，未重新读取上游代码/论文或做新数值裁决。
没有改冻结协议/比较矩阵、旧645-run、负责人签署或用户制品；没有启动真实数据/GPU跑批。
当前仍是独立探索入口，adapter 继续拒绝 versioned Survey；正式接入还需 v3 单元、
绑定 runner/适用训练视图、真实 support、预算冻结与对应验收，不是只加JSON就自动准入。

## 合并端独立验收与正式接入待办（2026-10-22）

`db87274` 将上述两笔提交合入 `feature/survey-protocol-v3-methods`，保留本分支的
PN-CNN 能力修复、五项最终 test 指标及显式结果覆盖准备层；自动合并无冲突。
本轮未启动或连接 AutoDL，未运行正式实验、改动冻结 split 或推送远程。

合并端专项（PULNS callback、support 守卫/双轨、CLI、结果覆盖准备）**108 passed / 0 skipped /
0 failures / 0 errors**，JUnit 为本机 `tmp/support_merge_target_20261022.xml`。
全仓快测 `pytest tests/ -m "not slow and not e2e"` 为 **5,454 passed / 31 skipped /
1 failure / 0 errors**（5,486项），JUnit 为 `tmp/support_merge_fast_20261022.xml`；
唯一失败是未被本次合并修改的
`test_diagnose_survey_split_rebuild.py::test_param_symlink_rebuild_file_is_rejected`，
在 fixture 创建符号链接时出现 Windows `WinError 1314`，未进入产品诊断逻辑。
单独复跑同样失败，不将此全仓结果报告为通过，也未为消除此环境限制改动无关测试。
格式、文档链接、目录清单、测试质量与 diff 检查通过。

合并端另以实际 Python CLI 子进程运行 Self-PU、LaGAM、PULNS 各一次。输入是固定
400 行、5 维合成数据，support=40；使用上文的小 CPU/MLP 参数，不是 Spambase
原始数据或正式100/500标签预算。三个输出均为 `accepted_engineering_run=true`，
0 warning、0 retry；`PA+clean-support`/OA 各自选中模型、分数/阈值/test accuracy
回放及选中制品摘要校验通过，两轨都记录 Accuracy/AUC/F1/Precision/Recall。
验收模式为 `synthetic_cpu_merge_acceptance`、`formal_eligible=false`；驱动、计划、
manifest、权重、两模型和收据留在本机 `tmp/support_merge_cli_receipt_20261022/`，不提交。

真实数据准备另只做 `prepare_clean_support.py` **dry-run**：五数据集均为 split seed 0、
support seed 0，Spambase=100，其余=500；原始读取、划分与准备逻辑均成功完成。
support seed 0 只是明确记录的工程探测参数，不是冻结决策，未按标签构成挑选 seed。
本轮不加 `--write`，因此没有五角色产品收据，不能据此把真实 support 标成已交付；
也未覆盖五 seed 或训练真实数据。五份输出保留为本机
`tmp/support_merge_dryrun_<dataset>_20261022.json`。

正式接入最小剩余项：

1. 确认 support seed/规模和标签预算，生成独立五角色产品、校验并绑定真实产品收据；
   原四路冻结 split 保持不变。IMDb/20News/ADNI 未因本批接线获得 support 资格。
2. 完整 Self-PU 新建独立 profile/unit、预算与比较身份，不覆盖当前 `require_validation=false`
   的 no-meta profile；绑定真实 support 和显式 checkpoint 双轨，不沿用默认 terminal OA。
3. 冻结候选/阶段/存储预算并接入 versioned Survey。当前 adapter 明确拒绝该入口，
   不能只将草案 PULNS/LaGAM 的 `runnable` 翻转为 true。
4. 当前工程链仅 SCAR/OS、CLI 仅 CPU/2D。TS 校准或不适用结论、native-CNN runner/
   图像规格/峰值估算及真实 smoke 分别验收，不能由本轮合成测试代替。
5. `PA+clean-support` 保留独立结果身份，不冒充纯 PA。现有
   `survey_result_coverage.py` 仅接受 `pa/oa` 且读取时直接转大写，
   P2.2 summary/CLI 也只枚举 PA/OA；冻结后的矩阵适配、键映射、汇总与 comparison
   必须补齐，并排除其参与纯 PU PA 优势比较。本轮不隐式改变旧工具行为。

以上是工程整合与输入准备验证，不代签 v2 review、预算或 v3 正式准入。
