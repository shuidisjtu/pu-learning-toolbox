# Clean-support 独立 checkpoint 双轨接线（2026-10-22）

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
  仍可用旧终点 OA 入口。
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
python scripts/run_clean_support_experiment.py \
  --artifact-dir path/to/independent-support-product \
  --method lagam --c 0.5 --parameters path/to/parameters.json \
  --selection-scope meta_checkpoints --checkpoint-bytes-per-component 65536

# 显式执行一次，输出目录不得存在：
python scripts/run_clean_support_experiment.py \
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

剩余：PULNS episode 接线、真实 support 制品、正式 profiles/units、Self-PU 全量正式
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
