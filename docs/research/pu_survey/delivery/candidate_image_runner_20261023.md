# PULDA / GradPU 图像 runner 与 120/200 快照验收（2026-10-23）

本批补齐独立四角色图像入口：`scripts/run_candidate_image_experiment.py`，
复用既有 seeded 构造、train-only 预处理和 weights-only checkpoint writer。
已在真实 CIFAR 固定小子集完整生成 PULDA 120 份、GradPU 200 份快照，
完成 PA/OA 独立选择、D25 回收及两份选中预测模型回放。
**这是工程 smoke/profile，不是正式 Survey 单元、预算批准或论文数值复现。**
不修改估计器目标/默认、冻结协议、台账或历史结果；PN-CNN 5 seed 继续暂停。

## 实现与原生行为

- PULDA：共享 ResNet-18 小图 stem，BN 训练、LDA warmup、无共享增强；
  MixUp/原损失不变，总体先验传给训练。每轮遍历 native U，更新数按
  `ceil(native_U / unlabeled_batch_size)`，不是按 TS 风险并集长度重新定义 epoch。
- GradPU：全宽 CNN13-no-BN（base channels 64），method-private/no-BN 单列；
  不传总体先验给训练，只给外部 PA 选择。保留原独立 P/U loader、短 loader 循环、
  TS 风险角色和梯度惩罚；更新数取两 loader 长度最大值。显式要求 LR 策略。
- 本 runner 将**所有已完成** PULDA warmup/pu_mixup、GradPU pu 快照交给离线选择。
  这是明确的工程阶段资格选择，不冒充正式冻结规格。本次 PULDA 两轨恰好都选中
  warmup（PA 第8份、OA 第1份）；如正式配方只允许 PU 阶段，应版本化登记，不能倒改本次结果。
- 复用既有构造/实际输入校验，禁止候选偷偷换 seed、先验、设备、encoder 或增强；
  拒绝 SAR、正式协议/registry、单 OA、非 TS 入口和缺失/不一致的字节上界。
  候选预算按最大实际 epoch 数预检，不按较小的 base model 数量估算。
- fit 只收到 train 的观测 PU 标签；pu_val/clean_val/test 不转发给训练。
  PA 在 pu_val、OA 在 clean_val 选择，test 仅用于选后测评/回放。
  内层 bound-fit receipt 仍只证明训练局部事实，不能单独当四角色或正式准入证明。
- 实测快照超出声明字节、输入/阶段/累计更新数不一致会拒收并使模型失效，
  不自动放宽预算、不删失败证据。告警或重试的 CLI 运行保留结果但不标 accepted。

来源/适配身份沿用[既有 PULDA/GradPU 审核说明](../admission/pulda_gradpu_image_review_20261018.md)
与 [GradPU #121](../admission/gradpu_training_recipe_review_20261018.md)。
按 [20261021 任务复核](../reviews/heng958_task_review_20261022.md)，#119/#120/#121 已处置，
无新反证不重开。这里没有重写论文目标或声称作者 CNN 数值复现；
同配置裸 fit 对照只证明录制桥不改变本机训练，不等于论文/作者程序对照。

## 实际测量

输入只读复用 GEN-PU 已建子集：原 train 行0–23、24–31、32–39分别为
train/pu_val/clean_val，官方 test 前16行作 test，24/8/8/16行，ID 隔离。
预先固定前缀，不按标签换 seed/重抽；不是合作者正式 split。
总体 π=0.4 来自 CIFAR 全体均衡十类与正类 {0,1,8,9}，不从小子集/test 估计。
两次均 seed0、SCAR c0.5、TS，实际观测 train P=4/native U=20；
归一化只拟合 train。机器收据见
[candidate_image_validation_20261023.json](../data/candidate_image_validation_20261023.json)。

| 项目 | PULDA | GradPU |
|---|---:|---:|
| 完成轮数 / 快照数 | 60 warmup + 60 PU / 120 | 200 PU / 200 |
| 配置 batch | P8 / U8 | 每分支8（实际 P 最大4、U8、风险 U∪P12） |
| 实际 optimizer steps | 360 | 600 |
| 每份 weights-only 文件字节 | 44,900,758 | 40,974,252 |
| 回收前全部快照 payload 字节 | 5,388,090,960（约5.02 GiB） | 8,194,850,400（约7.63 GiB） |
| 选后保留 / 回收份数 | 2 / 118 | 2 / 198 |
| 选后快照字节 | 89,801,516 | 81,948,504 |
| fit CUDA allocated 峰值字节 | 458,424,320 | 266,381,824 |
| 进程生命周期 RSS 峰值字节 | 2,921,340,928 | 2,455,769,088 |
| 捕获告警 / 重试 | 0 / 0 | 0 / 0 |

物理 GPU7 A6000；仅暴露逻辑 cuda:0，任务结束释放设备。
Python3.12.2 / torch2.6.0+cu124 / driver550.54.14。
基线提交 `baa9061` 加本轮未提交源码，dirty=true，源码/输入/输出 SHA 原样绑定，
未倒填为之后的提交或 frozen-lock 环境。

完整120/200计数只解决这两个模型尺寸下的快照 payload 测量，**不等于完整数据、
默认 batch 或多 seed 峰值**。GPU 数字是 fit 的 torch allocated，非 reserved、
整流程或整卡总用量；RSS 是进程生命周期，含加载/模板/选择/回放，非隔离 fit 增量。
Windows 无 Unix resource 计数时 RSS 输出 null，不伪填0；两类工程 CLI 均作兼容处理。
未计入 payload 的还有 selected pickle、JSON、optimizer、RAM/显存等；
回收后小体积不能替代回收前峰值。每份声明64 MiB，1候选×2尝试的磁盘预检分别
15 GiB / 25 GiB；没有拿选后2份抵扣，也没有实际触发重试。
小样本 proxy accuracy 可以超出[0,1]，不裁剪成真实 accuracy；不据此判断算法优劣。

## 运行方式

四个 NPZ 各含 `X/y/indices`，X 为 NCHW 原像素；由数据提供者保证 y 是对应角色的
二元真值。先提供参数 JSON，例如 PULDA：

```json
{"device":"cuda","warmup_epochs":60,"pu_epochs":60,"positive_batch_size":8,"unlabeled_batch_size":8}
```

GradPU 改为：

```json
{"device":"cuda","max_epochs":200,"batch_size":8,"lr_schedule":"constant"}
```

```bash
PYTHONPATH=. python scripts/run_candidate_image_experiment.py \
  --method pulda --data-dir data/genpu-cifar-smoke-20261023-p2ecu5tn/input \
  --dataset cifar10 --class-prior 0.4 \
  --prior-source 'CIFAR-10 balanced population; positives 0,1,8,9; not test estimate' \
  --seed 0 --parameters /path/to/parameters.json --c 0.5 \
  --checkpoint-bytes-per-component 67108864
```

默认只规划，不启动训练。显式分配可用 GPU 后增加
`--execute --out-dir <不存在的新目录>` 执行；请求 CUDA 不可用则失败，不降级 CPU。
GradPU 换 method 和 JSON 即可。输出 plan/environment/manifest/summary、选中快照及
两份 `selected_PA.pkl` / `selected_OA.pkl`；只反序列化自己刚生成的可信对象。
NPZ、模型、权重不提交 Git，Git 只登记本次摘要，不包含可重新下载数据的公开链接。

## 验证与后续

专项测试覆盖真实四角色链路、录制与裸 fit 的逐位对照、阶段/成本、候选预算上界、
禁止的覆盖/协议/视图、字节超界保留证据、计划/输入/源码篡改、双模型回放、
平台 RSS 单位与不支持分支。公开收据契约只核元数据，CI 无本机 GPU 制品，不称独立数值复核。

本机完整快层回归为 **5,551 passed / 59 skipped / 0 failures/errors**，共5,610项，
363.498秒；JUnit 为 `/tmp/candidate-image-full-fast-20261023.xml`。
该轮收集之后新增的4项公开收据契约另已通过，未伪算入上述单次全仓计数。
既有环境/成本/收敛告警保留，不把全仓测试称作零告警；两次真实 GPU 工程运行才是0捕获告警/0重试。
既有 PULDA/GradPU GPU 回归在 `PU_REQUIRE_CUDA=1` 下另有2项通过、0跳过，
不是拿普通环境的 CUDA skip 当通过。
本批新增 runner/CLI/RSS/公开收据合计37项最终专项全部通过、0跳过，
JUnit 为 `/tmp/candidate-image-final-new-tests-20261023.xml`。

格式（排除原有未提交 `benchmarks/deep_pu/official_data.py` 修改）、文档/API、元数据、
数学渲染、skills同步、测试/注释质量、结构/分层依赖、baseline、当前候选来源契约及
既有 draft registry 门禁通过；完整工作区格式门仍被该原有修改阻断，本批不编辑或提交它。
这是本地检查，不冒充远程多平台 CI。离线 wheel/sdist 构建、wheel 内容检查通过，
wheel 在独立临时 target 安装、仓库外完成两方法 TS fit/实际快照分数回放；
复用本机依赖，不是新建 frozen-lock 环境。数据/模型/上传未纳入 Git。

下一步仍是 protocol v3 的 profiles/units、正式预算/候选/阶段资格及相应正式验收；
Self-PU TS/full-meta 与 support 单元同步推进。不会因本轮 smoke 自动准入、改签署或启动 PN-CNN 5 seed。
