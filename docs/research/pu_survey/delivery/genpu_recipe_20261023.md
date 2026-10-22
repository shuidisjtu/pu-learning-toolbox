# GEN-PU #126：可执行工程配方与单候选验收（2026-10-23）

已新增 `pu_toolbox.experiment.genpu_recipe` 与
`scripts/run_genpu_recipe.py`。命令可输出四角色 ExperimentRunner manifest、PA/OA
独立选择、选中权重与可信预测模型回放，而不只是“实现存在”。
**本批是独立工程配方，不是正式 Survey recipe、预算批准或论文数值复现。**
估计器原有默认、冻结协议/台账、旧结果与人的签署不变；PN-CNN 5 seed 仍暂停。

## 1. #126 四项落地到什么程度

| blocker | 本批确定并验证的工程规格 | 未由本批放行 |
|---|---|---|
| prior_weighted_Du_and_synthetic_PN_spec | 采用已定的论文 minimax/anti-GAN 目标；Du 假样本按总体 π/1−π 加权；最终 PN 为同先验加权 softplus risk；TS 只将 Du 真样本池换成全 train 并集 | 有限 GAN 训练不保证分布恢复或真实数据精度 |
| six_network_GAN_plus_PN_selection_and_storage_spec | 显式 GAN/PN epoch、batch、seed；每步 3D+2G，PN 每步 1C；仅全部已完成 synthetic_pn 轮参与 PA/OA，GAN 无空壳分类器候选；只保留选中 PN 权重和全部元数据 | 正式 epoch/batch/candidates/seeds 数值仍须 P4.1-B 与 v3 登记 |
| dense_pixel_output_domain_and_four_CNN_encoder_recipe | 两层 256、latent100 的稠密生成器；identity 输出与真实 [0,1] 像素坐标一致；train-only 统计在 encoder 内归一化；Dp/Dn 线性头，Du/PN 两层256；四份独立 seeded ResNet-18 小图 stem，无增强，PN 从新模板初始化 | 不将原 MNIST/USPS 稠密配方或其他数据集 GAN 架构冒充本 CNN 适配 |
| generator_step_discriminator_mode_and_native_image_resource_spec | G 步冻结 D 参数及 BN/dropout 模式，但保留输入梯度；异常恢复；已测标准宽度/32×32 真实图像短 GPU profile、逐快照字节与回收 | 小子集/短预算峰值不是正式全数据/全预算上界 |

四项现在都有可执行工程选择及证据，不再要求合作者重复验证已覆盖的工程细节。
**#126 暂不记为已关闭**：正式单位/配方预算和对应全量资源、数值验收尚未完成；
本环境 GitHub CLI 也未登录，未评论或关闭远程 issue。

## 2. 与论文/源码不同的地方，以及原因

对照 [IJCAI 2018 正式论文](https://www.ijcai.org/proceedings/2018/0312.pdf)
Eq. (3)–(11)、§4 与 Table 1（PDF 第5页/印刷页2259）；
作者归档身份和差异沿用[已锁定来源](../genpu_source_cnn_review_20261007.md)，
本轮未复制、运行或重新分发无 LICENSE 的作者程序/权重。

- **目标**：保留论文带先验权重的 minimax/anti-GAN，不照搬 demo 的无权重 Du 与
  非饱和 BCE；最终 PN 的总体先验加权是项目明确的工程选择。
- **输出域**：本 recipe 选择 identity，不把输入偷偷映射到 demo 的 [-1,1]。
  实际输入是共享 [0,1] 像素，标准化在各 encoder 内执行一次。identity 输出无界，
  “同坐标”不表示生成值被限制在 [0,1]；不裁剪生成样本或真实样本。
  原估计器的显式 tanh 分支继续存在，只是不属于这个固定 recipe。
- **架构/初始化**：Table 1 的 MNIST/USPS 判别器是稠密网络；本批四份共享 ResNet-18
  是 benchmark 适配。隐藏256/latent100/lr0.0003/Adam 默认 betas 取该表对应配置，
  但不照搬表内的零初始化，而按共享架构 seeded 随机初始化，避免破坏共享表征协议。
  论文 CelebA 另用 improved WGAN，本配方不是该变体。
- **预算/选模**：Table 1 的 P/U batch 为50/100，§4 另调 λp/λn；本工程 recipe 固定
  三个 λ 为1，并显式传同尺寸 P/U/G batch，单候选不搜索。所有已完成 PN 轮由外部
  PA/OA 离线选择，test 不参与训练或选择。该配置仅用于验证，不暗定正式候选预算。
- **模式**：冻结 D 的 BN/dropout、保留对 G 输入的梯度，是将原目标用于 CNN 的工程处理；
  已有真实梯度、BN/模板隔离、异常恢复测试，不把“测试绿”说成论文精度复现。

## 3. 实际运行与身份

原始输入沿用上轮校验过整包官方 MD5 的 CIFAR-10。
另建**预先固定、不按标签重抽**的工程子集：原 train 行0–23、24–31、32–39分别作
train/pu_val/clean_val，官方 test 的前16行只作 test；全局 ID 隔离。
共24/8/8/16行，不是合作者原 split，不覆盖或混入正式制品。
先验0.4来自完整十类均衡总体与正类 {0,1,8,9} 的设定，不从小子集/test 估计。

- 实际路径：`data/genpu-cifar-smoke-20261023-p2ecu5tn`；原始输入不改。
- 物理 GPU7，A6000；进程内仅可见逻辑 cuda:0；Python3.12.2 / torch2.6.0+cu124。
- 每个 profile 单候选、seed0、c0.5、GAN1轮/PN3轮；完整宽度 ResNet-18，非缩宽替代。
- batch8：六项更新数为3/3/3/3/3/9，共24；3份快照各45,542,038字节，
  选后保留1份（PA/OA可选同一权重，各自阈值仍独立），回收2份。
- batch8 的 fit 峰值 **872,768,512字节（约832 MiB）**，是 PyTorch allocated，
  不是 reserved、整卡其他进程或整个流程峰值。进程生命周期 RSS 另登记，含加载/构造/回放。
- 已另测同输入、显式 batch256 的 profile，fit allocated 峰值为
  **8,652,706,816字节（约8.06 GiB）**；实际值见
  [机器收据](../data/genpu_recipe_validation_20261023.json)的 `batch256_profile`。
  P/U 有放回采样允许 batch 大于子集大小；这是资源探针，不增加独立样本或统计证据。
- 两个 profile 均0捕获告警/0重试，保存并回放 PA/OA 两份可信预测模型；测试精度只作链路
  自检，不用于论文对比或判断算法好坏。NPZ/pickle/权重不提交 Git，Git 仅登记摘要收据。

三份 snapshot payload 的峰值为136,626,114字节，选后45,542,038字节；
这**不含**所选模型 pickle、JSON、训练 RAM/显存或其他进程。容量预检按显式每组件
64 MiB ×3轮×1候选×2尝试=384 MiB，未拿选后体积冒充训练峰值。
正式 PN 轮数为 E 时，按同样已声明上界估算64 MiB×E×candidates×2；
实际每文件超界会拒收，保留失败证据，不自动放宽上界。正式 batch/环境仍须对应重测。

## 4. 可复制入口

命令默认只规划；四个 NPZ 各含 `X/y/indices`，X 为预先准备的真实像素或二维特征，
y 是四角色数据提供者持有的二元真值，runner 只向 fit 转发 train 的 SCAR PU 视图。
图像只从 train 拟合统计。先验来源须显式填写，不能取 test 标签反推。

```bash
PYTHONPATH=. python scripts/run_genpu_recipe.py \
  --data-dir data/genpu-cifar-smoke-20261023-p2ecu5tn/input \
  --dataset cifar10 --class-prior 0.4 \
  --prior-source 'CIFAR-10 balanced population; positives 0,1,8,9' \
  --seed 0 --device cuda --gan-epochs 1 --pn-epochs 3 --batch-size 8 \
  --checkpoint-bytes-per-component 67108864 --c 0.5
```

执行时增加 `--execute --out-dir <不存在的新目录>`，在已分配 GPU 的终端明确设置
`CUDA_VISIBLE_DEVICES`。需要设备权限的环境须先取得权限；请求 cuda 但不可用会失败，
不默默降级 CPU。不要用这里的短预算替代正式 recipe。

输出 `plan/environment/manifest/summary.json`、两份 `selected_*.pkl` 和选中 PN 权重；
绑定源码/锁摘要、四文件摘要、PU标记、recipe摘要、先验说明和实际训练身份。
pickle 只回读本次刚写出的可信对象，禁止加载未知来源权重包。

工程规格已验证；仍待 protocol v3 接入、P4.1-B 正式预算/候选冻结、对应全量资源与数值验收。
本轮未新增或修改人的 review/signature，不据此解除正式准入或关闭 #126。

## 5. 最终验证

- 最终完整快速层：**5,518 passed / 59 skipped / 0 failures/errors**，共5,577项，
  337.048秒；本机 `/tmp/genpu-recipe-final-full-fast-20261023.xml`。
- 新配方/CLI/收据专项40项全部通过；既有 GEN-PU 两项 CUDA 回归在
  `PU_REQUIRE_CUDA=1` 下 **2 passed / 0 skipped**。普通快测中 CUDA 条件跳过不计 GPU 通过。
- 新录制入口与同配置裸 fit 在两个 seed 上对照，六网络权重、history、成本与分数逐位一致；
  不等于作者程序或论文数值复现。固定参数/视图、阶段资格、实际快照超界、计划/输入篡改、
  角色边界、候选覆盖和完整双轨选择均有接受/拒绝测试。
- 格式、文档/API、数学渲染、元数据、skills同步、注释、目录/层依赖、测试质量、baseline、
  当前候选来源契约与既有 draft recipe registry 门禁通过；未宣称远程 CI 已通过。
- 离线 sdist/wheel 构建、包内容检查通过；未包含原始数据、NPZ、模型或用户上传。
  wheel 在独立临时 target 安装、仓库外执行 GenPU TS 拟合与 checkpoint 分数回放通过。
  该安装复用本机依赖，不是新建 frozen-lock 环境。

GPU 测量时基线为 `cefd2b6` 加未提交工程源码，清单按各文件 SHA 绑定，
`dirty_worktree=true` / `frozen_lock_environment_verified=false` 原样保留；
本轮提交不倒填测量身份。验证后已结束 GPU 进程，未保留占卡任务。
