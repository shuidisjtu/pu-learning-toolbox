# CNN 存储技术实测（2026-10-05）

状态：**工程证据，非正式预算/准入**。承接[优先级](post_pilot_priority_plan.md)第 2 项。
不使用合作者实验数据，不改变旧协议、比较矩阵、候选池、签署或历史 manifest。

## 范围与来源

入口 `scripts/profile_survey_cnn_storage.py`，五方法 × 三种子（0/1/2），CPU，24 条合成
NCHW 输入（3×8×8），CNN13 的默认 base_channels=64，GradPU 使用显式 cnn13_no_bn。
使用固定 0.5/0.5 通道归一化、小 head4、缩短阶段轮次；不是 CIFAR train-only 统计、
论文增强、正式训练预算或 frozen-lock 验收。8×8 用于工程冒烟，不登记为正式数据集。
全阶段 weights-only 快照均在 CPU 恢复为有限预测，最终快照与 fitted estimator 分数一致；
完整 estimator 仅反序列化本次自己生成的可信 pickle。临时目录退出后自动清理，仅保留 JSON。

机器记录：[CPU 15 组](data/candidate_cnn_storage_probe_20261005.json)、
[CUDA 单组](data/candidate_cnn_cuda_probe_20261005.json)。每组保留参数、视图、输入尺寸、
Python/PyTorch 版本、基线 commit 与实际源码 SHA256；训练前后来源摘要变化即拒绝记录。
基线 commit 为 19c68c2，新探针当时尚未提交，所以**实际源码以逐文件摘要识别**，
不能仅凭 code_commit 宣称整个工作区当时干净或代码已冻结。

**后续 CI 修正（2026-10-05）**：跨平台门禁发现旧快照恢复固定用 256 条推理，
而部分候选模型用 8 条，导致不同 CPU 内核出现约 1e-7 的舍入差异。新快照登记并恢复
原推理批大小；旧 reference 缺字段仍保留历史 256 行为。本页 JSON 是修正前实际源码
摘要对应的历史读数，不冒充修正后源码的重测结果或跨平台逐位一致性证明。

## 读数

同方法三种子的体积一致。单位是字节，不是容量上界或单模型训练恢复体积。

| 方法 | 推断快照最大字节 | 实际快照数 | 全 estimator pickle 字节 |
|---|---:|---:|---:|
| PULDA | 40,792,242 | 2 | 81,592,123 |
| GradPU | 40,719,276 | 1 | 81,441,072 |
| Robust-PU | 40,792,242 | 2 | 81,591,847 |
| Split-PU | 40,792,894 | 4 | 163,194,081 |
| PAN | 40,785,510 | 2 | 122,377,109 |

推断快照约 38.8–38.9 MiB，完整 estimator 约 77.7–155.6 MiB。
差异来自 constructor encoder 模板及辅助网络：PAN 有判别器，Split-PU 保留教师/splitter；
不能把完整对象 pickle 与推断权重混为一个 bytes/component，后者也不支持训练恢复。
推断快照不保留优化器、全部辅助网络、RNG 或数据。这不是恢复全部训练状态的预算。

CUDA 实测仅 PULDA/seed0，同一合成小预算，物理 GPU6 / RTX A6000。
训练前 PyTorch allocator 基线为 0；训练期间峰值 allocated=308,205,056 字节
（约 293.93 MiB）、reserved=333,447,168 字节（318 MiB）。这仅包含本进程 allocator，
不是 nvidia-smi 的整卡/整机峰值，也不代表其他方法或 CIFAR 正式多 seed 已验收。
CPU 快照回放/可信 pickle 回放均成功；测后 GPU6 为 1MiB/0%，没有常驻占卡。

## 重跑与门禁

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/profile_survey_cnn_storage.py --method all --seeds 0 1 2
# 先独立检查 GPU 空闲；不要照搬卡号抢占已有训练。
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/profile_survey_cnn_storage.py --method pulda --seeds 0 --device cuda
pytest -q tests/unit/scripts/test_profile_survey_cnn_storage.py
```

CUDA 请求不可自动回退 CPU。小宽度单测标明 matches_toolbox_default_width=false，
不替代本页默认宽度实测。种子/尺寸/设备输入拒绝、所有快照回放、源码身份和固定 seed
体积记录有测试；计时不宣称确定性。新旧存储脚本专项 **29 passed**（38.58s）。
随后补训练期间源码变化的 fail-closed 测试，独立 **1 passed**；全 unit/contract/integration
扩大 CPU 回归 **2535 passed /25 skipped /32 deselected**（185.36s），不含随后新增的该测试。

## 仍需完成

1. 负责人批准候选结构/head、阶段轮次、选模与重复/重试预算，再测批准结构与输入的资源。
2. 对正式结构登记保守 bytes/component 上界，而非直接把本页测量值写进协议。
3. 峰值磁盘按全部 epoch × component × candidate × retry 估计，回收不降低训练中峰值。
4. 多 seed GPU/显存/失败/OOM/frozen-lock 等 P3.3 验收仍待办；此处单组不能替代。
5. 来源行为/正式图像增强/SimSiam 或调度差异仍需逐方法处理，不靠存储测试签署方法学。
