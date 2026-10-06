# 新 CNN 路径资源证据准备（2026-10-07）

承接[后续优先级](post_pilot_priority_plan.md)的工程路径与预算/选模交接。
状态：探针与只读检查器已实现，默认宽度多 seed 实测尚待执行；不是 P3.3 正式验收或容量批准。
不读取合作者 Excel/结果树，不改冻结协议、历史存储记录、结果、manifest 或签署。

## 1. 范围和执行入口

复用 `scripts/profile_survey_cnn_storage.py`，增补 PULNS、GEN-PU 与 Holistic-PU。
原五方法仍可调用；输出 schema 升为 2，旧 schema1 JSON 保留原读数/来源，不回填。
新增 `scripts/profile_survey_extended_cnn_storage.py` 逐个启动独立 Python 子进程：

| 合成 recipe | 明确选项 | 快照范围 |
|---|---|---|
| `pulns_clean_reward` | 独立 8 行 clean PN support，train IDs 0–23 / support IDs 24–31 | 无 epoch callback，明确 null；仅全 estimator 可信 pickle |
| `genpu_identity` / `genpu_tanh` | 输出域显式区分；tanh 对合成正态样本做固定 tanh，不拟合真实数据统计 | 两轮合成 PN；累计含全部 GAN 训练 |
| `holistic_fixed_continue` / `holistic_fixed_reinitialize` | fixed warmup + 明确后段初始化 | 三轮 warmup + 一轮 pseudo-PN |
| `holistic_lzo_continue` / `holistic_lzo_reinitialize` | 正例 mixup 辅助大小 5，完整三轮训练并恢复终点 | 全部四轮技术快照，不认定正式候选资格 |

默认每 recipe × seeds 0/1/2，共 21 个独立进程；24 行合成 3×32×32 输入、CNN13 默认
base_channels=64、小 head4、batch8、缩短阶段。输入幅度/通道 0.5/0.5 归一化不是 CIFAR
train-only 统计或源论文配方。32×32 只是合成输入尺度，不是正式数据集、GAN 质量或数值复现。
子进程非零退出/超时直接停止，不自动重试、回退 GPU 或发布部分成功记录。

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/profile_survey_extended_cnn_storage.py --device cpu --seeds 0 1 2
# 可先用小宽度验证接口；不能冒充默认 backbone 的体积。
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/profile_survey_extended_cnn_storage.py \
  --recipes genpu_tanh --seeds 0 --base-channels 2 --image-size 8
```

## 2. 不混淆三种内存与两种磁盘读数

- `retained_module_storage`：模型持久 nn.Module 的参数/缓冲区，按实际 storage 指针及 device
  去重，包括 constructor encoder 模板与辅助网络；encoder_ 与 model_ 的引用不重复计。
  不含梯度、Adam、临时 reward probe / 最佳权重 / policy 图，不是训练内存峰值。
- `host_resident_memory`：训练前、训练后及全部保存/恢复验证后 `/proc/self/status` 的
  VmRSS/VmHWM，kB 按 1024 转字节。高水位覆盖该子进程生命周期，含 imports/encoder 与
  运行时开销，不能相减得到精确训练增量；非 Linux/不可读时 null，不填 0。
  [Linux 内核文档](https://www.kernel.org/doc/html/latest/filesystems/proc.html)明确 RSS 异步记账
  可能不精确，故 precise_upper_bound=false；独立进程防止其它 recipe 的旧峰值混入。
- CUDA 模式分别记录 fit allocator 峰值与包含恢复验证的 allocator 峰值，以及实际设备名、
  逻辑卡号/可见设备配置；CPU 必须全部为 null，不复用 10-05 旧 GPU 读数验收新路径。
- `epoch_weights_bytes`：临时目录里实际产生的全部推断快照 count/min/max/total，逐个 CPU 回放，
  最终分数与 fitted model 对照。PULNS 为 null，不用空壳快照凑候选。
- `estimator_pickle_bytes`：六网络/selector/模板/轨迹等完整可信 estimator 的序列化体积，
  仅加载本次生成的 bytes；不同于推断快照，也不含完整 optimizer/RNG 续训状态。

临时目录由该探针创建并在退出时清理，不碰用户训练制品。记录实际阶段更新数及全成本；
Holistic-PU 单列执行/选中 warmup 轮、discarded 成本、额外 LZO forward、后段 Adam 重建。
PULNS 仍需独立真值奖励预算、PA-ineligible/TS 拒绝；不提供 selection/test 标签。

## 3. 身份与校验

每 row 保留参数/视图/归一化/环境/进程、基线 commit 和实际运行源码 SHA256；
driver/child 身份、范围、宽度、参数不一致或测量前后源码变化即拒绝发布。
`scripts/check_survey_extended_cnn_storage.py --record <receipt.json>` 只读检查 sweep 覆盖、
source bytes、校准/support 角色、阶段付出、快照来源与 LZO 恢复成本；保持
formal_admission=false、training_peak_upper_bound_proven=false。当前源码变化后需要重测，
不能将旧记录说成当前实现资源证据；checker 的单元 fixture 不是测量结果。

实测读数、后续台账引用与完整回归将在测量完成后单独登记。已有 P1-1/P1-2/五方法
选模预算证据绑定不变，checkpoint writer 与算法源码不在本项修改范围内。

脚本工程验证：三份探针/检查器专项 **70 passed**；完整快层
`pytest tests/ -m 'not slow and not e2e'` 为 **3252 passed / 59 skipped / 36 deselected**
（161 warnings，238.51s）。11 项静态门禁通过，现有 P3 准入/公开对照/选模预算检查通过；
不声明 slow/e2e 或本轮 GPU 通过，亦不把 fixture 中的变体例子当正式测量记录。

## 4. 未完成的正式要求

负责人仍须批准完整 representation/head、阶段预算/候选资格、输入坐标、重复/重试及 PULNS
clean-support 预算。真实数据/完整轮次、数据加载/增强、候选数、失败/OOM、多 seed GPU 与
frozen-lock 资源验收不由本探针替代。机器 sweep 是技术比较，不是正式候选池。
正式存储须按峰值全部 epoch × component × candidate × retry 预算；自动回收仅减少幸存文件，
不降低已经发生的训练/临时写盘峰值。此处不裁决 PA/OA 阶段资格，不改旧 pilot 或主榜。
