# 五数据集 clean-support 制品（2026-10-22）

已在本机生成并验收 **5 数据集 × seeds 0–4，共 25 份**五角色制品。
这批只完成数据准备和短 CPU 工程验收，不是正式跑批、算法准入或预算批准。

## 给合作者的定位

- 制品根目录：`data/clean-support-20261022-yg0avvvx`，共 1,116,833,117 字节（约 1.04 GiB）。
- 每份在 `<dataset>/split_<seed>/`，包含 `train.npz`、`pu_val.npz`、`clean_val.npz`、
  `test.npz`、`support.npz` 和 `support_manifest.json`。
- [机器清单](../data/clean_support_materialization_20261022.json)登记每份路径、规模、support
  正负标签数、收据摘要与 manifest 文件 SHA。NPZ 的逐文件摘要保留在各自 manifest。
- **GitHub 只提交清单，不提交原始数据、NPZ 或模型 pickle**。这些本机路径不表示数据已
  自动送达另一台服务器；传输后须重新运行 `load_support_dataset` 核对文件/角色/摘要。

下表为 seed0 的五角色规模；全部 seed 的明细见机器清单：

| 数据集 | train | pu_val | clean_val | test | support |
|---|---:|---:|---:|---:|---:|
| Spambase | 3212 | 184 | 184 | 921 | 100 |
| Connect-4 | 48140 | 2702 | 2703 | 13512 | 500 |
| MNIST | 53500 | 3000 | 3000 | 10000 | 500 |
| Fashion-MNIST | 53500 | 3000 | 3000 | 10000 | 500 |
| CIFAR-10 | 44500 | 2500 | 2500 | 10000 | 500 |

## 验收了什么

- support 规模遵循执行计划：Spambase 100，其余 500；`support_seed=split_seed` 在看标签前
  固定。均匀无放回抽取，不按类别重抽；25 份均有正负支持标签，实际数量逐份披露。
- 五角色 ID 互斥且完整覆盖源池；support 从原 train 移除；每个角色的二元标签与 raw
  映射逐行一致。官方 test 边界保持，未改合作者旧 split 或历史结果。
- 表格五角色逐元素等于“剩余 train 拟合的均值/标准差”变换；图像五角色均保留原始 uint8
  像素，通道统计仅用剩余 train。独立重算统计最大误差为 0。
- CIFAR 的原始 NCHW 是 HWC-backed 布局；NPZ 回读变成连续 NCHW，float64 归约顺序变化
  可带来约 `1e-11` 的差异。验收按原始剩余 train 布局重算，与存储统计逐位一致；
  同时对 NPZ 像素逐元素核对。没有改制品，也没有放宽容差。
- 25 份均重新通过 `load_support_dataset` 的收据、五文件摘要、角色与 lineage 校验；
  raw reader 在处理前后摘要一致。另用公共预处理 CLI 独立重做 Spambase seed0，
  完整 manifest 及五个 NPZ 摘要与批量产物一致。

补取 [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase) 和
[官方 CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) 原始输入；CIFAR 最终完整包为
170,498,071 字节，发布方 MD5 `c58f30108f718f92721af3b95e74349a` 校验通过。
先前单连接得到的不完整包保留为 `.incomplete`，没有解包或纳入制品。
最终以最多 8 连接的精确 HTTP 字节段取齐，逐段检查范围/长度，再验整包 MD5 后仅解出固定
普通数据成员。Spambase 无发布方校验值，记录本机 SHA/TLS 下载来源，不伪称官方 checksum。

## 真实 Spambase 短 CPU 链路

Self-PU 完整 meta、LaGAM、PULNS 均用本批 Spambase seed0（100 个独立 support 标签）
执行现有 CLI：单候选、`c=0.5`、MLP hidden4/batch64、短预算3轮/episode。
三者都完成 `PA+clean-support`/OA 独立选模、快照回收、两份选中模型保存与分数/精度回放，
**0 告警、0 重试**。完整参数、文件摘要与来源见机器清单中的 `real_cpu_smoke`；本机制品在
`data/support-real-cpu-20261022-du_9pu5y`。

不把单 seed、短预算、CPU、额外监督的读数当正式多 seed/GPU/论文效果对照，
也不把它混入纯 PU PA 或旧 Pilot。没有在本批批准正式 epoch/stage 候选资格。

## 重建命令

在仓库根把以下保存为新的 Bash 脚本执行，输出根目录必须换成不存在的新目录。
这台机器现有 raw 路径如下；其他服务器先准备相同 reader 字节并核对机器清单摘要：

```bash
set -e
support_output_root=data/clean-support-new-output
for support_dataset in spambase connect_4 mnist fashion_mnist cifar10; do
  support_size=500
  case "$support_dataset" in
    spambase)
      support_size=100
      support_raw_root=data/support-raw-20261022-dbl2aovy ;;
    cifar10)
      support_raw_root=data/cifar-ranged-20261022-3ow3dxtc ;;
    *)
      support_raw_root=data/outgoing/raw-handoff-20261013/payload/data/raw ;;
  esac
  for support_split_seed in 0 1 2 3 4; do
    PYTHONPATH=. python scripts/prepare_clean_support.py \
      --dataset "$support_dataset" --raw-dir "$support_raw_root" \
      --seed "$support_split_seed" --support-seed "$support_split_seed" \
      --support-size "$support_size" --group-policy not_applicable \
      --out-dir "$support_output_root/$support_dataset/split_$support_split_seed" --write
  done
done
```

`formal_admission=false`、`budget_approval=not_granted_by_preparer`、
`support_source_verified=false` 保留：本机字节/数组核验不冒充独立来源授权或人的方法学签署。
v2 的 `v2_p4_clean_support_policy` / `v2_p10_self_pu_scope` 仍 pending。
配方预算、正式 v3 units/runner 接线、图像 CNN/GPU 和正式多 seed 效果验收仍待；
本批不需要重复 review 已完成的工程核对。

收尾验证：新增元数据契约 4 项通过；最终完整快速回归 **5,478 通过、59 条件跳过、
0 失败/错误**（5,537 项，不含 slow/e2e；本机 JUnit
`/tmp/p3-2-calibration-support-final-fast-20261022.xml`）。静态质量、文档/API、目录/层依赖、
候选 fit 契约、baseline 和 recipe draft 门禁通过。离线 sdist/wheel 构建及包内容核对通过；
wheel 安装到独立临时 target 后，在仓库外通过 CVIR 校准训练/快照和真实 support 加载。
该安装探针复用本机依赖，不是新建 frozen-lock 环境或远程 CI；未运行 GPU 验收。
