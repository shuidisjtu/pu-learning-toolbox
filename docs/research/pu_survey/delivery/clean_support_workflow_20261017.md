# 独立 clean-support 制品与 CPU 实验入口（2026-10-17）

## 状态与优先级

本轮 fetch 后本地/远程均为 `e84a76c`，无新提交。GitHub issue API 返回403且
本机 gh 未登录，无法据此断言无新 issue；按执行计划及上一轮算法收尾依赖推进。
P3MIX 正确官方 PDF 本轮仍转验证页，未取得新的原始论文/补充包，不重复冒充集成完成。

本交付补齐上轮支持集 adapter 之后的**制品准备和可复现实验入口**。只验证独立工程
路径，不批准正式标签预算，不改旧四角色 split、645-run、协议/矩阵或准入签署。
未在本机实际数据上私选支持集大小/seed、启动正式训练或占用 GPU。

## 1. 支持集制品

入口 `scripts/prepare_clean_support.py`；数组 API 位于
`pu_toolbox.experiment.support_data` 的 `prepare_support_dataset`、
`save_support_dataset`、`load_support_dataset`。

当前支持 Spambase、Connect-4、MNIST、F-MNIST、CIFAR-10。CLI 复用已审计 raw reader，
只读本地原始数据，**不下载、不读取旧 processed split 作为原始输入**；文件摘要在读取
前后核对。数组 API 则要求调用方提供真正未处理的 source，并诚实记录来源；无法自动
证明任意调用方提供的浮点矩阵从未标准化，不能把此接口当作来源鉴定或授权签名。

处理顺序：

1. 按已有目录、类别映射及 split seed，从原始数组重建四角色基准成员。
2. 从基准 train 中均匀无放回抽取显式大小的 support，**抽样不看标签**。
   support seed 必填；不按 test、类别比例或结果搜索 seed。若 support 或剩余 train
   仅有一类，直接失败，不重抽、不自动换 seed 或增大标签预算。
3. support 从 train 中移除；pu_val/clean_val/test 的成员保持不变。
4. 表格均值/标准差仅在剩余 train 上重新拟合，对全部角色使用同一变换。
   图像仅重算剩余 train 的通道统计，存储仍为原始 uint8 NCHW，不重复归一化。
5. 原总体 pi 保留为基准数据生成元信息；`class_prior.train` 只更新为描述性统计，
   不用 support/test 估计或替换训练先验。无额外先验估计池，角色显式记为 `[]`。

输出仅写入全新目录；失败后的残缺目录不会被当成功，也不自动删除或覆盖：

```text
new-support-product/
  train.npz
  pu_val.npz
  clean_val.npz
  test.npz
  support.npz
  support_manifest.json
```

使用独立 schema `independent-clean-support-v1`，故不生成会冒充旧协议的
`split_manifest.json`。收据绑定五份文件、特征/标签/样本身份、原始数组、原 train
成员、support seed 和剩余 train 的预处理。加载时重验文件/收据摘要、成员隔离、
原 train 分解及抽样 seed 的实际复现；摘要不是授权签名，不能代替来源或预算批准。
整数身份表示该原始 reader 中的行，不能发现不同 ID 对应的实际重复样本。

ADNI 不开放：替代数据缺受试者身份，不能把切片索引独立冒充患者独立。
IMDB/20News 暂不开放：须另接固定 SBERT 模型/缓存身份，不能伪称已完成八数据集。
调用方必须显式声明 `group_policy=not_applicable`；存在已知分组时不能用本入口绕过。

### 命令

以下 `40`/`7` 仅演示显式参数，**不是已批准实验预算或推荐样本量**。应先确定真实
支持集数量/采样 seed 与适用范围；默认只预览，不写制品。

```bash
python scripts/prepare_clean_support.py \
  --dataset spambase --raw-dir data/raw --seed 0 \
  --support-size 40 --support-seed 7 --group-policy not_applicable

# 确认准备方案后，写入全新独立目录：
python scripts/prepare_clean_support.py \
  --dataset spambase --raw-dir data/raw --seed 0 \
  --support-size 40 --support-seed 7 --group-policy not_applicable \
  --out-dir data/support-exploration/spambase-new --write
```

“dry-run”仍需读取、校验原始输入并计算划分/预处理，但不训练或写任何制品。

## 2. 单候选 CPU 实验

入口 `scripts/run_clean_support_experiment.py`，支持 self_pu / lagam / pulns。
当前仅2-D表格、CPU、SCAR OS、额外监督组、终点 OA；不开放 PA/SAR/TS、CNN、
搜索池或正式配方。准备器的图像支持不等于已接通 CNN 实验入口。

参数 JSON 必须显式写 `device="cpu"`、hidden_dim、batch_size，以及：

| 方法 | 必填阶段预算 |
|---|---|
| Self-PU | max_epochs、warmup_epochs、self_paced_start/end、distill_start |
| LaGAM | max_epochs、warmup_epochs、num_clusters |
| PULNS | pretrain_epochs、episodes、classifier_epochs |

Self-PU 的原总体先验、三个方法的随机种子由制品绑定，不允许参数覆盖；拒绝注入
backbone/encoder 及隐式 GPU。其他构造默认值会展开到计划，不把默认值当正式批准。
支持集 adapter 的真实元更新/仅 warmup 拒绝、六角色隔离和预算审计保持有效。

例如**仅用于工程 smoke**的 LaGAM 参数 JSON（保存为用户自己的参数文件）：

```json
{"device":"cpu","hidden_dim":4,"batch_size":32,"max_epochs":2,"warmup_epochs":0,"num_clusters":2}
```

```bash
# 默认只验证制品和参数，输出预先固定的计划，不训练：
python scripts/run_clean_support_experiment.py \
  --artifact-dir data/support-exploration/spambase-new \
  --method lagam --c 0.5 --parameters path/to/parameters.json

# 显式执行一次探索性训练，不能覆盖结果目录：
python scripts/run_clean_support_experiment.py \
  --artifact-dir data/support-exploration/spambase-new \
  --method lagam --c 0.5 --parameters path/to/parameters.json \
  --out-dir data/support-exploration/lagam-new --execute
```

计划在 fit 前绑定制品、所列源码/锁文件、有效构造参数、生成先验和 P/U 标签摘要。
执行时再次核对，不按 test 调参、换分支或选候选。Self-PU 内部 teacher 用独立
support 选择；外部 OA 使用另一个 clean_val。只保存一份 `selected_OA.pkl`，以及
plan/environment/manifest/summary。CPU 模型实际设备与环境机器是否有 CUDA 分开登记。

写入后只加载本次刚生成的可信模型，核对原始分数、OA 固定验证仿射/阈值和 test
准确率，不用 test 重新定阈值。不要加载他人未经信任的 pickle。warning 或 retry
会保留证据并返回失败，不能静默作为通过或继续下一轮；外层 runner 既有一次同 seed
重试行为未修改。中途异常仅有部分制品时，不算成功。

manifest 显著标注 extra_clean_support；暴露的训练标签预算为“实际观测 P 数 + support
的全部正负标签数”。这是本研究模拟协议的监督披露口径，不等于真实数据采集成本估计。
任何工程通过均保持 `formal_admission=false`；没有正式数据结果、GPU/多 seed 准入声明。

## 3. 自动验收与剩余依赖

测试覆盖五数据集存取、标签无关抽样复现、support/heldout 特征扰动不改变 train 统计、
图像 test 真值不进入预处理/总体先验、常见损坏/伪造批准字段/身份冲突/预算缺失拒绝、
CLI 默认不写不训、三种方法真实小型 CPU 训练、额外标签预算、OA 模型重载、确定性，
以及 warning 注入后拒绝接受。这里只使用合成/临时测试夹具，不是正式数据跑批。

后续依次为：取得 P3MIX 原始全文/补充包；确定真实 support 的来源/标签预算及正式
共享规格，运行实际数据；补文本/分组政策和获批 CNN 接线；最后按 P3.3/P4.1 的
正式前置进行资源、多 seed 与方法准入验收。不可代签合作者复核或改写旧 Pilot 结果。
