# 线性 LBE 与神经 LBE 临时成对实验

按用户 2026-10-16 新任务，停止主动推进 P2.2 review，先做算法对照。
本脚本只产生独立探索性结果，不修改冻结协议、v3 矩阵、旧 645-run 或人工签署。

## 固定范围

入口为 `scripts/run_lbe_comparison.py`。支持已有二维分区：Spambase、IMDB、20News、Connect-4；
当前不展平图像、不更换 encoder、不重新划分数据，不做自动推荐/调参。

- 每次一个数据集，默认 seed 0–4、SCAR c=0.1/0.3/0.5；每单元两方法各一个候选，共 30 runs。
- 读取实际 `split_<seed>`，核数据集/seed、四路身份、行数、索引隔离、有限二维特征及文件 SHA。
- SCAR 复用现有固定数量生成器。计划预先冻结 train/pu_val 标记摘要，实际两方法再次核对，
  不因随机生成的统计诊断或 test 指标重新抽 seed/标记，不把未标记 U 当成真实负类。
- 总体 prior 直接取已有 source-pool 元数据，不能用 train/test 子集重新推断。
  它用于 PA proxy accuracy；线性版同时作后验初始化，神经版只验证不消费，差异显式披露。
- OA 仅使用 clean_val accuracy，PA 仅使用 pu_val；两者分别复用当前 min-max/11 点阈值规则，
  test 只在选择结束后评测 accuracy/AUC，不翻转分数，不按 test 调参或改变选模范围。

## 选模与预算

默认 `--selection-scope terminal`：两方法各提供一个训练终点，独立 PA/OA 选择阈值。
这限制了 checkpoint 搜索机会，但**不是**冻结深度 Pilot 的逐 epoch 选模；明确作为终点探索比较。

可选 `--selection-scope em_rounds`：神经版用真实 EM-round callbacks 保存模型，PA/OA 独立选择轮次；
线性版没有 callback，仍仅终点。必须单独报告，不能声称选模机会或计算预算相同。

正常预算：线性版采用现有协议的 20 EM / 内层 max_iter=1000 / C=1；神经版采用当前默认
MLP100/tanh、20+20 预训练、20 EM×40 M updates、10% 重抽样、batch_size=256、Adam1e-3、wd1e-3。
其默认共 840 个 optimizer steps，不是 840 epoch；线性版保留实现原有收敛退出条件。
两种方法的计算/正则化单位不等价：本实验不是等计算预算的受控架构消融。
线性版内部 StandardScaler 会在加权训练副本上 refit，神经版使用已准备特征，亦作为混杂因素披露。

`--smoke` 仅用 2 EM、神经预训练2+2/M2，必须标为 smoke，不能用于效果结论。

## 使用

CPU 先跑小样本量的 Spambase，不占用其他用户 GPU：

```bash
python scripts/run_lbe_comparison.py data/splits/spambase \
  --dataset spambase --out-dir data/lbe-comparison-new --dry-run

python scripts/run_lbe_comparison.py data/splits/spambase \
  --dataset spambase --out-dir data/lbe-comparison-new --device cpu
```

`--dry-run` 只打印计划，不 fit、不建立输出目录；实际运行只接受全新目录，不覆盖/续跑旧实验。
GPU 需用户协调权限后明确指定 `--device cuda`；没有 CUDA 时拒绝，不静默降级。
CPU 默认 torch 1 线程；建议同时设 OPENBLAS/OMP/MKL 线程数，环境必须具备项目与 torch 依赖。
本轮现有 CPU 测试环境不声明为正式 frozen-lock。

## 产物和解释

- `plan.json`：训练前固定范围、预算、输入/标记摘要、选模口径及限制。
- `environment.json`：commit、dirty 状态、脚本/依赖锁摘要与实际软件/硬件。
- 每 run 的 `manifest.json`、`fit_diagnostics.json`：实际标记、选择、test、成本、迭代、收敛及 warning。
- `selected_PA.pkl` / `selected_OA.pkl`：**模型加对应 selection**。只加载自己生成的可信 pickle；
  用脚本 `predict_saved_selection(payload, X)` 重放 VAL 归一化与阈值，不能直接用裸模型默认 predict。
  神经版 EM 权重是推理 checkpoint，不保证 propensity/optimizer 续训恢复。
- `pair_c_<c>_seed_<seed>.json`：两边成对记录；失败保留已有产物并出 failure.json，不伪造完整汇总。
- `summary.json/csv`：每 c、每 PA/OA 的 accuracy/AUC 均值和样本 SD（ddof=1），
  同 seed 的 neural−linear 差值及其样本 SD；n=1 的 SD 为 null，缺/重复/错配 run 拒绝完整汇总。
- 成本每 run 计一次，不因 PA/OA 两行重复累计；CPU 显存缺测为 null，不记 0。

所有 manifest 仍为 technical_smoke/formal_eligible=false。预算完整的探索运行也不是正式准入，
不能与旧 Pilot 数值直接混组，不生成跨数据集总排名，也不把单一数据集胜负推广为神经版普遍更好。

实际首轮结果与验证在完成后补记；此前短预算 smoke 仅验证链路。
