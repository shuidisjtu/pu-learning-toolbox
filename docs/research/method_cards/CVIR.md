# Method Card: CVIR

> 技术预集成，不是 BBE/TEDⁿ 的实现，也不是 Survey v1 正式跑批方法。

## 来源与核心口径

| 字段 | 内容 |
|---|---|
| 论文 | Garg 等，*Mixture Proportion Estimation and PU Learning: A Modern Approach*，NeurIPS 2021，[原文 Algorithm 2](https://proceedings.neurips.cc/paper_files/paper/2021/file/47b4f1bfdf6d298682e610ad74b37dca-Paper.pdf) |
| 官方仓库 | [`acmi-lab/PU_learning`](https://github.com/acmi-lab/PU_learning) |
| 本项目入口 | `CVIRClassifier`，注册名 `cvir`，稠密二维表格 MLP，PyTorch |
| 必需先验 | `unlabeled_positive_prior = α_U = P(Y=1 \mid X\in U)`，即原始未标记池的混合比例 |

算法每轮先按当前正类分数给**原始 U** 排序，保留分数最低的约 `1−α_U` 部分为临时负例；随后按 `α_U·L_P + (1−α_U)·L_N` 更新模型。保留数有限样本下取 `max(1, floor((1−α_U)·n_U))`，相同分数稳定排序。默认先做一轮 P-v-U warm-start，再运行固定 20 轮 CVIR；这两个预算是本工具箱适配选择，原文 Algorithm 2 使用收敛停机。`class_prior` 参数会显式拒绝，避免把公共接口传来的总体 `π` 无声当作 `α_U`。

## 台账与校准

原文的 U 是总体边缘分布，两样本 case-control 下才有 `α_U=π_population`。若 PU 数据来自单一总体并移走已标记 P，剩余 U 的正类比例通常不同；仅凭观测 P/U 行数不能无条件转换。因此注册信息虽写 `requires_class_prior=False`，**不代表本算法不需要先验**，而是公共总体先验门禁不适用于它；直接拟合时仍必须显式给出 `unlabeled_positive_prior`。实验方法台账登记原生 `ts`，但当前没有可证明正确的 `os_or_ts` 损失替换：把已知 P 并入 CVIR 的临时负例候选池可能误选 P，故 `run_view=os-compatible`、`calibration_applied=false`，实验路由显式请求 `ts` 会报错。冻结的 `survey_protocol_v1.json` 保持不变。

## 适配边界与下一步

本实现没有 BBE 先验估计与 TEDⁿ 交替过程，没有论文图像/文本 backbone、原始训练预算、公开数值复现或多 seed GPU 证据。2026-09-28 已在全局 Python 3.12.2 / PyTorch 2.6.0+cu124 的 RTX A6000 上完成单次 CUDA 技术 smoke；它不是 frozen-lock 或多 seed 验收。需先由合作者复核 `α_U` 来源与 TS-OS 方案，再决定是否扩展 Survey 的先验字段、选模/视图协议与执行矩阵。单元测试证明负例选择、接口、确定性、checkpoint 与基础训练闭环；公开来源对照程度见 P3 交接记录，不能读作论文数值复现。
