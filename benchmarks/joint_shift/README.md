# 联合漂移公开数据 benchmark

该协议使用 scikit-learn 内置的公开 Breast Cancer Wisconsin (Diagnostic) 数据，不需要
联网下载。源域、目标 PU 训练集和目标测试集按每个 seed 分层且严格不重叠；源域在一个
固定特征区域改变类别关系，形成工具箱自己的 concept-shift smoke 协议。

```bash
python -m benchmarks.joint_shift.runner \
  --out-dir results/joint_shift \
  --seeds 1,2,3,4,5 \
  --methods dynamic,trpu,tepu,fine_tune,mmd \
  --target-pu-size 100 \
  --max-epochs 50 \
  --device cpu
```

产物：

- `trials.csv`：每个 method/seed 的 accuracy、ROC AUC、两个域先验和测试样本数；
- `summary.json` / `summary.md`：样本标准差和 Student-t 95% CI；
- `config.json`：完整协议参数；
- `split_audit`：内嵌于 summary，检查三个集合的样本 ID 重叠。

该数据、切分和 epoch 设置不是 Kumagai 等人 AISTATS 2025 的原始七个实验设置，所有报告
固定 `paper_claim=false`。它用于验证执行器、比较公平性和统计产物，不能与论文表格数值
直接比较。论文依据见[联合漂移方法卡](../../docs/research/method_cards/Importance_Weighted_PU_Shift.md)。
