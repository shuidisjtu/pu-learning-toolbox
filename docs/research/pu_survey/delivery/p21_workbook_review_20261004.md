# P2.1 / P2.2 交付 Excel 对账与阶段验收（2026-10-04）

结论：**交付表身份、计划覆盖及 raw → summary 数值对账通过；完整制品验收仍待 manifest / 日志 / 权重**。
复核为用户委托的 Codex 技术检查，不代填 HENG958 / shuidisjtu 本人签署，也不解除文献与 PA 阻断。
机器证据见 [对账 JSON](../data/p21_workbook_review_20261004.json)。

## 1. 收件身份及逐层结果

收到 `P2.1_results_645runs_20261003(1).xlsx`：126370 bytes，sha256
`cafedc98d8983764625d7b32e62301aa8de87e7da9d772e185f6a135df9c1244`，
与 [交付索引](../data/p2_2_artifacts_index.json) 完全一致。文件名的 `(1)` 不影响字节身份。
未编辑原表，未把原始 XLSX 提交至 Git。

| 层 | 独立复核结果 |
|---|---|
| 运行覆盖 | 645 个唯一键与冻结 `survey-v1.2` 展开计划逐键一致；无重复、遗漏、幽灵单元 |
| 批次 | B1 215 / B2 215 / B3a 110 / B3b 70 / B4 35，与 coverage 页逐项一致 |
| 视图与校准 | lbe / pn_oracle 为 OS、其余 pilot 方法为 TS；标志一致；Self-PU 消融 105 条均披露 |
| 选模输出 | OA 645 次、PA 270 次；PA 仅 SCAR PU 行，oracle/SAR 不冒充 PA |
| 汇总覆盖 | 183 条唯一键，各 5 seeds；OA 129 条 / PA 54 条，无遗漏或重复 |
| 状态 | 180 formal / 3 partial；3 partial 均为 oracle，15 次 run 的 c_grid 偏差明确保留 |
| 数值 | 183 行 accuracy 均值、样本 SD（n−1）、AUC 均值及观测数独立重算一致 |
| 分组投影 | 从字段重建 16/16/7/6/3 共 48 组；**不是**重跑完整公平性门禁 |
| 成本 | 129 个 run-cost 组；PA/OA 共用训练成本，不能把 183 行成本相加计费 |

accuracy/AUC 均值最大偏差约 `1.42e-14` 个百分点，样本 SD 最大偏差 `2.66e-15`，属于浮点舍入。
raw 的训练耗时保留 4 位小数，5 seed 相加允许最多 `0.00025` 秒偏差；实测最大 `0.000155` 秒，全部在界内。
tuning 最大偏差约 `3.64e-12` 秒；GPU 峰值的汇总列与逐 run 最大值一致。
按 645 runs 去重，single-config 总耗时约 **46.598 小时**（累计运行耗时，不等于自然日墙钟时间）。

## 2. 说明页须勘误，原数据不改

1. **c 不是 class prior**：readme 中 “Class-prior grid” / “Requested class-prior token” 应改为
   **label frequency / 标记频率** `P(S=1|Y=1)`；总体先验是 π，扫描 c 时 π 固定。
2. **OS/TS 不是“免先验/免 oracle”与“迁移”两种模式**：本项目表示训练采样/风险输入视图。
   TS 校准将适用无标签风险项的输入由 `D_U` 换为 `D_U ∪ D_P`，不要求重新按独立两样本生成数据。
3. **PA 是 proxy accuracy 选模**：按冻结准则在 PU validation 上选择 checkpoint/阈值；
   它需要总体先验，但不能仅称 “prior-aware” 而遗漏准则定义。OA 则用 clean validation accuracy。

这些是交付说明的修订请求，不是修改数值或事后改变实验协议。
请执行方后续重发勘误后的交付说明；若重发 XLSX，另记新摘要，旧索引与旧文件原样留档。

## 3. 已确认限制与质量诊断

- **315/645 条显存峰值未测量**：lbe / pusb_kernel / upu 各 105 条；不能为空填零。
  对 GPU 内存统计只能报告实际测量子集，不能从缺测推断未使用 GPU。
- Excel 不包含 precision / recall / F1；不能补造这些指标，须另取最终测试记录或预测制品。
- 表内 645 行的 Python 3.10.8、Torch 2.13.0+cu130、CUDA runtime 13.0、NVIDIA vGPU-32GB
  字段一致；这**不证明**依赖集、解释器路径或 frozen-lock 环境相同，环境日志仍需核验。
- B3b/B4 均无记录阻断；B1/B2/B3a 的 oracle 有阻断，B3a+B3b 合并组仍有阻断。
  支持上轮粒度纠错，但 workbook 没有完整 `formal_ready` 报告字段，仍不能冒称已重跑门禁。
- **性能与算术正确性分开**：31/183 个汇总行 AUC 低于 50%（PA/OA 行可能共用同次训练，不能解读为 31 次失败），应核对分数方向、SAR 分布、checkpoint
  与训练曲线，不据低值立即判实现错误。列表已列在 JSON 的 `performance_diagnostics`。
  例如 native CNN nnPU 的 SAR-LBE-A / c=0.05 / OA 为 accuracy **60.766%**、AUC **31.211%**；
  IMDB LBE 的 SAR-LBE-B / c=0.5 / OA 为 accuracy **49.903%**、AUC **15.342%**。
  这是需要解释的实验观测，**不是新注册的拒收阈值**，不改写原 formal/partial 状态，不生成跨数据集排名。

## 4. 对应复核清单的进度

| 任务 / 项 | 本轮代理结论 | 仍缺什么 |
|---|---|---|
| P2.1 覆盖、视图、变体、成本与输出 | 表格层接受 | 原始 manifest 身份、selected checkpoint、日志/退出码及源端 plan |
| P2.2 O1 | 有条件接受：精确运行键覆盖无差异 | 白名单精确根及 probe 隔离仍须源树验证 |
| P2.2 O2 | 有条件接受：183 行与状态计数复算一致 | 48 组实际门禁结果、refusal/reproducibility 字段须报告原件 |
| P2.2 O3/O4 | 表格层接受：状态不变式及 oracle 偏差一致 | 原始阻断字段仍须与 manifest 对账 |
| P2.2 O5 | 有条件接受：支持 B3b/B4 无记录阻断，粒度一致 | 原始 formal_ready 报告 |
| P2.2 O10 | Excel 收件身份确认 | 其余制品交付仍未完成 |
| P2.2 O6/O7/O8/O9 | 未代替审批、删除记录或归档核验 | 本人决定、A 层制品及源端归档材料 |
| P2.0c / PA 方法学 | 保持原待审 | PUSB 单位、来源协议修订、合作者本人签署 |

下一步优先获取 **645 份 manifest（索引记载约 90.76 MB）** 和白名单 config、五份 reference plans，
先验证树摘要再重跑 P2.2 三入口；不必先下载全部 epoch checkpoint。
随后按需取得 selected 权重、逐 epoch selection、环境/退出码日志和归档记录，处理原 not_run 项与性能诊断。
不能以 Excel 替代缺失 manifest 生成伪结果树，也不能把表格层通过升级为所有 P2 任务已验收。

## 5. 可重复执行

需已安装可选复核依赖 `openpyxl`，从仓库根运行：

```bash
PYTHONPATH=. python scripts/audit_survey_workbook.py \
  'P2.1_results_645runs_20261003(1).xlsx'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest -q \
  tests/unit/scripts/test_audit_survey_workbook.py
```

第一条只读 workbook，输出含 183 个独立重建行的 JSON；`ok` 仅指其声明的表格范围。
本轮该工具的 9 项合成工作簿测试与接收/文档检查器测试共 27 passed；ruff、文档链接/结构一致性及 git diff 空白检查通过。
均值/SD 使用独立 `statistics` 算术实现，不调用原汇总函数；合成工作簿测试覆盖均值/SD 篡改、
状态/视图错误、协议网格错误、coverage 差异、文件摘要差异及耗时舍入容差。
