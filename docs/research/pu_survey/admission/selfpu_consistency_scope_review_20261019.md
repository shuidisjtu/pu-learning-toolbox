# Self-PU 一致性定义域来源复核与适配补齐（2026-10-19）

## 来源结论

以 ICML 2020 正式论文 §3.3.1–3.3.2 为数学权威：[PMLR 正文 PDF](https://proceedings.mlr.press/v119/chen20b/chen20b.pdf)。式 (13) 将 student consistency 定义在各 student 自身的 `D - D_trust`，式 (14) 再以 hard-sample 条件筛选；由于 `D_trust` 只从 U 扩充，该域包含 observed P 与未信任 U。式 (16) 的 teacher consistency 定义在整个 D，因此还包含 trusted U。

旧实现只在 U batch 上计算两类一致性。这个收窄不改变算法目标，但不是论文定义域；先前方法卡/台账已披露该差异。本次将其变成显式工程选项，而不是覆盖旧行为：

| 路径 | Student consistency | Teacher consistency | 用途 |
|---|---|---|---|
| `distillation_scope="untrusted_u"`（默认） | 未信任 U | 当前 U batch | 历史默认与冻结 Pilot 消融兼容 |
| `distillation_scope="paper_population"` | P + 未信任 U | P + 全部 U（包含 trusted U） | 显式论文定义域适配；clean-support full-meta 探索入口自动选择 |

paper-population 路径按逐行 PU 风险分解计算 hard-mining 掩码：P 行使用 `π_p · ℓ_P^+`，未信任 U 行使用逐行负损失 `ℓ_U^-`。该映射与论文式 (14) 的逐样本表述相容，但仍属 clean-room 工程翻译，不声称作者源码逐步重放或数值复现。`ts` 下校准只改变 PU 风险中的负角色集合；一致性定义域与权重不随视图变化，known P 不进入 trusted/pseudo-label 集。

## 工程与台账变化

- `SelfPUClassifier` 增加显式 `distillation_scope`，校验两个受支持取值；训练 history 与 `get_pu_metadata()` 记录实际 scope、student/teacher 一致性行数。
- 默认值保持 `untrusted_u`，不改变旧模型的前向、损失或随机数轨迹。
- `SelfPUCleanSupportTrainer` 是有独立 clean support 的额外监督路径，在 experiment preflight 中强制 `paper_population`；冻结 Pilot 不经过该 adapter，仍是无 meta 消融。
- API 参考、方法卡及 `method_ledger.json` 明确区分两条路径。没有修改历史 Pilot、冻结矩阵、TS 校准合同、PA 资格或正式准入。

## 验证与剩余前置

新增回归锁定默认路径仍为 U-only，以及 paper-population 路径 student 的 P+未信任 U、teacher 的 P+全部 U 行数；支持集 adapter 的 manifest 和拟合模型也必须声明 `paper_population`。相关 Self-PU estimator / clean-support adapter / ledger-contract 测试通过后，代码级差异可追踪。

这项改动不补齐外部 clean support 的实际来源/样本身份/额外标签预算，也不解除 P4 资源、真实数据 benchmark、合作者 RV5/RV6/RV8 复核或正式准入前置。若要正式跑 Self-PU，仍须先批准支持集与所需 clean labels 的预算，并按已批准的模型/训练协议另行验收。
