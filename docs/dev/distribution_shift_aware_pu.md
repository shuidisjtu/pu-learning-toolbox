# 分布漂移感知 PU：设计说明

## 1. 目标

PU 模型通常在源域（历史、训练环境）上训练，却在目标域（当前、部署环境）上使用。
当两个域的分布不同，常规交叉验证仍可能表现正常，但部署效果已经下降。本功能希望回答三个
彼此不同的问题：

1. **是否存在可观测的分布差异？**
2. **源域对目标域是否有足够的覆盖，重要性加权是否稳定？**
3. **在明确假设成立时，能否用目标域信息改进 PU 训练？**

稳定 API 提供可发布的漂移审计、相对密度比和协变量漂移加权工作流。第二批增加配对决策、
连续监控、双域假设分析和人工复核工具。对于不能由现有 `sample_weight` 契约正确表达的联合
分布漂移，稳定报告必须停止在“发现风险”，不得把它描述成已经完成适配。另有明确隔离在
`estimators.research` 下的联合漂移求解器，供研究验证使用，不进入注册表和 `auto` 推荐。

## 2. 问题设定

记源域和目标域联合分布为 (p_s(x,y)) 与 (p_t(x,y))，真实类别
(y\in\{+1,-1\})。PU 标签 (s\in\{1,0\}) 中，`1` 表示已标记正例，`0`
表示未标记样本。输入包含：

- 必需：源域特征 `X_source`、源域 PU 标签 `y_source_pu`、目标域特征 `X_target`；
- 适配必需：目标域 PU 标签 `y_target_pu`；
- 可选审计真值：源域/目标域 `y_true`，只用于 oracle 评估，不参与训练；
- 可选：源域和目标域各自的正类先验 π；未知时必须分别估计并披露来源。

需要区分以下情况：

| 情况 | 条件 | 第一版能力 |
|---|---|---|
| 无明显可观测漂移 | 域分类器接近随机 | 报告证据不足，不声称分布相同 |
| 协变量漂移 | (p_s(x)\neq p_t(x))，且 (p_s(y\mid x)=p_t(y\mid x)) | 可用 (p_t(x)/p_s(x)) 加权 |
| 类先验漂移 | (p_s(y)\neq p_t(y)) | 分域估计并提示；普通样本权重不保证修正 |
| 概念/联合漂移 | (p_s(y\mid x)\neq p_t(y\mid x)) | 可检测风险但不能从边际域分类器识别或修正 |
| 支持集不重叠 | 目标域落在源域低密度区域 | 警告加权不可靠，建议采集目标域 PU 数据 |

域分类 AUC 只说明“样本来自哪个域是可预测的”，不能单独识别上述漂移类型，也不能证明
SCAR/SAR、类先验或覆盖假设成立。

## 3. 研究依据与实现边界

Kumagai 等人在 AISTATS 2025 提出的
[Importance-weighted Positive-unlabeled Learning for Distribution Shift Adaptation](https://proceedings.mlr.press/v258/kumagai25a.html)
使用少量目标域 PU 数据，在不预设漂移类型的情况下估计
(w(x,y)=p_t(x,y)/p_s(x,y))。论文使用有界的相对密度比

\[
w_\alpha(x,y)=\frac{p_t(x,y)}{\alpha p_t(x,y)+(1-\alpha)p_s(x,y)},
\qquad 0<\alpha\le 1,
\]

并交替优化共享特征提取器、类别分类头和权重模型。该方法还要求源域和目标域的 PU 数据及
两个域的类先验。

第一版采用同一类“有界相对密度比”思想，但仅估计可观测边际比
(r(x)=p_t(x)/p_s(x))：

\[
r(x)=\frac{q(D=t\mid x)}{1-q(D=t\mid x)}\frac{n_s}{n_t},
\qquad
w_\alpha(x)=\frac{r(x)}{\alpha r(x)+(1-\alpha)}.
\]

其中 (q(D=t\mid x)) 是带标准化的逻辑回归域分类器。第一版因此是**协变量漂移基线**，
不是上述论文联合漂移算法的复现。目标域没有 PU 标签时只能审计，不能启动适配。

## 4. 第一版架构

### 4.1 漂移审计

`analyze_pu_shift(...)` 负责输入校验、域可分性评估和稳定性诊断：

- 使用分层交叉验证的域分类概率计算 OOF ROC AUC，避免用训练内 AUC 夸大漂移；
- 在全部数据上重训域分类器，计算源域相对重要性权重；
- 报告权重分位数、裁剪率与有效样本量
  \(ESS=(\sum_i w_i)^2/\sum_i w_i^2\)；
- 分别报告源域/目标域样本量和已标记正例比例；
- 给出结构化 `issues` 与下一步建议，而不是只输出一个分数。

默认严重度仅作为操作提示：

| OOF 域 AUC | 等级 | 含义 |
|---:|---|---|
| `< 0.60` | `low` | 当前分类器未发现强可分性，不等价于无漂移 |
| `[0.60, 0.75)` | `moderate` | 存在可观测差异，应检查特征和时间/人群切分 |
| `>= 0.75` | `high` | 域差异明显，常规源域 CV 很可能低估部署风险 |

若 `ESS / n_source < 0.5`、较多权重触及上界，或归一化前源域相对权重均值低于
`0.1`，额外给出覆盖/方差警告。最后一项用于识别“所有源权重都接近零、归一化后却
看似均匀”的支持集断裂。阈值是工程启发式规则，报告中必须保留原始指标供用户自行判断。

### 4.2 重要性权重

默认 `alpha=0.1`，因此相对权重上界为 10。分类概率先裁剪到
`[probability_clip, 1-probability_clip]`，然后计算相对密度比；权重最后归一化为源域
均值 1，方便传给现有分类器。报告同时保存归一化前的上界触及情况。

目标域对源域没有支持覆盖时，任何密度比方法都无法创造缺失样本；此时权重极端值和低
ESS 是拒绝自动适配的重要信号。

### 4.3 `ShiftAwarePUPipeline`

该工作流组合 `analyze_pu_shift` 与现有 `PUPipeline`：

1. 审计源域和目标域；
2. 当且仅当用户提供目标域 PU 标签、明确选择协变量适配且分类器声明
   `sample_weight_support="supported"` 时，将源域相对权重传入每个 CV 训练折和最终重训；
3. 返回基础 `PipelineReport`、`PUShiftReport` 和目标域预测摘要；
4. 若分类器忽略或拒绝样本权重，立即失败，不允许静默降级。

交叉验证仍发生在源域，因此它只用于检查训练稳定性。目标域没有 `y_true` 时，目标性能
不可识别；报告不能把源域 CV 指标标成目标域效果。

## 5. CLI 与产物

新增独立命令，避免改变稳定的 `run` 参数语义：

```bash
pu-toolbox shift-audit \
  --source-data source.csv \
  --source-labels source_labels.csv \
  --target-data target.csv \
  --target-labels target_labels.csv \
  --out-dir shift_output
```

固定产物：

- `shift_report.json`：稳定、严格 JSON schema；
- `shift_report.md`：人可读结论、假设与警告；
- `source_importance_weights.csv`：与源域行一一对应的归一化权重。

`shift-audit` 只做审计和导出权重。`shift-run` 在同一配置、同一目标集上配对运行未加权
和加权两臂，输出 `shift_comparison.json/.md` 与 `target_predictions.csv`。只有目标真值
的 oracle 指标或目标类先验依赖指标可参与自动推荐；`pu_recall` 等 PU 可观测指标只展示，
不能单独触发换模。覆盖门禁失败时跳过加权臂并建议补充目标数据。

## 6. 后续扩展

- `PUShiftMonitor`：相对固定参考域逐窗口保存域 AUC、ESS、标记率变化和告警；历史文件
  不保存原始样本，恢复时校验监控配置。
- `analyze_domain_assumptions`：分别解析/估计两个域的类先验，再用
  `P(S=1)=P(Y=1)·c̄` 分解平均标记倾向；行级 bootstrap 在每次重采样中重新估计缺失先验，
  并把不确定性传播到标记率、平均倾向及域间差值；`c̄` 不识别特征依赖 SAR。
- `analyze_pu_uncertainty`：提供概率边际不确定性、拒绝预测、coverage 和三种主动人工
  复核排序。它不是贝叶斯或 conformal 置信区间。
- `JointShiftPUClassifier`：研究级 sklearn 求解器。它按软类别成员分别估计类条件域比，
  乘类先验比并有界化，随后交替更新 PU 分类器。该实现受 Kumagai 等人的相对联合权重
  思想启发，但不是论文共享神经特征与 PU 风险目标的精确复现。
- `DynamicJointShiftPUClassifier`：Torch clean-room 求解器，实现论文式 (13)、(19)--(23)
  的目标、绝对值风险修正、共享特征和 Algorithm 1 的两阶段梯度隔离。实现采用确定性全批次
  训练来核对公式与更新边界；作者源码未公开，因此不声明官方实现等价或论文数值复现。
- `JointShiftPUBaseline` 与 `build_joint_shift_estimator`：在同一神经规模下提供 `trPU`、
  `tePU`、fine-tune、五核 RBF-MMD、two-step 和损失修正消融。
- `benchmarks/joint_shift`：用公开 Wisconsin 表格数据执行多 seed、Student-t 95% CI 和
  集合重叠审计。它是工具箱自定义 concept-shift smoke，产物固定 `paper_claim=false`。
- `shift-monitor` 与 `review` CLI，以及 UI 部署面板：把窗口漂移历史、覆盖门禁、拒绝预测
  和主动复核导出接入实际部署旅程。`review` 只加载用户明确提供且信任的 pickle 模型。

## 7. 失败策略

- 源域/目标域特征列数不一致、NaN/Infinity、样本过少：失败并说明修复方法；
- 任一域不足以完成分层域 CV：降低到可行折数，仍不足则失败；
- `alpha`、裁剪范围或权重非有限值：失败，不导出部分结果；
- 目标域没有 PU 标签：允许审计，报告明确 `adaptation_ready=false`；
- 分类器不真正支持 `sample_weight`：适配工作流失败，不忽略权重；
- 高 AUC + 低 ESS：允许导出审计，但默认不建议自动适配。

## 8. 验收标准

- 相同分布的固定种子数据，域 AUC 接近随机且权重均值为 1；
- 已知均值漂移数据，域 AUC 和风险等级明显升高；
- 相对权重有限、符合 (1/\alpha) 上界变换，ESS 计算有数学金标准测试；
- 结构化报告可严格 JSON 序列化并正确保存 Markdown/CSV；
- CLI 能完成真实文件端到端旅程，并对列数不一致给出用户错误；
- 工作流只接受声明支持权重的分类器，CV 折和最终重训均收到对应权重；
- 完整测试、格式、文档链接、元数据、数学渲染和 skill 同步门禁通过。

## 9. 非目标

稳定路径不声称：从 PU 观测唯一识别漂移类型；自动修复概念漂移；用源域 CV 证明目标域
性能；在无支持重叠时可靠外推。研究路径只声称公开公式的 clean-room 实现，不声称与作者
私有源码逐梯度等价，也不声称复现论文七个实验设置的数值。剩余证据见
[补充清单](distribution_shift_aware_pu_checklist.md)。
