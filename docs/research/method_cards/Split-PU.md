# Method Card: Split-PU

资源接口补齐（2026-10-05）：快照上界为
`checkpoint_epoch_count=teacher_epochs+split_epochs+rounds*student_epochs`，
splitter 早停会减少实际快照，跑前不能用未发生的早停低估峰值。
`optimizer_steps_` 累计 teacher、splitter 与所有 student rounds 的真实更新，每次 fit 重置。
显式协议预算仍优先；不是正式 bytes/component 或训练恢复协议。
step 插桩、早停快照/回放和失败不重试见 `test_multistage_candidate_accounting.py`。

> 当前支持 MLP 或注入可训练 CNN 的技术预集成，不是论文 CIFAR 结果复现或 Survey P3.2 验收。

| 字段 | 内容 |
|---|---|
| 论文 | Xu 等，*Split-PU: Hardness-aware Training Strategy for Positive-Unlabeled Learning*，ACM MM 2022；[arXiv:2211.16756](https://arxiv.org/abs/2211.16756) |
| 官方代码 | [`loadder/SplitPU_MM2022`](https://github.com/loadder/SplitPU_MM2022)，核对 commit `82fb9730597a65a12156736419d4588675bc24d5` |
| 源码位置 | `main.py`、`splitpu_utils.py:split/train_splitpu`、`utils.py:JSDLoss/hard_loss/sim_loss`、`nnpu_utils.py:train_nnpu` |
| 先验 | `class_prior` 为 U 分布正类比例，只用于 nnPU teacher 预训练 |

## 训练阶段

1. 用 P/U 的 nnPU 风险训练 teacher；输入仅有观测 PU 标签。
2. 随机初始化 temporary 模型，拟合 teacher 的硬预测；在 U 上的预测一致率达到阈值可提前停止。teacher/temporary 对 U 的预测不一致即 hard，其他为 easy。
3. 随机初始化 student；对 P 用 BCE，对 easy U 用 teacher 软预测的加权 Jensen–Shannon 散度；对 hard U 用弱/强扰动预测一致性、teacher 低层特征 MSE 与 student 表征余弦一致性。下一轮以上一轮 student 为 teacher，默认两轮。

第一轮默认 hard/feature/sim 权重为 0.3/0.3/0.1；第二轮按官方 `main.py` 收窄为 0.01/0/0。

公开类 `SplitPUClassifier`，注册名 `split_pu`（别名 `split-pu`）；默认二维 MLP，可选 CPU/CUDA。`decision_function` 是 raw logit，0 为预测阈值，不宣称校准概率。非空 `sample_weight` 被拒绝。训练历史、每 epoch checkpoint 回调和 `state_dict` 权重恢复可用；若小样本集出现 U 全部一致/不一致，以 teacher 最低/最高 margin 的一个 U 维持两个分支非空。至少需要两条原始 U，否则明确拒绝，避免空 easy/hard 分支。

数据池保留在 CPU，教师风险、splitter 和 student 仅逐批搬入设备；冻结教师目标缓存于 CPU，分歧与 margin 扫描也分批执行。推断分批 eval，并恢复原模型模式，支持合法空预测、拒绝 float32 转换溢出。该资源工程改动不添加原生 CNN/官方增强，也不改变 teacher-only TS 风险角色。

### 注入 CNN 的工程路径（2026-10-05）

`encoder=...` 接受输出有限 `(batch, feature_dim)` 的模块，四维输入必须提供 encoder，
不会 flatten 图像。教师、splitter、各轮 student 分别 deepcopy 同一个用户模板后训练；
模板权重/冻结标志不改变，各阶段互不共享参数。`encoder_` 指当前 student，
`teacher_encoder_` 指冻结教师副本，`splitter_encoder_` 指 temporary 编码器。

hard 分支的低层 MSE 使用**真实编码器中间模块输出**，不是分类头隐层冒充卷积层；
`encoder_feature_layer="模块路径"` 可显式选择，默认选首个 MaxPool2d，否则首个 Conv2d/Linear。
路径必须实际执行且每次仅执行一次、返回有限且保留 batch 轴的张量。
高层余弦使用编码器最终特征向量。forward hook 仅在当前前向存在，finally 清除；
输出 clone 保留梯度，防止后续 inplace ReLU 覆盖低层特征。
两阶段图像尺寸必须固定；预测校验完整 NCHW shape。

单行组无法让 1×1 特征图执行训练态 BatchNorm，故该批 BN 暂用运行统计，
affine/编码器仍有梯度，模式随后恢复；不复制/丢弃行，不改变 PU 标签或分组。
较大组保留正常 BN 更新。用户编码器需有可用于 eval 的 BN 运行统计。
公共 pipeline 的 CNN 接入、fold/阶段隔离、固定 seed、pickle/各阶段推断快照、
低层解析梯度与 CUDA 回放由 `test_split_pu_cnn.py` 和共享 integration 测试覆盖。

## 差异与门禁

- 官方使用 CIFAR CNN、图像弱/强增强和带 predictor 的高层 SimSiam 一致性；本版仍以 Gaussian 弱/强扰动和余弦一致性替代，不含官方 SimSiam 投影/predictor。默认 MLP 用隐层特征，注入 CNN 用真实中间/最终编码特征；低层选择和相同模板初始化各阶段均属**显式工程适配**，不可直接拿论文准确率作数值裁决。
- 官方代码训练中反复读取测试标签报告准确率；本实现的 `fit` 无真实标签入口，不使用 test set 决定阶段或权重。PA/OA 由外部 runner 分开执行。
- 锁定 `nnpu_utils.py:30-34` / `splitpu_utils.py:170-172` 返回末轮 state_dict；虽然声明 best_acc/best_weight，但未用于恢复。**测试监控不是 test-best 选模**，不由这些变量名字推断存在该选模。
- 已配置公式、阶段、接口、确定性、checkpoint 和 CUDA smoke 测试。方法台账已登记；`fit(os_or_ts="ts")` 会在 **nnPU teacher 的未标记风险项**逐 mini-batch 使用 `U ∪ P`，正例项与先验保持不变。temporary 的一致率、easy/hard 划分及 student 的 U 分支仍只取原始 U，避免已知 P 被当伪负例。台账默认 `ts-compatible`，实际视图以 run manifest 为准。2026-09-28 全局 Python/PyTorch 环境下的 A6000 原有 GPU 测试和 TS 特征路径单次 CUDA smoke 均通过；不替代 frozen-lock、多 seed 或正式资源验收。公开论文数值对照、Survey 矩阵、合作者复核仍待完成；本技术预集成不进入冻结 pilot/主榜。

## 准入证据补全（2026-10-05）

2026-10-07 补[阶段/预算规格](../pu_survey/p3_stage_budget_review_20261007.md)：
teacher/splitter/student 快照保留局部 epoch、student round 与累计更新数，
实际 splitter 早停后不假造快照；恢复/回收不丢阶段。正式阶段及轮次资格仍待批准。

P1-2 [公开对照准备](../pu_survey/p3_public_comparison_20261005.md)核 Table1 Ours
n_P500/1000/3000 accuracy89.18±0.12/90.51±0.10/92.51±0.10，5次重复，全部 pending。
原论文PDF6 teacher50轮与锁定 main.py:14 teacher20轮不同，无日志确定哪一配方产出表格；
本版默认10轮又是独立适配，不能只按同名epoch或论文数字宣布复现。

见[五方法证据包](../pu_survey/p3_admission_evidence_20261005.md)。上游
`splitpu_utils.py:162-165` 用 `(sum_P BCE+sum_easy JSD)/(n_P+n_easy)`；本版为
`mean_P BCE+mean_easy JSD`，相对组权重不一致，不能仅凭 JS 公式一致声称训练等价。
上游 teacher20/splitter最多50/student100每轮及不同阶段 LR/weight_decay，不是我方10/10/10。
本次补独立 Bernoulli JS 与 teacher detach 对照、阶段/预算和选模边界登记；不静默改目标。
官方增强、SimSiam、正式预算及上述显式适配是否接受仍须负责人判断。
