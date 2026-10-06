# Holistic-PU：LZO 正例损失工程选择变体

状态：**implemented engineering variant / pending_method_owner_review**。承接
[来源复核](holistic_source_cnn_review_20261007.md)、[后段初始化](holistic_stage_initialization_20261007.md)
及[后续优先级](survey_execution_plan.md)。没有改冻结协议、历史结果、本人签署或正式候选。

## 1. 来源支持什么、没有支持什么

[LZO 原论文 v2](https://arxiv.org/pdf/2012.13309v2) 的 Algorithm 1（PDF页3）在辅助验证集上
比较风险取 argmin；Algorithm 2（页5）从同类样本以 Beta(α,α) 权重构造 label-invariant mixup。
这是候选选择，不是“验证若干轮不改善就停止”的 patience 规则。
Holistic-PU 的[会议摘要页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d5c0f9585592bad5251133813893a6c0-Abstract-Conference.html)
所附、带行号匿名稿 PDF 页23 §F 说明用原始已标记训练集的 mixup 选终点；不是另取真负验证集。
附件版本、摘要及作者锁定代码定位见来源复核，本文没有将匿名稿当正式主文。

来源未充分确定辅助集大小、固定或每轮重抽、完整候选范围、平局和选中后随机流。
锁定作者代码实际固定 warming_steps，validation 调用被注释。以下规则是**显式工程 recipe**，
不冒充确切原实验。仅正例 CE 不代表总体准确率，CE 本身无界，也不宣称自动继承
LZO 有界损失理论的无偏性/泛化保证。公开数值验证及负责人决定仍需后续完成。

## 2. 可执行接口与完整规则

`warmup_selection="fixed"` 默认不变；`"lzo_positive_loss"` 开启本变体。
`lzo_alpha=0.5`、`lzo_validation_size=None`（默认 n_P），均可显式覆盖；非法配置 fail-closed。
stopping_rule 为 `lzo_positive_loss_full_horizon_argmin`，variant 为
`holistic-lzo-positive-loss-engineering-v1`；不使用 test、外部 PA/OA selection、clean support。

1. 训练集内已标记 P 才能入辅助集；不把 U 当真负，不需要新增真值标签或外部类先验。
2. 克隆模型 seed 消费后的局部 NumPy RNG，独立生成 m 对 P 索引和 Beta 权重；
   配对有放回，允许同一行配对，计划固定跨各轮，不推进训练随机流。
3. 每轮分批在 CPU 按 λ*x_left+(1−λ)*x_right 构造特征/原图，只混合批次上设备；
   eval 正例平均 binary CE，按总行数池化，不平均大小不同的 batch mean。
4. 模型完整运行 warmup_epochs，所有轮均记录验证风险；可选终点为2..warmup，
   1轮不足以定义趋势。exact 最小值并列时取最早，非有限风险直接失败、不回退 fixed。
5. 趋势只消费选中 t 的概率前缀；continue 恢复该轮网络/BN/Adam，
   continue/reinitialize 均恢复该轮训练 NumPy 与单设备 CPU/CUDA torch RNG。
   后者仍按来源支持的步骤新建网络/Adam。内部终点恢复不是公共训练续跑 API。

辅助验证通过 torch RNG fork 隔离，模式/BN 在预测结束或异常时恢复；自定义模块若
在 eval 使用其它外部随机源，不在此单设备 RNG 契约内，也不声称 DataParallel/分布式源复现。
fit 的外部回调应只记录，不通过改变模型/RNG/计划干预内部终点状态。

## 3. 结果、成本与存储不能混用

| 字段 | 含义 |
|---|---|
| `selected_warmup_epoch_` | 内部选择的终点 t（1-based，至少2） |
| `executed_warmup_epochs_` | 实际执行的完整 horizon，不因选早终点减少 |
| `observed_prediction_trajectory_` | 全部 U×horizon 概率 |
| `prediction_trajectory_` | 趋势消费的 U×t 前缀；fixed/选中最后一轮时与全矩阵同对象 |
| `lzo_mixup_indices_` / `lzo_mixup_weights_` | 训练行身份与 λ，不能解释成外部验证样本 |
| `lzo_losses_` / `lzo_selection_spec_` | 逐轮风险及可读、可审计规则 |
| `discarded_warmup_optimizer_steps_` | 终点后未用于最终网络的已执行更新，仍是已付成本 |
| `lzo_evaluated_rows_` / `lzo_forward_batches_` | 额外辅助验证 forwards，不能因为没有 backward 就当免费 |

全局 optimizer_steps、stage_optimizer_steps、history/callback 仍含完整 horizon 和 PN；
checkpoint_epoch_count 仍为 horizon+max_epochs。选中早终点的 optimizer 局部 step 会回到
该轮，但**全局计数不倒退**。终点后 warmup 快照仍是工程记录，不据此默认正式 PA/OA eligible。
正式阶段资格与外部选模角色需新版 recipe 决定，旧 matrix/manifest 不重新贴标签。

continue 只保留一份最佳 CPU 网络/Adam 状态（含 state_dict 版本元数据），不断覆盖最佳槽；
reinitialize 不留最佳网络/Adam，只留该轮随机流。恢复后释放最佳槽，权重快照仍仅推断回放。
RAM 还须计全概率矩阵、混合计划、CPU 数据及分批临时量；prefix 在内存是 view，但
完整可信 pickle 的两个数组可能分别序列化。旧二维/旧 CNN 探针不可直接外推新 recipe 配额。

## 4. 同步与验证

能力仍为 MLP/注入 CNN、OS-compatible，不增加 TS 校准或额外 clean-support 预算。
台账、方法卡、API、公共 pipeline 和五方法扩展草稿同步；新增
`lzo_positive_loss_recipe_and_external_selection_isolation` 复核 blocker。
候选/预算/协议仍 null、admitted=false、负责人决定为空；完整作者图像/finetune/数值未完成。

`test_holistic_pu_lzo.py` 独立核同类混合与标量 CE；强制选择第2轮并与字面两轮运行对账，
MLP/CNN×continue/reinitialize 的分数、趋势、伪标签逐 bit 一致，实际成本按完整预算保留。
还测真 argmin/平局、默认路径、种子/refit/pickle/阶段快照、CPU最佳槽、验证 RNG/BN 隔离、
非有限值/错误配置拒绝。pipeline 两种初始化均记录显式 LZO 参数。CPU 合成验证不是正式数值验收。

本轮环境为 Python 3.11.12 / torch 2.13.0+cpu，CUDA 不可用；没有执行 GPU 验收。
最终相关组合回归 **160 passed / 3 skipped**（33.60s）；全量快层
`pytest tests/ -m 'not slow and not e2e'` 为 **3153 passed / 57 skipped /
36 deselected**（161 warnings，205.72s）。新增 LZO 单元覆盖23项 CPU案例、1项 CUDA跳过；
该 CPU结果不代表 GPU、完整图像配方、多 seed正式实验或远程 CI 门禁验收。
另用上一提交 `5c133fa` 的可信仓库代码核默认 fixed 路径：MLP/CNN × seeds 0/1/7 ×
continue/reinitialize，共12组最终分数、趋势与伪标签逐 bit 一致，累计更新数一致。
11项静态门禁通过；P3 准入证据、公开对照和五方法选模预算规格检查通过且
formal_admission=false。后者仅检查既有记录绑定，本轮未重下载这些方法的上游来源文件。
冻结协议、比较矩阵、既有 P1-1/P1-2 制品、uv.lock 和私有 Excel 未改。

下一步：Holistic-PU 新变体的独立资源/选模交接证据，GEN-PU/PULNS 的原生路径与来源闭合。
精确源配方、候选协议、GPU/多 seed 正式资源与公开读数仍需负责人审阅和新制品，不据本页代签。
