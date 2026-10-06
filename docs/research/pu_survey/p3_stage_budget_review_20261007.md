# 方法特定阶段、选模与预算准备（2026-10-07）

状态：**draft / pending_method_owner_review**。承接
[后续优先级](post_pilot_priority_plan.md) 第2项及
[公开对照准备](p3_public_comparison_20261005.md)，不是执行路线编号 P1.3。
本批补实际 checkpoint 留痕与五方法机器规格；不决定新方法的正式候选池、预算或阶段资格。
原始 pilot 制品、负责人签署和公开数值验收的阻断全部保留。

## 1. 修补的实际缺口

原推断快照只有全局 epoch、component、权重摘要及验证指标。对多阶段方法，
单凭 `model` / 第17个快照无法辨认是 warm-up、splitter 或第几轮 student，
更不能将 episode 数当作 optimizer 更新数。本批让声明 `checkpoint_stages` 的新方法
在每次 callback 前明确暴露阶段；checkpoint 保存当时的副本，选模、恢复和回收均保留它。

新阶段引用 schema 为 `1.1`，增加 `training_context`：

| 字段 | 含义与校验 |
|---|---|
| `stage` | estimator 明确声明的阶段标识；writer 拒绝未声明值 |
| `stage_epoch` | **1-based 阶段内**位置；Robust-PU 的 self_paced 按 episode 编号 |
| `round_index` | Split-PU student 的 **1-based round**，每轮 stage_epoch 从1开始；其它阶段为 null |
| `optimizer_steps` | fit 内累计实际更新数，不随阶段/round 清零；refit 重置；快照间不得下降 |

已有全局 `epoch_position`、展示标签 `epoch_label` 及 PA/OA 选择语义不变。
没有阶段声明的历史 pilot 方法仍生成旧 `1.0` 引用，不增加字段，不回填旧 manifest。
读取旧引用仍兼容；新引用缺字段、bool/负数计数或 schema/context 不一致会被拒绝。
上下文独立复制，修改返回字典、后续 fit 或回收未选权重，不会改写已有记录。

这是**来源留痕**，不是准入控制或训练恢复：weights_only 文件只保存推断网络，
不包含 optimizer、RNG 或 GAN/teacher 全部训练状态。权重摘要保护权重字节，
不单独证明 JSON 阶段名称真实或已获审批；外部 manifest 仍须身份校验与独立复核。
内置技术 selector 仍遍历全部实际记录快照，**没有默默改成只选最后阶段**。

## 2. 五方法规格与待决问题

[机器草案](data/p3_selection_budget_spec_20261007_draft.json) 记录当前 constructor 默认值、
fit 角色接口、阶段、表征/校准事实、实现与 writer 字节摘要、默认快照计数及实际更新公式。
读 P1-1/P1-2 证据并核冻结文件，不把旧结果重绑新版本。

| 方法 | 当前记录阶段 / 默认快照数 | 不能混淆的预算 | 负责人需要决定 |
|---|---|---|---|
| PULDA | warmup、pu_mixup / 60+60=120 | 两阶段 epoch；更新数还乘 U 批次数；cosine horizon 的已登记差异不改 | warm-up 能否选中、阶段预算、图像或冻结特征口径 |
| PUET | 无 epoch callback；单次 fit / 不计神经快照 | 100棵 CPU 树不是100 epoch；森林序列化需独立 profile | 树预算族、GPU 条款豁免、原像素或冻结特征 |
| GradPU | pu / 200 | 每 epoch 的 P/U 批次数与插值计算；更新计数不等于梯度计算成本 | no-BN CNN/调度、validation 标签口径、PA/OA 资格 |
| Robust-PU | pretrain、self_paced / 10+20=30 | inner_epochs 增加更新但不增加快照；两阶段不能称同一 epoch 预算 | pretrain 能否选中、episode 预算、来源 clean-val 使用与 prior 口径 |
| Split-PU | teacher、splitter、student rounds / 10+10+2×10=40上界 | splitter 可早停；每轮学生更新受 easy/hard 大小影响 | teacher/splitter/哪轮学生能选中、实际早停规则、论文/代码周期矛盾 |

数据行数指 **原始** n_P/n_U；TS 在风险角色中作并集，不因并集增加物理批次数。
Split-PU 的 student 更新公式用实际每轮 easy/hard 组大小；未知实际划分时不编造精确更新数。
预算公式只说明当前实现，不量化不同方法的公平训练成本。

所有 `formal_decisions` 的候选、预算、存储 bytes、PA/OA eligible stages、学生 rounds、
threshold policy 和表征引用仍为 null；`admitted=false`，GPU 豁免未批准，签名为空。
当前默认值是代码事实，**不是**为新协议选择的值。
神经峰值预检使用声明快照上界；回收只减少最终留存，不消除训练时全轨迹峰值。
历史缩短合成 probe 体积不升级为正式资源上界，也不重算覆盖其旧源摘要。

GenPU 与 Holistic-PU 同时补技术阶段留痕：前者只记录 `synthetic_pn`，GAN 全部更新
仍累计但没有伪造 GAN 分类器快照；后者记录 warmup / pseudo_pn。
本批五方法规格不替这两项审批 recipe，二者仍使用原独立扩展交接。

## 3. 复核和复现

```bash
PYTHONPATH=. python scripts/prepare_p3_selection_budget_spec.py
PYTHONPATH=. python scripts/prepare_p3_selection_budget_spec.py \
  --check docs/research/pu_survey/data/p3_selection_budget_spec_20261007_draft.json
pytest -q tests/unit/experiment/test_checkpoint_training_context.py \
  tests/unit/scripts/test_prepare_p3_selection_budget_spec.py \
  tests/contract/test_p3_candidate_admission.py
```

生成器只读 AST/JSON，不导入 torch、不训练、不读私人 Excel、不消费外部权重，不写文件。
检查入库草稿必须与当前事实**类型一致**：布尔不冒充整数，浮点默认值也不偷换为整数。
阶段、来源摘要、冻结摘要、台账指针或 null 决策有改动都 fail-closed；
输出 `formal_admission=false`。草稿的全部当前字段由生成器生成，人工审批另立版本，
不能靠修改此草稿使该检查器报告已正式准入。

可证伪复核要求：若阶段顺序、round/local epoch、optimizer 实际 step 数、
实现默认值或来源/预算描述不符，应提供代码位置和 callback/step 插桩证据。
不确认的项目继续 pending；不能用生成器通过代替阶段资格审批。

技术合成数据不参与公开数值对照。

## 4. 本批验证证据

- 阶段/恢复/回收、只读规格生成器与原八方法准入契约专项 **64 passed**。
  新增61个参数展开用例，含六个阶段方法的 CPU 可重复 fit、零第一阶段、
  Split-PU 早停/round 位置、实际 Adam step 插桩、非法计数/审批篡改拒绝。
- 全量快速回归 `pytest tests/ -m 'not slow and not e2e' -q --tb=short`：
  **3100 passed / 55 skipped / 36 deselected / 161 warnings**，191.35秒。
  跳过可选依赖/无 CUDA 用例，slow/e2e 未执行；不是完整训练或正式 GPU 验收。
- Python3.11.12 / PyTorch2.13.0+cpu 隔离环境，`CUDA_VISIBLE_DEVICES=''`；
  没有占用 GPU。CPU Torch 制品不是 Linux uv.lock 的 CUDA 制品，不称 frozen-lock GPU 通过。
- 11项静态门禁通过：格式、测试质量、文档链接、API、元数据、数学、Skill同步、
  baseline、注释、目录、层间依赖；注释检查的 advisory 保留，不是失败。
- 只读再生成/检查及禁用 torch 导入的 CLI 检查通过；冻结矩阵、公开对照 v1/v2/v3、
  uv.lock、P1-1/P1-2 证据字节保持原样。私人 Excel 不入库。

`pu-workflow` 的证据边界用于保持工程回归、来源核查、正式结果验收三者分离；
本批没有结果目录输入，不运行其 benchmark 制品验收扩展。
没有新远程 CI 结论或正式审批，不借用之前的远程绿色状态。

## 5. 下一步

1. 负责人审阅各方法 PA/OA 阶段资格、参数/预算族、表示与标签预算，留下明确决定。
2. 批准后才物化新版本 protocol/registry，绑定 runner 与 manifest；不向历史 v1 追加名字。
3. 不依赖实验数据的来源缺口和算法组件仍可继续；P3MIX 需先取得完整原始来源，
   原图增强/SimSiam/调度的缺口不能用已通过的留痕测试代替。
4. 正式资源与公开数值对账仍需批准配置、真实结果和原始制品；不在本批自动跑批或代签。
