# GEN-PU：像素生成与原生 CNN 工程路径复核（2026-10-07）

本项继续[独立优先级](post_pilot_priority_plan.md)中的 P3.1/P3.2 来源与图像路径准备。
已实现像素生成 / 独立 CNN 判别与 PN 分类器；不是公开数值验收、作者源码重放或正式准入。
不改旧协议、对照矩阵、probe、结果、manifest 或合作者签署。

## 1. 来源核查：生成器不是卷积网络

原文是 Hou et al., *Generative Adversarial Positive-Unlabelled Learning*, IJCAI 2018，
[正式 PDF](https://www.ijcai.org/proceedings/2018/0312.pdf)。Table 1（PDF 第 4 页、印刷页 2259）
给出 MNIST/USPS 的 Gp/Gn：100 维噪声 → 两个 256 维隐藏层 → 784/256 维 tanh 像素；
Dp/Dn/Du 也为稠密网络。不能把这些实验写成“原作者 CNN 生成器/判别器”。

[作者 Code 页](https://qibinzhao.github.io/codes/)链接的
[demo 归档](https://qibinzhao.github.io/assets/publications/IJCAI_2018_HouMing/gen_pu_demo.zip)
于本日重新下载，其 SHA256 与台账 2026-10-05 登记一致：
`a0975a3da5979febbd0504d5ee5b2638b4f9074a66e72d45198278933f54e6c0`。
只用 zipfile 读取文本，未执行作者程序、加载未知 pickle、复制来源代码或引入归档。
未确认授权；未发现 license 文件不是证明所有来源页面均无许可。

| 归档内相对位置 | 本次独立核查 | 文件 SHA256 |
|---|---|---|
| `gen_pu_demo/pun_demo_main.py:20-23,33-78,215-219` | 双稠密 tanh 像素生成器、稠密 D，真实图像展平 | `525f741bd746bd0895a560960641ef792226c30a912d589bef28bc95719ae65b` |
| `gen_pu_demo/pun_demo_main.py:249-297` | Du 假样本未按先验混合，G 为非饱和 BCE | 同上 |
| `gen_pu_demo/gpn_classifier.py:31-64,205-227` | 浅/深 MLP PN；新生成流，P/N 等权 BCE | `a1fab1808b9a206602b94c79e502d2e864df3aef0abf002ee4c63100fbf825cc` |
| `gen_pu_demo/utils/data_wrapper.py:110-125,160-168` | PUTrainSetMNIST 的 ToTensor/0.5 归一化意图为 [-1,1]；U 排除已标记 P | `609fe4ef790f8b6ba22a3e3ccd51900da407b70fab51ab46916260d031ee1cb2` |

归档为旧版 torchvision 风格，本文不证明其在当前依赖上可直接运行。
论文 §2.1/§3.2 要求 U 是总体边缘，归档排除 P；目标及采样分歧继续见
[方法卡](../method_cards/GEN-PU.md)，不因新架构而抹平。

## 2. 新工程路径与兼容边界

`GenPUClassifier(encoder=..., generator_output="identity")` 接受 NCHW，真实图像留 CPU、
只把批次送设备；不把真实图像展平接旧 MLP。Gp/Gn 仍是独立稠密像素 MLP，
输出 `C*H*W` 后显式 Unflatten 到训练图像形状；支持灰度/多通道和非正方形输入。
生成器不是卷积网络，更不是已经复现 GAN 图像质量。

Dp、Dn、Du 各 deepcopy 外部 encoder 模板并端到端训练，最终 PN 再从同一**未修改的模板**
建第四份独立编码器，不 warm-start 已训练判别器。它们共享初始化来源，不共享可变参数或 BN。
必须输出有限二维、保留 batch 的特征；生成流和 score 输出形状/有限性也 fail-loud。
CNN13/ResNet-18 仅是工具箱共享表征工程选项，不是作者 Table 1 架构。

- 默认 `identity` 无界输出，不静默将旧二维实验换成 tanh。显式 `tanh` 要求训练 X 已在 [-1,1]。
  不截断、不自动计算缩放、不用 selection/test 统计。train-only 通道标准化可能超出此域；
  此时 identity 或另外预注册的输入坐标须由负责人决定。
- D 更新只读取真实 P（Dp/Dn）和实际边缘池（Du），生成流停止梯度；G 更新冻结全部 D 参数，
  D 转 eval 以冻结 BN/dropout，保留对生成输入的梯度，异常也恢复原模式与 requires_grad。
  源全 MLP 没有这些 CNN 模式问题，该选择须显式进入 recipe。
- PN 用新生成流和既有总体先验加权 risk；不读取真实 PN 标签或新 clean support。
  G/D 在 PN 阶段保持 eval、no-grad 取样；PN 每轮恢复 train。singleton BN 用运行统计，不丢样本。
- 无 encoder 的默认二维网络/损失/随机顺序保留。新增元数据不等于完整 pickle 字节兼容承诺。

## 3. 校准、成本与保存

原生理论 TS 语义不变。`os_or_ts="ts"` 只替换 Du 的真实风险池为 U∪P，
Dp/Dn 真实池仍是 P；生成流和 PN 标签不改。组件直接 fit 默认 OS 仍是显式消融；
台账与 Survey 自动路由继续 TS，不把公共 Pipeline 的默认 OS 当正式校准实验。
`calibration_hooked=true`，无 sample_weight 支持；sigmoid 不保证概率校准或生成分布恢复。

每轮步数为 ceil(实际 Du 池大小 / batch_size)。GAN 每步执行 3D+2G，PN 每步执行 1C：
全局 `optimizer_steps_` 计全部付出，`stage_optimizer_steps_` 六项分账，PN 轮末
`history_["optimizer_steps"]` 保留累计成本。GAN 超参搜索、训练不收敛/重试亦须另定正式预算。

只有 PN 轮末产生分类器 callback，`checkpoint_epoch_count=classifier_epochs`；快照带
`synthetic_pn`、局部 epoch、全局更新数。CPU 恢复用于分类器推断，不含 optimizer/RNG 的训练恢复；
不创造 GAN 空壳分类器候选。完整可信模型 pickle 包含六网络；未知来源 pickle 不用于验证。

## 4. 验证与未完成项

本次新增测试覆盖六网络实际梯度/step、四份 CNN/BN 与模板隔离、fresh PN、G 步输入梯度与异常恢复、
全成本、PN 快照回放、OS/TS 池、像素域/形状拒绝、singleton BN、seed/clone/refit/pickle、空预测。
公共 Pipeline 覆盖 CNN13/ResNet-18 与 tanh，能力声明、UI CNN 候选、台账/草稿同步。
第一轮新增图像单元与公共 Pipeline 共 **43 passed / 1 skipped**；CUDA 不可用，因此不宣称 GPU 通过。
台账/能力/校准/草稿/UI helper/阶段快照与旧 GEN-PU 单元组合为 **112 passed / 1 skipped**。
UI 仅验证候选和 helper 接线，不声明浏览器交互已验收。当前环境为 Python 3.11.12 /
torch 2.13.0+cpu；无本轮图像 GPU 验收记录，这不是断言服务器没有 GPU。
默认二维与本轮改动前 `2cd4c3a` 的可信实现，在 seeds 0/1/7 × OS/TS 六组中，
分数、六网络全部权重、既有 history 字段和全部更新数逐 bit一致；新分项之和等于实际全局成本。
11 项静态门禁通过；既有 P3 准入证据、公开对照和五方法选模预算检查通过，
formal_admission=false，检查器的 source_files_checked 为空，不声称重验全部上游文件。
本轮单独重下载并核摘要的来源仅为本节 GEN-PU 归档。
最终快层 `pytest tests/ -m 'not slow and not e2e'` 为
**3198 passed / 59 skipped / 36 deselected**（161 warnings，222.67s）。
无快层失败；slow/e2e、不可用 CUDA 与缺失可选依赖不据此宣称通过。
uv.lock、冻结协议/对照矩阵、既有 P1-1/P1-2 制品、checkpoint writer 与五方法预算绑定源码未改。

仍待：批准 identity/tanh 与归一化、四份 CNN/head/初始化、G 步模式与 paper/demo 目标口径；
六网络预算/阶段候选资格、真实数据公开数值对照、授权和方法负责人复核。
双像素 MLP 的输出参数随 C*H*W 增长，四份 CNN 加独立优化器、PN 训练时仍保留五个 GAN 网络，
其峰值显存/CPU/disk 必须重新测量；旧二维 CPU/GPU/storage 探针不可外推到本路径。

[单独扩展草稿](data/p3_preintegration_extension_v1_draft.json)增加这些 blockers 和来源指针，
candidate protocol/budget/pool 仍 null、admitted=false、owner_decisions 为空。
工程通过不清空 P3.3/P4.1 正式阻断，不改 frozen-lock、旧矩阵与结果；Excel 未入库。

复核入口（仓库根，隔离环境 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/unit/estimators/test_gen_pu.py \
  tests/unit/estimators/test_gen_pu_cnn.py tests/integration/test_pan_pipeline_cnn.py -q
PYTHONPATH=. python scripts/prepare_p3_preintegration_extension.py
```

需要 GPU 时显式 `PU_REQUIRE_CUDA=1` 执行新 CUDA 测试；无设备必须失败，不以 skip 验收。
