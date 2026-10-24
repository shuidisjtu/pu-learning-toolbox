# GEN-PU #126 配方冻结（2026-10-24）

裁决人：项目负责人；日期 2026-10-24。**认定 [2026-10-23 单候选工程 recipe](../delivery/genpu_recipe_20261023.md)
给出的四项工程规格即正式配方**，不再要求合作者重复确认工程细节，也不把"数值未对齐论文"
当作阻断。本记录只冻结配方与登记差异，不改代码、不改执行单元、不宣称正式准入或论文复现。

## 冻结的四项

| # | blocker | 冻结值 |
|---|---|---|
| 1 | `prior_weighted_Du_and_synthetic_PN_spec` | 保留论文 Eq. (3)–(11) 的带先验权重 minimax / anti-GAN；`Du` 假样本按总体 `π/(1−π)` 混合；最终 PN 为同先验加权 softplus risk；`ts` 只把 `Du` 的真实池换成全 train 并集，生成流与 PN 标签不变 |
| 2 | `six_network_GAN_plus_PN_selection_and_storage_spec` | 阶段预算 `genpu` = 100 份 `synthetic_pn` 快照，逐组件 8–64 MiB；单候选不搜索；只有已完成的 `synthetic_pn` 轮进入 PA/OA，GAN 不产生空壳分类器候选；每步 3D+2G、PN 每步 1C；只保留选中 PN 权重与全部元数据 |
| 3 | `dense_pixel_output_domain_and_four_CNN_encoder_recipe` | 生成器两层 256、latent 100、**identity** 输出（真实像素 `[0,1]`，归一化在各 encoder 内做一次）；`Dp`/`Dn` 线性头；`Du`/PN 两层 256；四份独立 seeded ResNet-18 小图 stem、无增强、PN 从新模板初始化；三个 λ 固定为 1，P/U/G batch 同尺寸显式传入 |
| 4 | `generator_step_discriminator_mode_and_native_image_resource_spec` | G 步冻结 D 参数、BN/dropout 转 eval，但保留对生成输入的梯度，异常恢复原模式；资源 profile：A6000 真实 CIFAR 24 行子集 batch8 峰值 832 MiB allocated、batch256 8.06 GiB；快照 payload 峰值 136,626,114 B、选后 45,542,038 B；容量预检 `64 MiB × E × candidates × 2` |

证据：[机器收据](../data/genpu_recipe_validation_20261023.json) 的 `batch8_profile` / `batch256_profile`；
来源与架构真相仍以 [2026-10-07 复核](../genpu_source_cnn_review_20261007.md) 为准（生成器是稠密
像素 MLP、不是 CNN；归档 SHA-256 已锁定）。

## 已登记的差异（不是论文复现）

- 论文 Table 1 的 MNIST/USPS 是**全稠密**判别器；本配方四份 ResNet-18 是 benchmark 适配，
  不冒充作者架构。
- 输出域取 `identity`，不照搬作者 demo 的 `tanh` / `[-1,1]` 坐标。
- 不采用 demo 的非饱和 BCE 与等权 PN；λ 取 1 是工程固定值，未做论文 §4 的调参。
- 论文 CelebA 用 improved WGAN，不在本配方范围。
- 有限步 GAN 训练不证明分布恢复或真实数据精度，不做数值 parity 声明。

## 不随本裁决关闭的事项

- **全量数据/全预算资源实测**：现有峰值来自 24 行工程子集与短预算，不等于全 CIFAR+
  100 份快照的上界；已列入 GPU 任务（GEN-PU/PULDA 原生 CIFAR 单 seed）。
- **中央 recipe registry 打包**：registry schema v1 仍把 `base_ref`/`budget_binding_ref`
  限定在 `survey_protocol_v2`，现在无法登记 v3 配方；打包随 registry 扩到 v3 一起做
  （执行计划 §1.2 的 recipe registry 缺口）。

## 落点

- `survey_protocol_v3.json` 中 genpu 的两处 `configuration_review` 由"待 #126 裁决"改为已冻结。
- [GEN-PU 方法卡](../../method_cards/GEN-PU.md) 加冻结摘要与指针。
- 执行计划追加决策 D34；`#126` 关单。
- `docs/research/pu_survey/data/p3_preintegration_extension_v1_draft.json` 是按 `survey-v1.2`
  生成的日期化草稿，其 genpu 四个 `specific_blocker` 保留为当时快照、不再作为当前阻断；
  机器可读状态以本记录与 v3 为准。
