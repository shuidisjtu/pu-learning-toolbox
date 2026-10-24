# protocol v3：profile / unit 接入请求与缺口校验（2026-10-23）

新增 `prepare_survey_v3_proposal.py` 与机器草案
[survey_v3_profile_unit_proposal_20261023.json](../data/survey_v3_profile_unit_proposal_20261023.json)。
该交付把 v3 待接范围、源码参数观察、数据集组合和未定项集中起来，**不是可执行的 v3
协议，也没有让13方法进入正式 survey runner**。预算、候选、阶段资格和签署不自动补齐。
canonical 仍为 v2，冻结协议/台账/比较矩阵及历史结果不改。

## 范围已核清

按[签署 RV8](../reviews/rv8_method_admission_20261021.md)的口径：

| 集合 | 内容 | 数量 |
|---|---|---:|
| 既有 v2 | uPU、nnPU、KLDCE、Dist-PU、PUSB kernel、LBE、Self-PU no-meta、PN oracle | 8 |
| P3.1 宽口径 | P3MIX、PUET、PULDA、GradPU、Robust-PU、Split-PU、PAN、RP、PULNS、GEN-PU、Holistic-PU、CVIR、LaGAM | 13 |
| 另需补的目标方法 | VPU（不计入上行13项） | 1 |
| 正式目标算法总数 | 21 PU + 1 PN oracle | 22 |
| 额外 profile 变体请求 | Self-PU full-meta，仍对应 self_pu 算法，不把方法总数加成23 | 1 |
| 已注册但不在当前22目标内 | TEDn、linear PUSB、Neural LBE；保留实现与各自证据 | 3 |

因此本草案有23个 profile 请求，覆盖目录中的8个数据集，展开后184个
“dataset × profile”请求。它们不是184个可执行单元，也不再乘 seeds/c 宣称跑批规模。
PUSB 目标沿用 v2 的 `pusb_kernel`，不把线性 PUSB 再数成一个目标方法。
RV8 的14条含 TEDn，不能误读成 P3.1 的14方法；VPU 也不能挤掉 P3MIX。

## 本批实际做了什么

- 复制绑定 v2 的8个 profile、原25条 execution unit 与 review items 作为**原样基线观察**。
  原 `self_pu` 的 `require_validation=false` 保持，full-meta 另立请求，不覆盖旧消融。
- 23份请求绑定实际 class/source、台账的视图/校准/先验/模态及已有工程交付；
  AST 读取构造默认值，只作观察，不实例化模型、训练或生成数据。
  非 literal 默认值仅记录表达式，不执行表达式；**源码默认值不是批准的预算**。
- 每份请求的 `proposed_params/budget/candidates/stage_eligibility` 都为 null；
  不拿 GPU smoke 的短预算或原生默认200轮去伪造正式配方。full-meta 的必要语义
  （require_validation / paper_population）与可用工程视图另外记录，正式视图仍待绑定。
- 列出输入形状层面的路径选项，而不是替方法选 backbone：二维特征、冻结 CNN 特征，
  或源码声明支持的端到端 CNN。文本 `native_2d` 指预先准备的数值表示，不是直接传字符串。
  image 选项要求相应形状/配方，尤其不能把2D CNN选项当作直接接受原始3D ADNI体数据。
- clean-support 三个 profile（PULNS、LaGAM、Self-PU full-meta）覆盖范围外的三个数据集
  明确阻断；当前支持政策仍只覆盖五数据集，不能自行把 support 加到 IMDB/20NG/ADNI。
  每个 ADNI 请求另外保留 subject 隔离和替代来源口径阻断。
- #122、VPU 选模、CVIR TS 先验/风险池、GradPU 无BN分组的正式绑定要求单列；
  已定工程细节不重新要求合作者审阅。PN-CNN 5 seed 暂缓原样保留，不启动训练。
- `--check` 对照当前来源重建，检查幽灵/漏项、来源漂移、重复与擅自填入的预算/签署。
  只重算一个 JSON hash 不能把“已批准”塞进本规划产物。
  该 SHA 是一致性校验，不是人类电子签名或不可信输入的来源认证。

源文件摘要用 UTF-8/universal-newline，便于 Git CRLF checkout 对账；
不改变既有真实 GPU 收据按原始字节绑定的口径。
包的既有 import bootstrap 在已安装时可能导入 optional torch；本 builder 不依赖 torch、
不构造模型或初始化设备。缺少 torch 的独立进程规划测试已覆盖，不能说它“从不尝试 import torch”。

## 命令

```bash
# 默认输出完整草案，只写 stdout
PYTHONPATH=. python scripts/prepare_survey_v3_proposal.py

# 小摘要 / 展开184个阻断请求
PYTHONPATH=. python scripts/prepare_survey_v3_proposal.py --summary
PYTHONPATH=. python scripts/prepare_survey_v3_proposal.py --units

# 只读核对仓库记录，不训练、不写文件、不修改源草案
PYTHONPATH=. python scripts/prepare_survey_v3_proposal.py \
  --check docs/research/pu_survey/data/survey_v3_profile_unit_proposal_20261023.json --summary
```

不将此文件传给 `run_survey_experiment.py --protocol`：它采用独立 planning schema，
`load_protocol` 会明确拒绝。`runnable=false` 只约束本草案请求，不停用已有 v2 运行入口。
这不是供直接填预算并改 accepted 的表单；正式值应进入后续版本化协议/recipe。
命令需在源码仓库中使用：单独 wheel 不含仓库审核文档/脚本，不能据此在仓库外重建草案；
缺来源文件会报错，不编造替代来源。sdist 包含相关仓库内容。

## 从请求到可执行 v3 仍需什么

运行前需完成：

1. **profile identity 与算法 identity 分开**：当前 `_model_identity` 和 CLI 以 method ID
   找注册 class；`self_pu_full_meta` 不应伪注册成另一个算法，也不能冒用旧 no-meta 单元。
2. **预算、候选、阶段资格登记**：新方法的多阶段/episode/CPU树/OOF预算不能全部套
   `max_epochs`；接相应成本校验、组件快照和版本化 recipe，不把默认值当冻结决策。
3. **五角色正式绑定**：support_manifest、剩余 train、表示/预处理、额外标签预算和
   `PA+clean-support` 与纯 PA 的区分必须由正式入口核对；不能只加 JSON 名字。
4. **图像/先验正式接线**：允许共享/私有无BN的版本化规格；CVIR 每视图先验/风险池等
   与实际 trainer 对账；不绕过当前 unbound adapter 的正式协议拒绝守卫。
5. **registry / comparison 新版本**：当前 registry 的 base/selection refs 固定指向v2；
   新单元还要有相配的比较矩阵覆盖，不能只换文件名或挪用旧锚点映射。

资源峰值/D25/smoke 和预注册数值交叉验收是后续**实际验收记录**，不是新增一个必须
先有人批准 GPU 时间的门。CPU/GPU 时间不限的政策保持，不制造“先有正式结果才能训练”的循环前置。
预算与方法学接受仍须保留，未替 HENG958/shuidisjtu 签署 v2 pending 项或关闭 #122/#126。

本批测试覆盖精确22/13范围、23变体计数、184组合完整性、保留旧 no-meta/待审项、
support/ADNI阻断、预算/参数/签署篡改、非 literal不执行、缺源码/类、CRLF、
CLI只读/互斥输出/坏文件与可执行协议拒绝。不作为正式数据实验或数值复现证据。

最终专项23项通过、0跳过；`/tmp/survey-v3-proposal-release-target-20261023.xml`。
完整快层 **5,604 passed / 59 skipped / 0 failures/errors**，共5,663项，366.721秒；
`/tmp/survey-v3-proposal-full-fast-20261023.xml`。既有环境/成本/收敛告警保留，
CUDA条件跳过不是 GPU 通过。本批没有训练、下载数据或运行 GPU 实验。
格式（排除用户原有 benchmark 修改）、文档/API、元数据/数学渲染、skills、测试/注释质量、
目录/层依赖、baseline、候选来源契约、旧 draft registry及新草案检查通过。
原有 `benchmarks/deep_pu/official_data.py` 修改与上传不编辑、不提交。
离线 wheel/sdist 构建及内容检查通过；独立 wheel 安装后的模块可导入，缺少仓库证据时明确拒绝
重建草案，未把这个 import smoke 宣称为仓库外的完整规划或数值验证。
CI新增只读来源检查；未声称远程多平台检查已运行或通过。
