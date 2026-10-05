# 合作者五项建议：补全与边界（2026-10-04）

本页不代填本人签署，不修改冻结协议或历史 645 次结果。沿用 `pu-workflow`
的 benchmark audit 原则：区分表格复算、原始制品审计和数值复现，不把不同协议的差值当方法效果。

## 1. 核对结论

| 建议 | 已完成/本次补全 | 尚未完成 |
|---|---|---|
| 其他数据集 | 新增公开原始数据收集工具，覆盖 MNIST、Fashion-MNIST、20 Newsgroups、Connect-4；固定已有发布校验和，记录实际 SHA256，不自动解压归档 | 原始数据不等于四角色 split；三个试点数据集以外尚需预处理/划分与独立验收；ADNI 需持有人获批提供 |
| CNN PN oracle | 新增 `PilotOracleCNN`，端到端训练编码器和二分类 head，真实标签 BCE、固定 epoch、OA 选模、checkpoint 恢复/回收已通过合成图像 runner 测试 | 尚未进入冻结矩阵、未做真实 CIFAR 五 seed GPU 实验；不能把实现完成写成正式 baseline 数值完成 |
| Self-PU / Dist-PU 校准效果 | 复算既有同 seed OS/TS 观察差值，见下表；厘清 Excel 不能与历史单 seed 直接配对 | 收齐匹配配置与 split 身份的 OS/TS 多 seed 原始 manifest，再作配对分析 |
| Pilot 与文献复核 | 645-run Excel 已对账，183 汇总已独立复算；文献读数和协议差异见独立复核包；补充具体差值及解释边界 | 原始 manifest/门禁/权重重放及未决来源/合作者签署仍待交接；暂无可直接作数值一致性裁决的映射 |
| 实验逻辑与 checkpoint | 现有回收实现已复测：PA/OA 选中权重保留、指标不变、删除失败显式报错；新增 CNN OA-only 回收测试 | 回收默认关闭、只降低最终占用，不降低训练峰值；在线最优权重留存仍需设计/等价性验证 |

## 2. 公共数据收集契约

入口：`scripts/collect_survey_public_data.py`。输出到本地 `data/raw`，不入库原始数据：

```bash
python scripts/collect_survey_public_data.py
```

每个数据集生成本地 `provenance.json`，含来源、文件字节数、SHA256、发布方校验和及验证状态。
本机已实际下载全部四套公开原始数据，共 10 文件、57,337,501 字节（约 54.68 MiB），
并核验图像 IDX 头、20 Newsgroups 归档成员及 Connect-4 ZIP CRC，未展开归档。
MNIST/Fashion 各 60,000 train + 10,000 test、28×28；新闻归档 18,846 个文本文件。
入库收件快照见[公开数据收件](data/public_data_receipt_20261004.json)，原始文件在本地忽略目录。
MNIST/Fashion-MNIST 校验和取自 torchvision 的官方 dataset resources；20 Newsgroups 使用
scikit-learn 发布的归档 URL/SHA256。UCI Connect-4 未找到发布方固定校验和，记录为
`publisher_checksum_verified=false`：本地 SHA256 只保证再次使用时字节一致，不冒充发布方签名。
既有文件不匹配时报错且保留；下载临时文件仅在校验后发布，每个归档上限 256 MiB。

来源与许可：

- [Fashion-MNIST 发布方](https://github.com/zalandoresearch/fashion-mnist)及其 [MIT 许可](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE)。
- [MNIST 的 torchvision 下载定义](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)。镜像可下载不等于本项目已完成原数据授权核查，不据此声明 MNIST 为 MIT。
- [20 Newsgroups 的 scikit-learn 下载定义](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_twenty_newsgroups.py)。软件包 BSD 许可不自动等于原始新闻文本的授权；先保存归档，不宣称完成许可验收。
- [UCI Connect-4](https://archive.ics.uci.edu/dataset/26/connect%2B4)：67,557 条、42 个三值分类特征，CC BY 4.0；win 对 loss/draw 映射已在目录中定义。
- [ADNI 数据访问](https://adni.loni.usc.edu/data-samples/adni-data/)需申请/数据使用协议；工具不申请、不接受协议、不下载。

后续 split 注意：20 Newsgroups 原始 20 类与目录 7 类不能直接使用相同整数 id；须先核实预注册的
20→7 映射。Connect-4 的 b/o/x 是分类变量，不能未经登记就当作有序数值。
MNIST/Fashion 为单通道 28×28，而冻结 pilot 图像规格为三通道 32×32；需另立新执行规格。
本次没有重建或覆盖已验收的 15 个 pilot split。

## 3. CNN oracle 的实现与正式纳入是两件事

`pu_toolbox/experiment/survey_execution.py` 中 `PilotOracleCNN` 与 assembly 分支已实现：

- 必须显式提供 encoder；复制初始权重，解冻并端到端训练，不偷换为展平 MLP或 frozen adapter。
- 初始化种子、Adam/BCE、固定预算沿用 PN profile；不接受 PU class prior 或 fit 验证标签。
- 图像/标签保留在 CPU，仅逐 batch 转移设备，预测也分批；预测临时 eval 后恢复训练模式，避免 epoch 回调污染后续 BatchNorm/dropout 状态。
- 复用 `CleanLabelGenerator`、`SupervisedTrainer`、`ProtocolOA`，每 epoch 持久化编码器和 head；test 不用于选 epoch。
- 保持旧 `resolve_unit` 拒绝 CIFAR/native_cnn/oracle 行，不增加旧矩阵的实验次数、不回写历史比较映射。

正式入口仍是新协议登记后的 survey CLI，当前历史协议下的 native CNN oracle 仍 fail-closed。
下一步：新协议修订登记该行、OA-only、每 dataset/seed 一次、CNN checkpoint 大小预算及对照资格，
然后作真实 CIFAR 单 seed GPU smoke 和五 seed OA baseline；不需要另造会泄漏标签的深度 trainer。

## 4. 目前真正可配对的校准证据

来源为 [P2.0e 交付](p2_0e_delivery.md) §6/§7 的 Spambase split_0、c=0.1、seed=0，
同参数、同总体先验的两个视图。原始路径在 F:/Temp/lab；本机尚无这些旧原始制品。
以下仅复算已记录数字，差值单位为百分点（TS−OS），不是重新训练或原始制品重放：

| 方法/选模 | Accuracy 差 | AUC 差 |
|---|---:|---:|
| Dist-PU PA | 0.000000 | -0.009874 |
| Dist-PU OA | 0.000000 | +0.039002 |
| Self-PU PA | +1.085776 | -0.079485 |
| Self-PU OA | -1.085776 | +0.111081 |

已有证据支持「两个视图确实改变训练行为」，不支持「校准全面提升」；Self-PU 的 PA/OA 差方向相反。
两条 Self-PU 都是无 clean meta-validation 的消融，不是完整作者方法。不可把旧 OS 单 seed 与
新 Excel 的 TS 五 seed 均值相减作为校准效果。

完成正式配对需要双方提供：source commit、protocol digest、split/四角色摘要、seed、c、采样机制及
生成视图摘要、训练路径/表征身份、完整参数/候选预算、PA/OA 准则、selected epoch/checkpoint。
只让 OS/TS 标签预算适配变化，其余受控一致；分别报告每 seed Δ、均值/样本标准差及配对不确定性。

## 5. 文献交叉验证：实际差距与可解释性

Excel 的 CIFAR/SCAR/c=0.1/OA 五 seed 行如下。与公开数值的差值只是诊断，并非预注册的通过/失败：

| 本项目行 | Accuracy 均值±样本 SD (%) | 文献锚点 (%) | 诊断差值 (pp) |
|---|---:|---:|---:|
| Dist-PU / frozen adapter | 68.824±5.417 | Dist-PU Table 2：91.88±0.52 | -23.056 |
| Self-PU / frozen adapter / ablation | 61.582±3.169 | Self-PU Table 7：89.68±0.22 | -28.098 |
| nnPU / native CNN | 67.092±4.103 | Self-PU Table 7 第三方复现：88.60±0.40 | -21.508 |

出处：[Dist-PU](https://arxiv.org/abs/2212.02801) Table 2，第 6 页；
[Self-PU](https://proceedings.mlr.press/v119/chen20b/chen20b.pdf) Table 7，第 9 页。
nnPU 数值产出方是 Self-PU 作者第三方复现，不能写成 Kiryo 原论文锚点。

这些较大差距值得优先检查，但不能归因于校准：前两条是冻结随机 ResNet 特征，而论文是端到端 CNN；
Self-PU 为消融；nnPU 的 backbone、归一化、标记预算、验证选模等也不同。
固定 π 不等于固定标记频率 c；±的含义必须逐来源核实。对照矩阵中直接数值裁决资格当前为零。
继续检查 AUC<50 的分数方向/selected checkpoint/训练曲线，不能自动翻转 AUC 修结果。
完整复核事实与 PUSB 离散度、nnPU 补充材料、PUBench 论文/代码矛盾等未决项见
[独立复核](pilot_independent_review_20261004.md)与 [Excel 验收](p21_workbook_review_20261004.md)。

## 6. checkpoint 与实验逻辑后续

现有 `--reclaim-unselected-checkpoints` 在单次实验和 pilot CLI 均为 opt-in，默认保留全部。
应在新批次明确开启，并保存选中 epoch、权重 digest、回收前路径/标记；不要删除全部 checkpoint。
OA oracle 只留 OA 选中权重，PU SCAR 留 PA/OA 选中权重的并集。
自动回收在选模和 test 评价后发生；整个候选池训练时仍可能持有全部 epoch 权重，峰值磁盘预算不可按最终两份估算。
若要在线只留最优 checkpoint，先登记 tie-break、阈值候选、多教师组件、候选池独立选择和恢复测试，
验证与完整轨迹的结果等价后再启用，不能现在直接删除中间权重。

本次定向测试：58 passed、2 skipped；其中新增 CNN 4 条、下载安全 4 条。
10 条 warning 是合成估计器未声明磁盘估算，不是 GPU/真实数据资源验收。
增量 Ruff 通过；本次不执行 git commit/push。

## 7. 后续优先级

1. 收齐 AutoDL 原始 manifest/门禁报告及旧 OS 制品，完成 645-run 制品重放、校准配对与异常行定位。
2. 登记 CNN oracle 新协议/存储预算，真实 CIFAR smoke 后跑五 seed OA baseline。
3. 将新公开数据转成四角色 split，锁定 20→7 类映射、许可与图像/文本规格；ADNI 由获批合作者交付。
4. 固化新批次回收开关和峰值预算，再讨论流式 checkpoint 保存。
5. 上述闭环后恢复缺失算法正式准入与 P4 主榜工作；不抢跑完整主榜。
