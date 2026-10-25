# P4.2a 前置核对：两篇基准论文到底报了什么（2026-10-25）

P4.2a 的前置不是「仓库里有没有这些方法的代码」，而是**论文表格有没有报这些方法的数字**——
代码里有实现不等于论文报了结果。本次读了两个 PDF 全文（Chen 等 35 页、Wang 等 27 页），
逐表核对。源文件：`tmp/Chen 等 - 2026 - PU-BENCH ....pdf`、`tmp/Wang 等 - 2026 - Accessible ....pdf`
（本地）。

## 结论

| 方法 | Chen et al.（PU-Bench） | Wang et al. |
|---|---|---|
| `pulda` | **有**（Table 1，8 数据集） | **没有**（17 方法列表里无） |
| `robust_pu` | **有** | **有**（4 数据集，按 PA/PAUC/OA 三准则分别报） |
| `holistic_pu` | **有** | **有**（同上） |
| `lagam` | **有**（LaGAM-PU） | **没有** |

→ **P4.2a 的前置成立**：四个方法的数字都能在已引用的基准里找到，不需要新引入论文。
但**能用在哪一类单元**受选模协议限制（见下）。

## Chen et al.（PU-Bench）Table 1

设定（正文 §3、§4 与附录 D.2.1）：8 数据集（IMDb / 20News / MNIST / F-MNIST / CIFAR-10 / ADNI /
Connect-4 / Spambase）× 18 方法，**SCAR、固定 `c=0.1`**，报 accuracy（mean±std，
对 5 次运行），另报 precision/recall/macro-F1/AUC。

我们实际跑的 3 个数据集对应的四方法数值（accuracy %）：

| 方法 | IMDb | CIFAR-10 | Spambase |
|---|---:|---:|---:|
| PULDA [13] | 75.30 ± 1.47 | 87.02∗ ± 0.68 | 84.80 ± 2.30 |
| Robust-PU [39] | 77.37 ± 4.12 | 84.30 ± 3.19 | 81.66 ± 3.62 |
| Holistic-PU [32] | 64.40 ± 4.41 | 57.08 ± 3.41 | 56.59 ± 3.93 |
| LaGAM-PU [21] | 76.81 ± 4.71 | 86.22 ± 3.60 | 86.77∗ ± 4.15 |

（∗ = 论文标注的 paired t-test 显著优于基线；完整 8 数据集都在 Table 1，可陆续补。）

**选模是这一节的关键**：正文两处写明「a checkpoint is written whenever **validation-set
macro-F1** hits a new best」（§3）与「We select the checkpoint with the best **validation
macro-F1** and report … on the held-out test set」（§D.2.1）；该验证集是**带真标签**的
（Table B.2 把 `Validation` 与 PN 训练/测试统计并列，1%，Nv = 37/250/500），
Wang et al. 附录 A 更直接指出 Chen 等「assumed that the validation set had **true labels**,
which is unrealistic」。

→ Chen 的数字是**真标签选模**的产物，与我们 **OA**（clean_val 真标签）同类，
与 **PA**（PU 视图 proxy accuracy）不同类。

## Wang et al.

设定（§5、附录 C.1、Table 5）：17 方法（含 Robust-PU、Holistic-PU，**不含 PULDA/LaGAM**）×
4 数据集（CIFAR-10 20,000 例、ImageNette 6,000、USPS 4,000、Letter 13,000），
**π=30%**，图像用 ResNet-34、表格用宽度 500 的 MLP；验证集从正/未标记数据各切一部分
（正验证 + 未标记验证，符合 OS 定义），**对 PA / PAUC / OA 三种准则分别选模并分别报数**，
每个数字是多次数据划分的 mean±std。CIFAR-10 分两个正类子集：Case 1 `{0,1,2,8,9}`、
Case 2 `{2,3,5,7,9}`。标注信度（`c`）正文未给具体值，只给了 π 的取值网格——要用它的数字必须回查。

CIFAR-10 上两方法在三种准则下的 test ACC（Case 2，Table 2；Case 1 在 Table 1）：

| 方法 | PA | PAUC | OA |
|---|---:|---:|---:|
| Robust-PU（Case 2） | 78.94 ± 0.79 | 78.43 ± 0.61 | 79.60 ± 0.81 |
| Holistic-PU（Case 2） | 55.60 ± 0.16 | 56.04 ± 4.93 | 71.18 ± 1.20 |
| Robust-PU（Case 1） | 85.57 ± 0.18 | 85.61 ± 0.55 | 85.91 ± 0.35 |
| Holistic-PU（Case 1） | 50.20 ± 0.10 | 50.00 ± 0.00 | 81.81 ± 0.49 |

## 对比较矩阵的影响

1. **Chen 的四个方法锚点挂 OA 单元**（与现有 6 个方法的处理一致），`magnitude_and_trend`：
   准则 macro-F1 vs 我们的 accuracy、验证集 1% vs 10–20%、backbone 不同，且只有 `c=0.1`。
2. **Wang 的 PA 数字不升格为单元级锚点**：它的 CIFAR-10 是 20k 子集、π=0.3、正类子集与我们
   不同（我们用 `{0,1,8,9}`、π=0.4、45k）。v4 已把这类数字用在 `row_*_cifar10` 行级映射
   （`magnitude_and_trend`），位置正确，保持不变。
3. **54 条 PA 单元没有锚点**：两篇里都没有「在我们这 3 个数据集上、用 proxy accuracy 选模」的数字。
   裁定见 [pa_adjudication_20261025](pa_adjudication_20261025.md)。
4. **v4 的 18 个 Chen 锚点 `selection_protocol: proxy_acc` 是元数据错误**，v5 里按
   「真标签验证 + macro-F1」修正。

## 边界

- 只读了论文 PDF，没有读两篇的代码仓库；`c` 的具体值（Wang 等）与逐 seed 细节未核。
- 论文表格的 `∗` 显著性标记只说明「与基线相比」，不构成与我们实现的对照。
- 两篇都是 ICLR 2026；Chen 是 concurrent work，Wang 在附录 A 中对它的评价属于后者立场。
