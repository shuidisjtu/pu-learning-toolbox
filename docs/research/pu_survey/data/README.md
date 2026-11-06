# Survey 数据来源与解释边界

数据加载器及当前类别映射以 [`datasets.py`](../../../../pu_toolbox/experiment/datasets.py)
中的 `DatasetSpec` / `DataProvenance` 为准；运行产物的 split manifest 记录实际来源、
seed、划分索引和 train-only 预处理。仓库不存储原始数据或训练结果。

## 来源与许可

| 数据槽位 | 来源与许可解释 |
|---|---|
| MNIST | [原始来源](http://yann.lecun.com/exdb/mnist/)；下载定义可参考 torchvision，镜像或软件包许可证不自动构成原始数据授权 |
| Fashion-MNIST | [发布方](https://github.com/zalandoresearch/fashion-mnist)，发布方仓库声明 MIT |
| CIFAR-10 | [发布方](https://www.cs.toronto.edu/~kriz/cifar.html)；保留原始来源，不从依赖库许可证推断数据许可 |
| IMDB | [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)，保留其研究用途条件与来源声明 |
| 20 Newsgroups | [原始来源](http://qwone.com/~jason/20Newsgroups/)；scikit-learn 的 BSD 软件许可不能代替新闻文本授权；20→7 超类映射由加载器明确执行 |
| Spambase / Connect-4 | [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase) / [UCI Connect-4](https://archive.ics.uci.edu/dataset/26/connect-4)，页面声明 CC BY 4.0，保留署名；Connect-4 的 b/o/x 按分类特征处理 |
| `adni` 槽位 | 实际为下节 Alzheimer MRI 替代数据，不能称为获批 ADNI 数据 |

原始数据收集入口 `scripts/collect_survey_public_data.py` 使用已登记 URL/摘要；已存在文件
不匹配时拒绝覆盖。MNIST/Fashion 的校验值来自 torchvision，20 Newsgroups 来自 scikit-learn；
它们是依赖项目登记的校验值。UCI Connect-4 未登记发布方摘要，本地 SHA-256 只用于字节身份。
许可字段的未知状态必须保留；下载成功或字节核验通过不等于再分发权限已确定。

## Alzheimer MRI 替代数据

本项目的 `adni` 槽位使用 Kaggle *Alzheimer's Dataset (4 class of Images)* 的个人镜像
[`Jahnabi19/AD-Dataset`](https://github.com/Jahnabi19/AD-Dataset/tree/d90729118954bbedfe481dea8a53b4f3ee5a21b2)，
固定 commit `d90729118954bbedfe481dea8a53b4f3ee5a21b2`。
这不是 ADNI 数据，也不带 ADNI 数据使用许可。镜像未声明许可证，原 Kaggle 页面当时标为 Unknown。

- 共 6400 张图像；NonDemented 3200、VeryMildDemented 2240、ModerateDemented 64、MildDemented 896。
- NonDemented 为正类，其余三类为负类，总体先验 0.5；这与 PU-Bench 的另一二值定义不同。
- 原图为灰度 JPEG，加载器缩放到 128×128；归一化统计只从实际 train 计算。
- 来源没有 subject IDs。样本索引不交叉不能证明 subject 级隔离，可能存在跨划分同受试者切片。
  需要 subject/group 证据的路径应继续拒绝，不能凭工程接口可运行消除此限制。

归档摘要、类别及身份事实见
[`adni_substitute_receipt_20261012.json`](adni_substitute_receipt_20261012.json)。
其中历史验收状态描述当时的材料状态，不自动决定新结果有效性。

## 划分、support 与保留的机器事实

四角色为 train / pu_val / clean_val / test，真实测试标签只用于最终评测。
独立 clean support 从 train 划出时移除原行，并记录全部角色索引及实际正负标签预算；
支持集和选模验证集不能重用。项目 support 规模为 Spambase 100、其余已配置支持数据集 500，
实际执行还须以 support manifest 为准。

PULNS 原论文覆盖 MNIST、CIFAR-10、Spambase；其他数据集是项目扩展。
MNIST 的偶数/奇数、CIFAR 的 vehicle/animal 映射与原文一致；Spambase 保留 spam-positive，
原文的原始编码未独立证实。论文中的 γ 是 U 内正类比例，不是标记频率 c。
论文 support 规模不能静默成为新实验默认，作者代码未被确认。

实际数据来源、文件摘要与划分身份以各次数据准备生成的 provenance 和 split manifest 为准；
早期试点的划分索引及收件记录已退役，历史可从清理前提交 `7d11f2c4` 查阅。
P3MIX 文献锚点仅保留论文来源、数值及可比性限制，不绑定早期试点划分。
其余方法来源 JSON 和 P3MIX/PULNS 文献锚点保存外部来源事实，配合
[方法卡](../../method_cards/)解释实现差异，不作为训练、报告生成或人工审批的强制依赖。
