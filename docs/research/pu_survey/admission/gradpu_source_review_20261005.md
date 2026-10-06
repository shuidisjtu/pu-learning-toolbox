# GradPU 来源与协议复核准备（2026-10-05）

状态：**pending_method_owner_review**；不是签署、数值复现或正式准入。
机器事实见 [JSON](../data/gradpu_source_review_20261005.json)，不改冻结矩阵。

## 原始证据

AAAI [原文 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/25889/25661)，
印刷页 7300 / PDF 第 5 页，Implementation Details / SCAR Evaluation：

- 灰度数据为 4 个 300 宽隐层，CIFAR 为 CNN13；明确移除 BatchNorm。
- Adam、batch250、学习率渐降到初值 0.1 倍、三次运行；灰度100/CIFAR200 epoch。
- 从原训练集取 P、500 行 validation，剩余行为 U，不是独立采样 U 的实验流程。
- 正类为 MNIST 奇数、FashionMNIST 偶数索引、CIFAR 交通工具。
- 同页目标实现说明使用 tanh 前的输入梯度，beta 线性增长。

上述为人工来源读数，不由测试自动证明。未添加任何数值锚点；附录预处理、网格和
验证标签使用细节仍待补充材料核查。

## 接入影响

理论 U 为总体边缘分布，与上述实际剩余样本流程分开登记；台账 TS 理论口径保留，
但 TS 校准 U∪P 运行不是原实验流程直接复现。比较前须预注册这些采样/选模差异。
当前 MLP128、batch256、固定学习率不能称为论文配方。
后续 CNN 必须显式无 BatchNorm，验证原图像梯度、二阶反传、fold 隔离和回放；
不能为兼容默认 CNN13 而静默删层或关闭梯度惩罚。

本次再检索仍未确认本文作者源码，保留 not_found。检索得到的
[yunhe20/Grad-PU](https://github.com/yunhe20/Grad-PU) 属点云上采样工作，不是本论文来源。
没有复制/执行外部源码或权重；正式预算、源码授权和负责人接受继续待办。
