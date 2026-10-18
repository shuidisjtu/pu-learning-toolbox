# GradPU #121：调度选项、真实 batch 成本与运行声明

依据：[#121](https://github.com/shuidisjtu/pu-learning-toolbox/issues/121)，
本轮读取为 OPEN、0评论、更新于2026-10-18 03:38:54Z。
paper-only身份已由该issue明确接受，不代表配方、资源或正式准入接受。
不改冻结协议、v3矩阵、方法台账或历史结果；不关闭issue、不代签。

## 独立补齐

- 新增 `lr_schedule="constant"|"linear"|"cosine"`、`lr_final_ratio=0.1`。
  默认仍固定学习率。对E>=2，epoch e=0..E-1的进度t=e/(E-1)，
  linear系数为 `(1-t)+r*t`，cosine为 `r+(1-r)*(1+cos(πt))/2`。
  每轮训练前设置Adam实际LR，不重建optimizer或清空动量。
  首轮初值、末轮使用初值的r倍；单epoch只用初值，不假称完成衰减。
  曲线为工具箱显式选择，不是论文未给出的细节；beta逐step线性增长不变。
- `history_["learning_rate"]` 和纯JSON `training_provenance_` 记录策略/实际终值、
  batch、更新数、超参数、TS角色与理论边界。通过runner既有hook进入成功候选的
  `candidate_runs[].training_provenance`，不是只记录候选覆盖参数而遗漏实际默认值。
- 保留per-branch batch，不擅自改成论文语义未明的总batch250；记录实际最大行数。
- 补齐CNN原图梯度、二阶反传、BN拒绝、checkpoint回放、fresh-fold模板隔离、
  实际Adam LR和runner manifest的机器测试引用。移除旧来源草稿中“作者代码是
  paper-only必需前置”的过时阻断，保留not_found及其他未解决证据。

## batch与预算

设B=batch_size，每轮 `I_toolbox=max(ceil(n_P/B),ceil(n_native_U/B))`，短分支循环、
不丢尾批；总更新数为 `E*I_toolbox`。调度与诊断不增加优化更新，不宣称该I等于论文I。

| 项 | OS | TS |
|---|---|---|
| 每步物理抽取 | p+u，p/u各<=B | p+u，原始存储池不变 |
| 正例风险行 | p | p |
| U风险行 | u | u+p |
| P项与U项累计前向上界 | 2B | 3B，不含额外GP前向 |
| GP插值行（alpha>0） | max(p,u)，最多B | max(p,u+p)，最多2B |

alpha=0时GP不计算，实际插值行数记0。上界不是每步实际值，尾批实际最大值由训练记录。
均衡P/U分支与总体样本比例不同，TS简单batch并集不自动保证无偏的总体混合比例；
本轮披露执行事实，不用“角色已并集”证明总体风险已精确恢复。

## 声明与仍待办

TS替换U风险/原图像GP角色，不是论文“取P、500 validation、剩余U”的实验重放。
SCAR理论与正例标注偏差实验分开；manifest明确positive-label selection bias部分匹配，
`general_SAR_theoretical_guarantee=false`，不自动宣称SAR-native。

MLP128单隐层、200轮、分支batch256、alpha0.1/beta_max1及weight_decay/LR是工具箱适配，
可选衰减不使它变成MLP300×4、灰度100轮或数值复现。PA/OA阶段资格、正式候选池、
资源和准入继续null/false，新参数不自动进入正式搜索空间，快照/更新预算公式不变。

请合作者确认：接受固定LR还是明确曲线/终值；保留per-branch batch还是另立总batch变体；
数据集网络/epoch及alpha/beta候选池、阶段选择规则。与#119的PULDA共享规格独立，
#120的无BN/private方向不重开。未执行真实数据训练、GPU、200快照完整profile或公开
数值复现；补充材料的validation标签/网格与误差棒定义仍未解决。

## 本批工程验收

末次代码专项 **58 passed / 2 skipped / 0 failures / 0 errors**，2.658秒，
JUnit `/tmp/gradpu121-final-targeted.xml`。两项跳过为未执行的CUDA测试。
更早相关回归144项通过/2项跳过；最终全量快层另行记录，不把早期运行当末次源码验收。
实际NumPy float32学习率曾使新增记录不能直接JSON序列化，新增回归先失败、
修复后通过；该修复只将记录中的实际比例转为Python float。

三个seed、MLP/注入CNN、OS/TS共12条默认固定LR路径，与修改前自有代码的原损失
轨迹、预测和更新数逐bit一致。新增策略以实际Adam step拦截验证，不只检查日志。
来源/默认值、公开对照和预算草稿核验通过，保留所有正式批准字段为空或false。
