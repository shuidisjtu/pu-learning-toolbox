# 算法收尾推进（2026-10-17）

承接用户要求，先解决线性 LBE 的可解释性，再补 P3MIX、完整 Self-PU/支持集方法
实验接线与验收。工程完成、独立实验与正式准入分别登记；不改旧协议或645-run。

## 任务状态

| 项 | 本轮范围 | 状态 / 前置 |
|---|---|---|
| LBE 诊断 | EM20/200 与固定 propensity SCAR 约束对照；训练侧分解、收敛和角色对称性 | 脚本与数学回归已补，真实 Spambase 计划先固定再运行 |
| P3MIX | 完整来源、epoch/pool 时序、E/C 变体与分类器 | 正确官方 PDF/API 仍受访问限制；已请求用户提供论文/补充包，不把二手实现或索引摘要当全文验收 |
| Self-PU / LaGAM | 独立 clean support 与外部 OA 隔离的显式实验接线 | 延续额外监督审计，不用 clean_val/test 作训练支持集 |
| PULNS 实际支持集 | 已有安全接口后的实际数据与预算 | 需明确支持集来源/预算；本轮不在旧 frozen split 上偷偷取标签 |
| CVIR 扩展 | 固定alpha路径已接；BBE/TEDn是独立可选扩展 | OS→TS 与新方法正式规格仍需核验，不能把新增字段当校准完成 |
| 原论文配方差异 | GEN-PU/Holistic/Robust/Split等差异保留 | 先按来源登记，正式采用哪个目标/配方需负责人批准 |
| CNN oracle / GPU验收 | 正式协议、资源与多seed运行 | 10-17查询8张A6000均有占用；不抢卡，未经批准不启动正式200轮/5seed跑批 |

## LBE：诊断计划与解释边界

入口 `scripts/run_lbe_diagnostics.py`；固定已有 Spambase 五 split、SCAR c=.1/.3/.5，
共15单元，每单元三种预先登记的方案：原线性 EM20、相同实现 EM200、
固定 propensity 的 SCAR 诊断参考。每个方案各一个训练终点，外部 PA/OA 阈值
只看各自验证角色，不按 test 改配置、翻转分数、交换返回分支或挑最佳方案。

原线性两个相同 logistic 家族的观测概率为 p(x)c(x)。交换完整分支保持观测似然，
相同 C 的总 L2 正则也保持；初始化总体先验不是训练约束，无法仅凭似然判定分支语义。
这是该参数化目标的对称性，不是证明所有 LBE 变体不成立，也不是将 propensity
事后当分类器的依据。新增 train-only 后验最大变化、两内层 solver 迭代数与分解诊断；
默认算法、停止阈值、预算和预测接口不变。

SCAR 参考独立固定 `c_hat=训练观测标记率/已知总体pi`，仅优化类别 logit 的观测
Bernoulli likelihood，使用解析梯度与 L2；**不是原 LBE，也不支持一般 SAR**。
它增加 SCAR 假设/固定propensity，不能把差异只归因于容量。其 c_hat 是根据有限样本
标记率构造的诊断值，不声称等于真实生成概率；不从 train/test 真值重新估总体先验。
训练真值仅在拟合后评估分支 AUC，不能影响优化或返回哪个分支。
`_lbe_diagnostic_reference.py` 不注册为新算法，不纳入 frozen matrix。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_lbe_diagnostics.py data/splits/spambase \
  --dataset spambase --out-dir data/lbe-diagnostics-new --dry-run

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_lbe_diagnostics.py data/splits/spambase \
  --dataset spambase --out-dir data/lbe-diagnostics-new
```

实际运行要求全新目录，不覆盖源 split/旧结果；计划与输入/标记摘要先写入，逐run manifest、
训练诊断、环境/脚本/锁摘要与完整汇总独立存放。此为 CPU 独立探索，不是正式 frozen-lock。

## P3MIX 来源留痕

正确入口仍为 [OpenReview NH29920YEmj](https://openreview.net/forum?id=NH29920YEmj)
（数字0）；[ICLR作者报告页](https://iclr.cc/virtual/2022/poster/5904)及slides可读。
10-17 正确论坛/PDF仍转浏览器验证页，公开官方API返回403；索引发现的正文hash链接
`https://openreview.net/pdf/d687929225415551919e783b0c52f61382054101.pdf` 直接访问同样受限。
索引中的 Algorithm 1 片段不是已取得完整原文；不绕过验证。
作者代码页本轮仍无可确认入口；第三方 awesome-mixup 的“code”链接实际指向404的 `x`，
不能登记作者源码。PU-Bench第三方配置引用supplement路径，但当前锁定checkout不含该归档。
在获得原始完整算法/附录或可确认作者包前，保留组件阶段及未注册状态。
