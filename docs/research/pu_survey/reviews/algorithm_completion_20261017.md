# 算法收尾推进（2026-10-17）

承接用户要求，先解决线性 LBE 的可解释性，再补 P3MIX、完整 Self-PU/支持集方法
实验接线与验收。工程完成、独立实验与正式准入分别登记；不改旧协议或645-run。

## 任务状态

| 项 | 本轮范围 | 状态 / 前置 |
|---|---|---|
| LBE 诊断 | EM20/200 与固定 propensity SCAR 约束对照；训练侧分解、收敛和角色对称性 | 45次真实 Spambase CPU 拟合完成，零 warning/retry；见下表 |
| P3MIX | 完整来源、epoch/pool 时序、E/C 变体与分类器 | 正确官方 PDF/API 仍受访问限制；已请求用户提供论文/补充包，不把二手实现或索引摘要当全文验收 |
| Self-PU / LaGAM | 独立 clean support 与外部 OA 隔离的显式实验接线 | 接线与真实元更新 CPU 回归通过；实际支持集/正式标签预算仍待确认 |
| PULNS 实际支持集 | 已有安全接口后的实际数据与预算 | 需明确支持集来源/预算；本轮不在旧 frozen split 上偷偷取标签 |
| CVIR 扩展 | 固定alpha路径已接；BBE/TEDn是独立可选扩展 | 本 Survey 走 OS adaptation + 显式 α_U；OS→TS 不属于本实验，正式预算/选模/数值仍需核验，不能把新增字段当校准完成 |
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

### 45次真实数据诊断结果

代码 `2a89175`；输出 `data/lbe-diagnostics-spambase-20261017`（未提交原始数组），
每个 c 各5 seed，每种方案15次拟合。OA test accuracy，单位百分比，均值±样本标准差：

| c | 原 LBE EM20 | 原 LBE EM200 | 固定 propensity SCAR 诊断参考 |
|---|---:|---:|---:|
| 0.1 | 60.87±0.59 | 60.11±0.77 | 83.84±2.39 |
| 0.3 | 60.52±0.18 | 59.93±0.34 | 90.81±0.52 |
| 0.5 | 61.17±1.68 | 59.83±0.57 | 87.80±3.72 |

45次拟合零 warning、零 retry；36条汇总均由逐run值独立重算一致，15次 EM20
与此前线性 LBE 的生成记录、PA/OA 选择及 test 结果逐项一致。30次原 LBE 都达到 EM 轮数上限，未满足既有
后验收敛阈值；内层 logistic 最大59次迭代，远小于1000上限。EM200 的末次后验
变化仍为0.00383–0.01166；单纯加轮数没有改善本批准确率，但不代表已经收敛。
30次交换分支的观测似然差均为0；EM20 训练侧类别分支/propensity 分支真值 AUC
分别平均0.4754/0.9501（仅拟合后诊断）。固定 SCAR 参考15次 optimizer 均成功。

结论：当前结果不能仅归因于线性容量；原目标存在分支交换对称性，增加 EM 预算
不足以解决这批表现。SCAR 参考改变假设、目标参数化和求解器，不能据此断言唯一原因，
不能替换原 LBE、事后换分支，也不能外推到 SAR/其他数据集。未保存这45次模型 pickle，
故交付是可重跑的诊断记录，不是正式 checkpoint 回放验收。

输入与产物 SHA-256（完整 seed/protocol/metric 仍在本机 summary）：

| 文件 | SHA-256 |
|---|---|
| plan.json | `b98dc480a16f616a9f273e62ff00fe0d2304d3bc801fbd019774cca0145314e2` |
| environment.json | `35a2413c90007483a612c9bce3cf37b2cd4b4c79b7a5dbc58acc4a793cf427ef` |
| summary.json | `5821c8fe167179210f958b9f84ce5a0dff010358a2e0bada6bf616429390ea3e` |

## Self-PU / LaGAM：独立支持集接线

新增 `pu_toolbox.experiment.clean_support_adapters` 两个 opt-in Trainer，复用
PULNS 的六角色 ID/分组、特征摘要、来源、采样规则和额外标签预算审计。
Self-PU 用独立支持集启动已有 clean-meta 模块并选内部 teacher；LaGAM 用它做元标签。
外部 OA 的 clean_val 仍独立；显式拒绝 PA、TS、SAR 和 frozen Survey。
ID 证据来自调用方稳定身份；不同 ID 的同一副本/病人仍须 group 审计，不声称自动识别。

```python
from pu_toolbox.experiment.clean_support_adapters import SelfPUCleanSupportTrainer

trainer = SelfPUCleanSupportTrainer(
    support,  # 独立 DatasetPart: view="clean", for_selection=False，两类均有
    dataset="spambase",
    source="另行准备的独立支持集来源和制品标识",
    prior_estimation_indices=[],  # 显式声明无额外先验估计数据
    original_train_count=len(bundle.train.X),
    support_removed_from_train=False,
    group_policy="not_applicable",  # ADNI 必须 grouped，并提供全部 role_groups
    preparation_seed=0,
    sampling_rule="已批准的独立支持集采样规则",
)
# ExperimentRunner(protocols=[ProtocolOA()], config={"c": .1, "trainer": trainer}, ...)
# LaGAM 同样构造 LaGAMCleanSupportTrainer；不允许隐式拿 clean_val/test 作 support。
```

回归实际运行小型 Self-PU/LaGAM，核对元更新发生、六角色重叠预先拒绝、
test 改变不影响训练/选模、manifest 预算与模型 pickle 分数一致；同时拦截实际
Self-PU 元更新核对入参确为 support，并拒绝仅 warmup 的预算。加既有 PULNS
测试共37项通过。LBE 数学/真实小型执行相关49项通过。
这不是新正式数据跑批；还需独立 support 的真实来源/样本身份/组与标签预算。
若从旧 train 划 support，须重新做 train-only 预处理和新 split 身份，不能在已有
全 train 统计上切完就声明无泄漏。本文不修改旧645-run、签署、正式准入或预算。

## P3MIX 来源留痕

正确入口仍为 [OpenReview NH29920YEmj](https://openreview.net/forum?id=NH29920YEmj)
（数字0）；[ICLR作者报告页](https://iclr.cc/virtual/2022/poster/5904)及slides可读。
10-17 正确论坛/PDF仍转浏览器验证页，公开官方API返回403；索引发现的正文hash链接
`https://openreview.net/pdf/d687929225415551919e783b0c52f61382054101.pdf` 直接访问同样受限。
索引中的 Algorithm 1 片段不是已取得完整原文；不绕过验证。
作者代码页本轮仍无可确认入口；第三方 awesome-mixup 的“code”链接实际指向404的 `x`，
不能登记作者源码。PU-Bench第三方配置引用supplement路径，但当前锁定checkout不含该归档。
在获得原始完整算法/附录或可确认作者包前，保留组件阶段及未注册状态。

## 工程验收与交接

同日后续：[独立支持集制品与 CPU 入口](../delivery/clean_support_workflow_20261017.md)
补齐 raw 重建、剩余 train 预处理、制品复验和三方法终点 OA 回放；以下计数对应
前一轮 `e84a76c` 验收，不把本轮新测试或合成训练冒充真实数据/正式批准。

- 最终本地快速测试集4056项通过、58项跳过、零失败/错误（Python3.11.12，
  CPU；`not slow and not e2e`，296.34秒）。CUDA/可选依赖及不适用接口等跳过
  不作为 GPU 证据；GitHub 多系统/版本 CI 仍以推送后实际执行为准。
- 12项本地静态门禁通过：格式、测试质量、文档链接、API覆盖、项目元数据、
  MathJax、skill同步、基线配置、注释、目录结构、层依赖、配方草案。
- 支持集与 Iris 兼容性回归160项通过、23项按原 PU-only/概率接口限制跳过；
  支持集专门测试37项全部通过。顺带移除 Iris skip 理由中过时的 LaGAM
  CASE_CONTROL 文案，不改变其需要独立支持集的跳过条件。
- sdist/wheel 构建成功；wheel 在源码目录外独立导入三个支持集 adapter。
  新模块/台账入包，用户上传的六个根目录制品及数据数组/模型/压缩包均未夹带。
- 审计失败后清除 adapter 的可训练状态，拒绝捕获预检错误后直接复用旧状态。
  旧协议、矩阵、645-run、签署和用户已有未提交改动保持不动。

下一步依赖按顺序：①取得 P3MIX 完整原始论文/作者补充包，再补端到端实现；
②确认独立 support 来源、组身份、样本量/标签预算，重新做相应 train-only
预处理与 split 收据，再跑 Self-PU/LaGAM/PULNS；③在获准资源/正式配置下做
CNN oracle 和新增方法 GPU 多 seed 验收。上述依赖未满足前，不登记正式完成。
