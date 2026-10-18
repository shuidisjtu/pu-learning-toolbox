# RP：PU 特例来源与实际交叉拟合复核（2026-10-18）

P3.1 来源补证与数值边界修复。数学算法、默认值、OS/先验口径不改；仅对
`rank_pruning.py` 的 confident 正例均值增加舍入上界保护。不是作者程序重放、
论文数值复现、方法负责人签署或正式准入。

## 来源与数学口径

[机器复核包](../data/rp_source_review_20261018.json)锁定作者仓库
`cgnorthcutt/rankpruning@40ae17cc50fcfeec8d7a345a1c5426036d2bdb1a`，记录核心
算法、README、LICENSE、setup 四文件的原始与规范化文本摘要。
LICENSE 含 CRLF，两种摘要不同，均独立核对，不以换行归一化跳过原始字节核验。
MIT 声明与既有 registry 一致，不调整本项目许可；未引入或执行作者代码。
源码保留在库外审计目录，打包不包含它。

论文 [Table1 / Eq.(2)(3)(9)(10) / Algorithm1](https://auai.org/uai2017/proceedings/papers/35.pdf)
与源码保持分层。这里只核无误标 P 的 PU 特例 `rho0=0`，不是一般双向 noisy-PN。
源码泛用类的默认行为未必是 PU；不能把这一特例推论外推到全部 RP。

用独立 Python 计数/全排序而非生产 helper 验证：正例均值的闭区间 confident count、
rho 上限、总体/逆向噪声率推导、整数截断、严格 U 剪枝阈值、ties 保留、最少保留数及
P 不删除。保留源行为：推导总体比例可能 `>=1`，不能静默夹为“合法先验”；
只夹逆向噪声率，并保留 `prior_diagnostic_`；它不成为公共 `_class_prior`。
理论的识别条件/总体极限不等于有限 OOF 拟合就能保证真实噪声估计准确。

### 本轮实际修复：常数概率的均值上溢

初轮63项算术/调用测试通过后，补查十进制常数发现：三个正例概率均为0.1，
`np.mean` 可得0.10000000000000002，超过最大正例概率，使闭区间 confident 集合
为空而产生 NaN。真实 sklearn `DummyClassifier(strategy="prior")` 的90行、
9个P、三折回归，在默认/已知 rho 的三条路径上修复前均失败，不只是构造假后端。

保护为 `min(mean(P), max(P))`：只约束实数均值本来就应满足的上界，不加 epsilon，
不伪造计数、不改 `>=`、不悄悄忽略 fold 或改为 pooled count。不改变普通有限阈值。
源码原始 `np.mean` / NaN-fold 处理不是该保护的逐位行为，因此差异明确登记。
常数或近常数模型修复后“可计算”不代表能识别真实噪声；推导 π 仍可能无效。

## 实际拟合审计，不只看自报元数据

新增可 clone 的合成 probe classifier，按稳定行 ID 记录真实 fit/query：

- 每 fold 与最终模型都是不同的全新 clone；已拟合 template 的状态不能进入 clone。
- 每一 OOF query 都与该模型训练 ID 互斥；holdout 完整覆盖输入一次；只接观察标签 s。
- 两种 `classes_` 顺序均验证，以正类身份取概率，不暗定第二列。
- 从每个实际 holdout 独立算 rho，再取折均值。刻意构造三折 `2/3,1/3,4/7`，
  平均约0.52381，与全局合并 OOF count 约0.63158 不同，可抓住误换聚合口径。
- 最终 fit 的行、标签与内部权重逐项核对；P 权重 `1/(1-rho1)`、U 权重1。
- 2/3/4 folds × 默认/已知 rho × 两种类顺序的18条路径，实际拟合数均为
  `n_cv_folds+1`；已知 rho 也不跳过 OOF。次数不表示每次相同成本，更不等于 wall time。
- fold 或最终 base fit 失败不被吞掉/重试，公开预测不能复用之前的成功模型。
  无效概率的形状、类别、有限性、范围与归一化均 fail-loud。

仅合成数据与默认 LogisticRegression 的确定性回归，无新增真实数据/实验真值使用。
源码旧行为复用 classifier、随机 CV 未固定 seed、默认 base 库版本、单列 public proba、
泛用 PN 与作者 CNN 等差异保留；不升级 `official_related` 为 source-exact。

## 同步与复现

台账补四文件摘要和来源引用，[扩展草稿](../data/p3_preintegration_extension_v1_draft.json)
按原生成器再核一致；`run_view=os-compatible` / `calibration_applied=false` 不变。
不支持 TS 替换，不给 P 添加伪 U；`fit` 不接具名隐藏 y 或 PA/OA 角色，外部 population π
仅校验后忽略。调用方仍须保证所传标签是真实的观测 s，接口不能识别冒充 s 的隐藏真值。
正式 `frac_pos2neg` 的来源/默认估计政策、内部 CV 预算、表征、选模与公开数值仍待批准。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/unit/estimators/test_rank_pruning.py \
  tests/unit/estimators/test_rank_pruning_source_contract.py \
  tests/unit/scripts/test_prepare_p3_preintegration_extension.py
python scripts/prepare_p3_preintegration_extension.py
```

已有相同锁定 checkout 时，只读验证 commit 和四文件双摘要：

```python
import json
from pathlib import Path
from scripts.check_p3_admission_evidence import validate_source_checkout

review = json.loads(Path(
    "docs/research/pu_survey/data/rp_source_review_20261018.json"
).read_text(encoding="utf-8"))
assert validate_source_checkout(review["source"], Path("/tmp/pu-rp-source-WMm7KK")) == 4
```

## 本轮验收记录

- 修复前真实 DummyClassifier 的三条数值回归全部失败；最终专项
  **115 passed / 0 skipped / 0 failures / 0 errors**，0.931秒，其中新增来源/数值100项。
- 相对 `2c2df6b`，三个 seed 的普通 LogisticRegression 路径，OOF、各 fold rho、
  剪枝、保留行/权重、推导诊断及预测逐 bit相同；`fit` AST 不变。
- 五个仅内存的故障注入被抓住：pooled count 替换折均值、复用已拟合模板、
  删除正例重权、夹掉非法 prior，以及撤销数值均值上界保护。生产文件不做临时修改。
- 12项最终本地静态门禁通过；五方法准入证据/公开对照/阶段预算检查保持通过。
  四文件 commit/原始字节/规范化文本均验证，禁止 torch 也可核验来源与扩展草稿，
  并可完成真实 DummyClassifier 边界拟合。
- sdist → wheel 构建通过；源码目录外 wheel 核当前 RP SHA、真实三折与 pickle
  精确回放通过。wheel/sdist 未混入六个私有根目录制品、模型、数组或作者代码。
- 修复后全量快层 **4556 passed / 58 skipped / 0 failures / 0 errors**，
  305.006秒；本地 JUnit：`/tmp/rp-source-fixed-fast.xml`。
  远程多平台 CI 尚未核验。不声称全仓零 warning。

不改历史协议/矩阵、probe、结果与签署；
P3MIX 来源、Split-PU #122、支持集与正式配方/资源的前置不因本补证解除。
