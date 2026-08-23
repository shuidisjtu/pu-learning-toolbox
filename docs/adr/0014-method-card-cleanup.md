# ADR-0014:方法卡清洗

- 状态:已接受
- 触发复审:方法卡再混入开发前设计内容,或统一 8 节结构无法容纳某类论文知识时

## 背景

17 篇方法卡( `docs/research/method_cards/` )源于「开发前设计文档」,混有待办清单、
API 接口与构造签名、Toolbox 集成映射、测试与验收标准等工程内容。2026-08-14 逐篇测绘
(调研报告作为批次工件保留,不入库)显示:待删内容约占全目录 34%(约 2435/7234 行),
集中于「待办与注意」「API 接口与项目落点」「Toolbox 集成映射」「测试与验收标准」四大
固定节;且三形态并存——旧模板 14 篇、新模板 3 篇( DGPU / InfoMax-PU / WConPU ),
结构偏离 2 篇( KLDCE 整卡为求解器设计文档、Kernel_Mean_Class_Prior 为混合模板),
导航与检索成本持续累积。

## 决策

1. **方法卡清洗**:删除待办清单、API 接口/构造签名、Toolbox 集成映射、测试与验收标准
   四类「开发前」内容,统一为 8 节结构(具体节表与逐卡判定由清洗批次 spec/plan 定义,
   按调研报告「清洗建议要点」执行)。方法卡回归「论文知识」文档定位,不再承载工程规划。
2. **非常关键实现注记集中管理**:调研 §6 总表中 ★ 标注的 13 条「非常关键」注记
   (影响算法正确性/数值结果/注册正确性)全文收录于本 ADR(见下「注记全集」,每条含
   出处与正文);清洗后方法卡 §6 以 `> 见 ADR-0014 #N` 一行索引,不复制正文。
3. **结果记录真相源**:benchmark 数字、SHA/commit 锁、trial 计数等结果记录的权威源为
   `benchmarks/**/results/*/REPORT.md`;方法卡只保留指向 REPORT 的链接。注记全集中
   的结果型条目( #1 、#3 )以「防误读警示/诊断性结论」身份保留于本 ADR,并附结果目录
   路径,不随方法卡清洗而丢失。

## 注记全集(#1-#13)

> 编号按卡片文件名字母序;同卡多条按行号序。正文为原方法卡对应行号的完整内容(含日期标注),
> 上下文为理解所需的最小邻接信息,均以 2026-08-14 调研时点的方法卡为准。

### #1 class_prior_estimation.md L383

- **出处**:`docs/research/method_cards/class_prior_estimation.md` L383
- **摘要**:0.0380 MAE 为 sigma=1.0 时代数字,新默认后 benchmark 须显式 pin sigma(防误读)
- **正文**:
  > 注意：上述 MAE 是默认 `sigma=1.0` 时代的数字；2026-08-10 默认改为数据自适应 `sigma` 后该数值不再代表当前默认行为，benchmark 复现需在配置中显式 pin `sigma`。
- **上下文**(L378-381):seed 0..4 clean-room 合成实验 penL1 prior MAE 为 `0.0380 ± 0.0192`;
  该数字不含 MNIST、论文逐 `theta` CV 和完整基线,不是论文表格复现。

### #2 Elkan_Noto.md L169-173

- **出处**:`docs/research/method_cards/Elkan_Noto.md` L169-173
- **摘要**:OOF c_hat 系统性低估约 25%,折内 `sqrt(n_U/n_P_fold)` 加权校正(0.40→0.59 vs 真 0.535);最终 g 不加权
- **正文**:
  > **实现注记（2026-08-10）**：OOF 每折训练集的正例占比（约 n_P/n_cv 除以总训练数）
  > 明显低于全量数据占比，会把 LR 概率整体压低、系统性低估 `c_hat`（常规 SCAR 数据上
  > 低估约 25%）。折内对标注正例施加 `sqrt(n_U/n_P_fold)` 的 sample_weight（几何中点插值，
  > 与用户 `sample_weight` 相乘、不覆盖），将 `c_hat` 校正回真值附近（探针数据
  > 0.40 → 0.59，真 0.535）。最终全量 `g` 拟合保持规范不加权。
- **上下文**(L164-167,§5.1 概率校正步骤):标注正例设 `s=1`、无标签样本设 `s=0`,训练并
  校准 `g(x)≈p(s=1|x)`;用独立验证集标注正例预测值均值计算 `ĉ`;返回 `f̂(x)=g(x)/ĉ`;
  以目标真实概率阈值 `τ` 决策时比较 `g(x)` 与 `ĉτ`。

### #3 InfoMax-PU.md L346-359

- **出处**:`docs/research/method_cards/InfoMax-PU.md` L346-359
- **摘要**:Fashion-MNIST 60/60 trials:AUC 0.8547/0.6313/0.3340,7/20、15/20 seeds<0.5;KM1≈0.875 不随受控先验变化(诊断性负结果)
- **正文**:
  `configs/official_data_infomax_fashion_protocol_pi05.json` 已锁定论文网络深度、BN、优化器、
  gradient noise、样本数、epoch 和 20 seeds，并通过 CPU preflight。论文只写明将
  MNIST/Fashion-MNIST 十类分成两组，没有给出类别编号，也没有报告 mini-batch size；当前
  配置把 `[0,1,2,3,4]` 和 batch size `256` 明确标记为临时工程选择。runner 已从训练集
  之外确定性划分 `50 P + 200 U` validation，并接入原生 KM1/KM2 class-prior estimator；
  论文未说明使用 KM1 还是 KM2，当前 KM1 是显式临时选择。固定 epoch 协议不使用验证集早停。
  该暂定协议已在 Fashion-MNIST 完成 `pi=0.3/0.5/0.7` 各 20 seeds，共 60/60 trials。
  ROC-AUC（均值 ± 样本标准差）依次为 `0.8547 ± 0.0819`、
  `0.6313 ± 0.3077` 和 `0.3340 ± 0.2873`；AUC 小于 `0.5` 的 seed 数依次为
  `0/20`、`7/20` 和 `15/20`。KM1 类先验估计在三组均约为 `0.875`，未跟随
  受控 U 先验变化。完整结果位于
  `benchmarks/deep_pu/results/infomax_fashion_protocol_pi{03,05,07}_full/`。测试集保持
  Fashion-MNIST canonical 正率 0.5；类别分组、batch size、KM 变体和 test-prior 规则仍是
  临时选择，因此结果保持 `paper_claim=false`，不能标记为论文结果。

### #4 Kernel_Mean_Class_Prior.md L126-131

- **出处**:`docs/research/method_cards/Kernel_Mean_Class_Prior.md` L126-131
- **摘要**:width_selection="relative" 默认(2026-08-10);mmd_grid 系统性偏选宽带宽低估(km1 0.30→0.59、km2 0.31→0.46,真 0.5),论文协议须 pin mmd_grid
- **正文**:
  未显式指定 `kernel_width` 时，默认（`width_selection="relative"`，2026-08-10 起）取
  `kernel_width_scale × sqrt(median pairwise squared distance)`（`kernel_width_scale=0.1`）。
  该固定相对比例是尺度不变、数据自适应的；作者代码的 max-MMD 5 档宽度搜索
  （`width_selection="mmd_grid"`）保留为可选路径 —— 探针显示 max-MMD 准则在常规 SCAR
  数据上系统性偏选宽带宽、低估类先验（默认 km1 0.30→0.59、km2 0.31→0.46，真值 0.5），
  论文协议复现（如 InfoMax-PU）需显式 pin `width_selection="mmd_grid"`。
- **上下文**(L120-124):实现采用 RBF kernel
  $`k(x,x')=\exp\left(-\|x-x'\|^2/(2\sigma^2)\right)`$。

### #5 KLDCE.md L396-404

- **出处**:`docs/research/method_cards/KLDCE.md` L396-404
- **摘要**:RBF 用论文原生 σ,勿直接传 sklearn rbf_kernel 的 γ=1/(2σ²)
- **正文**:
  按附录推导，首批只支持：`rbf`。每个核必须满足近似对称性：

  ```text
  max_abs(K(X, X) - K(X, X).T) <= kernel_symmetry_tol
  ```

  RBF 采用 $`K(x,z)=\exp(-\|x-z\|^2/(2\sigma^2))`$（论文原生参数 $`\sigma`$）。
  不使用 `sklearn.metrics.pairwise.rbf_kernel`（其参数是 $`\gamma=1/(2\sigma^2)`$，直接传入 $`\sigma`$ 会得到错误核矩阵）。
  默认 $`\sigma=1/\sqrt{d}`$（`d` 为特征维数），对应 $`\gamma=d/2`$。

### #6 KLDCE.md L471

- **出处**:`docs/research/method_cards/KLDCE.md` L471
- **摘要**:2026-08-05:kldce 曾被误注册为 centroid_pu 别名导致静默解析错误,已修复;auto 模式跳过需显式传实例
- **正文**:
  - [x] 已注册为独立原生实现（registry `kldce`，2026-08-05）。`get_algorithm("kldce")` 解析到 `KLDCEClassifier`（此前被误注册为 `centroid_pu` 的别名，会静默解析到线性 `LDCEClassifier`，已修复）。注意：`KLDCEClassifier` 构造要求 `flip_probability`（与 `ldce` 相同），因此 `--classifier kldce` / auto 模式无法自动实例化——推荐器会将其列入候选（与 `centroid_pu` 同 metadata），但 auto 模式跳过它，需显式传实例（Python API）。censoring-PU/`h` 可用等数据条件的专门匹配规则暂未实现，推荐器仅按 SCAR/SINGLE_TRAINING_SET 元数据做包含性匹配。

### #7 LDCE.md L196-247

- **出处**:`docs/research/method_cards/LDCE.md` L196-247
- **摘要**:CEGE_PAMI20.rar 实为会议早期版(双向噪音+无椭球+无 MoM+简单 GD),非 PAMI 终稿 LDCE,不可作实现参考
- **正文**(§6.1 源码内容,已审计 2026-07-21):
  `CEGE_PAMI20.rar` 解压后共 5 个文件：

  | 文件 | 说明 |
  |---|---|
  | `main.m` | 主脚本：10-fold CV，GermanCredit 数据集 |
  | `SemiLinearTraining.m` | 核心训练函数（151 行 MATLAB） |
  | `GermanCredit.mat` | 数据集 |
  | `Idx0.05.mat` / `Idx0.1.mat` | 两种 labeled ratio（5%/10%）的 CV 索引 |

  §6.2 代码与论文差异(关键)——实际代码实现的是 **CEGE 早期会议版本**，与 PAMI 终稿
  LDCE 存在以下重大差异：

  | 维度 | 论文 LDCE | 实际代码 |
  |---|---|---|
  | **噪音模型** | 单向 censoring：P(Ỹ=-1\|Y=+1)=h, P(Ỹ=+1\|Y=-1)=0 | **双向噪音**：同时计算 `Yita_N` 和 `Yita_P`，U 分两次被当作 noisy N 和 noisy P |
  | **质心估计** | MoM 稳健初始化（Algorithm 1）→ 单一无偏质心校正 | 两个方向各算一个无偏质心估计，再以 `Beta` 加权组合；无 MoM |
  | **椭球约束** | 式 (13) 的 `(m-m̂)ᵀŜ(m-m̂) ≤ b` 约束 + 闭式更新 | **不存在**；质心仅为梯度中的常数项（`Mu_S`），不参与优化 |
  | **优化方式** | 交替优化：固定 w→闭式更新 m，固定 m→梯度更新 w（Algorithm 2） | 简单梯度下降（800 iter，自适应步长）；无交替结构 |
  | **损失函数** | hinge loss 偶/奇分解（§4.2 式 (11)-(12)） | 支持 squared / hinge / squared-hinge 三种 loss，但 hinge 梯度未做偶/奇分离 |
  | **正则化** | L2 正则 ‖w‖² | `J'*J*w` 形式的正则（`J` 为去掉 bias 行的单位阵） |

  关键代码片段——同时计算两个方向的质心：

  ````matlab
  % treat U as noisy N
  Mu_tilte_NoisyN = (sum(labeled.*y) - sum(unlabeled)) / n;
  % treat U as noisy P
  Mu_tilte_NoisyP = (sum(labeled.*y) + sum(unlabeled)) / n;
  % 加权组合两个无偏估计
  Mu_S = Beta*Tau_P*Mu_tilte_NoisyN + (1-Beta)*Tau_N*Mu_tilte_NoisyP;
  ````

  核心优化——简单梯度下降，无交替、无椭球约束：

  ````matlab
  w = -1+2*rand(DataDim,1);
  for Iter = 1 : MaxIter
      Grad = WeakGrad + Gamma_1*StrongGrad + 2*Gamma_2*(J'*J)*w;
      w = w - StepSize*Grad;
  end
  ````

  §6.3 集成边界:`.rar` 压缩包无 license 声明,仅作算法参考,不直接复用代码;URL 中
  "CEGE" 为论文方法的另一简称(Centroid Estimation with Generalized Eigenvalue),但
  压缩包内代码对应的是 **会议早期版本**(双向噪音 + 无约束 GD),**并非 PAMI 终稿的
  LDCE 算法**(单向 censoring + MoM + 椭球约束交替优化);这份代码不可直接用作 LDCE
  实现参考,论文 Algorithm 1(MoM 质心)和 Algorithm 2(交替优化)均需从零实现。

### #8 LLSVM.md L119-151

- **出处**:`docs/research/method_cards/LLSVM.md` L119-151
- **摘要**:论文式(9)与官方代码 5 项偏差(exp[-5f²]、A=10、归一化、增广常数 10),实现以代码为准
- **正文**(§4.3 论文 vs 官方代码偏差,实现以代码为准):
  官方 MATLAB 代码包（`LLSVM_TNNLS19.rar`）中的目标函数和梯度与论文式 (9) 存在以下差异。
  代码是作者实际运行实验的版本，**本项目以代码为准实现**。

  | # | 项目 | 论文 (§4.2) | 官方代码 | 说明 |
  |---|---|---|---|---|
  | 1 | 指数系数 | $`\exp[-3f^2]`$ | $`\exp[-5f^2]`$ | 梯度中 $`-6f`$ → $`-10f`$；更窄的 hat，对边界附近惩罚更集中 |
  | 2 | 压缩函数 | $`\Phi(z)=\frac{2}{\pi}\arctan z`$（固定） | $`\Phi_A(z)=\frac{A}{\pi}\arctan z`$，$`A=10`$ | 引入可配缩放参数 $`A`$，校准项和梯度均随之变化 |
  | 3 | P 项归一化 | $`\frac{\alpha}{p}\sum_P`$ | $`\alpha\sum_P`$（不除 $`p`$） | 等价于将 $`\alpha`$ 吸收了 $`p`$ 倍；跨数据集时需注意尺度 |
  | 4 | U 指数项归一化 | $`\frac{\beta}{u}\sum_U`$ | $`\beta\sum_U`$（不除 $`u`$） | 同理 |
  | 5 | 增广常数 | 1 | 10 | `fit_intercept` 对应的偏置列值 |

  代码的实际训练目标：

  ```math
  J_{\text{code}}(\omega)
  =\alpha\sum_{x\in P}[\max(1-f_\omega(x),0)]^2
  +\beta\sum_{x\in U}\exp[-5f_\omega(x)^2]
  +\frac{\gamma}{u}\sum_{x\in U}\!\left[\max\!\left(\tfrac{A}{\pi}\arctan f_\omega(x)-t,\,0\right)\right]^2
  ```

  对应的 U 项梯度（实现目标）：

  ```math
  \nabla_\omega\,\beta\,e^{-5f^2}=-10\beta\,f\,e^{-5f^2}\,\bar{x}
  ```

  ```math
  \nabla_\omega\,\frac{\gamma}{u}\!\left[\max\!\left(\Phi_A(f)-t,0\right)\right]^2
  =\frac{2A\gamma}{\pi\,u\,(1+f^2)}\max\!\left(\Phi_A(f)-t,0\right)\bar{x}
  ```

  > 正则化项：代码在 SGD 梯度中加 $`\omega\times\text{BatchSize}`$，`ComputeCost` 不含正则项。建议实现时采用标准 $`\frac{\lambda}{2}\lVert\omega\rVert^2`$ 正则并在 cost 中统一计算，以便验证收敛。

### #9 nnpu.md L237-238

- **出处**:`docs/research/method_cards/nnpu.md` L237-238
- **摘要**:Algorithm 1 校正分支梯度来自 -r_i,对 max(0,·) 反传不等价
- **正文**:
  > **关键实现约束**：在 `r_i < -beta` 时，Algorithm 1 的梯度来自 $`-r_i`$，不是来自  
  > $`\pi_p\widehat R_p^+ + \max(0,r_i)`$。直接对 `max` 反向传播会保留正类风险梯度且不给 $`r_i`$ 校正梯度，不等价于论文算法。
- **上下文**(L229-235):`r_i >= -beta` 时按 uPU 风险正常下降;`r_i < -beta` 时停止优化正类
  风险,反向推动 `r_i` 增大,避免负类风险向负方向发散;`beta=0` 为论文默认 nnPU;`gamma=1`
  完整校正步长;`gamma=0` 校正分支不更新。

### #10 PUSB.md L28-31 + L215-217

- **出处**:`docs/research/method_cards/PUSB.md` L28-31 与 L215-217
- **摘要**:官方梯度与正则相差系数 2,适配器用 0.5λ‖b‖² 兼容修正且须入 manifest
- **正文**(L28-31):
  - `PUSBClassifier` 保留为可运行的 linear ranking baseline；`PUSBKernelClassifier` 是独立的
    official-aligned clean-room 适配器，避免已有用户升级后行为突变。
  - 官方源码的正则目标与梯度相差系数 2。适配器使用与官方梯度一致的
    `0.5 * lambda * ||b||^2`，该修正必须进入结果 manifest。
- **正文续**(L215-217):
  实现使用 `logaddexp` 和 `expit` 避免指数溢出，并用有限差分锁定梯度。官方源码目标最后
  一项写成 `lambda * b^T b`，但梯度写成 `lambda * b`；上述 `lambda/2` 是显式兼容性
  修正，而不是无记录地改变公式。
- **上下文**(L209-213):对应梯度
  $`\nabla_b\widehat R=-\pi\,\overline{\tilde\phi}_P+\frac{1}{n_U}\sum_{x\in U}\mathrm{sigmoid}(g(x))\tilde\phi(x)+\lambda b`$。

### #11 ReCPE.md L23 + L180

- **出处**:`docs/research/method_cards/ReCPE.md` L23 与 L180
- **摘要**:2026-08-10 默认 base CPE 改 KernelMeanPriorEstimator(km2);原 1% 分位+未校准 LR 坍缩(0.036 vs 0.5),KM2 升至 ~0.40
- **正文**(L23):
  - **2026-08-10 变更**：默认 base CPE 改为 `KernelMeanPriorEstimator(variant="km2")`（论文官方参考基线）；工程侧 classifier-based baseline（`_DensityRatioCPE`，分位数默认 `0.25`）保留为显式可选。原 1% 分位数 + 未校准 LR 概率路径在常规 SCAR 数据上坍缩（估计 0.036 vs 真 0.5），换 KM2 后升至 ~0.40。
- **正文续**(L180,当前实现默认选择表行):
  | 底层 CPE | `KernelMeanPriorEstimator(variant="km2")`（2026-08-10 起）；显式 `base_estimator` 可注入任意 `fit/estimate` CPE |
- **上下文**(L173-181):默认排序分类器 `StandardScaler + LogisticRegression`、`copy_fraction=0.1`、
  复制数量 `max(1, ceil(copy_fraction * n_unlabeled))`;自定义底层方法要求支持 `fit(X, y_pu)` 与 `estimate()`。

### #12 ReCPE.md L24

- **出处**:`docs/research/method_cards/ReCPE.md` L24
- **摘要**:v1.2.1 边界声明:全部 base 变体常规 SCAR 系统性低估 0.08-0.19,修复目标仅「不坍缩」≥0.1
- **正文**:
  - **边界声明（v1.2.1）**：探针证实 ReCPE 的全部 base 变体（裸 LR + 分位数 0.01/0.25、校准 LR、KM1/KM2）在常规 SCAR 数据上均系统性低估（0.08–0.19 区间），这是 regrouping 构造与 irreducibility 折中的文献固有偏差；修复目标仅为"不坍缩"（≥0.1）与默认 base 正确化，不承诺 ±50% 带。常规 SCAR 场景优先选择 pen_l1/km2，ReCPE 保留其设计用途（irreducibility 失效时防正向偏差）。

### #13 WConPU.md L270-271

- **出处**:`docs/research/method_cards/WConPU.md` L270-271
- **摘要**:仓库旧元数据 requires_class_prior=False 必须修正(论文复用 Dist-PU 风险需要 π_P)
- **正文**:
  这说明 WConPU 实际上需要 `pi_P`。仓库旧元数据中的
  `requires_class_prior=False` 必须修正。
- **上下文**(L253-268,§11 标签分布对齐):对齐项含
  $`2\pi_P\left|\frac1{n_P}\sum_{x_i\in\mathcal X_L}z_{i,1}-1\right|+
  \left|\frac1{n_U}\sum_{x_i\in\mathcal X_U}z_{i,1}-\pi_P\right|`$，
  正则惩罚依赖先验 `π_P`,故元数据声明与算法实际需求矛盾。

## 备选方案

- **双份保留**(注记正文同时留方法卡与 ADR):同一事实两处维护,正文漂移风险
  (ADR-0013 已因 cli_design 参数表静默漂移付出代价)。否决。
- **只删不重组**(仅删待办/API/集成映射/测试,不统一 8 节结构):三形态并存,
  导航与检索问题未解决。否决。
- **结果全留卡内**(benchmark 数字继续写在方法卡):结果随基准执行更新而过期,
  class_prior_estimation L383 的过期警示(#1)即反例。否决。

## 后果

- 注记正文以本 ADR 为权威,方法卡 §6 只留 `> 见 ADR-0014 #N` 一行索引;后续新增
  非常关键注记须先更新本 ADR 再于方法卡引用。
- 本 ADR 收录 13 条注记(调研 §6 总表 ★ 条目),编号 #1-#13,与简报「约 14 条」的
  约数表述一致(简报卡片分布清单亦为 13 条)。
- 方法卡清洗由后续任务执行:KLDCE 整卡为求解器设计文档,按「论文/附录公式保留、
  工程设计删除」逐段判定,是全目录清洗工作量最大的一篇。
- 结果记录型内容迁移至 `benchmarks/**/results/*/REPORT.md`,方法卡保留链接;
  #1、#3 的数值警示与诊断性结论随本 ADR 留存,不受方法卡清洗影响。
