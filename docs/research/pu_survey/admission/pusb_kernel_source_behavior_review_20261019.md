# PUSBKernel 来源与行为核查（2026-10-19）

范围：核对 `PUSBKernelClassifier` 的来源锁、方法卡、生产实现、TS-OS 校准与测试边界。属于单方技术核查，不是方法负责人签署、正式准入或论文结果复现。

## 来源基线

- 论文：Kato, Teshima, Honda, “Learning from Positive and Unlabeled Data with a Selection Bias,” ICLR 2019，[OpenReview PDF](https://openreview.net/pdf?id=rJzLciCqKm)。论文目标是选择偏差下的部分识别/后验排序 scoring，不是校准后验。
- 作者代码：[MasaKat0/PUlearning](https://github.com/MasaKat0/PUlearning)，仓库源码锁定在 `benchmarks/assigned_methods/configs/official_sources.lock.json` 中的 commit `3401b77ccdd653d39f4f3a6258a42c7938fa9ede`。作者当前可变主页提示仓库正在重制；本审计以锁定版本及已保存的官方运行配置为准。
- 本方法在 registry/ledger 中为 `official_related`，意为官方算法来源相关且有 RBF 实现证据，不代表代码逐行复制；线性 `pusb` baseline 已另行审计，不能与此条目混同。

## 行为对应与明确适配

| 行为 | 工具箱证据 | 审计结论 |
|---|---|---|
| RBF basis | `pusb_kernel.py::_squared_distances`、`_rbf_design`；从完整输入训练矩阵随机抽取最多 `n_basis` 个中心并加截距列 | 适配记录称中心一次抽取、全 CV/最终 refit 共用。CV 验证行可成为训练阶段 basis center，这是当前发布结构所保留的 transductive basis 策略，不是独立 inductive fold-center 方案；比较时应一致披露。 |
| PU 风险 | `_pu_objective_and_gradient`：正例线性项、U 上 `logaddexp` 项、L2 正则及解析梯度 | 对应作者 RBF 风险结构。官方目标文字/代码梯度的正则系数存在已登记的 2 倍矛盾；实现选择 `0.5 * λ‖b‖²` 使解析梯度 `λb` 自洽，有限差分测试通过。属于显式 clean-room 数值裁决，不应隐藏。 |
| 超参选择 | `fit` 固定抽中心和随机折，遍历 σ/λ 网格、累计每折 PU validation risk，`argmin` 按网格顺序稳定解并最终全训练集 refit | 与现有官方配置/方法卡的调参骨架对齐；缺类别的 fold 会 fail-loud，而不继续产生无定义风险值。实际是否数值复现取决于数据与原始划分，不由本单元测试证明。 |
| 预测阈值 | `prior_quantile_predict` 使用 `floor(n(1-π))` 次序统计量并严格 `score > threshold`；`fit` 冻结训练 score 阈值用于公共 `predict`，另有 `predict_with_prior_quantile` | 批次依赖的官方规则由显式接口保留；稳定 sklearn `predict` 不随预测批组成变化，是接口适配。`predict_proba` 也不应解释为校准真实后验。 |
| TS-OS 角色校准 | `fit(os_or_ts=...)` 只将训练风险的 U 角色从 U 行切换到全体训练行，正例项/π、中心池、fold ids 和阈值池不变 | 协议驱动校准，不是论文方法语义。内部 CV validation 始终 OS，所以 TS 训练配方使用 OS 验证目标选参（跨目标选参）。方法卡与 RV6 技术审计披露了代价，专项测试覆盖角色、折内验证隔离与中心不随视图变化；方法学审阅仍待负责人确认。 |
| 官方数据结果边界 | `pusb_official_data_feasible_multiseed.json` 明示 IJCNN1 为仓库扩展、prior 0.2 三 seed tranche，`paper_claim=false`；Table 2 严格子集另有 45/72 cell、4500 trials 记录 | 不把 IJCNN1 或 Table 2 严格可行子集冒称完整论文复现；本审计不重写任何历史结果。 |

## 测试证据

- `tests/unit/estimators/test_pusb_kernel.py`：目标梯度有限差分、阈值边界/严格比较、随机种子重放与基础预测契约。
- `tests/unit/estimators/test_pusb_kernel_ts_view.py`、`test_pusb_kernel_view_invariants.py`：OS/TS 风险 golden、role mapping、中心/fold/threshold 池不随视图变化、参数视图和 validation OS 隔离。
- `tests/benchmarks/test_pusb_table2_benchmark.py` 与 `tests/benchmarks/test_pusb_official_data.py`：runner/产物结构及计划契约；不等于重跑完整论文 Table 2。

结论：RBF adapter 的关键实现边界已有较充分数学与协议测试，源码可获得性和代码忠实度仍需分开表述。保留 `official_related`；核方法的 TS 视图、跨目标 CV 选参、校准作用范围仍需 HENG958/RV6 方法学确认。该审计不改算法、冻结矩阵、实验产物或准入状态。
