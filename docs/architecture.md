# Architecture Design

## 1. 核心原则

- Core 包轻量，深度学习依赖放入 optional extension。
- 类先验、标记倾向、损失函数、分类器解耦。
- 所有算法通过 registry 管理，元数据驱动发现和推荐。
- 有官方源码的论文优先走 adapter，无源码的 clean-room 实现。SAR / Instance-Dependent PU 是中长期差异化重点。

完整目录结构以 [`project_structure.md`](project_structure.md) 为权威来源。

## 2. 模块分层

| 层 | 模块 | 作用 |
|---|---|---|
| Core | `core`, `preprocessing`, `registry`, `utils` | 稳定 API、标签规范、输入校验、SCAR/SAR 标签与数据生成、结构化数据画像、算法注册、元数据、共享工具 |
| Estimation | `prior`, `losses` | 类先验估计、PU 损失函数 |
| Algorithms | `estimators` | 实现具体 PU 分类器 |
| Evaluation | `metrics`, `model_selection`, `diagnostics` | PU 评估指标、PU 分层切分、结构化报告与假设敏感性 |
| User Layer | `examples` | 教程 |

## 3. 数据流

```
用户输入 (X, y_pu) → 标签规范化 + 校验 → Data Profiler
    ↓
Registry → 候选算法 → 实现解析 (native / torch)
    ↓
类先验估计 + 标记倾向估计 → 模型训练 → 输出 (predict / decision_function / predict_proba)
    ↓
评估 + 诊断 → 报告
```

Data Profiler 输出 `PUDataProfile`：包含基础统计、特征质量、问题级别、行动建议和
标记机制证据。无审计真值时，SCAR/SAR 提示明确标记为非识别性筛查；提供 `y_true`
时仅在真实正例内部评估 selection dependence，避免把类别可分性误认为 SAR。

`build_diagnostic_report` 位于 `diagnostics`，只读取 Data Profiler、已拟合 estimator
和指标接口。它不训练模型，并将观测 PU、类先验依赖、监督 oracle 和
不可用指标分别标记，输出稳定 schema 的 JSON/Markdown 报告。

## 4. 核心 API

### 4.1 BasePUClassifier

```python
class BasePUClassifier(BaseEstimator, ClassifierMixin, ABC):
    family: AlgorithmFamily = AlgorithmFamily.UNKNOWN
    assumption = (Assumption.UNKNOWN,)
    scenario = (Scenario.UNKNOWN,)
    requires_class_prior: bool = False
    implementation_status: ImplementationStatus = ImplementationStatus.API_ONLY
    source_status: SourceStatus = SourceStatus.UNKNOWN
    backend: Backend = Backend.NUMPY
    maturity: Maturity = Maturity.EXPERIMENTAL

    @abstractmethod
    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        ...

    def predict(self, X):                        # public: check → _predict
        self._check_is_fitted()
        return self._predict(X)

    @abstractmethod
    def _predict(self, X):                       # subclass implements
        ...

    def decision_function(self, X):              # public: check → _decision_function
        self._check_is_fitted()
        return self._decision_function(X)

    @abstractmethod
    def _decision_function(self, X):             # subclass implements
        ...

    def score_samples(self, X):                  # default = _decision_function
        ...

    def predict_proba(self, X):                  # raises NotImplementedError
        ...

    def predict_label_proba(self, X):            # returns None by default
        ...

    def get_pu_metadata(self) -> dict:
        ...
```

### 4.2 BasePriorEstimator

```python
class BasePriorEstimator(BaseEstimator, ABC):
    def fit(self, X, y_pu):
        ...

    def estimate(self):
        ...

    def confidence_interval(self, alpha=0.05):
        return None
```

### 4.3 BasePULoss

```python
class BasePULoss(ABC):
    requires_class_prior = True

    def __call__(self, positive_scores, unlabeled_scores, *, class_prior):
        ...
```

## 5. 输出接口规范

| 方法 | 是否必须 | 含义 |
|---|---|---|
| `fit(X, y_pu)` | 必须 | 训练模型 |
| `predict(X)` | 必须 | 输出离散标签；公共方法调用子类 `_predict(X)` |
| `decision_function(X)` | 必须 | 输出连续分数；公共方法调用子类 `_decision_function(X)` |
| `score_samples(X)` | 可选覆盖 | 默认复用 `decision_function` 分数；仅当分数约定不同才覆盖 |
| `predict_proba(X)` | 可选 | 输出 $P(y=1\mid x)$ |
| `get_params()` / `set_params()` | 必须 | 由 sklearn `BaseEstimator` 提供，兼容 Pipeline / GridSearchCV |

## 6. 算法注册表

每个算法注册元信息，registry 据此管理发现和推荐。

```python
{
    "name": "nnpu",
    "aliases": ["non_negative_pu", "nn-pu", "nnPU"],
    "family": "risk_estimation",
    "scenario": ["case_control"],
    "assumption": ["SCAR"],
    "requires_class_prior": True,
    "supports_sparse": False,
    "supports_gpu": True,
    "backend": "torch",
    "maturity": "stable",
    "source_status": "official_exact",
    "implementation_status": "native",
}
```

`implementation_status` 枚举：

`source_status` 表示论文源码可获得性，当前代码枚举以 `pu_toolbox/core/tags.py` 为准，常见值包括 `official_exact`、`official_bundle`、`official_related`、`third_party_only`、`not_found`、`unknown`。

| 状态 | 含义 |
|---|---|
| `api_only` | 仅 API 占位，无训练逻辑 |
| `native` | clean-room 实现 |

### 算法推荐器

`registry/recommender.py` 提供 `recommend_methods(X, y_pu, ...)` 和 `recommend_from_profile(profile, ...)`，将数据画像与元数据匹配：

1. **硬过滤**：trainable、scenario、sparse 支持、class_prior 可用性
2. **软评分**：assumption 匹配(30) + maturity(20) + source_status(15) + 数据规模(20) + GPU(5) + 标记充足度(10)
3. **风险提示**：自动生成全局和每方法的警告

返回 `RecommendationResult`，支持 `to_json()` / `to_markdown()` / `save()`。

## 7. 类先验、标记倾向与损失函数

| 概念 | 相关方法（✅ 已实现 / ⏳ 计划中） |
|---|---|
| 类先验 $\pi$ | ✅ ReCPE, ✅ penL1, ✅ KM1/KM2, † TIcE, † AlphaMax |
| 标记倾向 $c$ (SCAR) | ✅ Elkan-Noto |
| 标记倾向 $c(x)$ (SAR) | ✅ LBE, ✅ PUSB |
| PU 风险/损失 | ✅ uPU, ✅ nnPU, ✅ PNU, ✅ Dist-PU |

> † 扩展参考（不在 v1 范围内），非 15 篇核心论文方法。

## 8. 论文方法到模块的映射

| 方法 | 主要模块 |
|---|---|
| Class-Prior Estimation | `prior/pen_l1.py`, `prior/recpe.py`, `prior/kernel_mean.py` |
| ReCPE | `prior/recpe.py` |
| Elkan-Noto | `estimators/classic/elkan_noto.py` |
| uPU / nnPU / PNU | `losses/upu.py`, `losses/nnpu.py`, `losses/pnu.py` |
| uPU 分类器 | `estimators/risk/upu.py` |
| nnPU 分类器 | `estimators/risk/nnpu.py` |
| PNU 分类器 | `estimators/risk/pnu.py` |
| 共享 basis 工具 | `utils/basis.py` |
| PUSB / LBE | `estimators/bias_aware/pusb.py`, `estimators/bias_aware/lbe.py` |
| Dist-PU | `estimators/risk/dist_pu.py` |
| Self-PU | `estimators/deep/self_pu.py` (native core) |
| LDCE / Centroid PU | `estimators/risk/ldce.py` |
| KLDCE (核化 LDCE) | `estimators/risk/kldce.py` (QP oracle + RBF kernel) |
| 共享质心原语 | `utils/centroid.py` (MoM + 协方差) |
| LLSVM | `estimators/classic/llsvm.py` |
| InfoMax PU | `estimators/deep/infomax_pu.py` (PURL + nnPU pipeline) |
| Weighted Contrastive PU | `estimators/deep/weighted_contrastive_pu.py` (native core) |
| DGPU | `estimators/deep/dgpu.py` (native orchestration + generator protocol) |

完整映射及实现策略见 [`development_roadmap.md`](development_roadmap.md)。

## 9. 评价与切分

- `PUStratifiedKFold`、`PUStratifiedShuffleSplit`（已实现）：保证每个训练折含 labeled positive，保留 P/U 比例。
- PU-only 指标（不需要真实标签）：`pu_zero_one_risk`（PU 零一验证风险）、`pu_recall`（从已标记正样本估计召回率）、`pu_estimated_precision`（利用类先验估计精确率）、`pu_negative_rate`（无标记样本负预测率）。
- 有真实 $y$ 时使用标准监督指标包装（AUC, F1, Accuracy）。
- SCAR/SAR 证据：`scar_diagnostic` 在无真值时仅报告非识别性 P/U 筛查信号；
  提供审计 `y_true` 时，在真实正例内检查 selection dependence。
- Selection-bias 模拟：`make_sar_propensity`、`make_sar_labels` 和 `make_sar_dataset`
  支持常数、线性与非线性 propensity；隐藏 `y_true/propensity` 仅供 benchmark 使用。
- 结构化报告：`build_diagnostic_report` 组合数据画像、模型 metadata、PU 指标和
  可选监督指标，支持严格 JSON 与 Markdown 输出。
- 假设敏感性：`analyze_pu_sensitivity` 固定模型输出，以 $`P(S=1)=\pi\bar c`$
  检查类先验/平均标记倾向网格的相容性，并导出指标区间、JSON、Markdown 与 CSV；
  不承担 propensity 识别或逐假设模型重训。
