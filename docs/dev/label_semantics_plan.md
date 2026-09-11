# 标签语义契约：能力声明、检查点与实施计划

> 定位：根治"PU 标签与真实标签数值同形、语义相反"导致的**静默错配训练**。
> **阶段归属：P3 算法接入的前置项**（与"数据集内共享 backbone 规格"并列，同属接入时的接口约定），
> 本轮不实施——实验关键路径不受影响，理由见 §1.3。
> 上游依据：[pu_survey_protocol.md](../research/pu_survey/pu_survey_protocol.md) §2.4 第 10 条、
> [pn_oracle_integration.md](../research/pu_survey/pn_oracle_integration.md) §8 第 5 项
> （独立验收判定为未决）、项目 CLAUDE.md「API 契约」条。
> 状态日期：2026-09-11。

## 1. 问题

分类器的 `y` 语义是**约定**而非**可检查的接口**：项目契约只写"标签语义由分类器决定
（PU 为 `{+1, 0}`，PNU 为 `{+1, -1, 0}`）"，而 PU 的 `{1, 0}` 与监督 PN 的 `{0, 1}`
**是同一组数值**——`0` 在一处是"未标记"、在另一处是"真实负类"。两者都通过现有校验
（`normalize_pu_labels` 只判取值合法），于是"把真实标签喂给 PU 估计器"或反之，
都会训练出一个目标函数错配的模型，且**不报错**。

这不是假想：项目自己的防泄漏守卫已承认该不可判定性——
`benchmarks/traditional_pu/leakage_audit.py` 的 `guard_fit_labels` 明确按对象身份/内存
共享判定并写明"`label_frequency=1.0` legitimately makes `y_pu == y_true` in value"，
拒绝用值相等判定。即：**值层无法区分，只能在意图层声明。**

### 1.1 最小复现（2026-09-11，main 上实测）

```python
runner = ExperimentRunner(
    generator=CleanLabelGenerator(),          # clean 视图（真实标签）
    protocols=[ProtocolOA()],
    config={"trainer": SupervisedTrainer()},  # 声明 trains_on_real_labels=True
)
runner.fit(NonNegativePUClassifier(class_prior=0.3), train, pu_val, clean_val, test)
```

```
完成: {'OA': {'accuracy': 0.25, 'auc': 0.0}}
manifest.generation.train.mechanism = pn_oracle
failures = []
```

三条既有守卫全部放行（视图↔trainer 声明一致、runner 层无 `class_prior`、生成器自述
一致），真实标签被交给 nnPU：其损失把 `y=0` 读作"未标记"，`π` 被安在实际上是"全部负类"
的集合上。产出的模型既不是上界也不反映任何东西，却被记为 `pn_oracle`。

### 1.2 与已修缺陷的关系

PN oracle 的两轮修复解决的是**视图接线**（标签来源错、视图与 trainer 声明不一致、
生成器自述与声明不一致）。本条解决的是**估计器目标**：标签送对了，但估计器拿它算的目标
是错的。两者同属"假上界"缺陷族，层次不同，守卫载体也不同（前者在 Trainer 声明，
后者必须在估计器一侧）。

### 1.3 阶段归属（2026-09-11 定）

**该改动不是实验的必需项。** 按实验实际执行路径核可达性：

| 入口 | 语义组合 | 结论 |
|---|---|---|
| `--method <PU 方法>` + SCAR/SAR | PU 视图 → PU 估计器 | 由构造保证一致（生成器固定） |
| `--oracle` | clean 视图 → `SupervisedTrainer` → `OracleMLP`（监督） | 脚本内写死，配不错 |
| 误跑三值方法 `pnu` | PU 视图 → PNU 估计器 | 不静默：其校验要求三值齐备，直接 raise |
| 手工用 runner API 组合策略 | clean 视图 → PU 估计器 | **可达且静默**（F2） |

即：只要实验经示例脚本跑，本缺陷不可达；pilot 与主榜跑批均不受影响。它真正会咬人的时机是
**Phase 2 的 CNN oracle**（复用 PU 方法的编码器/估计器家族配监督头，正是"同一估计器换语义"
的场景）与**绕过脚本直接用 runner API 组装策略**。

因此归属 **P3 接入前置**——P3 本就要逐个方法过声明，边际成本最低；Phase 2 若先启动则随之提前。

## 2. 现状

| 项 | 现状 |
|---|---|
| 注册算法数 | 17 条（全部绑定类）；语义分布：16 个 PU、`pnu` 三值、`self_pu` 主输入 PU 而其 `validation_data` 参数为监督语义 |
| fit 内校验 | 一律为**取值合法性 / 组数非空**，无一处校验语义；不匹配时静默（唯 `pnu` 因"三值齐备"要求间接 raise） |
| 语义的记录位置 | 仅 docstring 与文档文字（`api.md` 契约段、`new_algorithm_template.md` 第 1 条）；无字段、无门禁、无测试 |
| 检查点 | experiment 层已有视图/声明守卫；pipeline 与 comparison 等入口无任何语义检查 |
| 可复用先例 | 能力声明字段 `native_architectures`/`input_ndims`/`encoder_parameter`/`trains_encoder`（类属性权威 + registry 同步 + 契约测试门禁，见 `core/base.py`、`registry/metadata.py`、`registry/registry.py` 的 `_SYNC_FIELDS`、`tests/contract/test_capability_declarations.py`） |
| 互补守卫 | `guard_fit_labels`（身份/内存层防泄漏）与本次（意图层防语义错配）判据不同，均不可省 |

### 2.1 参考文献 2（PU-Bench `2d95a19`）的做法

已 clone 源码核实（PU-Bench 侧的 `data/data_utils.py`、`train/pn_trainer.py`、
`train/base_trainer.py`、`train/base/data_model.py`、`config/methods/*.yaml`、
其 `check_arch_contracts.py` 与 `test_method_metadata_invariants.py`）。

| 维度 | PU-Bench 的做法 |
|---|---|
| 数据承载 | 单一数据集对象**同时**携带 `pu_labels` 与 `true_labels`，二者同长度同 dtype，靠 `__getitem__` 的**位置下标**区分；`test` 分区的 `pu_labels` 字段**直接填入真实标签**（`data_utils.py:966-972`）——即同一字段名在不同分区承载不同语义 |
| 语义分派 | 监督基线 `pn` 是**独立 trainer 类**（`train/pn_trainer.py`，与各 PU 方法平级），共享同一数据管线与 `epoch_loop`；语义差异只体现为取 batch 的下标（`pn` 取 `[2]`，PU 取 `[1]`） |
| 声明机制 | **没有**标签语义声明。唯一的 `label_scheme` 元数据因合并顺序被 dataset 配置覆盖（`experiment_plan.py:302-309`），仅通过静态检查的 token 存在性判定（`check_arch_contracts.py:905-907`），实际**从未被消费**——死字段 |
| 运行时校验 | PU trainer 各自复制一份 `_validate_*_data_contract`（校验取值编码与先验）；`pn` **完全没有**校验 |
| 防护实质 | 结构分离 + 位置/命名约定 + 取值级局部校验；**对用户误切换无防护**（`pn` 与 PU 走同一 loader，改一个 `--methods` 即换语义，无任何拦截） |

**对本次方案的直接含义**：

1. 同类错配在参考文献中**同样不可检测**——我们遇到的不是实现事故，而是该问题域的固有难点；
   差别只在于我们把它暴露成了用户可注入的策略组合（Generator × Trainer × 估计器），
   而他们把差异写进了类与下标
2. **A 路线无先例，但有一条硬约束**：他们的 `label_scheme` 证明"声明若没有强制读点，就会
   退化为死元数据"。故本方案的声明位必须**被守卫真实消费**（D2 的两处检查即读点），
   并由契约测试锁定——否则会精确重演 `pn.yaml` 的 `label_scheme`
3. 可借鉴的最小结构：其"元数据驱动 + 静态禁令 + 非默认取值的行为用例"
   （`check_arch_contracts.py` 用 `pu_labeled_label=7 / pu_unlabeled_label=-3` 的假数据
   验证实现确实从声明取值）——本方案 P2 的回归用例同理：用**非默认语义**的估计器驱动检查
4. B 路线的防护上限在他们的实践中已被证伪：中央注册表 + fail-fast 只保证"内部一致"，
   对使用者误用零防护

## 3. 设计决策

**D1 声明位**。在分类器契约上新增类属性：

```python
label_semantics: str = "pu"     # 取值域 {"pu", "pn", "pnu"}，声明 fit(X, y) 中 y 的语义
```

- **单值而非集合**：`fit` 的 `y` 只有一种语义，集合会弱化守卫（"两种都收"等于放行两类错配）
- **默认 `"pu"`**（与 `BasePUClassifier` 的定位一致）：危险方向仍是 fail-closed——clean
  视图要求显式 `"pn"`，未声明或声明 `"pu"` 的估计器一律拒绝；PU 流水线方向保持向后兼容，
  第三方自定义分类器不会因为新增字段而突然不可用
- **注册方法必须显式声明**（不靠默认），由契约测试门禁：现有 17 条逐一标注，
  新增算法漏声明即失败（沿 `native_architectures` 的做法）

**D2 检查点**。共享检查函数放 `core/validation.py`（与既有 `validate_pu_X_y` 等同层），
签名 `validate_label_semantics(estimator, expected)`，`expected ∈ {"pu", "pn", "pnu"}`，
不匹配时 raise 并给出该估计器的声明值。落点两处：

1. `pu_toolbox/experiment/runner.py`：`_validate_model_capability` 已做"模型 × bundle"预检，
   把 generator 声明的视图语义（`pu`→`"pu"`，`clean`→`"pn"`）传进去即可，逐候选生效
2. `pu_toolbox/workflows/`：沿 `_models.py` 的 `check_architecture_capability` 模式，
   在 pipeline 与 comparison 的公共入口调用（该层用户 `y` 按契约是 PU）

**D3 与 `Trainer.trains_on_real_labels` 的分工**（保留两者，不合并）：
trainer 声明"把哪套标签交给 loader、以及按什么指标选 checkpoint"，估计器声明"fit 期望
什么语义"。前者能拦 `DeepFitTrainer`（PU-risk 选 checkpoint）配 clean 视图这类
**训练机理**错配，后者能拦标签语义错配——互不替代。

**D4 `self_pu` 的双语义**：只声明**主输入** `y_pu` 的语义（`"pu"`）；`validation_data`
的监督语义留在 docstring 与参数文档里，不进声明位——否则声明为 `{"pu","pn"}` 等于对
clean 视图放行，正是要封的路径。

**D5 与 `guard_fit_labels` 互补**：前者按对象身份判"是否泄漏"，后者按声明判"语义是否
匹配"。不合并、不互相替代。

**D6 声明必须有强制读点**（参考文献 2 的直接教训）：PU-Bench 的 `label_scheme` 元数据因为
没有任何代码消费它，退化为死声明——只有 token 存在性检查，声明与行为早已脱节。故本方案的
`label_semantics` 必须由守卫**实际读取并据此 fail-loud**（D2 两处检查点即读点），契约测试
同时锁定"全部注册方法显式声明"；禁止出现"只写不读"的声明位。

## 4. 实施阶段（TDD；每阶段独立可验收）

**时机**：P3 算法接入前置（见 §1.3）。P1 可与方法接入并行；P2 须在 Phase 2 深度 oracle
之前完成；P3/P4 无时间压力。

**P1 —— 声明位与门禁（无行为变化）**
1. `core/base.py` 加类属性与取值域 docstring；`registry/metadata.py` 加字段；
   `registry/registry.py` 的 `_SYNC_FIELDS` 加字段名（漏加则注册表同步测试失败）
2. 回填 17 条注册算法的显式声明（`pnu` → `"pnu"`，其余 `"pu"`）
3. 契约测试：取值合法 + 全部注册方法显式声明 + 注册表同步（沿现有断言模式）
4. 模板与文档：`new_algorithm_template.md` 能力表增列；`docs/user/reference/api.md` 契约段
   改为指向字段

**P2 —— experiment 层检查（修复 F2）**
1. 先写失败测试：§1.1 复现转成回归用例（clean 视图 + `SupervisedTrainer` + nnPU →
   fail-loud；镜像方向 PU 视图 + 声明 `"pn"` 的估计器 → fail-loud）
2. `core/validation.py` 实现检查函数；`runner._validate_model_capability` 接入
3. 回退验证：该用例必须在修复前代码上失败

**P3 —— pipeline 层检查**
1. pipeline 与 comparison 入口接入（覆盖 CLI / UI / 调优 / 敏感度分析路径）
2. `cli/info.py` 的 list-methods 能力列展示该字段；UI 参数面板按需提示
3. 测试：错语义估计器在入口被拒

**P4 —— 文档与收口**
1. ADR（决策：用声明位而非运行期校验，理由见 §1 的值层不可判定性）
2. `docs/README.md` 索引登记本文件；`pn_oracle_integration.md` §8 第 5 项标记为该方案收口
3. 项目 CLAUDE.md「API 契约」条补一句：语义须由 `label_semantics` 声明

建议：全部在一个 `feature/label-semantics` 分支内按阶段提交，单个 PR。

## 5. 验收标准

1. §1.1 的复现路径在 P2 后 fail-loud（且该用例在 P2 前的代码上失败）
2. 全部注册方法显式声明且取值合法；新算法漏声明导致契约测试失败
3. 既有守卫不回退（视图↔trainer 双向、生成器自述一致性等用例保持绿）
4. pipeline 入口对错语义估计器 fail-loud，且不与 `guard_fit_labels` 重复判定
5. 八项门禁与结构文档检查通过

## 6. 边界、风险与取舍

- **不可判定边界**：值层永远无法区分 PU 视图与真实标签（`c=1.0` 时二者数值相同）。
  本方案只保证"**已声明的估计器**不会被错配使用"；声明失真（无论有意无意）不在检测范围内
- **兼容性**：默认 `"pu"` 使第三方自定义分类器在 PU 流水线中保持可用；代价是监督估计器
  若忘记声明，会在 clean 视图路径被拒（错误信息须直接给出应声明的取值）
- **误报面**：语义位与"训练机理"无关——`self_pu` 等混合估计器按主输入声明（D4），
  其内部辅助路径不参与判定
- **范围**：`benchmarks/` 路径已有身份层守卫，本次仅保证与其互补；是否也在 benchmarks
  内部统一调用新检查，见 §7
- **声明腐化**：声明位若失去强制读点（重构中被绕过或改成"仅记录"），会退化为参考文献 2
  的 `label_scheme` 式死元数据。缓解即 D6：读点写进验收标准（§5 第 1、4 条），
  并在 P4 的文档收口里明确"只写不读的声明位不允许存在"

## 7. 开放问题

1. `benchmarks/` 与 `diagnostics/` 内部的估计器构造点是否统一接入检查函数（当前仅
   pipeline/comparison/experiment 三处）
2. 是否需要 `requires_clean_labels` 之类的派生 property（沿 `is_tabular_only` 模式），
   供 UI/CLI 直接展示"该算法可用哪些数据角色"
3. 与 P3 的"数据集内共享 backbone 规格"是否共用同一份能力声明表（两者都描述
   "接入某个数据集实验时的接口约定"）
