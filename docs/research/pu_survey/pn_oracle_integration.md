# PN oracle 接入实施计划

> 定位：修复并接入协议 §2.4 第 10 条的 PN oracle 对照路径。
> 上游依据：[pu_survey_protocol.md](pu_survey_protocol.md) §2.4/§5、
> [implementation_plan.md](implementation_plan.md) §7 验收清单、
> [survey_execution_plan.md](survey_execution_plan.md) P2；
> 参考文献 2 = PU-Bench `2d95a19`（与本项目其他锁定值同一 commit）。
> 状态日期：2026-09-11。

## 1. 问题

**P0 声称已实现的 PN oracle 路径实际不工作，且会静默产出错误的"上界"数值。**

`docs/dev/experiment_layer.md:43-45` 把「PN oracle（`SupervisedTrainer` +
`protocols=[ProtocolOA()]`，调用方显式传递）」列在「P0 已实现」清单里；
但按该用法接入 runner 后，训练用的**不是**真实标签。

### 1.1 代码路径证据

`ExperimentRunner.fit`（`pu_toolbox/experiment/runner.py`）第一步就生成 PU 视图：

```python
# runner.py:87-89
y_pu_train, meta_train = self.generator.generate(train.X, train.labels, c, self.seed)
train_pu = DatasetPart(X=train.X, labels=y_pu_train, view="pu", indices=train.indices)
# runner.py:119
trajectory = self._train(est, train_pu, pu_val_pu)
# runner.py:283-292
def _train(self, est, train_pu, pu_val_view):
    trainer = self.config.get("trainer", DeepFitTrainer())
    return trainer.fit(est, train_pu.X, train_pu.labels, ...)   # ← PU 标签
```

`SupervisedTrainer`（`strategies.py:286-292`）只做转调：

```python
def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
    return FitTrainer().fit(estimator, X, y, class_prior=class_prior)
```

它不做任何标签替换——其 docstring 写 "only the label view differs from a
SCAR/SAR run"，即**设计意图是调用方喂它不同的 label view**，而 runner
没有提供这条路径。结果是训练出一个"把未标记样本当负类"的朴素监督模型。

### 1.2 最小复现实证（2026-09-11）

构造 200 样本（train 120，含 51 个真正例）、`c=0.1`、`protocols=[ProtocolOA()]`、
`config={"trainer": SupervisedTrainer()}`，用记录型 trainer 捕获 runner 实际传入的标签：

```
true positives in train split        : 51
SCAR PU view would label about       : 5   (c=0.1)
labels actually handed to the oracle : n_labeled_positive = 5
manifest.generation                  : {'train': {'mechanism': 'scar', 'c_realized': 0.098, ...}}
→ BROKEN: oracle trained on the PU view, not true labels
```

即 **46 个真正例被当作负类**参与"oracle"训练；同时印证差距 4（manifest 记了 SCAR 元数据）。
该复现在接入修复后应转为回归测试的断言基础。

### 1.3 偏差量级

`c` 定义为真实正例中被标注的比例（`strategies.py:30-34`：`n_labeled = round(c·n₊)`），
故 U 集合中残留真正例比例为 `(1−c)·π/(1−c·π)`。取 π_train = 0.3：

| c | U 中被当作负类的真正例占比 |
|---|---|
| 0.1 | ≈ 27.8% |
| 0.3 | ≈ 23.1% |
| 0.5 | ≈ 17.6% |

### 1.4 为什么逃逸了测试

`tests/unit/experiment/test_trainers.py:41-46` 有测试
`test_supervised_trainer_uses_true_labels`，但它**直接调用 trainer**
（`SupervisedTrainer().fit(est, X, y)`，y 为真实标签），断言仅
`len(traj.epochs) >= 1`——测试的是 trainer 单元行为，恰好绕过了 runner
的标签源问题。**没有任何测试经过 runner 验证 oracle 路径**。

## 2. 现状差距清单

| # | 差距 | 位置 | 后果 |
|---|---|---|---|
| 1 | 标签源：runner 恒传 PU 视图 | `runner.py:87-89, 283-292` | **核心缺陷**：oracle 训练标签错误 |
| 2 | 默认 protocols 含 `ProtocolPA()` | `runner.py:50` | 违反协议"不得伪造 PA 结果" |
| 3 | pu_val 的生成与正例校验对 oracle 无意义却仍执行 | `runner.py:88, 92-96` | 低 c / pu_val 无正例时误报失败 |
| 4 | `generation` 字段写入 SCAR 元数据 | `runner.py:266` | 留痕污染：复核者会误读 oracle 的生成机制 |
| 5 | `class_prior` 沿链路下传 | `runner.py:290` | oracle 不需要 π；部分模型会接受并误用 |
| 6 | 深度路径 checkpoint 以 `val_pu`（PU risk）选择 | `strategies.py:312-350` | 违反"仅以 clean_val 选模"（图像端到端路径） |
| 7 | 脚本层无 oracle 入口 | `scripts/run_survey_experiment.py:112` | 跑批无法启动 |

## 3. 参考文献 2（PU-Bench `2d95a19`）口径对照

已 clone 源码核实（`train/pn_trainer.py`、`config/methods/pn.yaml`、
`train/base/` 等）。PU-Bench 的全监督基线叫 **`pn`**（`PNTrainer`），
README 称其为 "fully supervised PN oracle baseline"。

| 维度 | PU-Bench `pn` | 本协议 §2.4 第 10 条 | 判定 |
|---|---|---|---|
| 训练数据 | 同一底层 train 分区（`split_source_train_val` → PU 采样），训练循环取 `true_labels`（`pn_trainer.py:64`），损失 `BCEWithLogitsLoss` | 同 | ✅ 一致 |
| **选模** | `monitor: val_proxy_acc`（**PU-only proxy accuracy**，`config/methods/pn.yaml:19`，patience 5） | **仅以 clean_val 的真实 Accuracy 选模** | ⚠️ **实质分歧** |
| 分类器 bias 初始化 | `pn` 不在 `SOURCE_FAITHFUL_NO_BIAS_INIT` 内 → 末层 bias 用 PU 先验 `pi_unlabeled` 初始化（`train/base/data_model.py:71-97`） | 无规定 | ⚠️ 我方未覆盖 |
| 候选池/调参 | `pn.yaml` 为扁平 fallback 配方，**不参与任何超参搜索** | 同一候选预算 | ⚠️ 我方更严 |
| backbone | PN 不覆写 `create_model`，走 `select_public_model`；但 6 个 PU 方法用私有模型族，**代码无"强制同 backbone"断言** | 同一表征/backbone | ⚠️ 我方更严 |
| seed 数 | 10 个（`datasets_typical/*.yaml:3`） | 5 个 | 差异（我方按协议） |
| 报告口径 | 无 "OA only" 字面；但 **test split 只出 oracle 指标**（proxy 的 Test 列打印 `"--"`，`epoch_loop.py:168-177`），PN 在 train/val 仍算 proxy | "标为 `PN oracle (OA only)`，不得伪造 PA 结果" | ✅ 语义一致，我方措辞更显式 |
| 资源计量 | 同 `finalize()` 路径：`duration_seconds`/`time_to_best_seconds`/`max_gpu_memory_bytes` | 协议 §5.4 | ✅ 一致 |
| 结果聚合 | **仓库内无聚合代码**（无 mean±std 脚本，聚合在论文侧） | 协议 §5.2 要求五次均值±标准差 | 我方需自建 |

### 3.1 分歧的处理建议

**选模口径**（唯一实质分歧）。PU-Bench 的 `pn` 严格说是"用真标签训练、
但用 PU proxy 选 checkpoint"的半监督式参照；本协议要求的是**纯监督上界**
（真实 Accuracy 选模）。

建议：**保留协议口径**（它才是"上界"的正确语义，且协议是项目内更高权威），
但必须做两件事：

1. 在 `implementation_plan.md` 记录该差异，注明**本实验 PN oracle 数值不可与
   PU-Bench 论文表中的 `pn` 直接对比**（选模口径不同）；
2. 在 oracle 的 manifest / 结果标注中写死 `selection_metric: "clean_val_accuracy"`，
   使口径可审计（PU-Bench 侧对应值是 `val_proxy_acc`）。

**bias 初始化**（我方未覆盖）。建议沿用本工具箱默认（不做 PU 先验 bias 初始化）
并在台账/manifest 注明——理由：本实验的 oracle 定位是纯监督上界，引入 PU 先验
会使其不再是上界。

## 4. 设计约束（不可破坏的既有不变量）

- `bundle.py:54-77` `validate_bundle`：四路必须 `view="clean"`；索引两两不重叠；
  `test.for_selection=False`
- `strategies.py:236` `ProtocolPA.select`：`val_part.view != "pu"` 时 `raise ValueError`
  —— PA **结构性**拿不到真实标签，由 `test_pa_never_receives_clean_labels`
  （`tests/unit/experiment/test_runner.py:51`）守护
- `experiment_layer.md:22` D4 决策：Generate 阶段产 PU 视图，Trainer/PA 路径结构性接收不到真实标签

**推论**：oracle 需要"用真实标签训练"这条路径必须**显式声明**，不能靠隐式复用
PU 通道——否则 D4 不变量在审计上失效。

**附带约束**：`feature_adapter.py:160` `partition_fair_leaderboard_runs` 已能强制
同数据集内 `split_sha256`/`seeds`/`max_epochs`/`batch_size_candidates`/
`tuning_candidate_count`/`representation_sha256` 一致——协议"同一表征、backbone、
split、候选预算与五个 seed"有现成执行机构，oracle 只需正确构造 `LeaderboardRunSpec`。

## 5. 设计决策（2026-09-11 已拍板）

### D-A 接入形态

| 方案 | 机制 | 评价 |
|---|---|---|
| **A1（推荐）** | 新增 `CleanLabelGenerator`（恒等策略：真实标签原样作为 view 返回），`Generator` 增加 `output_view` 类属性（默认 `"pu"`），runner 尊重之 | 复用 `Generator` 变化点；**PA 防护免费继承**（若误配 PA，`view!="pu"` 直接 fail-loud）；改动集中在 `runner.py:89-90` 两行 |
| A2 | runner 增加 `config["train_view"]="clean"` 分支 | 显式但需在 `fit` 内多处判断（生成/校验/manifest） |
| A3 | 独立 `fit_oracle()` 方法 | 重复整条管线，违反 DRY；协议要求"与 PU 运行相同的底层 train 分区"，共享管线是设计意图 |

推荐 **A1**。需同步把 `Generator` 的 ABC 契约文字从 "clean labels -> PU label view"
扩为 "clean labels -> label view"，并由 oracle 这个正当用例显式驱动。

> **决策（2026-09-11）：采用 A1 —— `CleanLabelGenerator` + `Generator.output_view`。**

### D-B 深度路径（图像）范围

协议 §2.5 要求图像统一 ResNet-18，故 CIFAR-10 上的 oracle 需端到端 CNN。
`DeepFitTrainer`（`strategies.py:312-350`）优先用 `pu_validation_data` 并以
`history_["val_risk"]` 的 **argmin** 选 `best_epoch`——对 oracle 必须改为用
`validation_data`（clean,`self_pu.py:361`）并按真实 accuracy 取 argmax。

- **B1（推荐）**：分两阶段。Phase 1 交付 MLP 路径（Spambase/IMDB，P2 的 2/3 数据集）；
  Phase 2 交付 CNN 路径 + oracle 专用深度 trainer（不阻塞 P2 启动）
- B2：一次性做完（P2 启动被图像路径阻塞）
- B3：图像 oracle 走现有 PU-risk 选模 —— **不建议**，违反协议选模要件

> **决策（2026-09-11）：采用 B1 —— Phase 1 交付 MLP 路径（Spambase/IMDB）不阻塞 P2 启动，
> Phase 2 补 CNN 路径与 oracle 专用深度 trainer。**

### D-C 跑批次数

oracle 用真实标签训练，**结果对 c 恒定**（c 只影响标记）。协议 §2.4 第 4 条要求
"同 seed 下所有方法、PA/OA、PN oracle 及所有 c 共享同一底层 split"——与"按 c 重复跑
oracle"并不冲突，只是冗余。

- **C1（推荐）**：每个 (dataset, seed) 跑 1 次，共 3×5 = **15 次**，聚合时广播到各 c 列，
  并在结果标注 `c_independent: true`
- C2：按 c 各跑一次（45 次），结构简单但浪费 3 倍 GPU 时间

> **决策（2026-09-11）：采用 C1 —— 每 (dataset, seed) 跑 1 次共 15 次，聚合时广播到各 c 列。**

### D-D 与 PU-Bench 数值并列展示

建议榜单中 PN oracle 单独一组（协议 §5.1 已如此规定），并注明选模口径差异，
不与 PU-Bench 论文表直接对比。

## 6. 实施步骤（TDD；每步独立可测）

> 分支：`feature/pn-oracle`。每步先写失败测试，再实现，再跑门禁。

### Task 1 — Generator 视图契约扩展（D-A）

- 测试 `tests/unit/experiment/test_strategies_labeling.py`：
  `CleanLabelGenerator().generate(X, y_true, c, seed)` 返回的标签等于 `y_true`、
  meta 含 `mechanism == "pn_oracle"` 且 `c_realized == 1.0`；
  默认 `Generator.output_view == "pu"`，`CleanLabelGenerator.output_view == "clean"`
- 实现 `strategies.py`：加 `CleanLabelGenerator` + `Generator.output_view` 类属性
- 改 `runner.py:89-90`：`view=getattr(self.generator, "output_view", "pu")`
- 验证：现有 `test_pa_never_receives_clean_labels` 等全部保持绿

### Task 2 — oracle 端到端集成测试（守护核心缺陷）

- 新增集成测试 `test_runner_oracle.py`（置于 `tests/unit/experiment/` 目录）：
  1. **核心断言**：用记录型 trainer 捕获 runner 传入的标签，
     断言标记正例数 == train 真实正例数（当前 1.2 节复现的反面）
  2. 断言 `result.test_metrics` 只含 `{"OA"}`，无 `PA`
  3. 断言 `manifest["generation"]["train"]["mechanism"] == "pn_oracle"`（差距 4）
  4. 断言 pu_val 无真实正例时 oracle 仍能跑通（差距 3 的条件化校验）
  5. 断言显式传入 `protocols=[ProtocolPA()]` 时 fail-loud（防护不变量）
- 该文件需登记 `docs/dev/project_structure.md`（Rule 2 门禁）

### Task 3 — runner 侧 oracle 适配

- `runner.py:92-96`：正例校验加条件（仅 PU 视图路径执行）
- `runner.py`：oracle 路径下不向 trainer 传 `class_prior`（或由脚本传 `None` 并在
  runner 记录 `class_prior_applied: false`）
- `runner.py:266`：`generation` 字段写入 generator 的 meta（A1 下自动正确）
- 补测：`test_runner_oracle.py` 增断言

### Task 4 — 脚本层入口

- `scripts/run_survey_experiment.py`：新增 `--oracle` 开关
  （自动 `CleanLabelGenerator` + `protocols=[ProtocolOA()]` + `class_prior=None`
  + `config["trainer"]=SupervisedTrainer()`），并在输出目录写
  `oracle_integration.json`（选模口径、候选池、seed 数）
- 测试扩展 `tests/unit/experiment/test_survey_script.py`（现有 3 例 + oracle 例）
- 冒烟：`data/splits/spambase/split_0` 上跑一次，人工核对 oracle accuracy
  **显著高于**同条件 PU 方法（行为级证据）

### Task 5 — 深度路径（Phase 2，D-B）

- 新增 oracle 深度 trainer（用 `validation_data` + clean accuracy argmax 选 checkpoint）
- 测试：`tests/unit/experiment/test_trainers.py` 增 oracle 深度用例
- CNN 冒烟（T600 或 CPU 小规模）

### Task 6 — 文档与留痕

- `docs/dev/experiment_layer.md:43-45`：修正"P0 已实现"表述为真实状态 + 接入方式
- `docs/research/pu_survey/survey_execution_plan.md:13`：修正 oracle 现状描述；
  §2 P2 条目补 oracle 跑批口径（15 次 + 广播）
- `docs/research/pu_survey/implementation_plan.md`：记录 §3.1 的两处口径差异
- `docs/user/reference/api.md:1100`：`SupervisedTrainer` 条目补接入约束
- `docs/README.md`：登记本文件

## 7. 验收标准

1. §1.2 复现脚本在修复后输出 `OK: oracle received true labels`
2. 新增集成测试能**在旧代码上失败、在新代码上通过**（否则等于没守护）
3. 既有实验层测试全绿，尤其 `test_pa_never_receives_clean_labels` 与
   `test_runner_raises_when_pu_view_has_no_labeled_positive`
4. 脚本层 `--oracle` 在 Spambase/IMDB 上产出 OA-only 结果，accuracy 高于
   同条件 PU 方法（行为级证据，人工核对后记录）
5. 门禁：`check_test_quality` / `check_doc_links` / `check_api_docs` /
   `check_format` / `check_project_metadata` / `generate_structure --check`

## 8. 与 P2 跑批的接口

- 交付物：`--oracle` 入口 + 跑批清单（dataset × seed 共 15 条）+ 结果目录约定
- 执行方：HENG958（GPU 窗口）；榜单聚合由 shuidisjtu 的聚合脚本消费 oracle 结果
- 公平性：聚合时构造 `LeaderboardRunSpec(method="pn_oracle", ...)` 并过
  `partition_fair_leaderboard_runs`，与同数据集同训练路径的 PU 方法同组校验

## 9. 开放问题

1. ~~D-A / D-B / D-C 待拍板~~ —— 已于 2026-09-11 拍板，见 §5
2. 图像数据集上 oracle 的训练路径分组（端到端 vs feature-adapter）需与协议 §5.1
   的"训练路径分层"对齐—— oracle 是否需要在两组各出一条？（Phase 2 实施前决策）
3. 并列展示的口径差异说明是否要写进最终榜单脚注（面向论文读者）
4. 协议 §2.4 第 10 条的选模口径（clean_val 真实 Accuracy）与参考文献 2 实际做法
   （`val_proxy_acc`）的分歧，是否与学长确认过原意（见 §3.1）
