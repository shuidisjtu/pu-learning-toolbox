# 从原始数据到汇总表：命令序列

> 一页纸：把 `data/raw` 变成交付物 A 的工作簿要跑哪几条命令、每一步留下什么凭证、复现一个已交付结果从哪开始读。
> 实验口径以[协议](pu_survey_protocol.md)和[执行计划](survey_execution_plan.md)为准，本页只给命令和产物。

所有命令都在仓库根执行，Python 一律走 `uv run`。下文用 `<results>` 表示 `run_survey_pilot.py --results` 指的那个目录。

## 0. 先确认矩阵与注册表没变

```bash
uv run python scripts/check_survey_recipe_registry.py --require-present --require-locked
uv run python scripts/check_survey_profile_map.py --require-present
```

两条都是 CI 门禁，跑批前手动跑一次，是为了把"矩阵和配方注册表对不上"挡在第一批之前。

## 1. 四路 split（协议 §2.4 第 3 条）

```bash
uv run python scripts/prepare_survey_splits.py --datasets spambase,cifar10,imdb --seeds 0,1,2,3,4
```

产物 `data/splits/<dataset>/split_<seed>/`：`train.npz` / `pu_val.npz` / `clean_val.npz` / `test.npz` 加 `split_manifest.json`。
manifest 里的 `indices_sha256` 是 run manifest 的 `split_ref` 指向的对象，`class_prior.population` 是驱动的先验来源——缺这条记录的 split 会让驱动在开跑前拒跑，而不是在第一百批才发现。

## 2. 五角色 support 制品（只有 clean-support 三方法需要）

```bash
uv run python scripts/prepare_clean_support.py --dataset spambase --raw-dir data/raw \
  --seed 0 --support-size 100 --support-seed 0 --group-policy not_applicable \
  --out-dir data/clean-support-20261024-u5axe92t/spambase/split_0 --write
```

每个 (dataset, seed) 一条，`--out-dir` 必须是不存在的新目录，不加 `--write` 只预览。
support 规模与 support seed 记在 `data/clean_support_materialization_20261024.json`：spambase 100，connect_4 / mnist / fashion_mnist / cifar10 各 500，support seed 等于 split seed。

## 3. 计划与磁盘预算（不训练、不写证据包）

```bash
uv run python scripts/run_survey_pilot.py \
  --datasets mnist --support-root /root/autodl-tmp/clean-support \
  --splits data/splits --results <results> --device cuda \
  --adapter-cache /root/autodl-tmp/cache --reclaim-unselected-checkpoints \
  --class-prior mnist=0.4915333333333333 --dry-run
```

`--dry-run` 打印 planned / completed / pending、还要跑几个批、checkpoint 峰值，以及先验读到了什么；`--plan-json <file>` 把展开后的每个 run 写成文件，两台机器分片时用它核对两片不重叠。
计划里有 support 单元却没给 `--support-root` 时，驱动在任何批开始前报错并点名；这时要么补上制品，要么用 `--methods` 把 support 方法排除出去。

### 子批次与续跑接收边界

2026-10-28 用户确认完整 seed × c 网格可以由多次合规子批次组成；分批本身不是实质偏离。
审计对历史 `seed_subset` / `c_grid` 仅在完整无重复网格、每份引用包可核验、代码与依赖、
软硬件及线程环境一致且其他正式门禁均通过时，输出派生重裁定；原始 manifest 不修改。
新批次包记录 CPU 型号和数量、GPU 驱动、软件版本及线程环境。缺字段的旧包不能仅凭相同的
已有字段自动认定环境一致。该重裁定目前适用于已有 clean-support 重审范围；原始单 run
正式聚合入口不消费派生裁定，应使用审计和汇总报告，不能直接宣称原 manifest 已被改写。

完整的 nnPU / Dist-PU `native_2d` 批次也可通过审计生成派生准入裁定：核对冻结参数与预算、
完整逐 epoch 候选及对应选模验证记录，Dist-PU 还核对冻结 clean-room fullbatch 路径的实际
epoch 和更新步数。SAR 只要求 OA 验证，SCAR 同时要求 PA / OA；其他路径不因此自动解锁。
只有完整网格及所需审计均通过，才解除已登记的历史全局进度声明；未知或真实 blocker 保留。
审计配置须接入 `plan`、`run_log`，已有退出码通过 `exit_code_file` 接入；未保存的原进程
退出码保持 unverified，不补造。分开保存的机制日志可按实际顺序拼接成注明来源的派生日志，
保留原始内容；审计同时识别标准驱动完成行与 `TRAINING_COMPLETE X/Y; formal admission requires audit`，
并逐条核对日志路径、重复项和实际产物数量，不能凭完成标记单独授予资格。

逐行 `formal` 资格与整份报告的 `overall` 分别解读。任何 `not_run` 检查仍使整体报告为
`partial`；例如未配置归档摘要时 A11 保持 `not_run`，不能将其描述为通过，也不自动否决
已经通过所需准入检查的完整结果行。

## 4. 正式批次

同一条命令去掉 `--dry-run`。分片用 `--datasets` / `--methods` / `--training-paths`，三者取交集；命名了一个矩阵里没有的东西、或交集为空，会被拒绝而不是"缩小成空"。

版本化 support 批次依协议记录所有已完成 epoch，包括 LaGAM / Self-PU 的 warmup；分别独立评估 PA+clean-support 与 OA，Self-PU 同时记录两条 teacher 流。旧探索入口的显式 meta-only 候选池保持原义。D53 之前已过滤 warmup 的批次不可补造缺失快照或回写证据，须以修复后的代码在新结果目录重跑。

协议 §2.4 已批准的独立 support 不构成自动降级理由：来源、角色隔离、规模与标签预算核验通过后，`PA+clean-support` 是正式实验数据的独立报告部分，与纯 PU PA 同表另列，禁止混入纯 PU 优势比较。准备器/adapter 的 `formal_admission=False` 表示该组件不自行授予整批准入，不能覆盖已批准的协议政策；最终资格由完整批次审计判定。旧 manifest 保持只读，过时资格声明应在新复审报告中记录解除依据，未知或真实 blocker 保留。

产物：

```text
<results>/<dataset>/<method>/<training_path>/<view>/[<机制>/]c_<token>/seed_<n>/manifest.json
<results>/<dataset>/<method>/<training_path>/<view>/[<机制>/]c_independent/seed_<n>/manifest.json   # PN oracle
<results>/batch_evidence/<dataset>__<method>__<training_path>__<机制或 oracle>__<标识前 12 位>.json
```

SAR 的 run 多一层 `<机制>/`（`sar_lbe_a` / `sar_lbe_b`），oracle 没有 c 轴、用 `c_independent`。

**每批一个证据包**（协议 §5 第 6 条，D44/D52）：一条 unit 脚本调用 = 一个批 = 一个包，包里有

- `code_commit` / `code_commit_dirty` / `code_commit_state`
- `dependency_lock`：对象 `{"path": "uv.lock", "sha256": ..., "state": "recorded"}`，不是裸哈希
- `protocol_version` / `protocol_sha256`、绑定了配方注册表时的 `recipe_registry`
- `environment`：Python / torch / CUDA 与 GPU 型号，外加 `state`（探测失败时是 `unavailable` + 原因）
- `unit` / `seeds` / `c_tokens` / `run_count`：这一批到底覆盖哪些 run

这一批写出的每份 manifest 带一个 `batch` 块引用它：

```json
"batch": {
  "schema_version": "batch-evidence-v1",
  "batch_id": "<包内容除自身外的严格摘要>",
  "evidence_file": "<包的文件名>",
  "evidence_sha256": "<文件字节的 sha256>"
}
```

三条性质值得记住：

- **按内容寻址，一个包只描述一次尝试**：`batch_id` 是包内容（去掉它自己）的严格摘要，文件名以同一摘要的前 12 位结尾。所以「同一个批」重跑一次（同 unit、同一组 seeds/c、同 commit、同 lock、工作区脏状态相同、同机器）会命中同一个文件、内容逐字节相同、直接复用；而下面这些情况都会**另写一个包，不覆盖旧的**：换了 commit、工作区从脏变干净、`uv.lock` 变了、换了机器/GPU（`environment` 在摘要里）、绑定的配方注册表变了、以及**部分完成后的续跑**——续跑会把没跑完的单元按 seed 拆成更小的批，那是另一个批、另一个包。没有任何 pending 时根本不写包。
  换句话说：`batch_id` 标识的是「这一次尝试」，不是「这个单元」；比较两组结果时看 unit / seeds / c，不要看 `batch_id`。
- **不许"借用"身份**：unit 脚本在建任何目录、训练任何候选之前会重读这个包，核对 dataset / method / training_path / 机制 / seeds / c / 协议摘要，还核对包里的 `code_commit` 与 `dependency_lock.sha256` 是否就是**当前这个 checkout** 的。对不上就停。所以 manifest 里的 `batch_id` 是被验证过的，不是从命令行抄来的。
- **包必须找得到**：manifest 只记包的文件名与字节摘要，包必须位于 `<results>/batch_evidence/`。`audit_survey_batches.py` 的 E01 检查会逐条核对「manifest 引用的包还在、字节没变」，对不上就报 finding；没有引用包的 manifest 只计数（D52 之前的历史 run 不回填）。

## 5. 聚合门禁（先诊断，不出表）

```bash
uv run python scripts/aggregate_survey_runs.py <results> --diagnostic
```

按 `comparability_group` 分组比较公平字段。一个组里混进两个配方（例如 backbone 宽度或 batch 不同）会让**整次聚合**失败并点名该组，这就是每个 support 方法单独一组的原因（D51）。

## 6. 汇总与导出

先写批次白名单（schema `survey-batch-roots-1`）：

```json
{
  "schema_version": "survey-batch-roots-1",
  "batches": [
    {"name": "B1", "root": "<results>", "expected_manifests": 3320, "role": "formal"}
  ]
}
```

`expected_manifests` 写这一批应有多少份 manifest，数量不符会被当成缺口报出来，不会静默通过。

```bash
uv run python scripts/audit_survey_batches.py --config batches.json --out-dir <out>/audit
uv run python scripts/summarize_survey_results.py --config batches.json --out-dir <out>/summary
uv run python scripts/export_survey_results.py --config batches.json --out-dir <out>/delivery
```

- `audit` 只读结果树，把每一条发现都列出来（不写回任何结果目录）。
- `summary` 按 (comparability group, view, mechanism, method, 选模协议, c) 出均值 ± 标准差；缺 seed 只报告、不填补。
- `export` 写出 `survey_results.xlsx` 与诊断卡；它不重算任何数字。

## 7. 复现一份已交付的结果

1. 打开该 run 的 `manifest.json`，读 `batch.evidence_file` 与 `batch.evidence_sha256`；
2. 在 `<results>/batch_evidence/` 找到同名文件，`sha256sum` 应等于 `evidence_sha256`（不等说明包被改过，run 引用的身份已不可信）；
3. `git checkout <code_commit>`。**先看 `code_commit_dirty`**：为 `true` 说明当时的代码含未提交改动，checkout 那个 commit 复现不出同一份字节，这时只能按 `code_commit` 的 parent 重建工作区，或者承认这份结果不可逐字节复现；
4. `uv sync`；`uv.lock` 的 sha256 应等于 `dependency_lock.sha256`（对象里的那个字段）；
5. 核对 `split_ref.indices_sha256` 与手上的 `data/splits` 是否是同一份 split（五角色单元则对制品收据）；
6. 重放这一批：照包的 `unit.seeds` 与 `unit.c_tokens` 直接调单元脚本，例如
   `uv run python scripts/run_survey_experiment.py <data_dir> --protocol survey-v3.0 --dataset <ds> --method <m> --training-path <p> --labeling-mechanism <机制> --seeds 0,1,2,3,4 --c 0.1,0.3,0.5 --class-prior <π> --out-dir <新目录>`。
   驱动 `run_survey_pilot.py` 没有按 seeds/c 过滤的开关，它的 `--plan-json` 只能核对计划，不能当重放入口。

换一台主机或换一块 GPU，`environment`（在 `batch_id` 里）会不同，所以会写出**另一个包**——这是对的：那是另一次尝试，不是同一个批的副本。可比的是 unit / seeds / c 与 `protocol_sha256`，不是 `batch_id`。
