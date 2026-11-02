# 复现 PU Survey 实验流程

> 前置条件：已按仓库说明安装环境（建议 `uv sync --python 3.11 --extra dev --extra torch`），
> 原始数据已放在 `data/raw/`，并在仓库根执行下面的命令。
> 概念：实验口径见[协议](pu_survey_protocol.md)与[执行计划](survey_execution_plan.md)；本文只讲
> 如何用现有脚本完成 split → support → 计划 → 跑批 → 审计 → 汇总 → 复现。

## 1. 流程一览

| 阶段 | 做什么 | 主要命令 | 产物 |
|---|---|---|---|
| 数据 | 把原始数据整理到 `data/raw/<dataset>/` | 手工准备 | 原始文件 |
| Split | 生成四路 train / pu_val / clean_val / test | `prepare_survey_splits.py` | `data/splits/<dataset>/split_<seed>/` |
| Support | 生成 clean-support 五角色制品 | `prepare_clean_support.py` | `data/clean-support.../<dataset>/split_<seed>/` |
| 计划 | 核对单元、先验、磁盘与批次 | `run_survey_pilot.py --dry-run` | 计划输出、可选的 `plan.json` |
| 跑批 | 按批训练并写 manifest / 证据包 | `run_survey_pilot.py` 或单元脚本 | `<results>/<dataset>/.../manifest.json` |
| 审计 | 检查覆盖、证据包与正式准入 | `audit_survey_batches.py` / `audit_survey_queue.py` | `audit.json / audit.md` |
| 汇总 | 生成五 seed 统计与工作簿 | `summarize_survey_results.py` / `export_survey_results.py` | `summary.json` / `survey_results.xlsx` |

## 2. 先确认版本与门禁

当前最终候选是 `survey-v3.2`；v3.0 与 v3.1 保留为历史绑定。复现已交付结果时，以该 run 的
manifest 和批次证据包里的版本身份为准，不要一律换成最新版。

| 用途 | 协议 | 配方注册表 | 比较矩阵 / profile 映射 |
|---|---|---|---|
| 当前最终流程 | `survey-v3.2` | `survey_recipe_registry_v1_2.json` | v8 / v5 |
| v3.1 缺口批 | `survey-v3.1` | `survey_recipe_registry_v1_1.json` | v7 / v4 |
| v3.0 基线 | `survey-v3.0` | `survey_recipe_registry_v1.json` | v6 / v3 |

跑批前先确认矩阵和注册表能对上（以 v3.2 为例；v3.1 换成 v1.1 / v4）：

```bash
uv run python scripts/check_survey_recipe_registry.py \
  --registry pu_toolbox/experiment/survey_recipe_registry_v1_2.json \
  --require-present --require-locked

uv run python scripts/check_survey_profile_map.py \
  --map pu_toolbox/experiment/survey_comparison_profile_map_v5.json \
  --require-present
```

两条都应通过；失败时先修矩阵/注册表，不要绕过门禁开跑。

## 3. 准备数据

### 3.1 四路 split

`prepare_survey_splits.py` 从 `data/raw/` 读取原始数据，生成协议要求的四路划分：

```bash
uv run python scripts/prepare_survey_splits.py \
  --datasets spambase,cifar10,imdb,adni,mnist,fashion_mnist,twenty_newsgroups,connect_4 \
  --seeds 0,1,2,3,4
```

默认输出到 `data/splits/`；每个 `split_<seed>/` 包含：

```text
train.npz  pu_val.npz  clean_val.npz  test.npz  split_manifest.json
```

四个 `.npz` 都保存真实标签与全局唯一 `indices`；PU 标签视图由实验层按协议现场生成，
不写进 split。`split_manifest.json` 记录角色大小、索引摘要和 `class_prior.population`；
后续驱动的类先验默认从这里读取。

如果只需要某个数据集，把 `--datasets` 改成对应名字即可。未知数据集会直接报错，不会
静默跳过。

### 3.2 clean-support 制品

只有 PULNS、LaGAM、Self-PU full-meta 需要 clean-support。每个 (dataset, seed) 生成一份
五角色制品；以 Spambase seed 0 为例：

```bash
uv run python scripts/prepare_clean_support.py \
  --dataset spambase --seed 0 --support-size 100 --support-seed 0 \
  --group-policy not_applicable --raw-dir data/raw \
  --out-dir data/clean-support/spambase/split_0 --write
```

- 规模：Spambase 100，Connect-4 / MNIST / F-MNIST / CIFAR-10 各 500；`support-seed` 与
  split seed 相同。
- `--out-dir` 必须是**不存在的新目录**；不加 `--write` 只预览，不落盘。
- 已有制品在 `data/clean-support-20261024-u5axe92t/`，可直接作为 `--support-root` 使用；
  重建时用新根目录，并保留[机器清单](data/clean_support_materialization_20261024.json)核对。

## 4. 计划一次运行

先用 `--dry-run` 看要跑多少 run、批次和 checkpoint 空间。下面的命令用当前最终版本，
跑完整矩阵的 Spambase 切片：

```bash
uv run python scripts/run_survey_pilot.py \
  --protocol survey-v3.2 \
  --recipe-registry pu_toolbox/experiment/survey_recipe_registry_v1_2.json \
  --splits data/splits \
  --support-root data/clean-support-20261024-u5axe92t \
  --results results/survey-v3.2 \
  --datasets spambase --methods upu \
  --device cuda --dry-run
```

要做最小验证，保留上面的 `--datasets spambase --methods upu`；要跑全量，去掉这两个过滤
参数。常用选项：

- `--datasets` / `--methods` / `--training-paths`：三轴取交集，只接受矩阵里真实存在的组合；
  命名未知项或交集为空会报错。
- `--plan-json <file>`：把展开后的每个 run 写成 JSON，便于两台机器核对分片不重叠。
- `--class-prior`：一般不用传，驱动会从 split 的 `class_prior.population` 读取；只有 split
  缺少先验或需要显式覆盖时才传，例如 `--class-prior spambase=0.394`。覆盖与 split 冲突时
  还要 `--allow-prior-override`。
- `--keep-going`：某批失败后继续后面的批；默认遇到失败就停。
- `--reclaim-unselected-checkpoints`：选模和 test 评分结束后回收未选中的逐 epoch 权重。

计划里有 support 单元却没给 `--support-root` 时，驱动会在任何训练开始前拒绝并点名；
此时补上制品根，或用 `--methods` 把 support 方法排除。

## 5. 执行

### 5.1 批量执行（推荐）

把上面的命令去掉 `--dry-run`：

```bash
uv run python scripts/run_survey_pilot.py \
  --protocol survey-v3.2 \
  --recipe-registry pu_toolbox/experiment/survey_recipe_registry_v1_2.json \
  --splits data/splits \
  --support-root data/clean-support-20261024-u5axe92t \
  --results results/survey-v3.2 \
  --device cuda
```

驱动按批调用 `run_survey_experiment.py`，每个批写一个证据包，并让该批的每份 manifest
引用它。结果树大致如下：

```text
<results>/<dataset>/<method>/<training_path>/<view>/[<机制>/]c_<token>/seed_<n>/manifest.json
<results>/batch_evidence/<dataset>__<method>__<training_path>__<机制或 oracle>__<标识前 12 位>.json
```

SAR run 多一层机制目录；PN oracle 没有 c 轴，使用 `c_independent`。

中途停止后重新执行同一条命令即可续跑：已完成的 run 会跳过，未完成的部分会按 seed
拆成新的批并另写证据包，不覆盖旧包。

### 5.2 单单元执行（调试或复现）

只跑一个单元、确认参数或复现某份 manifest 时，用单元脚本：

```bash
uv run python scripts/run_survey_experiment.py data/splits/spambase/split_0 \
  --protocol survey-v3.2 \
  --recipe-registry pu_toolbox/experiment/survey_recipe_registry_v1_2.json \
  --dataset spambase --method upu --training-path native_2d \
  --labeling-mechanism scar --seeds 0 --c 0.1 \
  --class-prior 0.3940447729 --device cpu \
  --out-dir results/survey-debug
```

- `--class-prior` 取 `split_manifest.json` 里的 `class_prior.population`；上面的数值只对应
  当前 Spambase split。
- support 方法还要加 `--support-dir data/clean-support-.../<dataset>/split_<seed>`。
- 单单元脚本遇到失败会立刻停止；正式批量应使用 `run_survey_pilot.py`。
- 正式批次需要 `--batch-evidence`；手工调试不传时仍可跑，但 manifest 没有 `batch` 块，
  不能当作正式批次证据。

### 5.3 从上一版队列增量补批（可选）

当前 v3.1/v3.2 的缺口批还有一个队列入口：它从上一版队列读取数据路径，只展开新增单元，
并带 canary / 失败隔离。**只有已经持有上一版队列文件时才适用**；从零复现请用上面的
`run_survey_pilot.py`。

```bash
# 准备 v3.1 队列（v3.2 则把 revision 和 previous-queue 换成 v3.1）
uv run python scripts/run_survey_backlog_campaign.py \
  --output <queue-v3.1> \
  --previous-queue <v3.0-queue.json> \
  --support-root <support-root> \
  --revision survey-v3.1

# 启动 / 查看状态
uv run python scripts/run_survey_backlog_campaign.py --output <queue-v3.1> --run
uv run python scripts/run_survey_backlog_campaign.py --output <queue-v3.1> --monitor
```

该入口要求提交干净、在独立 tmux 中运行；GPU 白名单和失败隔离由队列配置决定。关闭
编辑器不影响 tmux，但服务器重启或控制器异常退出后不能假设自动恢复。

### 5.4 失败项恢复（2026-11-02）

只对已经失败或挂起的批次使用；父批次、历史 manifest 和冻结输入保持只读。恢复批次是
替换，不是新增科学实验，不增加分母。

KLDCE 先逐项预检，再按 c 分组恢复：

```bash
uv run python scripts/prepare_kldce_recovery.py \
  --parent <旧队列目录> \
  --output <新恢复目录>
uv run python scripts/run_survey_backlog_campaign.py --output <新恢复目录> --run
```

预检只读 train 标签和特征头，不构造模型；按实际 SCAR 计数判断既有 QP 等式/盒约束与
冻结载荷上限。不可行项记 `blocked_preflight`，保留可见、不删除、不算等 GPU；可行项按
`family + c` 各有一个 canary，`c=0.1` 失败不阻断 `c=0.5`。本轮 120 项中 90 项在冻结
配方下数学不可行，30 项可行（6 个数据集 × `c=0.5` × 5 seed）。恢复批次 CPU-only，
同一父批次的替换控制器串行启动。预检通过不等于拟合成功、收敛或峰值内存批准；不要删
约束、改 h 或裁剪右端来“修复”。要改变不可行项的可行域，须另行批准版本化方法/协议变更。

GradPU 先恢复失败的 canary：

```bash
uv run python scripts/recover_survey_backlog_canary.py \
  --parent <父队列目录> --label <失败 canary> --gpu <卡> \
  --output <恢复目录>
uv run python scripts/recover_survey_backlog_canary.py --output <恢复目录> --run
```

完整 canary 验收通过后，再准备其余 14 项：

```bash
uv run python scripts/recover_survey_backlog_canary.py \
  --completed-recovery <恢复目录> --output <后续目录>
uv run python scripts/run_survey_backlog_campaign.py --output <后续目录> --run
```

要求完整 200 轮预算、选中权重和 manifest 验收通过；强制 `cpu_offload`，沿用原 GPU
范围，主机余量至少 192 GiB；运行中、失败、制品变更或部分 epoch 都不放行。恢复不改
父队列，也不自动释放 family。

监控时把父批次和恢复批次一起传入：

```bash
uv run python scripts/monitor_survey_campaign.py \
  --output <父队列目录> --output <恢复或后续目录>
```

原失败与替换尝试分开显示；`blocked_preflight` 表示数学/资源上不适用，不是等 GPU。

## 6. 审计、汇总与导出

### 6.1 队列批次（v3.1/v3.2）：先审计覆盖

如果你用 5.3 的队列入口跑 v3.1/v3.2，先用队列审计检查覆盖、重复归属和完成制品：

```bash
uv run python scripts/audit_survey_queue.py \
  --queue <queue-dir> \
  --protocol pu_toolbox/experiment/survey_protocol_v3_2.json \
  --output <new-report.json>
```

退出码 0 表示覆盖完整且完成制品可读；输出已存在时会拒绝覆盖。需要把队列结果交给别人
复核时，用：

```bash
uv run python scripts/export_survey_queue_handoff.py \
  --queue <queue-dir> --output <new-handoff-dir>
```

### 6.2 结果树批次：审计、汇总与导出

用 5.1 的 pilot 跑出的结果树时，批次审计使用一个白名单配置，列出每个结果根和应有的 manifest 数。以 v3.0 基线为例：

```json
{
  "schema_version": "survey-batch-roots-1",
  "batches": [
    {"name": "v3.0", "root": "results/survey-v3.0", "expected_manifests": 3320, "role": "formal"}
  ]
}
```

```bash
uv run python scripts/audit_survey_batches.py --config batches.json --out-dir <out>/audit
uv run python scripts/summarize_survey_results.py --config batches.json --out-dir <out>/summary
uv run python scripts/export_survey_results.py --config batches.json --out-dir <out>/delivery
```

- `audit` 只读结果树，不写回任何结果目录；先跑它，再看 `audit.md`。
- `summary` 按 comparability group / view / mechanism / method / 选模协议 / c 输出均值±标准差；
  缺 seed 只报告，不填补。
- `export` 生成 `survey_results.xlsx` 与诊断卡；不重算数字。

v3.1/v3.2 的 `summary` / `export` 还要加 `--protocol survey-v3.2`（v3.1 用 `survey-v3.1`）；
完整 D54 批次审计仍在执行计划 §1.2 待办。当前 `audit_survey_batches.py` 的
正式重裁定仍以 v3.0 canonical 为准；不要用 v3.0 的审计结果给 v3.1/v3.2 授予 formal。
这两版先以队列审计和导出的观测结果为准，正式准入仍看 formal blockers。

## 7. 复现一条已交付结果

1. 打开该 run 的 `manifest.json`，找到 `batch.evidence_file` 与 `batch.evidence_sha256`。
2. 在 `<results>/batch_evidence/` 找到同名包，核对它的字节摘要；不一致说明包被改过，
   该 run 的批次身份已不可信。
3. `git checkout <code_commit>`；先看 `code_commit_dirty`。为 `true` 时，原 run 含未提交
   改动，仅 checkout 该 commit 复现不出同一份字节，只能重建当时工作区，或承认不可逐字节复现。
4. `uv sync`；确认 `uv.lock` 的摘要与包里的 `dependency_lock.sha256` 一致。
5. 核对 `split_ref.indices_sha256` 与手上的 split（support 单元还对制品收据）。
6. 用单元脚本重放该批：按包里的 `unit.seeds` 与 `unit.c_tokens` 传参，并使用包里的
   `protocol` 与 `recipe_registry`。换机器或换 GPU 会生成新的证据包，这是另一次尝试；
   可比的是 unit / seeds / c / `protocol_sha256`，不是 `batch_id`。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `support profile requires --support-root` | 补 `--support-root`；或只用 `--methods` 选非 support 方法 |
| 缺少或矛盾的类先验 | 从 split 的 `class_prior.population` 传 `--class-prior`；冲突需 `--allow-prior-override` |
| `recipe registry changed since preflight` | 使用 manifest / 证据包里记录的同一份 registry，不要换成最新版 |
| `output exists` | 换新的输出目录；脚本不会覆盖已有结果 |
| `no manifests under ...` | 检查 `--results` / 配置里的 `root` 是否指向正确的批次根 |
| 队列控制器拒绝启动 | 真实后台调度要求 POSIX 文件锁；Windows 上只能做计划/监控，不能当正式控制器 |
| 审计或汇总报告为 partial | 先看 `audit.md` 的具体检查项；不要靠重跑或删结果“修复” |

## 9. 延伸阅读

- [实验协议](pu_survey_protocol.md)：角色、PA/OA、SAR、PN oracle 与验收线。
- [执行计划](survey_execution_plan.md)：能力、缺口、预算/资源与决策账本。
- [实验 README](README.md)：交付物定义、当前版本与最终清理计划。
- [方法卡](../method_cards/)：各方法的公式、来源差异与实现边界。
- [用户文档](../../user/howto/)：工具箱通用流程；实验层的定制接口会在实验完成后单独整理。
