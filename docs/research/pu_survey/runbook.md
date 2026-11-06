# 运行实验与生成报告

本文介绍官方 Survey 示例脚本的操作流程。直接调用 `ExperimentRunner`、准备四路数据及
扩展生成器、训练器和选模策略，见 [DIY 实验教程](../../user/howto/custom_experiments.md)。

在仓库根目录执行命令，使用 Python 3.11：

```powershell
uv sync --python 3.11 --extra dev --extra torch --extra report
```

新运行使用 `survey-v3.2`；旧 v3 产物按其原始记录只读汇总，不重新解释为新运行。

## 准备四路数据

`ExperimentRunner` 接收用户已经切分的 train、pu_val、clean_val、test。
工具箱也提供数据准备脚本：

```powershell
uv run python scripts/prepare_survey_splits.py --datasets spambase --seeds 0,1,2,3,4
```

默认从 `data/raw` 读取，并写入 `data/splits/<dataset>/split_<seed>/`。每份包含
`train.npz`、`pu_val.npz`、`clean_val.npz`、`test.npz`、`split_manifest.json`。
NPZ 含真实标签和互斥的全局 indices；PU 标签由生成器在训练时生成。
先验来自 split 的 `class_prior.population`，不从测试集反推。

PULNS、LaGAM、Self-PU full-meta 使用独立 clean support。例如：

```powershell
uv run python scripts/prepare_clean_support.py --dataset spambase --seed 0 --support-size 100 --support-seed 0 --group-policy not_applicable --raw-dir data/raw --out-dir data/clean-support/spambase/split_0 --write
```

support 与其他四角色互斥。Spambase 使用 100 个，Connect-4/MNIST/F-MNIST/CIFAR-10 使用 500 个；输出目录须尚不存在。去掉 `--write` 可预览。

## 执行官方示例

校验当前配方：

```powershell
uv run python scripts/check_survey_recipe_registry.py --require-present --require-locked
```

执行一个已有 split 的单元：

```powershell
uv run python scripts/run_survey_experiment.py data/splits/spambase/split_0 --protocol survey-v3.2 --recipe-registry pu_toolbox/experiment/survey_recipe_registry_v1_2.json --dataset spambase --method upu --training-path native_2d --labeling-mechanism scar --seeds 0 --c 0.1 --class-prior 0.3940447729 --device cpu --out-dir results/example
```

上述先验仅示例当前 Spambase；执行时使用自己 split 中的 `class_prior.population`。
单元入口须显式提供该值，批量入口会从 split 读取。
PN oracle 使用 `--oracle` 替代 `--method upu` 并去掉 `--class-prior`，仅 OA 选模；support 方法需要对应 `--support-dir`。
每个 seed 使用它自己的 split，不能将 `split_0` 同时作为多个 seed 的输入。

确定性网格运行与续跑沿用现有批量入口：

```powershell
uv run python scripts/run_survey_pilot.py --protocol survey-v3.2 --recipe-registry pu_toolbox/experiment/survey_recipe_registry_v1_2.json --splits data/splits --results results/example-grid --datasets spambase --methods upu --device cpu --dry-run
```

核对输出后去掉 `--dry-run`。可通过 `--datasets`、`--methods`、`--training-paths` 收窄范围，`--plan-json` 保存计划。重新执行同一请求会按实际完成的 manifest 跳过；`--keep-going` 允许记录失败后继续。需要 support 的方法须提供 `--support-root`。

批量运行共享代码/依赖凭证放在 `<results>/batch_evidence`，manifest 引用该文件；单单元入口直接记录身份。迁移结果时一并保存共享凭证和所选权重，结果读取不要求原机器的绝对路径存在。

Python 接口的四路数据构造、参数调整和策略扩展示例见 [DIY 实验教程](../../user/howto/custom_experiments.md)，精确签名见 [API 参考](../../user/reference/api.md#四角色运行与策略注入)。

## 汇总结果与导出

新运行的 manifest 在 `research_settings` 中记录采样、标记政策、风险池、训练与选择先验、
验证监督和表征。`summary.json` 保留各 seed 的完整设置，`summary.md/csv`、工作簿和
诊断卡展示摘要。历史结果显示“历史记录：按原 manifest 设置读取”，继续使用原协议身份。
字段契约见 [API 参考](../../user/reference/api.md)。

直接从 manifest 文件或目录汇总，输出到独立目录：

```powershell
uv run python scripts/summarize_survey_results.py --manifests results/example-grid --out-dir results/report
uv run python scripts/export_survey_results.py --summary results/report/summary.json --out-dir results/report
```

也可一条命令完成：

```powershell
uv run python scripts/export_survey_results.py --manifests results/example-grid --out-dir results/report
```

输入路径可有多个，重复路径只读取一次；不同文件却声明相同结果身份时拒绝，避免静默覆盖。
默认预期 seed 为 0–4；单 seed 验证可显式 `--expected-seeds 0`，报告保留单次结果。
待审核与排除路径分别用 `--pending <path...>`、`--exclude <path...>`，保留在 sources/coverage，不进入均值。
只有 summary 时，无法重新分离其已聚合的来源；先核对原汇总的审核状态。

汇总生成 `summary.json/md/csv`、`coverage.csv`、`refusals.json`；完整明细、成本、来源、选模与复现身份以 JSON 为准。
导出生成一个含四组结果页的工作簿、诊断卡和结论。缺 seed 行进入 coverage，不补成五 seed；无定义的附加指标保留可用数量与原因，成本不因 PA/OA 双行翻倍。

可选的 `--literature facts.json` 接收 `{"facts": [...]}`；每条需 fact_id、method、statement、signals、source。
可选的 `--native-views views.json` 是 `{"nnpu": "ts"}` 这类 method→os/ts/both 映射，用于解释原生假设，不改实际 run_view。缺少这些信息时不推测文献一致性或原生分组。

读取与导出只使用输入中实际记录的信息；不会重新训练、回填原 manifest 或自动批准待审核结果。
