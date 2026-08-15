# 使用命令行接口

> 前置条件：先完成 [快速开始](../quickstart.md)（3 条命令的完整流程）。
> 概念：CLI 是 [PUPipeline](../howto/pipeline.md) 的薄封装——所有学习逻辑在库内，
> CLI 只负责参数解析、CSV 读写与错误映射。

## 快速上手（3 条命令）

```bash
# 1. 生成 SCAR 演示数据（X.csv / y_pu.csv / y_true.csv）
pu-toolbox make-demo-data --out-dir demo/ --n 200 --seed 42

# 2. 一键式全流程训练评估（auto 模式自动选算法）
pu-toolbox run --data demo/X.csv --labels demo/y_pu.csv --out-dir results/

# 3. 查看结果
#    results/report.md    完整 Markdown 报告
#    results/report.json  严格 JSON（无 NaN），可程序化消费
```

## 命令与参数

### run

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--data X.csv` | ✅ | — | 特征矩阵（行 = 样本；首行必须是表头）。也接受 `.npy` 4-D NCHW 图像数组（配 `--architecture cnn`） |
| `--labels y_pu.csv` | ✅ | — | PU 标签单列 {1, 0}（首行必须是表头） |
| `--out-dir results/` | ✅ | — | 输出目录（report.json + report.md） |
| `--true-labels` | — | — | 真值单列 {0, 1}，启用 oracle 指标（auc 等） |
| `--class-prior` | — | — | 显式类先验 (0, 1)，跳过估计 |
| `--prior-estimator` | — | `pen_l1` | `pen_l1`/`recpe`/`km1`/`km2`/`none`，也接受注册表名如 `class_prior_estimation`（别名 `cpe`/`pe`） |
| `--prior-param` | — | — | 估计器超参数，可重复（如 `--prior-param sigma=3.0 --prior-param n_centers=100`）；值自动转为 int/float/str；与估计器实例方式互斥 |
| `--classifier` | — | `auto` | 注册方法名或 `auto`（推荐器选算法） |
| `--classifier-param` | — | — | 分类器构造参数，可重复；支持 JSON 数字/布尔值/列表/对象（如 `--classifier-param reg_lambda=0.01`） |
| `--config` | — | — | 导入 UI/CLI 共用的 JSON 运行配置；显式非默认参数覆盖配置值 |
| `--architecture` | — | `mlp` | 深度算法网络架构：`mlp`（表格数据，默认）或 `cnn`（4-D NCHW 图像，需 `--classifier wconpu/infomax_pu`） |
| `--backbone` | — | `cnn13` | CNN 骨架：`cnn13`/`resnet18`/`resnet50`（仅 `--architecture cnn` 有效；mlp 下指定会报错） |
| `--device` | — | `auto` | 深度算法 torch 设备：`auto`/`cpu`/`cuda`（`auto` 在有 GPU 时自动用 CUDA） |
| `--max-epochs` | — | 估计器默认 | 深度算法训练轮数上限（仅对构造签名含 `max_epochs` 的算法生效，如 `wconpu`/`self_pu`/`nnpu`） |
| `--cv` | — | `5` | CV 折数 |
| `--metrics` | — | 默认四件套 | 逗号分隔（`pu_risk,recall,auc`） |
| `--seed` | — | `42` | 随机种子（同种子输出可复现） |
| `--save-model` | — | 关 | 额外保存 `model.pkl`（最终模型，pickle） |
| `--quiet` | — | 关 | 只打印错误，不打印 summary |

`--data` 与 `--labels` 的 CSV 首行必须是表头（列名）；首行若为数字会被拒绝并报错（防静默丢失首样本）。

### 辅助命令

- `list-methods`：列出全部注册算法（名称 / family / 是否需要先验 / 实现状态 /
  能否自动实例化）。新算法注册后自动出现。
- `list-priors`：列出 `--prior-estimator` 可用的估计器（`km1`/`km2` 映射到
  `KernelMeanPriorEstimator(variant=...)`）。
- `make-demo-data --out-dir demo/ [--n 200] [--c 0.5] [--n-features 5] [--separation 1.0] [--seed 42]`：
  用 `make_scar_dataset` 生成演示 CSV（`--n` 为每类样本数，总 2n；`--c` 为
  SCAR 标注概率；`--separation` 默认 1.0，避免强分离度下类先验估计系统性低估）。
- `profile --data X.csv --labels y_pu.csv [--true-labels y_true.csv] [--out-dir .]`：
  数据画像 + SCAR/SAR 假设诊断，写 `profile.json`（pu-workflow 环节 1）。
- `recommend --profile profile.json [--data X.csv --labels y_pu.csv] [--class-prior 0.3 | --prior-estimator recpe] [--top-k 5] [--has-gpu] [--out-dir .]`：
  算法推荐 + 类先验估计（估计需 `--data/--labels`），写 `recommendation.json`
  （pu-workflow 环节 2）。
- `sensitivity --data X.csv --labels y_pu.csv [--classifier elkan_noto] [--class-priors 0.1,...,0.9] [--out-dir .]`：
  假设敏感性分析（先验/标记倾向扫描），写 `sensitivity.json`（pu-workflow 环节 4）。
- `audit-benchmark --result-dir 结果目录 [--output audit.json]`：检查 benchmark 的必需产物、
  配置哈希、trial/seed 完整性、重复行、指标有限值，以及 official-data PU split 的样本重叠
  和目标先验一致性。失败时退出码为 1；`paper_claim=false` 与脏工作区作为警告保留。
- `skill install [--force] [--dest 目录]`：安装内置 `pu-workflow` 技能到用户级
  `~/.claude/skills/` 与 `~/.agents/skills/`（默认跳过已存在安装，`--force` 覆盖；
  详见 [启用与使用 pu-workflow Skill](using_skill.md)）。

## 深度算法与图像数据

`run` 支持两个深度算法（WConPU、InfoMax PU），需先安装可选依赖
`pip install pu-toolbox[torch]`。

- **表格数据**（默认）：`--classifier wconpu --architecture mlp`，MLP 骨架
- **图像数据**：`--data` 传 4-D NCHW float 数组（.npy 文件，如
  `benchmarks/deep_pu` 数据加载器导出的数组），配合
  `--architecture cnn --backbone cnn13|resnet18|resnet50`
- `--architecture cnn` 仅对声明了 `encoder` 参数、支持骨架注入的深度算法
  （`wconpu` / `infomax_pu`）有效；`auto`、非深度算法、或未适配的深度算法
  （如 `self_pu`）配合 `--architecture cnn` 会报错
- 深度训练较慢（WConPU 默认 100 epoch，可用 `--max-epochs` 调整），可减少 `--cv` 折数

## 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 用户/运行错误（文件缺失、标签非法、方法名无效等；stderr 输出 `error: ...`） |
| 2 | 参数语法错误（argparse） |

`audit-benchmark` 只验证产物自洽性和可追溯性，不证明实现等同于官方源码，也不替代论文
协议核对、数据授权确认或统计结论审阅。只有这些外部证据也完成后，才能升级论文复现声明。

## 类先验解析顺序

`--class-prior`、`--prior-estimator` 与 `--prior-param` 的组合遵循显式优先：

1. **`--class-prior 0.5`（显式）**：最高优先级，报告标 `source: user`，`--prior-estimator` 被忽略（二者同时给出时以显式值为准）；
2. **估计器实例携带的 prior**（Python API，`source: constructor`）；
3. **`--prior-estimator` 自动估计**（`source: estimated`）。

`--prior-param KEY=VALUE` 只作用于第 3 步的估计器构造（例如 `--prior-param sigma=3.0` 覆盖 pen_l1 的核宽度）；数值参数不接受非数值字符串（`sigma=abc` 会在估计前报错退出，而非静默降级）。

## 与 Python API 的关系

`pu-toolbox run` 等价于 `PUPipeline(classifier=..., prior_estimator=..., cv=..., metrics=..., random_state=...).fit_evaluate(X, y_pu, y_true=..., class_prior=...)`。
图形界面导出的配置可直接传给 `run --config`；配置内含参数网格时会运行 `PUTuner` 并
生成 `tuning.json`，包含多模型设置时会运行 `PUModelComparator` 并生成
`comparison.json`。手写搜索或自定义 CV splitter 仍可使用 Python API 的 `PUTuner` /
`PUPipeline`，详见[模型调整指南](model_tuning.md)。不想使用命令行可启动
[图形界面](ui.md)。

## 下一步

- Python 版完整流程（参数解析、降级语义、指标证据）：[pipeline.md](pipeline.md)
- 精确参数契约：[API 参考](../reference/api.md)
