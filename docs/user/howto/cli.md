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
| `--data X.csv` | ✅ | — | 特征矩阵（行 = 样本；首行必须是表头） |
| `--labels y_pu.csv` | ✅ | — | PU 标签单列 {1, 0}（首行必须是表头） |
| `--out-dir results/` | ✅ | — | 输出目录（report.json + report.md） |
| `--true-labels` | — | — | 真值单列 {0, 1}，启用 oracle 指标（auc 等） |
| `--class-prior` | — | — | 显式类先验 (0, 1)，跳过估计 |
| `--prior-estimator` | — | `recpe` | `recpe`/`pen_l1`/`km1`/`km2`/`none`，也接受注册表名如 `class_prior_estimation`（别名 `cpe`/`pe`） |
| `--classifier` | — | `auto` | 注册方法名或 `auto`（推荐器选算法） |
| `--cv` | — | `5` | CV 折数 |
| `--metrics` | — | 默认四件套 | 逗号分隔（`pu_risk,recall,auc`） |
| `--seed` | — | `42` | 随机种子（同种子输出可复现） |
| `--save-model` | — | 关 | 额外保存 `model.pkl`（最终模型，pickle） |
| `--quiet` | — | 关 | 只打印错误，不打印 summary |

`--data` 与 `--labels` 的 CSV 首行都必须是表头（列名），否则首行数据会被当作表头解析。

### 辅助命令

- `list-methods`：列出全部注册算法（名称 / family / 是否需要先验 / 实现状态 /
  能否自动实例化）。新算法注册后自动出现。
- `list-priors`：列出 `--prior-estimator` 可用的估计器（`km1`/`km2` 映射到
  `KernelMeanPriorEstimator(variant=...)`）。
- `make-demo-data --out-dir demo/ [--n 200] [--c 0.5] [--n-features 5] [--separation 4.0] [--seed 42]`：
  用 `make_scar_dataset` 生成演示 CSV（`--n` 为每类样本数，总 2n；`--c` 为
  SCAR 标注概率）。

## 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 用户/运行错误（文件缺失、标签非法、方法名无效等；stderr 输出 `error: ...`） |
| 2 | 参数语法错误（argparse） |

## 与 Python API 的关系

`pu-toolbox run` 等价于 `PUPipeline(classifier=..., prior_estimator=..., cv=..., metrics=..., random_state=...).fit_evaluate(X, y_pu, y_true=..., class_prior=...)`。
需要更细控制（传入分类器实例、自定义 CV splitter）时直接用 Python API。

## 下一步

- Python 版完整流程（参数解析、降级语义、指标证据）：[pipeline.md](pipeline.md)
- 精确参数契约：[API 参考](../reference/api.md)
