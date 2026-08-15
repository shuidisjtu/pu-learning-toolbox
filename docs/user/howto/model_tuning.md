# 调整模型与搜索超参数

`PUPipeline` 允许按注册名配置分类器参数；`PUTuner` 在相同的 PU 分层验证、类先验和
指标语义上比较参数组合。参数与最终选择都会进入结果记录，便于复查。

## 固定模型参数

```python
from pu_toolbox import PUPipeline

report = PUPipeline(
    classifier="upu",
    classifier_params={
        "loss": "logistic",
        "reg_lambda": 0.01,
        "max_iter": 2000,
    },
    cv=5,
).fit_evaluate(X, y_pu, class_prior=0.4)
```

原来需要额外必填参数的方法也可以按名称使用：

```python
pipe = PUPipeline(
    classifier="ldce",
    classifier_params={"flip_probability": 0.2},
)
```

`class_prior`、`random_state` 和 CNN `encoder` 由流水线管理，不能放入
`classifier_params`；请使用 `fit_evaluate(class_prior=...)`、`random_state=...` 和
`architecture` / `backbone` 参数。

命令行使用可重复的 `--classifier-param KEY=VALUE`。值支持 JSON 数字、布尔值、
列表和对象，普通字符串可以不加引号：

```bash
pu-toolbox run \
  --data demo/X.csv --labels demo/y_pu.csv --out-dir results/ \
  --classifier upu --class-prior 0.4 \
  --classifier-param loss=logistic \
  --classifier-param reg_lambda=0.01 \
  --classifier-param max_iter=2000
```

## PU-aware 网格搜索

```python
from pu_toolbox.model_selection import PUTuner

tuner = PUTuner(
    classifier="upu",
    param_grid={
        "loss": ["double_hinge", "logistic"],
        "reg_lambda": [0.001, 0.01, 0.1],
    },
    scoring="pu_zero_one_risk",
    cv=5,
    random_state=42,
)
result = tuner.fit(X, y_pu, class_prior=0.4)

print(result.best_params)
print(result.best_score)
best_model = result.best_report.final_model
```

默认对 `pu_zero_one_risk` 取最小值，对其他指标取最大值；可用
`higher_is_better=` 显式覆盖。每个失败或指标不可用的组合会保留在
`result.trials` 中，只有所有组合都不可用时才抛出 `PipelineError`。

如果选择 `pu_auc_roc`、`pu_accuracy` 或 `pu_f1`，必须向 `fit` 提供
`y_true`。没有真实标签时优先使用 `pu_zero_one_risk`，并检查类先验估计是否可靠。

## 图形界面

图形界面包含固定参数编辑器和网格搜索面板，不需要编写 Python。安装与操作见
[图形界面](ui.md)。
