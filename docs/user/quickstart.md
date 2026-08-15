# 快速开始

> 前置条件：Python 3.10+。`pip install pu-toolbox`（torch 方法另加 `pip install "pu-toolbox[torch]"`；图形界面使用 `pip install "pu-toolbox[ui]"`）。

## 步骤 1：生成演示数据

```bash
pu-toolbox make-demo-data --out-dir demo/ --n 200 --seed 42
# 产出 demo/X.csv、demo/y_pu.csv、demo/y_true.csv
```

`--n` 为每类样本数（总 2n），`--c` 为 SCAR 标注概率（默认 0.5）。

## 步骤 2：一条命令跑完整实验

```bash
pu-toolbox run --data demo/X.csv --labels demo/y_pu.csv --out-dir results/
# auto 模式自动估先验 → 画像 → 推荐算法 → 训练 → PU 分层 CV → 评估
```

## 步骤 3：查看结果

```bash
# results/report.md   完整 Markdown 报告
# results/report.json 严格 JSON（无 NaN），可程序化消费
```

## 步骤 4：Python 最小片段

```python
from pu_toolbox import PUPipeline

pipe = PUPipeline()                  # classifier="auto"，自动选算法
report = pipe.fit_evaluate(X, y_pu)  # X: (n, d)，y_pu: {1, 0} PU 标签

print(report.summary())
report.save("results/pipeline.json")
```

## 接下来呢？

- 不想使用命令行：安装后运行 `pu-toolbox-ui`，见 [howto/ui.md](howto/ui.md)
- 调整模型或搜索参数：[howto/model_tuning.md](howto/model_tuning.md)
- 完整实验的每一步怎么调：见 [howto/pipeline.md](howto/pipeline.md)
- 想理解 PU 问题与标记机制：见 [concepts/pu_problem.md](concepts/pu_problem.md)
- 全部命令与参数：见 [howto/cli.md](howto/cli.md) 与 [reference/api.md](reference/api.md)
