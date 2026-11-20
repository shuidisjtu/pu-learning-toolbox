# Quickstart

Install `pu-toolbox` in your Python environment. Generate and profile a small synthetic dataset:

```bash
pu-toolbox make-demo-data --out-dir demo/ --n 200 --seed 42
pu-toolbox profile --data demo/X.csv --labels demo/y_pu.csv --out-dir results/
pu-toolbox list-methods
```

PU labels use 1 for labeled positives and 0 for unlabeled observations, not known negatives.
For direct training with correct OS risk-pool prior conversion, see the [migration guide](howto/pipeline.md).
For controlled studies with clean labels in all four roles, see [custom experiments](howto/custom_experiments.md).
The [CLI guide](howto/cli.md) covers independent sensitivity, shift audit, monitor and review tools.
