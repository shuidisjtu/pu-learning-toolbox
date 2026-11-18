# Quickstart

Requires Python 3.10+. Install the package:

```bash
pip install pu-toolbox
```

For torch methods, use `pip install "pu-toolbox[torch]"`. For the graphical interface, use
`pip install "pu-toolbox[ui]"`.

## Step 1: Generate demo data

```bash
pu-toolbox make-demo-data --out-dir demo/ --n 200 --seed 42
# Writes demo/X.csv, demo/y_pu.csv, and demo/y_true.csv.
```

`--n` is the number of samples per true class (2n in total). `--c` is the requested positive-class
labeling fraction (default 0.5). This SCAR constructor selects a fixed quota uniformly without
replacement, so rounding can affect the realized fraction; it does not use independent Bernoulli
draws. The default demo provides enough labeled positives for the default five-fold CV.

## Step 2: Run the complete workflow

```bash
pu-toolbox run --data demo/X.csv --labels demo/y_pu.csv --out-dir results/
```

In auto mode, the pipeline resolves the prior, profiles the data, recommends a method, trains
with PU-stratified cross-validation, and produces evaluation reports. Prior estimation is used
when needed; the report records its source and the prior values actually consumed by training.
The unlabeled samples include both true classes. The demo's `y_true.csv` is not passed to this
training command.

## Step 3: Inspect the results

- `results/report.md`: the full Markdown report.
- `results/report.json`: strict JSON without NaN literals, suitable for programmatic use.

## Step 4: Run the Python API

After generating the demo above, run this snippet from the same working directory:

```python
import pandas as pd
from pu_toolbox import PUPipeline

X = pd.read_csv("demo/X.csv").to_numpy()
y_pu = pd.read_csv("demo/y_pu.csv")["label"].to_numpy()

pipe = PUPipeline()  # classifier="auto" selects a method.
report = pipe.fit_evaluate(X, y_pu)  # X: (n, d); y_pu: {1, 0} PU labels.

print(report.summary())
report.save("results/pipeline.json")
```

Here `1` means labeled positive and `0` means unlabeled. The default pipeline expects OS input:
a single population sample with some positives labeled. Independent P and population U require
`input_sampling="ts"`; see [input sampling](concepts/pu_sampling.md).

## Next steps

The [PU problem setting](concepts/pu_problem.md), [SCAR/SAR](concepts/scar_sar.md),
and [prior semantics](concepts/class_prior_semantics.md) explain the core concepts in English.
Continue with these task guides and consult the API reference for exact contracts.

- Describe an analysis or research task in natural language: [pu-workflow skill](howto/using_skill.md).
- Use the graphical interface: run `pu-toolbox-ui` after installing the UI extra; see [UI](howto/ui.md).
- Configure models or search parameters: [model tuning](howto/model_tuning.md).
- Control each step of the workflow: [pipeline](howto/pipeline.md).
- Look up commands and parameters: [CLI](howto/cli.md) and [API reference](reference/api/index.md).
