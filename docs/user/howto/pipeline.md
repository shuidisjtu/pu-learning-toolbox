# Migrating from the retired Pipeline

`PUPipeline`, `PUTuner`, `PUModelComparator`, `ShiftAwarePUPipeline`, `RunConfiguration`,
CLI `run`/`shift-run`, and the Streamlit UI are removed without a compatibility period.
Pipeline reports and their generation are also retired; historical artifacts remain historical.
There is no replacement single-table candidate evaluator.

For existing PU data, call an estimator directly and manage splits, prior estimation,
selection and evaluation explicitly. For OS input, an estimator using the native-unlabeled
risk pool needs the native-U prior, not the population prior. The example below assumes a
known population prior and SCAR labels; conversion uses the labeled fraction of the **training**
partition. Keep validation and test data separate, and do not use test truth for selection.

```python
import numpy as np
from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.prior import population_to_native_unlabeled

# X_train and y_train_pu are one OS training pool, with 1=labeled positive, 0=unlabeled.
pi_population = 0.4
q_train = float(np.mean(y_train_pu == 1))
pi_u = population_to_native_unlabeled(pi_population, q_train, input_sampling="os")
model = UPUClassifier(class_prior=pi_u, random_state=42)
model.fit(X_train, y_train_pu)
predictions = model.predict(X_test)
```

With independent TS input the native U pool is a population sample and its prior is the
population prior. For uPU on OS data, `fit(..., os_or_ts="ts", class_prior=pi_population)`
instead uses the complete training table as the marginal risk pool. That flag is not a
declaration of independent TS input; other estimators can have different risk pools and
prior targets. Recompute conversions within each training fold. See [prior semantics](../concepts/class_prior_semantics.md)
and [example 05](../../../examples/minimal/05_recpe_pipeline.py).

For controlled experiments with **clean labels in all four input roles**, use
[ExperimentRunner](custom_experiments.md): it supports a parameter grid for one model,
not multi-model comparison. Independent [CLI tools](cli.md), profiling, sensitivity,
shift audit, monitoring and review remain available.

For CLI `review`, serialize your own fitted estimator after training and match its
feature preparation at inference time:

```python
import pickle

with open("model.pkl", "wb") as output:
    pickle.dump(model, output)
```

```bash
pu-toolbox review --model model.pkl --data current.csv --query-budget 20 --out-dir review/
```

Use only trusted pickle artifacts; loading an untrusted pickle can execute code.
