# ruff: noqa: N806

"""Convert a population prior for direct native-U risk training."""

import numpy as np

from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.preprocessing import make_scar_dataset
from pu_toolbox.prior import population_to_native_unlabeled

X, y_pu, _ = make_scar_dataset(n=200, c=0.5, random_state=42)
pi_population = 0.5
pi_u = population_to_native_unlabeled(pi_population, float(np.mean(y_pu == 1)))
# UPUClassifier directly consumes the native-U prior.
model = UPUClassifier(class_prior=pi_u, fit_intercept=False, random_state=42)
model.fit(X, y_pu)
print(f"native-U prior={pi_u:.3f}; population prior={pi_population:.3f}")
print(f"predictions={len(model.predict(X))}")
