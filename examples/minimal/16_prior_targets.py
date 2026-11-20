# ruff: noqa: N806

"""Convert a population prior for direct native-U risk training."""

import numpy as np

from pu_toolbox.core.random import check_random_state
from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.preprocessing import GaussianSourceConfig, PULabelingConfig, simulate_pu_dataset
from pu_toolbox.prior import population_to_native_unlabeled

rng = check_random_state(42)
result = simulate_pu_dataset(
    GaussianSourceConfig(400, 5, 0.5, 4.0, rng, row_order="class_blocked"),
    PULabelingConfig("scar", "fixed_count", rng, c=0.5),
)
X, y_pu = result.observations.features, result.observations.labels
pi_population = 0.5
pi_u = population_to_native_unlabeled(pi_population, float(np.mean(y_pu == 1)))
# UPUClassifier directly consumes the native-U prior.
model = UPUClassifier(class_prior=pi_u, fit_intercept=False, random_state=42)
model.fit(X, y_pu)
print(f"native-U prior={pi_u:.3f}; population prior={pi_population:.3f}")
print(f"predictions={len(model.predict(X))}")
