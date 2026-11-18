"""nnPU (non-negative PU) classifier — PyTorch backend."""
# API reference: docs/user/reference/api/classifiers.md#nonnegativepuclassifier-nnpu-alias-nn-pu

import numpy as np
from sklearn.metrics import accuracy_score

from pu_toolbox.estimators.risk import NonNegativePUClassifier
from pu_toolbox.prior import population_to_native_unlabeled

# -- Generate SCAR PU data --
rng = np.random.RandomState(42)
n = 200
X_pos = rng.randn(n, 5) + 2.0
X_neg = rng.randn(n, 5) - 2.0
X = np.vstack([X_pos, X_neg])
y_true = np.array([1] * n + [0] * n)
class_prior = 0.5

labeled_mask = rng.rand(n) < 0.5
y_pu = np.zeros(2 * n, dtype=int)
y_pu[:n][labeled_mask] = 1

# -- Train and evaluate --
# The default risk consumes native-U prior, supplied here through fit().
# The constructor also accepts class_prior.
native_prior = population_to_native_unlabeled(class_prior, float(y_pu.mean()), input_sampling="os")
clf = NonNegativePUClassifier(max_epochs=100, random_state=42, device="cpu")
clf.fit(X, y_pu, class_prior=native_prior)

y_pred = clf.predict(X)
acc = accuracy_score(y_true, y_pred)
risk = clf.evaluate_pu_risk(X, y_pu)
print(f"nnPU  |  Training accuracy: {acc:.3f}  |  Training nnPU risk: {risk:.4f}")
