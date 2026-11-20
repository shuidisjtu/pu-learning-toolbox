"""UPU (unbiased PU) classifier — requires class prior."""
# API reference: docs/user/reference/api/classifiers.md#upuclassifier-upu-alias-convex_pu

import numpy as np
from sklearn.metrics import accuracy_score

from pu_toolbox.estimators.risk import UPUClassifier
from pu_toolbox.metrics import pu_zero_one_risk
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
# The default risk uses native U; class_prior above describes the whole OS pool.
native_prior = population_to_native_unlabeled(class_prior, float(y_pu.mean()), input_sampling="os")
clf = UPUClassifier(risk_prior=native_prior, loss="squared", fit_intercept=False, random_state=42)
clf.fit(X, y_pu)

y_pred = clf.predict(X)
acc = accuracy_score(y_true, y_pred)
# Metrics on this same pool are training diagnostics, not independent validation.
risk = pu_zero_one_risk(y_pu, clf.decision_function(X), class_prior, input_sampling="os")
print(f"UPU  |  Training accuracy: {acc:.3f}  |  Training PU zero-one risk: {risk:.4f}")
