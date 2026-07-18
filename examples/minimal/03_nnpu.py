"""nnPU (non-negative PU) classifier — PyTorch backend."""

import numpy as np
from sklearn.metrics import accuracy_score

from pu_toolbox.estimators.risk import NonNegativePUClassifier

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
# class_prior is passed to fit(), not the constructor
clf = NonNegativePUClassifier(max_epochs=100, random_state=42)
clf.fit(X, y_pu, class_prior=class_prior)

y_pred = clf.predict(X)
acc = accuracy_score(y_true, y_pred)
risk = clf.evaluate_pu_risk(X, y_pu)
print(f"nnPU  |  Accuracy: {acc:.3f}  |  nnPU risk: {risk:.4f}")
