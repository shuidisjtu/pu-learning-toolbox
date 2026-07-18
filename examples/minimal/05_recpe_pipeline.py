"""ReCPE + UPU pipeline — estimate class prior, then train classifier."""

import numpy as np
from sklearn.metrics import accuracy_score

from pu_toolbox.prior import ReCPEEstimator
from pu_toolbox.estimators.risk import UPUClassifier

# -- Generate SCAR PU data --
rng = np.random.RandomState(42)
n = 300
X_pos = rng.randn(n, 5) + 2.0
X_neg = rng.randn(n, 5) - 2.0
X = np.vstack([X_pos, X_neg])
y_true = np.array([1] * n + [0] * n)
true_prior = 0.5

labeled_mask = rng.rand(n) < 0.5
y_pu = np.zeros(2 * n, dtype=int)
y_pu[:n][labeled_mask] = 1

# -- Step 1: Estimate class prior with ReCPE --
recpe = ReCPEEstimator(copy_fraction=0.1)
recpe.fit(X, y_pu)
pi_hat = recpe.estimate()
print(f"ReCPE estimated prior: {pi_hat:.3f}  (true: {true_prior})")

# -- Step 2: Train UPU with estimated prior --
clf = UPUClassifier(class_prior=pi_hat, loss="squared", fit_intercept=False, random_state=42)
clf.fit(X, y_pu)

y_pred = clf.predict(X)
acc = accuracy_score(y_true, y_pred)
print(f"UPU (with ReCPE prior)  |  Accuracy: {acc:.3f}")
