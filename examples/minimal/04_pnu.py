"""PNU classifier — uses positive, negative, AND unlabeled samples."""

import numpy as np
from sklearn.metrics import accuracy_score

from pu_toolbox.estimators.risk import PNUClassifier

# -- Generate PNU data: some labeled positive, some labeled negative, rest unlabeled --
rng = np.random.RandomState(42)
n = 150
X_pos = rng.randn(n, 5) + 2.0
X_neg = rng.randn(n, 5) - 2.0
X = np.vstack([X_pos, X_neg])
y_true = np.array([1] * n + [0] * n)
class_prior = 0.5

# Label 40% of positives as +1, 40% of negatives as -1, rest unlabeled (0)
y_pnu = np.zeros(2 * n, dtype=int)
y_pnu[:n][rng.rand(n) < 0.4] = 1    # labeled positive
y_pnu[n:][rng.rand(n) < 0.4] = -1   # labeled negative

# -- Train and evaluate --
clf = PNUClassifier(class_prior=class_prior, eta=0.0, random_state=42)
clf.fit(X, y_pnu)

y_pred = clf.predict(X)
acc = accuracy_score(y_true, y_pred)
print(f"PNU  |  Accuracy: {acc:.3f}  |  eta: {clf.eta_}")
print(f"       n_P={clf.n_positive_}, n_N={clf.n_negative_}, n_U={clf.n_unlabeled_}")
