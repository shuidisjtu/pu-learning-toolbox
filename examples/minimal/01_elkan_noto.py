"""Elkan-Noto classifier — no class prior needed."""

import numpy as np
from sklearn.metrics import accuracy_score

from pu_toolbox.estimators.classic import ElkanNotoClassifier

# -- Generate SCAR PU data --
rng = np.random.RandomState(42)
n = 200
X_pos = rng.randn(n, 5) + 2.0
X_neg = rng.randn(n, 5) - 2.0
X = np.vstack([X_pos, X_neg])
y_true = np.array([1] * n + [0] * n)

# SCAR labeling: 50% of positives are labeled
labeled_mask = rng.rand(n) < 0.5
y_pu = np.zeros(2 * n, dtype=int)
y_pu[:n][labeled_mask] = 1

# -- Train and evaluate --
clf = ElkanNotoClassifier(random_state=42)
clf.fit(X, y_pu)

y_pred = clf.predict(X)
acc = accuracy_score(y_true, y_pred)
print(f"Elkan-Noto  |  Accuracy: {acc:.3f}  |  Estimated c: {clf.propensity_:.3f}")
