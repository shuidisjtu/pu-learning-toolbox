"""Fixed positive counts and uniform sampling shared by PU data constructors."""

import numpy as np

from pu_toolbox.core.config import POSITIVE_LABEL
from pu_toolbox.core.random import check_random_state


def fixed_positive_count(n_positive: int, label_frequency: float) -> int:
    """Round the requested count, keeping at least one available positive."""
    if n_positive == 0:
        return 0
    return min(n_positive, max(1, int(np.round(n_positive * label_frequency))))


def sample_positive_labels(
    y_true: np.ndarray,
    n_labeled: int,
    random_state: int | np.random.RandomState | None,
) -> np.ndarray:
    """Draw uniform positive labels after the caller validates its input contract."""
    rng = check_random_state(random_state)
    labels = np.zeros(len(y_true), dtype=int)
    if n_labeled > 0:
        positive = np.flatnonzero(y_true == POSITIVE_LABEL)
        # Full selection still consumes the historical random permutation.
        labels[rng.choice(positive, size=n_labeled, replace=False)] = POSITIVE_LABEL
    return labels
