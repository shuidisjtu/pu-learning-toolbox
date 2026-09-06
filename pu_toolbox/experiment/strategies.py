"""Built-in strategies for the experiment layer (P0).

Design notes: SCAR/SAR share the fixed-count policy (protocol §2.1,
n_L = round(c·n_+), uniform without replacement); SAR-LBE matches
PU-Bench commit 2d95a19 (implementation_plan.md §2.2). The posterior
helper model is fitted on REAL labels (source train) — never on PU views.
SAR-LBE sampling pool = true positives only (S=1 ⟹ Y=1), as in PU-Bench.
Selection (protocol §2.4): OA min-max normalises scores before
thresholding on real-label val; PA only ever sees the PU val view, so
clean labels are structurally unreachable.
"""

# ruff: noqa: N803

from __future__ import annotations

import inspect

import numpy as np
from sklearn.linear_model import LogisticRegression

from pu_toolbox.core.random import check_random_state

from .bundle import DatasetPart
from .protocols import Generator, SelectionProtocol, Trainer
from .tracking import EpochRecord, RunTrajectory, SelectionArtifact


def _n_labeled(y_true: np.ndarray, c: float) -> int:
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0
    return min(n_pos, max(1, int(np.round(n_pos * c))))


def _fit_posterior(X: np.ndarray, y_true: np.ndarray, seed: int | None):
    return LogisticRegression(solver="lbfgs", max_iter=100, random_state=seed).fit(X, y_true)


class SCARGenerator(Generator):
    """Fixed-count SCAR (protocol §2.1): uniform sampling without replacement."""

    def generate(self, X, y_true, c, seed=None):
        rng = check_random_state(seed)
        n_labeled = _n_labeled(y_true, c)
        pos = np.where(y_true == 1)[0]
        y_pu = np.zeros(len(y_true), dtype=int)
        if n_labeled > 0:
            y_pu[rng.choice(pos, size=n_labeled, replace=False)] = 1
        n_pos = int(len(pos))
        return y_pu, {
            "mechanism": "scar",
            "c_realized": n_labeled / n_pos if n_pos else 0.0,
            "n_labeled": n_labeled,
        }


def _lbe_sample(scores: np.ndarray, n_labeled: int, rng, weights: np.ndarray) -> np.ndarray:
    """Weighted sampling without replacement used by both LBE variants.

    ``scores``/``weights`` are the positive-subset arrays; returned indices
    are offsets within that subset (caller maps them back to global indices).
    """
    if n_labeled >= len(scores):
        return np.arange(len(scores))
    if np.all(weights == 0):
        return rng.choice(len(scores), size=n_labeled, replace=False)
    return rng.choice(len(scores), size=n_labeled, replace=False, p=weights / weights.sum())


class SARLBEAGenerator(Generator):
    """LBE-A: p ∝ scores**k with smoothing p = 0.9p + 0.1·uniform (PU-Bench 2d95a19).

    Sampling pool = true positives only (S=1 ⟹ Y=1); uniform term is
    1/len(pos) as in PU-Bench ``uniform_p``.
    """

    def __init__(self, k: float = 10, smoothing: tuple[float, float] = (0.9, 0.1)) -> None:
        self.k = k
        self.smoothing = smoothing

    def generate(self, X, y_true, c, seed=None):
        rng = check_random_state(seed)
        n_labeled = _n_labeled(y_true, c)
        pos = np.where(y_true == 1)[0]
        n_pos = int(len(pos))
        y_pu = np.zeros(len(y_true), dtype=int)
        if n_pos > 0:
            model = _fit_posterior(X, y_true, seed)
            scores = model.predict_proba(X)[:, 1]
            scores_pos = scores[pos]
            weights = np.clip(scores_pos, 1e-9, None) ** self.k
            w1, w0 = self.smoothing
            weights = w1 * weights / weights.sum() + w0 * np.ones_like(weights) / len(weights)
            chosen_pos = _lbe_sample(scores_pos, n_labeled, rng, weights)
            y_pu[pos[chosen_pos]] = 1
        return y_pu, {
            "mechanism": "sar_lbe_a",
            "c_realized": n_labeled / n_pos if n_pos else 0.0,
            "k": self.k,
            "smoothing": self.smoothing,
            "posterior_fit_on": "real_labels",
            "posterior_version": "PU-Bench 2d95a19/lbfgs(max_iter=100)",
            "scores_file": None,
        }


class SARLBEBGenerator(Generator):
    """LBE-B: p ∝ (1.5 + shrink_coef - scores)**k, negative clipped, uniform fallback.

    Sampling pool = true positives only (S=1 ⟹ Y=1).
    """

    def __init__(self, shrink_coef: float = 1.0, k: float = 10) -> None:
        self.shrink_coef = shrink_coef
        self.k = k

    def generate(self, X, y_true, c, seed=None):
        rng = check_random_state(seed)
        n_labeled = _n_labeled(y_true, c)
        pos = np.where(y_true == 1)[0]
        n_pos = int(len(pos))
        y_pu = np.zeros(len(y_true), dtype=int)
        if n_pos > 0:
            model = _fit_posterior(X, y_true, seed)
            scores = model.predict_proba(X)[:, 1]
            scores_pos = scores[pos]
            weights = np.clip(1.5 + self.shrink_coef - scores_pos, 0.0, None) ** self.k
            chosen_pos = _lbe_sample(scores_pos, n_labeled, rng, weights)
            y_pu[pos[chosen_pos]] = 1
        return y_pu, {
            "mechanism": "sar_lbe_b",
            "c_realized": n_labeled / n_pos if n_pos else 0.0,
            "shrink_coef": self.shrink_coef,
            "k": self.k,
            "posterior_fit_on": "real_labels",
            "posterior_version": "PU-Bench 2d95a19/lbfgs(max_iter=100)",
            "scores_file": None,
        }


# ---------------------------------------------------------------------------
# Selection protocols (protocol §2.4: OA / PA model selection on the val view)
# ---------------------------------------------------------------------------


def select_threshold(
    scores: np.ndarray, labels: np.ndarray, candidates: np.ndarray
) -> tuple[float, float]:
    """Pick the threshold maximizing accuracy.

    Ties resolve to the lowest threshold (first candidate encountered, i.e.
    the lowest one when ``candidates`` is non-decreasing).
    """
    best_thr, best_acc = candidates[0], -1.0
    for thr in candidates:
        pred = (scores >= thr).astype(int)
        acc = float(np.mean(pred == labels))
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return float(best_thr), best_acc


class ProtocolOA(SelectionProtocol):
    """Oracle selection on real-label validation (clean_val).

    Scores are min-max normalised to [0, 1] before thresholding, so the
    default ``np.linspace(0, 1, 11)`` grid is meaningful for any
    ``decision_function`` range.
    """

    name = "OA"

    def select(
        self, trajectories: list[RunTrajectory], val_part: DatasetPart, threshold_candidates=None
    ) -> SelectionArtifact:
        x_val, labels = val_part.X, val_part.labels
        if threshold_candidates is None:
            threshold_candidates = np.linspace(0.0, 1.0, 11)
        best_arti_cand = None
        for i, traj in enumerate(trajectories):
            model = traj.model
            # normalise scores to [0,1] via decision_function min-max
            scores = model.decision_function(x_val)
            scale = np.ptp(scores)
            if scale > 0:
                scores = (scores - scores.min()) / scale
            thr, acc = select_threshold(scores, labels, threshold_candidates)
            cand = (acc, i, thr)
            if best_arti_cand is None or acc > best_arti_cand[0]:
                best_arti_cand = cand
        _, run_idx, thr = best_arti_cand
        return SelectionArtifact(
            protocol="OA",
            run_index=run_idx,
            epoch=trajectories[run_idx].best_epoch,
            threshold=thr,
            metrics={"val_accuracy": best_arti_cand[0]},
        )


class ProtocolPA(SelectionProtocol):
    """PA: selection on the PU view only (pu_val, also keeps real labels away)."""

    name = "PA"

    def select(
        self, trajectories: list[RunTrajectory], val_part: DatasetPart, threshold_candidates=None
    ) -> SelectionArtifact:
        if val_part.view != "pu":
            raise ValueError("ProtocolPA must receive a PU view (never clean labels).")
        if threshold_candidates is None:
            threshold_candidates = np.linspace(0.0, 1.0, 11)
        # PA uses unlabeled count + labeled-positive risk proxy; keep it simple:
        # pick the trajectory with best mean PU-view separation on val.
        best_run, best_score = 0, -1.0
        for i, traj in enumerate(trajectories):
            scores = traj.model.decision_function(val_part.X)
            mask = val_part.labels == 1
            if mask.sum() == 0:
                continue
            sep = float(np.mean(scores[mask]) - np.mean(scores[~mask]))
            if sep > best_score:
                best_score, best_run = sep, i
        return SelectionArtifact(
            protocol="PA",
            run_index=best_run,
            epoch=trajectories[best_run].best_epoch,
            threshold=None,
            metrics={"pu_val_separation": best_score},
        )


# ---------------------------------------------------------------------------
# Training strategies (protocol §2.5 + spec §6.3: single-point / deep / oracle)
# ---------------------------------------------------------------------------


class FitTrainer(Trainer):
    """Single-point trainer for classical (no-epoch) estimators."""

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
        if class_prior is not None:
            try:
                estimator.fit(X, y, class_prior=class_prior)
            except TypeError:
                estimator.fit(X, y)
        else:
            estimator.fit(X, y)
        return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=estimator)


class SupervisedTrainer(Trainer):
    """PN oracle: train on real labels (unbiased supervised baseline)."""

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
        return FitTrainer().fit(estimator, X, y, class_prior=class_prior)


class DeepFitTrainer(Trainer):
    """Probe-based trainer: uses validation_data + history_ when available."""

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
        params = inspect.signature(type(estimator).fit).parameters
        if val_pu is not None and "validation_data" in params:
            kwargs = {"validation_data": val_pu}
            if class_prior is not None:
                kwargs["class_prior"] = class_prior
            try:
                estimator.fit(X, y, **kwargs)
                hist = getattr(estimator, "history_", None)
                if isinstance(hist, dict) and "val_risk" in hist and len(hist["val_risk"]):
                    epochs = [
                        EpochRecord(
                            epoch=int(e),
                            metrics={"val_risk": float(v), "train_risk": float(r)},
                        )
                        for e, v, r in zip(
                            hist["epoch"], hist["val_risk"], hist["nnpu_risk"], strict=False
                        )
                    ]
                    return RunTrajectory(
                        epochs=epochs,
                        model=estimator,
                        best_epoch=int(np.argmin(hist["val_risk"])) + 1,
                    )
            except TypeError:
                pass
        return FitTrainer().fit(estimator, X, y, class_prior=class_prior)
