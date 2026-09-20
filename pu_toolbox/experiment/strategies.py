"""Built-in strategies for the experiment layer (P0).

Design notes: SCAR/SAR share the fixed-count policy (protocol §2.1,
n_L = round(c·n_+), uniform without replacement); SAR-LBE matches
PU-Bench commit 2d95a19 (implementation_plan.md §2.2). The posterior
helper model is fitted on REAL labels (source train) — never on PU views.
SAR-LBE sampling pool = true positives only (S=1 ⟹ Y=1), as in PU-Bench.
Selection (protocol §2.4): OA min-max normalises scores before
thresholding on real-label val; PA only ever sees the PU val view, so
clean labels are structurally unreachable. Injectable-strategy pattern
per implementation_plan.md §1.4.
"""

# ruff: noqa: N803

from __future__ import annotations

import inspect
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from pu_toolbox.core.random import check_random_state
from pu_toolbox.utils.serialization import canonical_hash

from .bundle import DatasetPart, LabelView
from .checkpoints import record_validation, selection_models
from .protocols import Generator, SelectionProtocol, Trainer
from .tracking import EpochRecord, RunTrajectory, SelectionArtifact


def _n_labeled(y_true: np.ndarray, c: float) -> int:
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0
    return min(n_pos, max(1, int(np.round(n_pos * c))))


def _flatten_features(X: np.ndarray) -> np.ndarray:
    """Return the ``(n_samples, -1)`` view of *X* (2-D input is unchanged).

    The posterior helper is a linear model, so survey 4-D NCHW tensors must
    be flattened before fit/predict — the same view for both, mirroring
    PU-Bench ``data_utils.py`` (``features.reshape(features.shape[0], -1)``).
    """
    if X.ndim <= 2:
        return X
    return X.reshape(X.shape[0], -1)


def _fit_posterior(X: np.ndarray, y_true: np.ndarray, seed: int | None):
    """Fit the PU-Bench posterior helper on REAL labels (any input dim)."""
    return LogisticRegression(solver="lbfgs", max_iter=100, random_state=seed).fit(
        _flatten_features(X), y_true
    )


def _common_generation_metadata(
    generator, c: float, n_labeled: int, n_pos: int, seed, y_pu: np.ndarray
) -> dict:
    """Audit fields shared by SCAR / LBE-A / LBE-B metadata (manifest).

    ``n_labeled_requested`` is the protocol formula ``round(c·n₊)`` BEFORE the
    ``_n_labeled`` clamp, ``n_labeled`` the count actually applied. The two
    differ whenever ``round(c·n₊) == 0`` (small ``n₊`` or small ``c``): the
    run then marks one positive although the protocol asked for none, and the
    manifest must show both sides of that gap. ``c_requested`` is recorded
    verbatim because ``c_realized`` is the clamped, sample-size-dependent one.

    ``label_view_sha256`` digests the label view itself, which the counts
    cannot: protocol §2.4 item 4 requires every method compared under one
    (dataset, seed, c) to see the SAME P/U marking, and the digest is what
    makes that auditable after the fact.
    """
    return {
        "generator": type(generator).__name__,
        "c_requested": float(c),
        "c_realized": n_labeled / n_pos if n_pos else 0.0,
        "n_positive": n_pos,
        "n_labeled_requested": int(np.round(n_pos * c)),
        "n_labeled": n_labeled,
        "generation_seed": seed,
        "label_view_sha256": canonical_hash({"y_pu": np.asarray(y_pu).astype(int).tolist()}),
    }


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
            **_common_generation_metadata(self, c, n_labeled, n_pos, seed, y_pu),
        }


class CleanLabelGenerator(Generator):
    """PN oracle: pass the real labels through unchanged (§2.4 item 10).

    Design notes: the oracle trains on the same underlying train partition as
    the PU runs but keeps every real label, so it declares
    ``output_view = "clean"``.  PA then fails loudly in ``ProtocolPA`` instead
    of silently emitting a fake PA row, and the runner skips the PU-view
    positive check that only applies to generated views.  ``c`` is recorded
    but never applied — the oracle is c-independent by construction.
    See docs/research/pu_survey/pn_oracle_integration.md §5 (D-A).
    """

    output_view: LabelView = "clean"

    def generate(self, X, y_true, c, seed=None):
        labels = np.asarray(y_true).astype(int)
        return labels, {
            "mechanism": "pn_oracle",
            "c_realized": 1.0,
            "n_labeled": int(np.sum(labels == 1)),
            "c_requested": c,
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
            # Score the SAME flattened view the posterior was fitted on.
            scores = model.predict_proba(_flatten_features(X))[:, 1]
            scores_pos = scores[pos]
            weights = np.clip(scores_pos, 1e-9, None) ** self.k
            w1, w0 = self.smoothing
            weights = w1 * weights / weights.sum() + w0 * np.ones_like(weights) / len(weights)
            chosen_pos = _lbe_sample(scores_pos, n_labeled, rng, weights)
            y_pu[pos[chosen_pos]] = 1
        return y_pu, {
            "mechanism": "sar_lbe_a",
            "k": self.k,
            "smoothing": self.smoothing,
            "posterior_fit_on": "real_labels",
            "posterior_version": "PU-Bench 2d95a19/lbfgs(max_iter=100)",
            "scores_file": None,
            **_common_generation_metadata(self, c, n_labeled, n_pos, seed, y_pu),
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
            # Score the SAME flattened view the posterior was fitted on.
            scores = model.predict_proba(_flatten_features(X))[:, 1]
            scores_pos = scores[pos]
            weights = np.clip(1.5 + self.shrink_coef - scores_pos, 0.0, None) ** self.k
            chosen_pos = _lbe_sample(scores_pos, n_labeled, rng, weights)
            y_pu[pos[chosen_pos]] = 1
        return y_pu, {
            "mechanism": "sar_lbe_b",
            "shrink_coef": self.shrink_coef,
            "k": self.k,
            "posterior_fit_on": "real_labels",
            "posterior_version": "PU-Bench 2d95a19/lbfgs(max_iter=100)",
            "scores_file": None,
            **_common_generation_metadata(self, c, n_labeled, n_pos, seed, y_pu),
        }


# ---------------------------------------------------------------------------
# Selection protocols (protocol §2.4: OA / PA model selection on the val view)
# ---------------------------------------------------------------------------


def _best_threshold(candidates: np.ndarray, value_at) -> tuple[float, float]:
    """Grid argmax; a plateau resolves to the earliest (lowest) candidate.

    Shared by OA's accuracy scan and PA's proxy-accuracy scan so both
    protocols resolve ties identically (``selection_spec.tie_breaking`` ends
    with "...then earliest threshold").
    """
    best_thr = float(candidates[0])
    best_value = -float("inf")
    for thr in candidates:
        value = value_at(float(thr))
        if value > best_value:
            best_thr, best_value = float(thr), value
    return best_thr, best_value


def select_threshold(
    scores: np.ndarray, labels: np.ndarray, candidates: np.ndarray
) -> tuple[float, float]:
    """Pick the threshold maximizing accuracy.

    Ties resolve to the lowest threshold (first candidate encountered, i.e.
    the lowest one when ``candidates`` is non-decreasing).
    """

    def _accuracy(thr: float) -> float:
        return float(np.mean((scores >= thr).astype(int) == labels))

    return _best_threshold(candidates, _accuracy)


def proxy_accuracy(
    scores: np.ndarray, labels: np.ndarray, threshold: float, class_prior: float
) -> float:
    """Proxy accuracy, Wang et al. 2026 (ICLR) Definition 1, OS branch.

    ::

        PA(theta) = (2*pi/n'_P) * sum_{D'_P}          I(f(x) >= theta)
                  + (1/(n'_P+n'_U)) * sum_{D'_P u D'_U} I(f(x) < theta)

    ``labels`` is the PU view: 1 marks a labeled positive (``D'_P``), 0 an
    unlabeled sample (``D'_U``).  The second sum runs over EVERY validation
    sample, labeled positives included -- that is the paper's definition, not
    a transcription slip.  Both ``n'_P`` and the sample count are fixed for a
    given validation set, and with a perfect classifier the expression
    evaluates to ``ACC + pi``: a constant shift, which is exactly why
    Proposition 1 (PA orders classifiers the way ACC does) holds.

    A consequence worth stating: ``pi`` is the WEIGHT of the positive term, so
    it moves the argmax rather than merely rescaling the score.  A missing
    prior must therefore be refused at the protocol level, never defaulted.
    """
    labeled = labels == 1
    n_labeled = int(np.count_nonzero(labeled))
    above = scores >= threshold
    positive_term = 2.0 * class_prior * np.count_nonzero(above & labeled) / n_labeled
    negative_term = np.count_nonzero(~above) / len(scores)
    return float(positive_term + negative_term)


class ProtocolOA(SelectionProtocol):
    """Oracle selection on real-label validation (clean_val).

    ``class_prior`` is accepted for call-shape parity and ignored: OA scores on
    real labels, so its criterion has no pi term.  Compare :class:`ProtocolPA`,
    whose criterion does need it.

    Scores are min-max normalised to [0, 1] before thresholding, so the
    default ``np.linspace(0, 1, 11)`` grid is meaningful for any
    ``decision_function`` range. The val-side affine constants are
    recorded in ``metrics`` so the runner applies the SAME transform to
    test scores — the threshold was picked in the val-side normalised
    space; re-normalising on the test set's own min/max would apply
    different affine constants and shift the threshold (F1 fix).
    """

    name = "OA"

    def select(
        self,
        trajectories: list[RunTrajectory],
        val_part: DatasetPart,
        threshold_candidates=None,
        *,
        class_prior: float | None = None,  # ignored: OA's criterion has no pi term
    ) -> SelectionArtifact:
        if val_part.view != "clean" or not val_part.for_selection:
            raise ValueError("ProtocolOA must receive a selection-enabled clean validation view")
        if not trajectories:
            raise ValueError("OA selection requires at least one trajectory.")
        x_val, labels = val_part.X, val_part.labels
        if threshold_candidates is None:
            threshold_candidates = np.linspace(0.0, 1.0, 11)
        best_arti_cand = None
        best_min = best_scale = None
        best_epoch = best_checkpoint = None
        for i, traj in enumerate(trajectories):
            for checkpoint_index, epoch, model in selection_models(traj):
                started_at = time.perf_counter()
                # Normalise each checkpoint in its own VAL-side space.
                scores = model.decision_function(x_val)
                if scores.shape != labels.shape or not np.isfinite(scores).all():
                    raise ValueError(
                        "OA checkpoint scores must be finite and match validation labels"
                    )
                scale = np.ptp(scores)
                if not np.isfinite(scale):
                    raise ValueError("OA checkpoint validation score range is not finite")
                if scale > 0:
                    s_min = float(scores.min())
                    scores = (scores - s_min) / scale
                    affine = (s_min, float(scale))
                else:
                    # Legacy constant-score behavior remains explicit.
                    affine = (None, None)
                thr, acc = select_threshold(scores, labels, threshold_candidates)
                record_validation(
                    traj,
                    checkpoint_index,
                    "OA",
                    {
                        "val_accuracy": acc,
                        "threshold": thr,
                        "val_score_min": affine[0],
                        "val_score_scale": affine[1],
                    },
                    started_at,
                )
                cand = (acc, i, thr)
                if best_arti_cand is None or acc > best_arti_cand[0]:
                    best_arti_cand = cand
                    best_min, best_scale = affine
                    best_epoch, best_checkpoint = epoch, checkpoint_index
        _, run_idx, thr = best_arti_cand
        return SelectionArtifact(
            protocol="OA",
            run_index=run_idx,
            epoch=best_epoch,
            threshold=thr,
            metrics={
                "val_accuracy": best_arti_cand[0],
                "val_score_min": best_min,
                "val_score_scale": best_scale,
            },
            checkpoint_index=best_checkpoint,
        )


class ProtocolPA(SelectionProtocol):
    """PA: proxy-accuracy selection on the PU view only (pu_val).

    The criterion is :func:`proxy_accuracy` (Wang et al. 2026 Definition 1, OS
    branch), scanned over the same per-checkpoint min-max normalised threshold
    grid OA uses — so the runner re-emits the val-side affine constants at test
    time through the very path OA already exercises.  Clean labels stay
    structurally unreachable: the view guard below is the only gate.

    ``class_prior`` is the population class prior pi (protocol §3.1).  It is
    REQUIRED: pi weights the positive term of PA, so a run without it would
    silently select by a different criterion rather than fail loudly.
    """

    name = "PA"

    def select(
        self,
        trajectories: list[RunTrajectory],
        val_part: DatasetPart,
        threshold_candidates=None,
        *,
        class_prior: float | None = None,
    ) -> SelectionArtifact:
        if val_part.view != "pu" or not val_part.for_selection:
            raise ValueError("ProtocolPA must receive a PU view (never clean labels).")
        mask = val_part.labels == 1
        if not mask.any() or mask.all():
            raise ValueError("PA validation requires both labeled positive and unlabeled samples")
        if class_prior is None:
            raise ValueError(
                "PA selection needs the population class prior pi (protocol §3.1: "
                "data-generation metadata, never back-inferred from a subset). Pass "
                "class_prior= to select(), or --class-prior to the run."
            )
        prior = float(class_prior)
        if not np.isfinite(prior) or not 0.0 < prior <= 1.0:
            raise ValueError(f"PA class prior pi must lie in (0, 1], got {class_prior!r}")
        if not trajectories:
            raise ValueError("PA selection requires at least one trajectory")
        if threshold_candidates is None:
            threshold_candidates = np.linspace(0.0, 1.0, 11)
        labels = val_part.labels
        best_value = -float("inf")
        best_run, best_epoch, best_checkpoint = 0, None, None
        best_threshold = float(threshold_candidates[0])
        best_min = best_scale = None
        for i, traj in enumerate(trajectories):
            for checkpoint_index, epoch, model in selection_models(traj):
                started_at = time.perf_counter()
                # Normalise each checkpoint in its own VAL-side space, exactly as
                # OA does: the runner has to reuse the same affine constants when
                # it applies this threshold to the test scores.
                scores = model.decision_function(val_part.X)
                if scores.shape != labels.shape or not np.isfinite(scores).all():
                    raise ValueError(
                        "PA checkpoint scores must be finite and match validation labels"
                    )
                scale = np.ptp(scores)
                if not np.isfinite(scale):
                    raise ValueError("PA checkpoint validation score range is not finite")
                if scale > 0:
                    s_min = float(scores.min())
                    scores = (scores - s_min) / scale
                    affine = (s_min, float(scale))
                else:
                    # Constant-score checkpoint: the criterion is flat, so the
                    # tie rule (earliest candidate/epoch/threshold) decides.
                    affine = (None, None)
                threshold, value = _best_threshold(
                    threshold_candidates,
                    # ``_scores`` binds the loop variable: the lambda is called
                    # immediately, but a late-binding closure is exactly the
                    # defect B023 flags, so bind it rather than silence it.
                    lambda thr, _scores=scores: proxy_accuracy(_scores, labels, thr, prior),
                )
                record_validation(
                    traj,
                    checkpoint_index,
                    "PA",
                    {
                        "val_proxy_accuracy": value,
                        "threshold": threshold,
                        "val_score_min": affine[0],
                        "val_score_scale": affine[1],
                        "class_prior": prior,
                    },
                    started_at,
                )
                if value > best_value:
                    best_value, best_run = value, i
                    best_threshold = threshold
                    best_min, best_scale = affine
                    best_epoch, best_checkpoint = epoch, checkpoint_index
        return SelectionArtifact(
            protocol="PA",
            run_index=best_run,
            epoch=best_epoch,
            threshold=best_threshold,
            metrics={
                "val_proxy_accuracy": float(best_value),
                "val_score_min": best_min,
                "val_score_scale": best_scale,
                "class_prior": prior,
            },
            checkpoint_index=best_checkpoint,
        )


# ---------------------------------------------------------------------------
# Training strategies (protocol §2.5: single-point / deep / oracle;
# see docs/dev/experiment_layer.md §3)
# ---------------------------------------------------------------------------


class FitTrainer(Trainer):
    """Single-point trainer for classical (no-epoch) estimators.

    Design notes: classical estimators expose no per-epoch history, so a
    run is one point — the trajectory holds a single record and
    ``best_epoch`` stays None. ``class_prior`` is forwarded only when the
    estimator accepts it (TypeError-catch, sklearn duck contract) so the
    runner's candidate pool may mix prior-aware and prior-free methods.
    See implementation_plan.md §1.4.
    """

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
    """PN oracle: train on real labels (unbiased supervised baseline).

    Design notes: the oracle reuses the same single-point fit contract —
    only the label view differs from a SCAR/SAR run, so the comparison
    isolates the labeling mechanism instead of the training machinery;
    delegating to FitTrainer avoids a second fit path to maintain. See
    implementation_plan.md §1.4.
    """

    trains_on_real_labels = True

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
        return FitTrainer().fit(estimator, X, y, class_prior=class_prior)


class DeepFitTrainer(Trainer):
    """Probe-based trainer: uses validation_data + history_ when available.

    Design notes: probes the fit signature instead of requiring a deep
    interface, so one runner implementation serves both classical and
    deep estimators. When the probe passes but the implementation
    disagrees at runtime, the call degrades to a bare fit in place
    (replacement semantics — the FitTrainer fallback below is NOT
    re-run), so a broken deep path cannot silently fall back twice. See
    implementation_plan.md §1.4.
    """

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None):
        params = inspect.signature(type(estimator).fit).parameters
        validation_parameter = None
        if "pu_validation_data" in params:
            validation_parameter = "pu_validation_data"
        elif "validation_data" in params:
            validation_parameter = "validation_data"
        if val_pu is not None and validation_parameter is not None:
            kwargs = {validation_parameter: val_pu}
            if class_prior is not None:
                kwargs["class_prior"] = class_prior
            try:
                estimator.fit(X, y, **kwargs)
            except TypeError:
                # Signature probe passed but the implementation is inconsistent at
                # runtime: degrade to a bare fit — replacement (single fit), so the
                # FitTrainer fallback below is NOT re-run. Any other exception
                # (including internal training bugs) must bubble up.
                kwargs.pop(validation_parameter, None)
                estimator.fit(X, y, **kwargs)
                return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=estimator)
            hist = getattr(estimator, "history_", None)
            if isinstance(hist, dict) and "val_risk" in hist and len(hist["val_risk"]):
                if len(hist.get("epoch", [])) != len(hist["val_risk"]):
                    raise ValueError("estimator history_ epoch and val_risk lengths must match.")
                train_risk = hist.get("train_risk", hist.get("nnpu_risk"))
                epochs = []
                for position, (epoch, val_risk) in enumerate(
                    zip(hist["epoch"], hist["val_risk"], strict=True)
                ):
                    metrics = {"val_risk": float(val_risk)}
                    if train_risk is not None and position < len(train_risk):
                        metrics["train_risk"] = float(train_risk[position])
                    epochs.append(EpochRecord(epoch=int(epoch), metrics=metrics))
                return RunTrajectory(
                    epochs=epochs,
                    model=estimator,
                    best_epoch=int(np.argmin(hist["val_risk"])) + 1,
                )
            # The validation-aware fit already completed. Missing optional
            # history means a single-point trajectory, never a second fit.
            return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=estimator)
        return FitTrainer().fit(estimator, X, y, class_prior=class_prior)
