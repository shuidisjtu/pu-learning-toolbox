"""Data-leakage audit gate for the traditional PU benchmark (design §7, phase A).

Two independent rule sets live here — do not conflate them:

- ``check_feature_blacklist`` guards *feature column names* (§2.2): exact
  full-name, case-insensitive match against the default blacklist plus
  caller-supplied terms.  It must NOT be applied to trial rows: legitimate
  metadata columns like ``class_prior`` / ``label_frequency`` would be
  mis-killed.
- ``check_trial_columns`` guards *trial row keys* (§2.1): any ``y_*``-style
  raw label column is blocked, and nothing else.

Gate placement discipline (runner integration): ``guard_fit_labels`` must
sit immediately above the estimator ``fit`` call it guards, and the trial
column gate must run on every row *before* it is appended or persisted —
a future refactor (e.g. multiple workers) must not bypass either point.
"""

# Dataset matrices follow sklearn's conventional X/y names.
# ruff: noqa: N803

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks._common import canonical_hash

# §2.2 default feature blacklist (exact full-name, case-insensitive).
DEFAULT_FEATURE_BLACKLIST_TERMS: tuple[str, ...] = (
    "y_true",
    "y_pu",
    "label",
    "target",
    "class_prior",
    "prior",
    "propensity",
    "selection_probability",
    "split",
    "fold",
    "is_train",
    "is_test",
)

# §2.1 trial rows must never persist raw labels: block any ``y_*`` key.
TRIAL_LABEL_COLUMN_PATTERN = re.compile(r"^y_", re.IGNORECASE)

AUDIT_RULE_VERSION = 1


class LeakageAuditError(Exception):
    """Raised when a hard audit check hits — the run must be blocked (design §4)."""


def check_feature_blacklist(column_names, *, extra_patterns=()):
    """Return blacklist hits for *column_names* (design §2.2).

    Each hit is ``{"column": name, "pattern": term, "reason": text}``.
    Matching is exact full-name and case-insensitive, so ``split_ratio``
    does not match ``split``.  Callers may add project-specific terms via
    *extra_patterns* (same matching semantics).  A column reports at most
    one hit — the first matching term wins, so the redundant defaults
    (``prior`` vs ``class_prior``) never double-report.
    """
    terms = DEFAULT_FEATURE_BLACKLIST_TERMS + tuple(extra_patterns)
    compiled = [(term, re.compile(re.escape(term), re.IGNORECASE)) for term in terms]
    hits = []
    for name in column_names:
        for term, pattern in compiled:
            if pattern.fullmatch(str(name)):
                hits.append(
                    {
                        "column": str(name),
                        "pattern": term,
                        "reason": (f"feature column {name!r} matches blacklist term {term!r}"),
                    }
                )
                break
    return hits


def check_duplicate_samples(train_X, test_X):
    """Detect rows duplicated across train/test splits (design §2.3).

    Returns ``{"n_overlap": int, "train_indices": [...], "test_indices":
    [...]}`` with ascending, deterministic index lists.  A programming error
    (non-2-D input or mismatched feature counts) raises ``ValueError``
    rather than silently reporting "no overlap".
    """
    train = np.asarray(train_X)
    test = np.asarray(test_X)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError(
            "split shapes must be 2-D with equal feature counts; "
            f"got {train.shape!r} vs {test.shape!r}"
        )
    train_index: dict[bytes, list[int]] = {}
    for i, row in enumerate(train):
        train_index.setdefault(row.tobytes(), []).append(i)
    pairs = []
    for j, row in enumerate(test):
        for i in train_index.get(row.tobytes(), ()):
            pairs.append((i, j))
    pairs.sort()
    return {
        "n_overlap": len(pairs),
        "train_indices": [i for i, _ in pairs],
        "test_indices": [j for _, j in pairs],
    }


def guard_fit_labels(y_fit, y_true):
    """Block y_true from reaching estimator fit (design §3.3, §5 item 5).

    Checks object identity and memory sharing — never value equality:
    ``label_frequency=1.0`` legitimately makes ``y_pu == y_true`` in value
    (independent allocations), which must pass.  A deep copy of y_true
    cannot be intercepted here; that residual risk is a code-review
    responsibility.  Keep this call directly above the ``fit`` call it
    guards — the guard must move with the call.
    """
    if y_fit is y_true:
        raise LeakageAuditError(
            "leakage: fit labels are the ground-truth labels (direct alias of y_true)"
        )
    if (
        isinstance(y_fit, np.ndarray)
        and isinstance(y_true, np.ndarray)
        and np.shares_memory(y_fit, y_true)
    ):
        raise LeakageAuditError(
            "leakage: fit labels share memory with the ground-truth labels (view alias of y_true)"
        )


def check_trial_columns(column_names):
    """Return the ``y_*`` raw-label column names among *column_names* (§2.1).

    Case-insensitive; only the ``y_`` prefix is checked — NOT the feature
    blacklist (trial rows legitimately carry ``class_prior`` /
    ``label_frequency`` metadata columns).
    """
    return [str(name) for name in column_names if TRIAL_LABEL_COLUMN_PATTERN.match(str(name))]


_PHASE_B_FEATURE_REASON = (
    "synthetic per-trial generation: no persistent feature-column set; "
    "takes effect in phase B when the official data line lands"
)
_PHASE_B_SPLIT_REASON = (
    "synthetic per-trial generation: no persistent train/test splits; takes effect in phase B"
)


def _resolve_seeds(config: dict, seed_set: str) -> list[int]:
    """Mirror load_config's seeds-only fallback (run_trials indexes directly)."""
    seeds = config.get(f"seed_set_{seed_set}") or config.get("seeds") or []
    return [int(s) for s in seeds]


def run_leakage_preflight(
    config: dict,
    results_dir: str | Path,
    *,
    seed_set: str = "development",
    feature_columns=None,
    split_pairs=None,
) -> dict:
    """Run the preflight audit and persist ``data_leakage_audit.json`` (design §3.1).

    Synthetic phase A scope: *feature_columns* / *split_pairs* are None on
    the production path (no persistent feature set or train/test splits),
    so both checks record ``not_applicable`` with the phase-B reason.  The
    two runner gates (y_true path, trial columns) are enforced inside the
    runner itself; the report records that enforcement.  Any hard hit
    blocks the run: the report is persisted first (reproducible cause,
    §4) and then ``LeakageAuditError`` is raised.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seeds = _resolve_seeds(config, seed_set)
    hits: list[dict] = []
    checks: dict = {}

    if feature_columns is None:
        checks["feature_blacklist"] = {
            "status": "not_applicable",
            "reason": _PHASE_B_FEATURE_REASON,
        }
    else:
        feature_hits = check_feature_blacklist(feature_columns)
        checks["feature_blacklist"] = {
            "status": "blocked" if feature_hits else "pass",
            "hits": feature_hits,
        }
        hits.extend({"check": "feature_blacklist", **h} for h in feature_hits)

    if split_pairs is None:
        checks["duplicate_samples"] = {
            "status": "not_applicable",
            "reason": _PHASE_B_SPLIT_REASON,
        }
    else:
        split_hits = []
        for pair_index, (train, test) in enumerate(split_pairs):
            overlap = check_duplicate_samples(train, test)
            if overlap["n_overlap"] == 0:
                continue
            split_hits.append(
                {
                    "check": "duplicate_samples",
                    "split_pair": pair_index,
                    "train_indices": overlap["train_indices"],
                    "test_indices": overlap["test_indices"],
                    "reason": (
                        f"{overlap['n_overlap']} duplicate row(s) shared by "
                        f"train/test split pair {pair_index}"
                    ),
                }
            )
        checks["duplicate_samples"] = {
            "status": "blocked" if split_hits else "pass",
            "hits": split_hits,
        }
        hits.extend(split_hits)

    # Runner-enforced gates: implemented and negative-tested in the runner,
    # so the preflight records them as enforced rather than unchecked.
    checks["y_true_path"] = {"status": "pass", "enforcement": "runner_gate"}
    checks["trial_label_columns"] = {"status": "pass", "enforcement": "runner_gate"}

    status = "blocked" if any(check["status"] == "blocked" for check in checks.values()) else "pass"
    report = {
        "schema_version": 1,
        "rule_version": AUDIT_RULE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config.get("protocol"),
        "config_sha256": canonical_hash(config),
        "seed_set": seed_set,
        "seeds": seeds,
        "generation_parameters": config.get("data"),
        "scenario_hash": canonical_hash(
            {"data": config.get("data"), "seed_set": seed_set, "seeds": seeds}
        ),
        "scope": "synthetic_phase_a",
        "status": status,
        "checks": checks,
        "hits": hits,
    }
    (results_dir / "data_leakage_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    if status == "blocked":
        raise LeakageAuditError(
            f"leakage preflight blocked: {len(hits)} hit(s); "
            "see data_leakage_audit.json for the reproducible cause"
        )
    return report
