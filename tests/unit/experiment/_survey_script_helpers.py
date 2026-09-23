# tests/unit/experiment/_survey_script_helpers.py

# ruff: noqa: N803, S101

"""Shared fixtures/helpers for the ``scripts/run_survey_experiment.py`` tests.

Not collected by pytest (``python_files = ["test_*.py"]``): importing
``survey_script``/``make_splits``/``make_sar_splits`` into a test module
registers the fixture there, so both script test modules exercise the same
script source without duplicating the loader.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/run_survey_experiment.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_survey_experiment", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses resolves field annotations through
    # sys.modules[cls.__module__], which the spec loader alone does not populate
    # (stdlib documented idiom, https://docs.python.org/3/library/importlib.html).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def survey_script():
    return _load_script_module()


def make_splits(data_dir: Path) -> None:
    """Four-part bundle with every role containing real positives; global ids 0..29."""
    rng = np.random.RandomState(0)
    x = rng.randn(30, 3)
    # train(0-17): 6 pos + 12 neg; pu_val(18-21): 2 pos; clean_val(22-25): 2 pos;
    # test(26-29): 2 pos + 2 neg (so AUC is defined).
    y = np.array([1] * 6 + [0] * 12 + [1] * 2 + [0] * 2 + [1] * 2 + [0] * 2 + [1] * 2 + [0] * 2)
    for name in ("train", "pu_val", "clean_val", "test"):
        start, end = {
            "train": (0, 18),
            "pu_val": (18, 22),
            "clean_val": (22, 26),
            "test": (26, 30),
        }[name]
        np.savez(
            data_dir / f"{name}.npz",
            X=x[start:end],
            y=y[start:end],
            indices=np.arange(start, end),
        )
    # Real split products carry a manifest, and §3.1 puts pi in it.  A method
    # that needs no prior still has one for PA to select with, which is exactly
    # the path a run without --class-prior is supposed to take.
    (data_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "spambase",
                "seed": 0,
                "role_sizes": {"train": 18, "pu_val": 4, "clean_val": 4, "test": 4},
                "class_prior": {"population": float(np.mean(y))},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def make_sar_splits(data_dir: Path) -> None:
    """Four-part bundle sized for the SAR c values {0.05, 0.5}; global ids 0..119.

    The PU estimators need at least two labeled positives, which the six-positive
    ``make_splits`` bundle cannot provide at c=0.05 (40 positives at c=0.5 / 10 at
    c=0.05 keep both SAR c values runnable).  c=0.05 also exercises both sides of
    the ``_n_labeled`` clamp: the train role asks for round(40*0.05)=2 labels and
    gets 2, while pu_val asks for round(10*0.05)=0 and gets the clamped 1
    (``c_realized``=0.1).

    Labels follow the positives-first layout of ``make_splits``:
    train(0-59): 40 pos + 20 neg; pu_val/clean_val/test: 10 pos + 10 neg each.
    """
    rng = np.random.RandomState(0)
    x = rng.normal(size=(120, 5))
    y = np.array(
        [1] * 40 + [0] * 20 + [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10
    )
    for name, (start, end) in (
        ("train", (0, 60)),
        ("pu_val", (60, 80)),
        ("clean_val", (80, 100)),
        ("test", (100, 120)),
    ):
        np.savez(
            data_dir / f"{name}.npz",
            X=x[start:end],
            y=y[start:end],
            indices=np.arange(start, end),
        )


def run_manifests(out_dir: Path) -> list[Path]:
    """Every run manifest under ``out_dir``, in a stable order.

    Runs are found by searching, not by rebuilding their path: the script owns
    the directory layout (view / mechanism / c / seed), and a test that spells
    the whole layout out breaks on every layout change without telling us
    anything about behaviour.  Aggregation and the pilot driver locate runs the
    same way.
    """
    return sorted(Path(out_dir).rglob("manifest.json"))


def run_directory(out_dir: Path, *segments: str) -> Path:
    """The run directory whose path ends with ``segments``."""
    return run_manifest(out_dir, *segments).parent


def run_manifest(out_dir: Path, *segments: str) -> Path:
    """The manifest of the single run whose path ends with ``segments``.

    Callers name only the segments they actually chose in the command
    (``"c_0.3"``, ``"seed_0"``); the rest of the layout stays the script's
    business.
    """
    wanted = tuple(segments)
    matches = [
        path
        for path in run_manifests(out_dir)
        if tuple(path.parts[-len(wanted) - 1 : -1]) == wanted
    ]
    assert len(matches) == 1, f"expected one run matching {wanted}, found {matches}"
    return matches[0]
