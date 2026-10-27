# tests/unit/experiment/test_survey_pilot_support.py

# ruff: noqa: F811

"""Clean-support units under the batch driver: where their data comes from.

The driver used to plan support runs and then have the unit script refuse them,
because a five-role profile's data is not the four-role split under ``--splits``.
These tests pin the three things that have to stay true for a shard to be honest:
a support plan without the artifacts is refused by name, a plan with them reads
``<root>/<dataset>/split_<seed>`` (by ``{seed}`` placeholder, so one command
covers the batch's seeds), and a resumed support run is held to the artifact's
digest rather than to the four-role split's.
"""

import json
from pathlib import Path

import pytest
from _survey_pilot_helpers import (  # noqa: F401 - imported fixtures are used by name
    raw_driver,
    splits_tree,
)

from pu_toolbox.experiment.pilot_plan import (
    Batch,
    batch_command,
    clean_support_methods,
    completed_runs,
    planned_runs,
    support_products,
)
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATHS, ROLES, load_protocol

pytestmark = pytest.mark.unit

SUPPORT_DIGEST = "b" * 64
FOUR_ROLE_DIGEST = "a" * 64


@pytest.fixture(scope="module")
def v3():
    return load_protocol(PROTOCOL_PATHS["survey-v3.0"])


@pytest.fixture
def mnist_matrix(raw_driver, v3):
    """The shipped matrix narrowed to one dataset, one support profile, one path."""
    return raw_driver.select_execution_units(
        v3,
        raw_driver.PilotScope(
            datasets=("mnist",), methods=("lagam",), training_paths=("native_cnn",)
        ),
    )


@pytest.fixture
def stub_protocol(raw_driver, monkeypatch, mnist_matrix):
    """Serve the narrowed matrix for whatever path ``--protocol`` resolves to."""
    monkeypatch.setattr(raw_driver, "load_protocol", lambda path: mnist_matrix)
    return mnist_matrix


def batch(*, seeds=(0,)) -> Batch:
    return Batch(
        dataset="mnist",
        method="lagam",
        training_path="native_cnn",
        mechanism="scar",
        seeds=seeds,
        tokens=("0.1",),
    )


def support_product(
    root: Path, *, seed=0, declared_seed=None, digest=SUPPORT_DIGEST, indices=True
) -> Path:
    """A product directory, optionally with a receipt that misnames its own seed."""
    directory = root / "mnist" / f"split_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "mnist",
        "seed": seed if declared_seed is None else declared_seed,
        "indices_sha256": digest,
        "role_sizes": {"train": 10, "support": 5},
    }
    if indices:
        payload["indices"] = {
            role: list(range(index * 4, index * 4 + 4)) for index, role in enumerate(ROLES)
        }
    (directory / "support_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def finished_run(results: Path, *, digest: str) -> Path:
    """A finished run of the one support unit this matrix holds."""
    path = results / "mnist" / "lagam" / "c_0.1" / "seed_0" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "execution_mode": "versioned_pilot",
                "seed": 0,
                "execution_unit": {
                    "dataset": "mnist",
                    "method": "lagam",
                    "training_path": "native_cnn",
                },
                "generation": {"train": {"mechanism": "scar"}},
                "c_requested_token": "0.1",
                "selection": {"OA": {"candidate_index": 0}},
                "run_view": "os-compatible",
                "calibration_applied": False,
                "representation": {"split_sha256": digest},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_basic_a_support_plan_without_the_artifacts_is_refused(
    raw_driver, stub_protocol, tmp_path, capsys
):
    splits = splits_tree(tmp_path / "splits", prior=0.5, datasets=("mnist",))
    code = raw_driver.main(
        [
            "--protocol",
            "survey-v3.0",
            "--splits",
            str(splits),
            "--results",
            str(tmp_path / "results"),
            "--datasets",
            "mnist",
            "--methods",
            "lagam",
            "--dry-run",
        ]
    )
    assert code == 1
    error = capsys.readouterr().err
    assert "--support-root" in error
    assert "mnist/lagam" in error


def test_basic_a_support_plan_reads_the_artifacts(raw_driver, stub_protocol, tmp_path, capsys):
    splits = splits_tree(tmp_path / "splits", prior=0.5, datasets=("mnist",))
    planned = raw_driver.planned_runs(stub_protocol)
    products = {run.seed for run in planned}
    for seed in sorted(products):
        support_product(tmp_path / "support", seed=seed)
    code = raw_driver.main(
        [
            "--protocol",
            "survey-v3.0",
            "--splits",
            str(splits),
            "--results",
            str(tmp_path / "results"),
            "--datasets",
            "mnist",
            "--methods",
            "lagam",
            "--support-root",
            str(tmp_path / "support"),
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert f"support: {len(planned)} run(s)" in out
    assert f"{len(products)} five-role product(s)" in out
    assert "pending:   " in out


def test_param_one_command_covers_the_batchs_seeds():
    argv = batch_command(
        batch(seeds=(0, 2)),
        splits_root=Path("data/splits"),
        out_root=Path("results"),
        protocol="survey-v3.0",
        support_root=Path("/products"),
    )
    assert argv[0] == str(Path("/products/mnist/split_{seed}"))
    assert argv[argv.index("--support-dir") + 1] == argv[0]
    assert argv[argv.index("--split-ref") + 1] == str(
        Path("/products/mnist/split_{seed}/support_manifest.json")
    )
    assert argv[argv.index("--seeds") + 1] == "0,2"


def test_param_a_four_role_batch_keeps_the_splits_root():
    argv = batch_command(
        batch(),
        splits_root=Path("data/splits"),
        out_root=Path("results"),
        protocol="survey-v3.0",
    )
    assert argv[0] == str(Path("data/splits/mnist"))
    assert "--support-dir" not in argv
    assert "--split-ref" not in argv


def test_edge_a_receipt_naming_another_seed_is_refused(tmp_path):
    support_product(tmp_path / "support", seed=0, declared_seed=1)
    with pytest.raises(ValueError, match="not 'mnist'/0"):
        support_products(tmp_path / "support", {("mnist", 0)})


def test_edge_a_receipt_without_the_four_role_indices_is_refused(tmp_path):
    """Without them the run's own digest cannot be recomputed, so it cannot be resumed."""
    support_product(tmp_path / "support", indices=False)
    with pytest.raises(ValueError, match="no four-role indices"):
        support_products(tmp_path / "support", {("mnist", 0)})


def test_edge_a_receipt_without_its_own_digest_is_refused(tmp_path):
    support_product(tmp_path / "support", digest="")
    with pytest.raises(ValueError, match="no receipt digest"):
        support_products(tmp_path / "support", {("mnist", 0)})


def test_edge_a_missing_product_is_refused_by_path(tmp_path):
    with pytest.raises(ValueError, match="unreadable at"):
        support_products(tmp_path / "support", {("mnist", 0)})


def test_basic_a_scar_only_profile_plans_no_sar_cell(v3):
    """The support adapters refuse SAR outright, so no batch may be bought for it.

    Declaring the restriction on the profile is what makes the plan and the
    comparison expansion agree with the runner: before it, the plan held four SAR
    cells per support unit, and the first real driver batch aborted with
    "exploratory support adapter supports SCAR only".
    """
    planned = planned_runs(v3)
    support_mechanisms = {
        run.mechanism for run in planned if run.method in ("lagam", "pulns", "self_pu_full_meta")
    }
    assert support_mechanisms == {"scar"}
    # The no-meta ablation is a separate profile and does run the SAR cells.
    assert {run.mechanism for run in planned if run.method == "self_pu"} >= {
        "scar",
        "sar_lbe_a",
    }
    assert len(planned) == 3320


def test_determ_each_kind_of_run_is_held_to_its_own_product(v3, tmp_path):
    """A support manifest on the four-role digest is not evidence of completion."""
    on_product = finished_run(tmp_path / "on_product", digest=SUPPORT_DIGEST)
    on_split = finished_run(tmp_path / "on_split", digest=FOUR_ROLE_DIGEST)
    support_methods = clean_support_methods(v3)
    assert "lagam" in support_methods and "nnpu" not in support_methods
    support_splits = {("mnist", 0): SUPPORT_DIGEST}
    splits = {("mnist", 0): FOUR_ROLE_DIGEST}

    kept = completed_runs(
        on_product.parent,
        splits=splits,
        support_splits=support_splits,
        support_methods=support_methods,
    )
    dropped = completed_runs(
        on_split.parent,
        splits=splits,
        support_splits=support_splits,
        support_methods=support_methods,
    )
    assert len(kept) == 1
    assert dropped == {}
