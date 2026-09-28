# tests/unit/experiment/test_checkpoint_storage_profiles.py

# F811: ``survey_script`` is a pytest fixture looked up by name, so it repeats as
# both an import and a test parameter.
# ruff: noqa: N803, N806, S101, F811

"""What one per-epoch checkpoint component costs, per training path.

That figure feeds two decisions which are not symmetric: a host is sized against
the accumulated total, and a run is refused by the pre-run guard.  Both used to
read one constant for every ``resnet18*`` row, which is only right for the rows
that train the ResNet itself -- the adapter rows train a head on frozen features
and save that head.

The adapter bound is pinned against the real construction path rather than
against a hand-computed parameter count, so an architecture change has to fail
these tests instead of quietly keeping a stale constant plausible.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from _survey_script_helpers import (  # noqa: F401 - survey_script is a fixture
    make_splits,
    survey_script,
)

from pu_toolbox.experiment import resources
from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer
from pu_toolbox.experiment.manifest import load_manifest
from pu_toolbox.experiment.pilot_plan import _declared_components
from pu_toolbox.experiment.resources import checkpoint_disk_requirement
from pu_toolbox.experiment.survey_execution import (
    assemble_model,
    cached_adapter,
    prepare_image_bundle,
)
from pu_toolbox.experiment.survey_protocol import (
    ADAPTER_HEAD_COMPONENT_BYTES,
    RESNET18_COMPONENT_BYTES,
    ROLES,
    load_protocol,
    resolve_unit,
    unit_checkpoint_bytes,
    unit_checkpoint_profile,
)

pytestmark = pytest.mark.unit

#: The adapter head reads the frozen encoder's output, so every adapter row's
#: estimator is sized from this width rather than from the raw pixels.
ADAPTER_FEATURE_DIM = 512

_GIB = 1024**3


def _adapter_row(protocol: dict, method: str = "dist_pu") -> dict:
    return resolve_unit(protocol, "cifar10", method, "cnn_feature_adapter")


def _native_cnn_row(protocol: dict) -> dict:
    return resolve_unit(protocol, "cifar10", "nnpu", "native_cnn")


def _runnable_adapter_rows(protocol: dict) -> list[dict]:
    """Adapter rows that actually write per-epoch checkpoints, from the matrix.

    Read rather than listed, so a row added later is covered by these tests
    without anyone remembering to extend them.
    """
    return [
        row
        for row in protocol["execution_units"]
        if row["training_path"] == "cnn_feature_adapter"
        and row["runnable"]
        and protocol["budgets"][row["budget"]].get("epochs")
    ]


#: Discovered at collection time so a new adapter method shows up as a new
#: parameter rather than as a silent gap in coverage.
_ADAPTER_METHODS = sorted(row["method"] for row in _runnable_adapter_rows(load_protocol()))


def _cheap_epochs(protocol: dict, epochs: int = 2) -> dict:
    """The shipped matrix with the epoch caps lowered, and nothing else changed.

    The architecture, the row and the construction path are what the bound is
    evidence for, and all three stay the shipped ones.  The epoch cap only
    decides how long the serialisation takes: a component's size does not depend
    on how many epochs ran, since each epoch writes its own file.

    Every profile is lowered, not only the ones these tests build -- a method
    left at its shipped 200 epochs would quietly make this file take minutes.
    """
    for profile in protocol["method_profiles"].values():
        for key in ("epochs", "max_epochs"):
            if key in profile["params"]:
                profile["params"][key] = epochs
    protocol["method_profiles"]["self_pu"]["params"].update(
        {
            "warmup_epochs": 0,
            "self_paced_start": 0,
            "self_paced_end": 1,
            "distill_start": 1,
        }
    )
    return protocol


def _features(width: int, rows: int = 12) -> tuple[np.ndarray, np.ndarray]:
    X = np.random.RandomState(0).normal(size=(rows, width)).astype(np.float32)
    return X, np.array([1, 1, 0] * (rows // 3))


def _cifar_source(rows: int = 8) -> DatasetBundle:
    """Synthetic CIFAR-shaped splits, enough to build and step a real ResNet."""
    rng = np.random.RandomState(8)
    return DatasetBundle(
        **{
            role: DatasetPart(
                X=rng.randint(0, 256, (rows, 3, 32, 32), dtype=np.uint8),
                labels=np.array([1, 0] * (rows // 2)),
                indices=np.arange(index * rows, (index + 1) * rows),
                view="clean",
                for_selection=role != "test",
            )
            for index, role in enumerate(ROLES)
        }
    )


@pytest.fixture(scope="module")
def adapter_width(tmp_path_factory) -> int:
    """The head's input width, read back from the production adapter path.

    The constant profiles ignore ``input_dim``, so a wider frozen encoder would
    grow every real file while the estimate stayed at the published bound --
    hard-coding the width here would let the bound and its evidence drift
    together.  Taking it from the adapter cache is what makes the byte assertion
    below about the head that actually trains.
    """
    protocol = _cheap_epochs(copy.deepcopy(load_protocol()))
    prepared, encoder, image = prepare_image_bundle(_cifar_source(), protocol, 0)
    adapted, _ = cached_adapter(
        prepared,
        encoder,
        image,
        cache_dir=tmp_path_factory.mktemp("adapter"),
        batch_size=8,
        device="cpu",
    )
    return int(adapted.train.X.shape[1])


# --- the defect this file exists for -----------------------------------------


def test_basic_adapter_rows_do_not_cost_a_full_resnet():
    """A frozen encoder is never saved, so a row naming one may not cost one.

    The two rows differ in what ``fitted.model_`` holds: the native CNN builds
    ``Sequential(encoder_, head)`` and so saves the ResNet, the adapter saves the
    head it trains on top of frozen features.  One constant for both overstates
    the adapter by ~178x, which is what a host used to be sized against.
    """
    protocol = load_protocol()

    adapter = unit_checkpoint_bytes(protocol, _adapter_row(protocol), input_dim=ADAPTER_FEATURE_DIM)
    end_to_end = unit_checkpoint_bytes(
        protocol, _native_cnn_row(protocol), input_dim=ADAPTER_FEATURE_DIM
    )

    assert end_to_end == RESNET18_COMPONENT_BYTES
    assert adapter < end_to_end


# --- the bound, against what the training path really writes ------------------


@pytest.mark.parametrize("method", _ADAPTER_METHODS)
def test_basic_real_adapter_checkpoints_stay_under_the_published_bound(
    method, adapter_width, tmp_path
):
    """Serialise through the production path; assert every component, not the max.

    Checking only the largest written file would pass a run that silently
    dropped a component, so the written set is compared with the estimator's own
    declaration and every file is measured -- including both Self-PU teachers.
    """
    # The bound is evidence for one head width.  A different one means the
    # frozen encoder changed and the published byte figure is stale.
    assert adapter_width == ADAPTER_FEATURE_DIM, (
        f"the frozen encoder now emits {adapter_width}-dimensional features, not "
        f"{ADAPTER_FEATURE_DIM}: re-measure ADAPTER_HEAD_COMPONENT_BYTES"
    )
    protocol = _cheap_epochs(copy.deepcopy(load_protocol()))
    row = _adapter_row(protocol, method)
    model = assemble_model(
        protocol,
        row,
        adapter_width,
        seed=0,
        params={},
        class_prior=None if method == "pn_oracle" else 0.3,
        device="cpu",
        encoder=None,
    )
    X, y = _features(adapter_width)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trajectory = EpochCheckpointTrainer(
            supervised=method == "pn_oracle", checkpoint_dir=tmp_path
        ).fit(model, X, y)

    declared = set(getattr(model, "epoch_components", ("model",)))
    assert {checkpoint.component for checkpoint in trajectory.checkpoints} == declared
    for checkpoint in trajectory.checkpoints:
        size = Path(checkpoint.path).stat().st_size
        assert size <= ADAPTER_HEAD_COMPONENT_BYTES, (
            f"cifar10/{method}/{checkpoint.component} wrote {size} bytes, over the "
            f"{ADAPTER_HEAD_COMPONENT_BYTES}-byte bound the host is sized against"
        )


def test_basic_real_native_cnn_checkpoint_holds_the_encoder_within_the_bound(tmp_path):
    """45 MiB is evidence only while the ResNet is really in the saved component.

    A bound that passed because the run had quietly saved a head would be the
    same defect pointing the other way, so the encoder's presence is asserted
    rather than assumed from the row's backbone name.
    """
    protocol = _cheap_epochs(copy.deepcopy(load_protocol()), epochs=1)
    prepared, encoder, _ = prepare_image_bundle(_cifar_source(), protocol, 0)
    row = _native_cnn_row(protocol)
    model = assemble_model(
        protocol, row, 3, seed=0, params={}, class_prior=0.3, device="cpu", encoder=encoder
    )
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(
        model, prepared.train.X, prepared.train.labels
    )

    saved = torch.load(trajectory.checkpoints[0].path, map_location="cpu", weights_only=True)
    # model_ is Sequential(encoder_, head), so the encoder's parameters arrive
    # under a wrapper prefix; match on the suffix rather than on the prefix.
    under_wrapper = {key.split(".", 1)[1] for key in saved if "." in key}
    assert set(model.encoder_.state_dict()) <= under_wrapper
    assert Path(trajectory.checkpoints[0].path).stat().st_size <= RESNET18_COMPONENT_BYTES


# --- which profile a row gets, and what happens without one ------------------


def test_param_every_runnable_adapter_row_maps_to_a_registered_profile():
    """The set is read from the matrix: a new adapter row cannot slip through.

    A row whose backbone or model family is not registered must fail rather than
    inherit a bound measured on a different head.
    """
    protocol = load_protocol()
    rows = _runnable_adapter_rows(protocol)

    assert rows, "no runnable adapter row carries an epoch budget any more"
    profiles = {
        row["method"]: unit_checkpoint_profile(protocol, row, input_dim=ADAPTER_FEATURE_DIM)
        for row in rows
    }
    assert set(profiles.values()) == {"adapter_trainable_head"}


def test_param_an_unregistered_architecture_is_refused_with_its_context():
    """The message has to name the fields needed to add the missing profile."""
    protocol = load_protocol()

    unknown_backbone = {**_adapter_row(protocol), "backbone": "resnet34_random_frozen"}
    with pytest.raises(ValueError, match="resnet34_random_frozen"):
        unit_checkpoint_bytes(protocol, unknown_backbone, input_dim=ADAPTER_FEATURE_DIM)

    unknown_path = {**_adapter_row(protocol), "training_path": "native_3d"}
    with pytest.raises(ValueError, match="native_3d"):
        unit_checkpoint_bytes(protocol, unknown_path, input_dim=ADAPTER_FEATURE_DIM)

    # Every field a reader would need is in one message, not spread over a trace.
    message = str(
        pytest.raises(
            ValueError,
            unit_checkpoint_bytes,
            protocol,
            unknown_backbone,
            input_dim=ADAPTER_FEATURE_DIM,
        ).value
    )
    # Every field a reader needs to add the profile: dataset, method, training
    # path, backbone, model family and budget.
    for field in (
        "cifar10",
        "dist_pu",
        "cnn_feature_adapter",
        "resnet34_random_frozen",
        "mlp128",
        "fullbatch",
    ):
        assert field in message


def test_edge_a_budget_without_epochs_is_not_a_sizing_failure():
    """``None`` still means "writes nothing", which is not the same as unknown."""
    protocol = load_protocol()

    # A closed-form solve and a kernel fit keep no per-epoch state at all, so
    # neither of them needs a storage profile to answer for.
    for method in ("upu", "pusb_kernel"):
        row = resolve_unit(protocol, "cifar10", method, "cnn_feature_adapter")
        assert unit_checkpoint_bytes(protocol, row, input_dim=ADAPTER_FEATURE_DIM) is None
        assert (
            unit_checkpoint_profile(protocol, row, input_dim=ADAPTER_FEATURE_DIM)
            == "no_epoch_budget"
        )


# --- the totals a host is sized against --------------------------------------


def test_basic_full_pilot_totals_are_pinned_to_exact_bytes():
    """The shipped matrix, to the byte: a host decision is made on these.

    A tolerance here would be a hole in exactly the test that exists to stop the
    old four-times figure from coming back, so the byte counts are exact and the
    GiB readings only appear in the failure message.
    """
    from pu_toolbox.experiment.pilot_plan import estimate_checkpoint_bytes

    estimate = estimate_checkpoint_bytes(load_protocol(), input_dims={"spambase": 57, "imdb": 384})

    assert estimate["runs"] == 645
    assert estimate["units"] == 21
    writing = {label: entry for label, entry in estimate["per_unit_bytes"].items() if entry["runs"]}
    assert sum(entry["runs"] for entry in writing.values() if entry["bytes_per_run"]) == 330
    assert sum(1 for entry in writing.values() if entry["bytes_per_run"]) == 12

    assert estimate["total_bytes"] == 348_443_368_000, f"{estimate['total_bytes'] / _GIB:.2f} GiB"
    assert estimate["per_dataset_bytes"]["cifar10"] == 341_835_776_000
    assert estimate["peak_unit"] == "cifar10/nnpu/native_cnn"
    assert estimate["peak_bytes_per_run"] == 9_437_184_000
    assert estimate["guard_peak_bytes_per_run"] == 18_874_368_000


def test_determ_the_run_script_hands_the_runner_the_shared_component_size(survey_script, tmp_path):
    """The guard's number is the one this helper produced, carried through config.

    The pilot sizes a host from that figure and the runner refuses a run with it,
    so a second formula on either side is a drift nobody notices until a batch
    stops on a full disk.  Read end to end, from the script's own run.
    """
    protocol = load_protocol()
    row = resolve_unit(protocol, "spambase", "dist_pu")
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)
    out_dir = tmp_path / "out"
    # The width the script itself sizes against, so this reproduces its
    # expression rather than a number copied out of one run.
    width = survey_script.load_split_parts(data_dir)[0].X.shape[1]
    component = unit_checkpoint_bytes(protocol, row, input_dim=width)

    rc = survey_script.main(
        [
            str(data_dir),
            "--protocol",
            "survey-v1",
            "--dataset",
            "spambase",
            "--training-path",
            "native_2d",
            "--method",
            "dist_pu",
            "--model-params",
            '{"class_prior": 0.3}',
            "--c",
            "0.3",
            "--seeds",
            "0",
            "--class-prior",
            "0.3",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    manifests = list(out_dir.rglob("manifest.json"))
    assert len(manifests) == 1
    payload = load_manifest(manifests[0])["resources"]["checkpoint_disk_preflight"]
    assert payload["required_bytes"] == checkpoint_disk_requirement(
        bytes_per_component=component,
        epochs=protocol["budgets"][row["budget"]]["epochs"],
        components=len(_declared_components("dist_pu")),
        candidates=len(protocol["candidate_pool"]),
        attempts=resources.DEFAULT_CHECKPOINT_ATTEMPTS,
    )


def test_param_the_run_script_reports_an_unsized_row_without_a_traceback(
    survey_script, tmp_path, capsys, monkeypatch
):
    """The pilot refuses this before any batch; the unit script is the second line.

    A sizing failure has to arrive as an error line rather than as a traceback:
    the caller is a batch loop that would otherwise report a crashed unit for a
    configuration mistake it could have named.
    """
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    make_splits(data_dir)

    def _refuse(protocol, row, *, input_dim):
        raise ValueError(f"no checkpoint storage profile for {row['dataset']}/{row['method']}")

    # The script imports the helper inside main(), so the patch goes on the
    # module it imports from rather than on the script's namespace.
    monkeypatch.setattr("pu_toolbox.experiment.survey_protocol.unit_checkpoint_bytes", _refuse)

    rc = survey_script.main(
        [
            str(data_dir),
            "--protocol",
            "survey-v1",
            "--dataset",
            "spambase",
            "--training-path",
            "native_2d",
            "--method",
            "dist_pu",
            "--model-params",
            '{"class_prior": 0.3}',
            "--c",
            "0.3",
            "--seeds",
            "0",
            "--class-prior",
            "0.3",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "no checkpoint storage profile" in err
    assert "Traceback" not in err
    assert not list(tmp_path.rglob("manifest.json"))


def test_determ_the_pilot_and_the_run_script_size_from_one_component():
    """One helper feeds both consumers; a second copy is how they drift apart.

    The guard is the estimate's ``guard_*`` column by construction, so reproducing
    it here from the same component figure is what proves the host total and the
    refusal threshold come from the same number.
    """
    from pu_toolbox.experiment.pilot_plan import estimate_checkpoint_bytes, planned_runs

    protocol = load_protocol()
    estimate = estimate_checkpoint_bytes(
        protocol, input_dims={"spambase": 57, "imdb": 384}, runs=planned_runs(protocol)
    )
    component = unit_checkpoint_bytes(protocol, _native_cnn_row(protocol), input_dim=512)
    entry = estimate["per_unit_bytes"]["cifar10/nnpu/native_cnn"]

    assert entry["bytes_per_run"] == checkpoint_disk_requirement(
        bytes_per_component=component, epochs=200, components=1, candidates=1, attempts=1
    )
    assert entry["guard_bytes_per_run"] == checkpoint_disk_requirement(
        bytes_per_component=component, epochs=200, components=1, candidates=1, attempts=2
    )
