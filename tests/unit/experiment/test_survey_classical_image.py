"""Current classical image configurations, frozen ResNet features and real fits."""

# ruff: noqa: F811

import json

import numpy as np
import pytest
from _survey_script_helpers import survey_script  # noqa: F401

from pu_toolbox.experiment.pilot_plan import PilotScope, planned_runs, select_execution_units
from pu_toolbox.experiment.survey_protocol import load_protocol, resolve_protocol_path, resolve_unit
from pu_toolbox.experiment.survey_recipe_registry import load_registry, validate_registry

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["puet", "rp"])
def test_param_edge_classical_image_units_require_the_feature_adapter(method):
    protocol = load_protocol(resolve_protocol_path("survey-v3.2"))
    resolve_unit(protocol, "cifar10", method, "cnn_feature_adapter")
    with pytest.raises(ValueError):
        resolve_unit(protocol, "cifar10", method, "native_cnn")


def test_basic_classical_image_scope_covers_all_dataset_seed_and_mechanism_cells():
    protocol = load_protocol(resolve_protocol_path("survey-v3.2"))
    datasets = ("cifar10", "mnist", "fashion_mnist", "adni")
    narrowed = select_execution_units(
        protocol,
        PilotScope(
            datasets=datasets, methods=("puet", "rp"), training_paths=("cnn_feature_adapter",)
        ),
    )
    runs = planned_runs(narrowed)
    assert len(runs) == len({run.key for run in runs}) == 280
    assert {(run.dataset, run.method, run.mechanism) for run in runs} == {
        (dataset, method, mechanism)
        for dataset in datasets
        for method in ("puet", "rp")
        for mechanism in ("scar", "sar_lbe_a", "sar_lbe_b")
    }
    assert all(run.training_path == "cnn_feature_adapter" for run in runs)
    assert {run.seed for run in runs} == set(range(5))


def test_determ_current_registry_preserves_classical_image_budgets_and_pending_review():
    protocol_path = resolve_protocol_path("survey-v3.2")
    p = load_protocol(protocol_path)
    registry = load_registry(protocol_path.with_name("survey_recipe_registry_v1_2.json"))
    assert not validate_registry(registry, p)
    assert p["review_status"] == "pending_collaborator_review" and p["formal_blockers"]
    for method in ("puet", "rp"):
        assert registry["profiles"][method]["recipe_lifecycle"] == "locked"


@pytest.mark.parametrize("method", ["puet", "rp"])
@pytest.mark.parametrize(
    "dataset,mechanism",
    [(d, "scar") for d in ("cifar10", "mnist", "fashion_mnist", "adni")]
    + [("cifar10", m) for m in ("sar_lbe_a", "sar_lbe_b")],
)
def test_basic_real_resnet_classical_cli_and_exact_budget(
    tmp_path, survey_script, method, dataset, mechanism, capsys
):
    data = tmp_path / "images"
    data.mkdir()
    rng = np.random.RandomState(18)
    shape = {
        "cifar10": (3, 32, 32),
        "mnist": (1, 28, 28),
        "fashion_mnist": (1, 28, 28),
        "adni": (1, 128, 128),
    }[dataset]
    for i, role in enumerate(("train", "pu_val", "clean_val", "test")):
        labels = np.arange(64) % 2
        images = rng.randint(0, 80, (64, *shape), dtype=np.uint8)
        images[labels == 1] += 150
        np.savez(data / f"{role}.npz", X=images, y=labels, indices=np.arange(64) + 64 * i)
    (data / "split_manifest.json").write_text(
        json.dumps({"dataset": dataset, "seed": 0, "class_prior": {"population": 0.5}})
    )
    output = tmp_path / "out"
    registry = resolve_protocol_path("survey-v3.2").with_name("survey_recipe_registry_v1_2.json")
    args = [
        str(data),
        "--method",
        method,
        "--protocol",
        "survey-v3.2",
        "--dataset",
        dataset,
        "--training-path",
        "cnn_feature_adapter",
        "--recipe-registry",
        str(registry),
        "--device",
        "cpu",
        "--c",
        "0.5",
        "--class-prior",
        "0.5",
        "--labeling-mechanism",
        mechanism,
        "--adapter-cache",
        str(tmp_path / "features"),
        "--out-dir",
        str(output),
    ]
    code = survey_script.main(args)
    captured = capsys.readouterr()
    assert code == 0, captured.err + captured.out
    m = json.loads(next(output.rglob("manifest.json")).read_text())
    assert m["protocol_version"] == "survey-v3.2"
    assert m["recipe_registry_version"] == "survey-recipes-v1.2"
    assert set(m["test_results"]) == ({"PA", "OA"} if mechanism == "scar" else {"OA"})
    assert not m["failures"] and not m["formal_eligible"]
    assert m["run_view"] == ("ts-compatible" if method == "puet" else "os-compatible")
    representation = m["representation"]
    adapter = representation["adapter"]
    image = representation["image_preprocessing"]
    assert adapter["encoder_training"] == "none; seeded_random_frozen"
    assert adapter["feature_dimension"] == 512
    assert adapter["feature_sha256"] == representation["feature_sha256"]
    assert adapter["encoder_state_sha256"] == image["encoder_state_sha256"]
    assert image["initialization_seed"] == 0
    assert image["backbone"]["weights"] is None
    receipt = m["candidate_runs"][0]["training_provenance"]
    assert receipt["status"] == "completed"
    assert not receipt["validation_used_for_training"]
    assert receipt["neural_epoch_count"] == 0
    assert receipt["work_started"] == receipt["work_completed"]
    if method == "puet":
        assert receipt["trees_requested"] == receipt["trees_completed"] == 100
    else:
        assert receipt["cross_fits_completed"] == 3
        assert receipt["final_refits_completed"] == 1
